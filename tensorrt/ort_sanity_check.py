"""
ort_sanity_check.py — ONNX Runtime Baseline Sanity Check
=========================================================
Confirms whether output divergence is unique to TensorRT or also present in
the ONNX Runtime execution path.

Runs the same fixed inputs through:
  1. PyTorch FP32 (the ground truth)
  2. ONNX Runtime CUDA EP   (if onnxruntime-gpu is installed)
  3. ONNX Runtime CPU EP    (always available as fallback)

and reports cosine similarity + L2 error for each backend vs. PyTorch FP32.

If PyTorch ≈ ORT-CUDA ≈ ORT-CPU → the ONNX graph is correct; problem is TRT.
If PyTorch ≠ ORT-CPU             → there is an export-level bug (eval mode,
                                   dynamic op, non-traceable branch, etc.)

Key differences vs. diag_precision.py:
  • Does NOT require a compiled TensorRT engine
  • Exercises both CUDA and CPU ORT execution providers independently
  • Runs multiple seeds and reports per-seed + aggregate statistics
  • Prints a clear verdict table with provider-level notes
  • Installs onnxruntime-gpu automatically if missing (opt-in via --auto_install)

Usage:
  # Minimal (no TRT engine needed):
  python tensorrt/ort_sanity_check.py \\
      --onnx      ./checkpoints/onnx_models/fmt.onnx \\
      --ckpt_path ./checkpoints/float.pth

  # Multi-seed sweep to stress-test numerical stability:
  python tensorrt/ort_sanity_check.py \\
      --onnx      ./checkpoints/onnx_models/fmt.onnx \\
      --ckpt_path ./checkpoints/float.pth \\
      --seeds 42 7 123 999 1234

  # Auto-install onnxruntime-gpu if CUDA EP is missing:
  python tensorrt/ort_sanity_check.py \\
      --onnx ./checkpoints/onnx_models/fmt.onnx \\
      --ckpt_path ./checkpoints/float.pth \\
      --auto_install_gpu

  # Skip PyTorch baseline (faster, only needs onnx file):
  python tensorrt/ort_sanity_check.py \\
      --onnx ./checkpoints/onnx_models/fmt.onnx \\
      --skip_pytorch
"""

from __future__ import annotations

import os
import sys

# Pre-load CUDA and cuDNN libraries to prevent segmentation faults and linker issues
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import _cuda_patch

import json
import argparse
import subprocess
import textwrap
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# ── resolve project root ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(script_dir, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _fmt_utils import (
    load_fmt_wrapper, build_dummy_inputs, add_model_args, cosine_check,
)

# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime provider helpers
# ─────────────────────────────────────────────────────────────────────────────
_ORT_CUDA_EP   = "CUDAExecutionProvider"
_ORT_CPU_EP    = "CPUExecutionProvider"
_ORT_AZURE_EP  = "AzureExecutionProvider"   # proxy — not a real compute EP

INPUT_NAMES = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]

# Default CFG scale values injected when the ONNX was exported with --cfg_as_inputs
_CFG_DEFAULTS = {
    "a_cfg":       2.0,
    "e_cfg":       1.0,
    "a_cfg_scale": 2.0,
    "e_cfg_scale": 1.0,
    "r_cfg_scale": 1.0,
}


def _import_ort():
    """Import onnxruntime; raise ImportError with install hint if missing."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed.\n"
            "  CPU-only : pip install onnxruntime\n"
            "  GPU      : pip install onnxruntime-gpu\n"
        ) from exc


def _has_cuda_ep(ort) -> bool:
    """True if the installed onnxruntime has a real CUDA execution provider."""
    providers = ort.get_available_providers()
    return _ORT_CUDA_EP in providers


def _try_install_ort_gpu() -> bool:
    """Attempt to pip-install onnxruntime-gpu. Return True on success."""
    print("  Attempting: pip install onnxruntime-gpu …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "onnxruntime-gpu"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  onnxruntime-gpu installed successfully ✓")
        return True
    print(f"  [WARN] pip install failed:\n{result.stderr[:500]}")
    return False


def _make_ort_session(onnx_path: str, providers: list, ort) -> object:
    """
    Create an ONNX Runtime InferenceSession with the given provider list.
    Silences provider-not-found warnings from ORT's verbose logger.
    """
    so = ort.SessionOptions()
    so.log_severity_level = 3   # ERROR only (suppress INFO/WARNING from ORT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    actual = sess.get_providers()
    # Report the graph input names — important for cfg_as_inputs detection
    graph_inputs = [inp.name for inp in sess.get_inputs()]
    print(f"    Active providers  : {actual}")
    print(f"    Graph input names : {graph_inputs}")
    return sess


def _run_ort(sess, dummy_inputs: tuple) -> torch.Tensor:
    """
    Feed numpy arrays to ORT session; return float32 CPU tensor.

    Handles both standard 7-input exports and --cfg_as_inputs exports
    (which add a_cfg_scale and/or e_cfg_scale as extra graph nodes)
    by auto-detecting the required graph inputs from the session metadata
    and supplying default CFG values for any extra scalar inputs.
    """
    # Get the actual graph input names from the ONNX session
    graph_input_names = [inp.name for inp in sess.get_inputs()]

    # Build a feed dict for the standard inputs
    feeds: dict[str, np.ndarray] = {}
    for i, name in enumerate(INPUT_NAMES):
        if name in graph_input_names:
            feeds[name] = dummy_inputs[i].cpu().float().numpy()

    # Supply any extra CFG scalar inputs with default values
    for extra_name, default_val in _CFG_DEFAULTS.items():
        if extra_name in graph_input_names and extra_name not in feeds:
            feeds[extra_name] = np.array([default_val], dtype=np.float32)
            # Only print once per session (first call)

    # Warn if any required input is still missing
    missing = [n for n in graph_input_names if n not in feeds]
    if missing:
        raise ValueError(
            f"Required inputs {missing} are missing from input feed ({list(feeds.keys())}). "
            f"ONNX graph inputs: {graph_input_names}. "
            f"Add them to _CFG_DEFAULTS in ort_sanity_check.py if they are scalar constants."
        )

    out = sess.run(None, feeds)[0]
    return torch.from_numpy(out).float()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics / reporting
# ─────────────────────────────────────────────────────────────────────────────
_VERDICT_THRESHOLDS = {
    # (cos_min, rel_max) → verdict tag
    (0.9999, 0.001): "✅ NEAR-LOSSLESS",
    (0.999,  0.01):  "✅ EXCELLENT",
    (0.99,   0.05):  "⚠️  GOOD (minor rounding)",
    (0.95,   0.30):  "⚠️  ACCEPTABLE (FP16-class noise)",
    (0.0,    1e9):   "❌ FAIL (significant divergence)",
}


def _verdict(cos: float, rel: float) -> str:
    for (cos_min, rel_max), tag in _VERDICT_THRESHOLDS.items():
        if cos >= cos_min and rel <= rel_max:
            return tag
    return "❌ FAIL"


def _print_comparison(label_a: str, label_b: str,
                       a: Optional[torch.Tensor],
                       b: Optional[torch.Tensor]) -> Optional[dict]:
    """Print a formatted comparison row and return the metrics dict."""
    if a is None or b is None:
        print(f"\n  {label_a}  vs  {label_b}  ── SKIPPED (one output is None)")
        return None

    r = cosine_check(a, b)
    v = _verdict(r["cosine"], r["rel_l2"])

    width = 54
    print(f"\n  {'─' * width}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"  {v}")
    print(f"  {'─' * width}")
    print(f"    Cosine similarity  : {r['cosine']:.10f}  (target ≥ 0.9999)")
    print(f"    Relative L2 error  : {r['rel_l2']:.10f}  (target ≤ 0.001)")
    print(f"    Max absolute error : {r['max_abs_err']:.8f}")
    allclose = torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    print(f"    torch.allclose     : {allclose}  (atol=1e-3, rtol=1e-3)")
    print(f"    Norm [{label_a:10s}]  : {a.norm():.6f}")
    print(f"    Norm [{label_b:10s}]  : {b.norm():.6f}")
    return r


def _aggregate(records: list[dict], label: str):
    """Print aggregate stats over multiple seeds."""
    if not records:
        return
    cos_arr = np.array([r["cosine"]     for r in records])
    rel_arr = np.array([r["rel_l2"]     for r in records])
    mae_arr = np.array([r["max_abs_err"] for r in records])
    print(f"\n  Aggregate [{label}] over {len(records)} seeds:")
    print(f"    cosine  — mean={cos_arr.mean():.8f}  min={cos_arr.min():.8f}  max={cos_arr.max():.8f}")
    print(f"    rel_l2  — mean={rel_arr.mean():.8f}  min={rel_arr.min():.8f}  max={rel_arr.max():.8f}")
    print(f"    max_abs — mean={mae_arr.mean():.6f}  max={mae_arr.max():.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Provider-level diagnostic notes
# ─────────────────────────────────────────────────────────────────────────────
def _print_provider_notes(ort_cuda_available: bool, ort_cpu_cos: Optional[float],
                           ort_cuda_cos: Optional[float], pt_ort_cpu_cos: Optional[float]):
    print("\n" + "=" * 60)
    print("Diagnosis & Provider Notes")
    print("=" * 60)

    # Stage 1 — ORT-CPU (export correctness, most reliable baseline)
    print("\n  [Stage 1] ONNX Export Validation  (PyTorch vs ORT-CPU)")
    if pt_ort_cpu_cos is None:
        print("  ⚠️  Could not run (pytorch or ORT-CPU unavailable)")
    elif pt_ort_cpu_cos >= 0.9999:
        print("  ✅ Export is numerically correct (cosine ≥ 0.9999).")
        print("     The ONNX graph faithfully reproduces PyTorch FP32.")
        print("     Any divergence in TensorRT is introduced during compilation.")
    elif pt_ort_cpu_cos >= 0.999:
        print(f"  ⚠️  Minor export discrepancy (cosine={pt_ort_cpu_cos:.6f}).")
        print("     Likely harmless (ORT uses slightly different op kernels).")
        print("     Check: constant folding, opset ops, eval() mode.")
    else:
        print(f"  ❌ Significant export discrepancy (cosine={pt_ort_cpu_cos:.6f}).")
        print("     Likely causes:")
        print("       1. Model not in eval() mode — BatchNorm/Dropout active.")
        print("       2. Data-dependent branching not captured in the ONNX trace.")
        print("       3. A custom op or in-place op breaking the tracer.")
        print("     Fix: review export_onnx.py — ensure wrapper.eval() + no_grad().")

    # Stage 2 — ORT-CUDA vs ORT-CPU (cuda kernel parity)
    if ort_cuda_available and ort_cuda_cos is not None and ort_cpu_cos is not None:
        print("\n  [Stage 2] ORT-CUDA vs ORT-CPU Provider Parity")
        if ort_cuda_cos >= 0.9999:
            print("  ✅ CUDA and CPU ORT providers produce identical outputs.")
        elif ort_cuda_cos >= 0.999:
            print(f"  ⚠️  Small CUDA/CPU divergence (cosine={ort_cuda_cos:.8f}) — normal for FP32 cuBLAS vs. CPU BLAS.")
        else:
            print(f"  ❌ Large CUDA/CPU divergence (cosine={ort_cuda_cos:.6f}).")
            print("     This may indicate a bug in the ORT CUDA kernel for an op in this model.")
            print("     Try: ORT_CUDA_GEMM_VERSION=0 to force legacy cuBLAS path.")

    # Stage 3 — TRT isolation message
    print("\n  [Stage 3] TensorRT Isolation")
    if pt_ort_cpu_cos is not None and pt_ort_cpu_cos >= 0.9999:
        print("  The ONNX graph is verified correct.  If TRT output diverges:")
        print("  → Run diag_precision.py to quantify TRT error against ORT-CPU.")
        print("  → Known TRT 10.3 regression: adaLN Slice/Gather INT64 chain")
        print("    compounds error across all 8 transformer blocks (cosine ~0.981).")
        print("  → Recommended fix: upgrade TRT to ≥10.7 or use --disable_cudnn_tactic")
        print("    + --force_fp32_layers in build_engine.py for strict FP32 testing.")

    # ORT-GPU install hint
    if not ort_cuda_available:
        print("\n  ℹ️  CUDA Execution Provider not available in this ORT install.")
        print("     To benchmark with ORT-CUDA, install the GPU build:")
        print("       pip install onnxruntime-gpu")
        print("     (Note: onnxruntime-gpu replaces onnxruntime — uninstall first.)")
        print("     Re-run with --auto_install_gpu to do this automatically.")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="ONNX Runtime sanity check — isolate divergence from TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          # Quick check (no TRT engine needed):
          python tensorrt/ort_sanity_check.py \\
              --onnx ./checkpoints/onnx_models/fmt.onnx \\
              --ckpt_path ./checkpoints/float.pth

          # Multi-seed stability sweep:
          python tensorrt/ort_sanity_check.py \\
              --onnx ./checkpoints/onnx_models/fmt.onnx \\
              --ckpt_path ./checkpoints/float.pth \\
              --seeds 42 7 123 999

          # Skip PyTorch reference (faster):
          python tensorrt/ort_sanity_check.py \\
              --onnx ./checkpoints/onnx_models/fmt.onnx \\
              --skip_pytorch
        """),
    )
    p = add_model_args(p)

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--onnx",      type=str, default="./checkpoints/onnx_models/fmt.onnx",
                   help="Path to the ONNX file produced by export_onnx.py")
    p.add_argument("--ckpt_path", type=str, default="./checkpoints/float.pth",
                   help="FLOAT checkpoint path (for PyTorch FP32 reference)")

    # ── run control ───────────────────────────────────────────────────────────
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="List of random seeds to test (default: [42]). "
                        "Multiple seeds stress-test numerical stability.")
    p.add_argument("--skip_pytorch", action="store_true",
                   help="Skip PyTorch FP32 baseline (no checkpoint needed, faster).")
    p.add_argument("--skip_cpu_ep",  action="store_true",
                   help="Skip ONNX Runtime CPU execution provider.")
    p.add_argument("--skip_cuda_ep", action="store_true",
                   help="Skip ONNX Runtime CUDA execution provider even if available.")

    # ── provider install ──────────────────────────────────────────────────────
    p.add_argument("--auto_install_gpu", action="store_true",
                   help="Auto-install onnxruntime-gpu via pip if CUDA EP is missing.")

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to write JSON results file.")

    return p.parse_known_args()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.rank = device

    print("=" * 60)
    print("FLOAT FMT — ONNX Runtime Sanity Check")
    print("=" * 60)
    print(f"  ONNX file    : {args.onnx}")
    print(f"  Checkpoint   : {args.ckpt_path if not args.skip_pytorch else '(skipped)'}")
    print(f"  Device       : {device}")
    print(f"  Seeds        : {args.seeds}")

    # ── validate ONNX file ────────────────────────────────────────────────────
    if not os.path.isfile(args.onnx):
        print(f"\n[ERROR] ONNX file not found: {args.onnx}")
        print("  Run export_onnx.py first:")
        print("    python tensorrt/export_onnx.py --ckpt_path ./checkpoints/float.pth")
        sys.exit(1)

    # ── import ORT ────────────────────────────────────────────────────────────
    ort = _import_ort()
    print(f"\n  onnxruntime  : {ort.__version__}")

    ort_cuda_available = _has_cuda_ep(ort)
    print(f"  ORT CUDA EP  : {'available ✓' if ort_cuda_available else 'NOT available'}")
    print(f"  ORT CPU EP   : available ✓")
    print(f"  All providers: {ort.get_available_providers()}")

    # ── optionally install onnxruntime-gpu ────────────────────────────────────
    if not ort_cuda_available and args.auto_install_gpu:
        print("\n  CUDA EP missing — attempting auto-install …")
        ok = _try_install_ort_gpu()
        if ok:
            # Re-import after install
            import importlib
            import onnxruntime as ort_new  # noqa: F401
            ort = ort_new
            ort_cuda_available = _has_cuda_ep(ort)
            print(f"  ORT CUDA EP after install: {'available ✓' if ort_cuda_available else 'still NOT available'}")

    # ── load PyTorch wrapper once ─────────────────────────────────────────────
    wrapper = None
    if not args.skip_pytorch:
        if not os.path.isfile(args.ckpt_path):
            print(f"\n[WARN] Checkpoint not found: {args.ckpt_path} — skipping PyTorch baseline.")
            args.skip_pytorch = True
        else:
            print(f"\n[1/3] Loading PyTorch model …")
            wrapper = load_fmt_wrapper(args, device)
            wrapper.eval()

    # ── build ORT sessions (once) ─────────────────────────────────────────────
    use_cuda_ep = ort_cuda_available and not args.skip_cuda_ep
    use_cpu_ep  = not args.skip_cpu_ep

    ort_cuda_sess = None
    ort_cpu_sess  = None

    print(f"\n[2/3] Creating ORT sessions …")
    if use_cuda_ep:
        print("  ORT-CUDA session:")
        try:
            ort_cuda_sess = _make_ort_session(
                args.onnx,
                [_ORT_CUDA_EP, _ORT_CPU_EP],   # CUDA with CPU fallback
                ort,
            )
        except Exception as e:
            print(f"  [WARN] ORT-CUDA session creation failed: {e}")
            ort_cuda_sess = None
    else:
        reason = "disabled by --skip_cuda_ep" if args.skip_cuda_ep else "CUDA EP not installed"
        print(f"  ORT-CUDA session: SKIPPED ({reason})")

    if use_cpu_ep:
        print("  ORT-CPU session:")
        ort_cpu_sess = _make_ort_session(args.onnx, [_ORT_CPU_EP], ort)
    else:
        print("  ORT-CPU session: SKIPPED (--skip_cpu_ep)")

    # ── multi-seed loop ───────────────────────────────────────────────────────
    print(f"\n[3/3] Running inference across {len(args.seeds)} seed(s) …")

    all_results = []
    pt_vs_ort_cpu_records  = []
    pt_vs_ort_cuda_records = []
    ort_cpu_vs_cuda_records = []

    for seed in args.seeds:
        print(f"\n{'─'*60}")
        print(f"  Seed {seed}")
        print(f"{'─'*60}")

        dummy = build_dummy_inputs(args, device, batch=1, seed=seed)

        # PyTorch
        pt_out = None
        if wrapper is not None:
            with torch.no_grad():
                pt_out = wrapper(*dummy).cpu().float()
            print(f"  PyTorch FP32     : shape={tuple(pt_out.shape)}  norm={pt_out.norm():.6f}")

        # ORT-CUDA
        ort_cuda_out = None
        if ort_cuda_sess is not None:
            try:
                ort_cuda_out = _run_ort(ort_cuda_sess, dummy)
                print(f"  ORT-CUDA         : shape={tuple(ort_cuda_out.shape)}  norm={ort_cuda_out.norm():.6f}")
            except Exception as e:
                print(f"  ORT-CUDA         : [FAIL] {e}")

        # ORT-CPU
        ort_cpu_out = None
        if ort_cpu_sess is not None:
            try:
                ort_cpu_out = _run_ort(ort_cpu_sess, dummy)
                print(f"  ORT-CPU          : shape={tuple(ort_cpu_out.shape)}  norm={ort_cpu_out.norm():.6f}")
            except Exception as e:
                print(f"  ORT-CPU          : [FAIL] {e}")

        # ── comparisons ───────────────────────────────────────────────────────
        r_pt_cpu  = _print_comparison("PyTorch",   "ORT-CPU",  pt_out, ort_cpu_out)
        r_pt_cuda = _print_comparison("PyTorch",   "ORT-CUDA", pt_out, ort_cuda_out)
        r_cpu_gpu = _print_comparison("ORT-CPU",   "ORT-CUDA", ort_cpu_out, ort_cuda_out)

        seed_rec = {
            "seed": seed,
            "pt_vs_ort_cpu":  r_pt_cpu,
            "pt_vs_ort_cuda": r_pt_cuda,
            "ort_cpu_vs_cuda": r_cpu_gpu,
        }
        all_results.append(seed_rec)

        if r_pt_cpu  is not None: pt_vs_ort_cpu_records.append(r_pt_cpu)
        if r_pt_cuda is not None: pt_vs_ort_cuda_records.append(r_pt_cuda)
        if r_cpu_gpu is not None: ort_cpu_vs_cuda_records.append(r_cpu_gpu)

    # ── aggregate stats ───────────────────────────────────────────────────────
    if len(args.seeds) > 1:
        print("\n" + "=" * 60)
        print("Aggregate Statistics")
        print("=" * 60)
        _aggregate(pt_vs_ort_cpu_records,   "PyTorch vs ORT-CPU")
        _aggregate(pt_vs_ort_cuda_records,  "PyTorch vs ORT-CUDA")
        _aggregate(ort_cpu_vs_cuda_records, "ORT-CPU vs ORT-CUDA")

    # ── provider-level diagnosis ───────────────────────────────────────────────
    pt_ort_cpu_cos  = pt_vs_ort_cpu_records[-1]["cosine"]  if pt_vs_ort_cpu_records  else None
    ort_cpu_cos     = ort_cpu_vs_cuda_records[-1]["cosine"] if ort_cpu_vs_cuda_records else None
    ort_cuda_cos    = pt_vs_ort_cuda_records[-1]["cosine"] if pt_vs_ort_cuda_records  else None

    _print_provider_notes(ort_cuda_available, ort_cpu_cos, ort_cuda_cos, pt_ort_cpu_cos)

    # ── optional JSON output ───────────────────────────────────────────────────
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "onnx_path":           args.onnx,
                "ort_version":         ort.__version__,
                "ort_cuda_available":  ort_cuda_available,
                "seeds":               args.seeds,
                "results":             all_results,
            }, f, indent=2)
        print(f"\n  Results saved: {args.output_json}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
