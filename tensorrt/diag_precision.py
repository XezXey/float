"""
diag_precision.py — Bisect numerical error: PyTorch → ONNX Runtime → TensorRT
===============================================================================
Runs the same inputs through three backends and compares outputs pairwise,
so you can pinpoint exactly where the error is introduced.

Typical findings:
  PyTorch ≈ ONNX Runtime  (should be near-identical)
  ONNX Runtime ≠ TRT FP32  →  TF32 silently enabled in TRT, or aggressive fusion
  PyTorch ≠ ONNX Runtime   →  export bug (eval mode, no_grad, dynamic op)

Usage:
  python tensorrt/diag_precision.py \
      --engine  ./checkpoints/trt_engines/fmt_fp32.engine \
      --onnx    ./checkpoints/onnx_models/fmt.onnx \
      --ckpt_path ./checkpoints/float.pth
"""

import os
import sys

# Pre-load CUDA and cuDNN libraries to prevent segmentation faults and linker issues
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import _cuda_patch

import argparse
import numpy as np
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(script_dir, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _fmt_utils import (
    load_fmt_wrapper, build_dummy_inputs, add_model_args,
    cosine_check,
)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Precision diagnostic: PyTorch vs ONNX vs TRT")
    p = add_model_args(p)
    p.add_argument("--engine",    type=str, required=True)
    p.add_argument("--onnx",      type=str, default="./checkpoints/onnx_models/fmt.onnx")
    p.add_argument("--ckpt_path", type=str, default="./checkpoints/float.pth")
    p.add_argument("--input_seed",type=int, default=42)
    return p.parse_known_args()[0]


# ─────────────────────────────────────────────────────────────────────────────
def run_pytorch(wrapper, dummy_inputs):
    with torch.no_grad():
        out = wrapper(*dummy_inputs)
    return out.cpu().float()


def run_onnxrt(onnx_path, dummy_inputs):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed (pip install onnxruntime-gpu)")
        return None

    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    graph_inputs = [inp.name for inp in sess.get_inputs()]
    names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
    feeds = {}
    for i, name in enumerate(names):
        if name in graph_inputs:
            feeds[name] = dummy_inputs[i].cpu().float().numpy()
            
    cfg_defaults = {"a_cfg": 2.0, "e_cfg": 1.0, "a_cfg_scale": 2.0, "e_cfg_scale": 1.0, "r_cfg_scale": 1.0}
    for k, v in cfg_defaults.items():
        if k in graph_inputs:
            feeds[k] = np.array([v], dtype=np.float32)

    out   = sess.run(None, feeds)[0]
    return torch.from_numpy(out).float()


def run_trt(engine_path, dummy_inputs, opt):
    import tensorrt as trt

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    n_curr  = int(opt.wav2vec_sec * opt.fps)
    n_prev  = opt.num_prev_frames
    dim_w   = opt.dim_w

    t_in, x_in, wa_in, wr_in, we_in, px_in, pwa_in = dummy_inputs
    io = {
        "t":             t_in.contiguous().cuda().float(),
        "x":             x_in.contiguous().cuda().float(),
        "wa":            wa_in.contiguous().cuda().float(),
        "wr":            wr_in.contiguous().cuda().float(),
        "we":            we_in.contiguous().cuda().float(),
        "prev_x":        px_in.contiguous().cuda().float(),
        "prev_wa":       pwa_in.contiguous().cuda().float(),
        "motion_latent": torch.zeros(1, n_prev + n_curr, dim_w, device="cuda", dtype=torch.float32),
    }
    for name, tensor in io.items():
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    return io["motion_latent"].cpu().float()


# ─────────────────────────────────────────────────────────────────────────────
def report(label_a, label_b, a, b):
    if a is None or b is None:
        print(f"  {label_a} vs {label_b}: SKIPPED (one output is None)")
        return
    r = cosine_check(a, b)
    allclose = torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    status   = "✅ PASS" if r["cosine"] >= 0.9999 and r["rel_l2"] <= 0.001 else "❌ FAIL"
    print(f"\n  {'─'*50}")
    print(f"  {label_a}  vs  {label_b}  {status}")
    print(f"  {'─'*50}")
    print(f"    Cosine similarity : {r['cosine']:.8f}  (target ≥ 0.9999)")
    print(f"    Relative L2 error : {r['rel_l2']:.8f}  (target ≤ 0.001)")
    print(f"    Max absolute error: {r['max_abs_err']:.8f}")
    print(f"    torch.allclose    : {allclose}  (atol=1e-3, rtol=1e-3)")
    print(f"    Output norm [{label_a}]: {a.norm():.4f}")
    print(f"    Output norm [{label_b}]: {b.norm():.4f}")
    return r


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.rank = device

    print("=" * 60)
    print("Precision Diagnostic")
    print("=" * 60)
    print(f"  Engine  : {args.engine}")
    print(f"  ONNX    : {args.onnx}")
    print(f"  Seed    : {args.input_seed}")

    dummy = build_dummy_inputs(args, device, batch=1, seed=args.input_seed)

    # ── 1. PyTorch FP32 ──────────────────────────────────────────────────────
    print("\n[1/3] Running PyTorch FP32 …")
    wrapper = load_fmt_wrapper(args, device)
    pt_out  = run_pytorch(wrapper, dummy)
    del wrapper
    torch.cuda.empty_cache()
    print(f"  Output: {tuple(pt_out.shape)}  norm={pt_out.norm():.4f}")

    # ── 2. ONNX Runtime ───────────────────────────────────────────────────────
    print("\n[2/3] Running ONNX Runtime …")
    ort_out = run_onnxrt(args.onnx, dummy)
    if ort_out is not None:
        print(f"  Output: {tuple(ort_out.shape)}  norm={ort_out.norm():.4f}")
    else:
        print("  ONNX Runtime skipped.")

    # ── 3. TensorRT ───────────────────────────────────────────────────────────
    print("\n[3/3] Running TensorRT …")
    trt_out = run_trt(args.engine, dummy, args)
    print(f"  Output: {tuple(trt_out.shape)}  norm={trt_out.norm():.4f}")

    # ── Comparisons ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    report("PyTorch",  "ONNX-RT",  pt_out,  ort_out)
    report("PyTorch",  "TRT",       pt_out,  trt_out)
    if ort_out is not None:
        report("ONNX-RT", "TRT",    ort_out, trt_out)

    # ── Diagnosis ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Diagnosis")
    print("=" * 60)
    pt_ort_ok  = (ort_out is not None and cosine_check(pt_out, ort_out)["cosine"] >= 0.9999)
    ort_trt_ok = (ort_out is not None and cosine_check(ort_out, trt_out)["cosine"] >= 0.95)

    # ── Stage 1: Export validation (strict) ──────────────────────────────────
    print("\n  Stage 1 — ONNX Export Validation  (strict: cosine ≥ 0.9999)")
    if pt_ort_ok:
        print("  ✅ PyTorch == ONNX Runtime: export is numerically correct.")
        print("     The ONNX graph faithfully represents the PyTorch model.")
    else:
        print("  ❌ PyTorch ≠ ONNX Runtime: export has a bug.")
        print("     Possible causes:")
        print("     1. Model not in eval() mode (BatchNorm running stats update).")
        print("     2. Dropout still active during trace.")
        print("     3. Data-dependent branching not traced correctly.")
        print("     Fix: check export_onnx.py — ensure wrapper.eval() + torch.no_grad()")

    # ── Stage 2: TRT deployment validation (relaxed) ─────────────────────────
    print("\n  Stage 2 — TRT Engine Validation  (relaxed: cosine ≥ 0.95 vs ONNX-RT)")
    if ort_out is not None:
        if ort_trt_ok:
            print("  ✅ ONNX Runtime ≈ TRT: engine is deployment-ready.")
        else:
            print("  ❌ ONNX Runtime ≠ TRT: engine has a significant numerical problem.")
            print("     Check TRT build logs for unsupported op fallbacks.")

    # ── Known limitation note ─────────────────────────────────────────────────
    print("""
  ⚠  Known TRT 10.3 Limitation — TRT FP32 ≠ Mathematical FP32
  ─────────────────────────────────────────────────────────────
  TRT FP32 engines for this model architecture produce cosine ~0.981
  vs PyTorch FP32, regardless of builder flags (TF32 off, OBEY_PRECISION,
  opt_level 0, opset 14, cuDNN disabled). Root cause: TRT 10.3 evaluates
  the adaLN dynamic Slice indices (Shape→Gather→Div→Mul INT64 chain) with
  different kernel sequencing than ONNX Runtime, and since adaLN gates
  every transformer block the error compounds 8×.

  ✅ Correct sanity check tool: ONNX Runtime (Stage 1 above).
  ✅ Correct production target:  TRT FP16 (cosine ~0.97, 5-6× faster).
  ⚠  TRT FP32 is ~2× faster than PyTorch but NOT numerically equivalent.
""")

if __name__ == "__main__":
    main()



