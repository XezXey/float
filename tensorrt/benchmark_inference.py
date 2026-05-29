"""
benchmark_inference.py — Profile a TensorRT engine against the PyTorch baseline
================================================================================
Step 3 of 3 in the TensorRT pipeline.

Consumes : TensorRT engine from build_engine.py
Produces : JSON metrics file + Markdown performance report

Metrics collected:
  - Latency: mean, p50, p95, p99, min, max, std  (milliseconds)
  - Throughput: frames per second (FPS)
  - VRAM usage: engine load delta + peak allocation
  - Cosine similarity vs PyTorch FP32 baseline
  - Relative L2 error

Usage:
  python tensorrt/benchmark_inference.py \\
      --engine     ./checkpoints/trt_engines/fmt_fp16.engine \\
      --ckpt_path  ./checkpoints/float.pth \\
      --precision  fp16 \\
      --warmup     10 \\
      --iters      100 \\
      --output_dir ./results/trt_reports
"""

import os
import sys
import json
import time
import datetime
import argparse
import subprocess
import torch
import numpy as np

# ── resolve project root ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(script_dir, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _fmt_utils import (
    load_fmt_wrapper, build_dummy_inputs, output_shape,
    perf_stats, cosine_check, add_model_args, PRECISION_THRESHOLDS,
)

PRECISION_MODES = ["fp32", "fp16", "int8", "best"]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a TensorRT engine vs PyTorch FP32 baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = add_model_args(parser)

    # ── I/O ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--engine", type=str, required=True,
        help="Path to the compiled TensorRT .engine file",
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default="./checkpoints/float.pth",
        help="FLOAT checkpoint for PyTorch baseline comparison",
    )
    parser.add_argument(
        "--precision", type=str, default="fp16",
        choices=PRECISION_MODES,
        help="Precision used during build (for threshold selection and report labelling)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./results/trt_reports",
        help="Directory to write JSON metrics and Markdown report",
    )
    parser.add_argument(
        "--output_prefix", type=str, default=None,
        help="Report filename prefix. Defaults to fmt_<precision>_<timestamp>.",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to build manifest JSON from build_engine.py (for report enrichment)",
    )

    # ── benchmark tuning ──────────────────────────────────────────────────
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Warm-up iterations (discarded from metrics, default: 10)",
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Timed benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--input_seed", type=int, default=42,
        help="Random seed for reproducible dummy inputs (default: 42)",
    )
    parser.add_argument(
        "--skip_baseline", action="store_true",
        help="Skip PyTorch FP32 baseline benchmark (faster but no cosine similarity)",
    )

    return parser.parse_known_args()[0]


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch FP32 baseline
# ─────────────────────────────────────────────────────────────────────────────
def run_pytorch_baseline(wrapper, dummy_inputs, warmup, iters):
    """
    Benchmark PyTorch FP32 inference and return (output_tensor, latency_stats).
    Uses torch.cuda.synchronize() for accurate wall-clock measurement.
    """
    with torch.no_grad():
        for _ in range(warmup):
            wrapper(*dummy_inputs)
        torch.cuda.synchronize()

        latencies = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0  = time.perf_counter()
            out = wrapper(*dummy_inputs)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    return out.cpu(), perf_stats(latencies)


# ─────────────────────────────────────────────────────────────────────────────
# TRT engine benchmark
# ─────────────────────────────────────────────────────────────────────────────
def run_trt_benchmark(engine_path, dummy_inputs, opt, warmup, iters, batch):
    """
    Load TRT engine and benchmark via execute_async_v3.
    Uses torch CUDA tensors as I/O binding memory (no pycuda required).
    Optionally falls back to pycuda if available.

    Returns (output_tensor_cpu, latency_stats, vram_stats).
    """
    import tensorrt as trt

    # ── VRAM before engine load ────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1e6  # MB

    # ── load engine ─────────────────────────────────────────────────────────
    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    vram_after_load = torch.cuda.memory_allocated() / 1e6
    engine_vram_mb  = vram_after_load - vram_before
    print(f"  Engine VRAM delta : +{engine_vram_mb:.1f} MB  (total allocated: {vram_after_load:.1f} MB)")

    context = engine.create_execution_context()

    # ── compute output shape ─────────────────────────────────────────────────
    n_curr  = int(opt.wav2vec_sec * opt.fps)   # 50
    n_prev  = opt.num_prev_frames               # 10
    n_total = n_prev + n_curr                   # 60
    dim_w   = opt.dim_w                         # 512
    out_shape_hw = (batch, n_total, dim_w)
    print(f"  Output shape      : {out_shape_hw}")

    # ── allocate I/O tensors ────────────────────────────────────────────────
    # t_in, x_in, wa_in, wr_in, we_in, px_in, pwa_in, a_cfg_scale_in, r_cfg_scale_in, e_cfg_scale_in = dummy_inputs
    t_in, x_in, wa_in, wr_in, we_in, px_in, pwa_in = dummy_inputs
    io = {
        "t":             t_in.contiguous().cuda().float(),
        "x":             x_in.contiguous().cuda().float(),
        "wa":            wa_in.contiguous().cuda().float(),
        "wr":            wr_in.contiguous().cuda().float(),
        "we":            we_in.contiguous().cuda().float(),
        "prev_x":        px_in.contiguous().cuda().float(),
        "prev_wa":       pwa_in.contiguous().cuda().float(),
        # "a_cfg_scale": a_cfg_scale_in.contiguous().cuda().float(),
        # "r_cfg_scale": r_cfg_scale_in.contiguous().cuda().float(),
        # "e_cfg_scale": e_cfg_scale_in.contiguous().cuda().float(),
        "motion_latent": torch.empty(out_shape_hw, dtype=torch.float32, device="cuda"),
    }
    print("  Input tensors for trt:")
    for name in ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]:
        t = io[name]
        print(f"    {name}: {tuple(t.shape)}, dtype={t.dtype}, min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.mean().item():.6f}")

    # Bind tensor addresses (TRT 10.x execute_async_v3 API)
    for name, tensor in io.items():
        context.set_tensor_address(name, tensor.data_ptr())

    # Use a dedicated non-default CUDA stream to avoid extra syncs
    stream = torch.cuda.Stream()

    def _run_once():
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        return io["motion_latent"].clone()

    # ── warm up ────────────────────────────────────────────────────────────
    print(f"  Warming up ({warmup} iters) …")
    for _ in range(warmup):
        _run_once()

    # ── timed benchmark ─────────────────────────────────────────────────────
    print(f"  Timing ({iters} iters) …")
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = _run_once()
        latencies.append((time.perf_counter() - t0) * 1000)

    # ── VRAM peak ───────────────────────────────────────────────────────────
    peak_vram_mb  = torch.cuda.max_memory_allocated() / 1e6
    final_vram_mb = torch.cuda.memory_allocated() / 1e6

    vram_stats = {
        "vram_before_load_mb":  round(vram_before,      1),
        "vram_after_load_mb":   round(vram_after_load,  1),
        "engine_vram_delta_mb": round(engine_vram_mb,   1),
        "peak_vram_mb":         round(peak_vram_mb,     1),
        "final_vram_mb":        round(final_vram_mb,    1),
    }

    return out.cpu(), perf_stats(latencies), vram_stats


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────
def _get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        return result.stdout.strip().splitlines()
    except Exception:
        return ["unknown"]


def write_json(path, metrics):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  JSON metrics : {path}")


def write_markdown(path, precision, pt_s, trt_s, num, vram, args, manifest):
    """Write a detailed Markdown performance report."""
    thresh   = PRECISION_THRESHOLDS.get(precision, PRECISION_THRESHOLDS["fp16"])
    gpus     = " | ".join(_get_gpu_info())
    ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    speedup  = pt_s["mean_ms"] / trt_s["mean_ms"] if trt_s else None
    fps_gain = trt_s["fps"] / pt_s["fps"] if (trt_s and pt_s) else None

    cos_val  = num.get("cosine",  "N/A")
    rel_val  = num.get("rel_l2",  "N/A")
    l2_val   = num.get("l2",      "N/A")
    max_err  = num.get("max_abs_err", "N/A")

    if isinstance(cos_val, float):
        cos_pass = "✅ PASS" if cos_val >= thresh["cos_min"] else "⚠️  WARN"
        rel_pass = "✅ PASS" if rel_val <= thresh["rel_max"] else "⚠️  WARN"
    else:
        cos_pass = rel_pass = "N/A"

    # Build info from manifest if available
    build_info = ""
    if manifest:
        build_info = f"""
## Build Configuration

| Parameter | Value |
|---|---|
| ONNX source | `{manifest.get('onnx_path', 'N/A')}` |
| Precision | `{manifest.get('precision', precision).upper()}` |
| Workspace | {manifest.get('workspace_gb', 'N/A')} GB |
| Timing iterations | {manifest.get('timing_iters', 'N/A')} |
| Optimisation level | {manifest.get('opt_level', 'N/A')} |
| Build time | {manifest.get('build_time_s', 'N/A')} s |
| TensorRT version | {manifest.get('trt_version', 'N/A')} |
| Engine size | {manifest.get('engine_mb', 'N/A')} MB |
| Dynamic batch | {manifest.get('dynamic_batch', False)} |

---
"""

    def fmt_ms(v):
        return f"{v:.2f} ms" if isinstance(v, float) else "N/A"
    def fmt_fps(v):
        return f"{v:.1f}" if isinstance(v, float) else "N/A"
    def fmt_speedup(v):
        return f"**{v:.2f}×**" if isinstance(v, float) else "N/A"

    lines = [
        f"# FLOAT FMT — TensorRT Benchmark Report [{precision.upper()}]",
        f"",
        f"**Generated:** {ts}  ",
        f"**GPU:** {gpus}  ",
        f"**Engine:** `{args.engine}`  ",
        f"**Precision:** `{precision.upper()}`  ",
        f"**Iterations:** {args.iters} (+ {args.warmup} warm-up)  ",
        f"**Batch size:** {args.batch}  ",
        f"",
        f"---",
        build_info,
        f"## Latency Benchmark",
        f"",
        f"| Metric | PyTorch FP32 | TensorRT {precision.upper()} | Speedup |",
        f"|---|---|---|---|",
        f"| Mean | {fmt_ms(pt_s.get('mean_ms'))} | {fmt_ms(trt_s.get('mean_ms') if trt_s else None)} | {fmt_speedup(speedup)} |",
        f"| Median (p50) | {fmt_ms(pt_s.get('median_ms'))} | {fmt_ms(trt_s.get('median_ms') if trt_s else None)} | — |",
        f"| p95 | {fmt_ms(pt_s.get('p95_ms'))} | {fmt_ms(trt_s.get('p95_ms') if trt_s else None)} | {fmt_speedup(pt_s['p95_ms']/trt_s['p95_ms'] if trt_s else None)} |",
        f"| p99 | {fmt_ms(pt_s.get('p99_ms'))} | {fmt_ms(trt_s.get('p99_ms') if trt_s else None)} | {fmt_speedup(pt_s['p99_ms']/trt_s['p99_ms'] if trt_s else None)} |",
        f"| Std dev | {fmt_ms(pt_s.get('std_ms'))} | {fmt_ms(trt_s.get('std_ms') if trt_s else None)} | — |",
        f"| Min | {fmt_ms(pt_s.get('min_ms'))} | {fmt_ms(trt_s.get('min_ms') if trt_s else None)} | — |",
        f"| Max | {fmt_ms(pt_s.get('max_ms'))} | {fmt_ms(trt_s.get('max_ms') if trt_s else None)} | — |",
        f"| **Throughput** | **{fmt_fps(pt_s.get('fps'))} FPS** | **{fmt_fps(trt_s.get('fps') if trt_s else None)} FPS** | {fmt_speedup(fps_gain)} |",
        f"",
        f"---",
        f"",
        f"## VRAM Usage",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| VRAM before engine load | {vram.get('vram_before_load_mb', 'N/A')} MB |",
        f"| VRAM after engine load | {vram.get('vram_after_load_mb', 'N/A')} MB |",
        f"| Engine VRAM footprint | **{vram.get('engine_vram_delta_mb', 'N/A')} MB** |",
        f"| Peak VRAM (during inference) | **{vram.get('peak_vram_mb', 'N/A')} MB** |",
        f"",
        f"---",
        f"",
        f"## Numerical Verification  (PyTorch FP32 vs TensorRT {precision.upper()})",
        f"",
        f"| Metric | Value | Threshold `{precision}` | Status |",
        f"|---|---|---|---|",
        f"| Cosine similarity | {cos_val:.6f} | ≥ {thresh['cos_min']} | {cos_pass} |" if isinstance(cos_val, float) else f"| Cosine similarity | N/A | ≥ {thresh['cos_min']} | N/A |",
        f"| Relative L2 error | {rel_val:.4f} | ≤ {thresh['rel_max']} | {rel_pass} |" if isinstance(rel_val, float) else f"| Relative L2 error | N/A | ≤ {thresh['rel_max']} | N/A |",
        f"| Absolute L2 error | {l2_val:.4f} | — | — |" if isinstance(l2_val, float) else f"| Absolute L2 error | N/A | — | — |",
        f"| Max absolute error | {max_err:.6f} | — | — |" if isinstance(max_err, float) else f"| Max absolute error | N/A | — | — |",
        f"",
        f"> Thresholds: FP32 (≥0.9999 / ≤0.001) · FP16 (≥0.950 / ≤0.300) · INT8/BEST (≥0.900 / ≤0.500)",
        f"> Inputs used for comparison: random synthetic tensors (seed={args.input_seed})",
        f"",
        f"---",
        f"",
        f"## Input / Output Shapes (batch={args.batch})",
        f"",
        f"| Tensor | Shape | Description |",
        f"|---|---|---|",
        f"| `t` | `(1,)` | ODE timestep ∈ [0, 1] |",
        f"| `x` | `({args.batch}, 50, 512)` | Motion latent (50 frames) |",
        f"| `wa` | `({args.batch}, 50, 512)` | Audio feature sequence |",
        f"| `wr` | `({args.batch}, 512)` | Reference motion latent |",
        f"| `we` | `({args.batch}, 1, 7)` | Emotion latent |",
        f"| `prev_x` | `({args.batch}, 10, 512)` | Previous motion context |",
        f"| `prev_wa` | `({args.batch}, 10, 512)` | Previous audio context |",
        f"| **output** | `({args.batch}, 60, 512)` | Vector field (prev 10 + curr 50 frames) |",
        f"",
        f"---",
        f"*Generated by `tensorrt/benchmark_inference.py`*",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Markdown report : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.rank = device

    print("=" * 60)
    print("FLOAT FMT — TensorRT Inference Benchmark")
    print("=" * 60)
    print(f"  Engine    : {args.engine}")
    print(f"  Precision : {args.precision.upper()}")
    print(f"  Warmup    : {args.warmup}")
    print(f"  Iters     : {args.iters}")
    print(f"  Batch     : {args.batch}")
    print(f"  Seed      : {args.input_seed}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)
    ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix or f"fmt_{args.precision}_{ts}"
    json_path = os.path.join(args.output_dir, f"{prefix}_metrics.json")
    md_path   = os.path.join(args.output_dir, f"{prefix}_report.md")

    # Load build manifest if available
    manifest = None
    manifest_path = args.manifest or args.engine.replace(".engine", "_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"  Build manifest : {manifest_path}")

    # Build dummy inputs (reproducible)
    dummy_inputs = build_dummy_inputs(args, device, batch=args.batch, seed=args.input_seed)

    # ── PyTorch FP32 baseline ───────────────────────────────────────────────
    pt_s   = {}
    pt_out = None
    if not args.skip_baseline:
        print("[1/3] PyTorch FP32 baseline …")
        wrapper = load_fmt_wrapper(args, device)
        print("  Input shapes:")
        for tensor in dummy_inputs:
            print(f"    {tensor.shape}, dtype={tensor.dtype}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
        pt_out, pt_s = run_pytorch_baseline(wrapper, dummy_inputs, args.warmup, args.iters)
        del wrapper  # free GPU memory before loading TRT engine
        torch.cuda.empty_cache()
        print(f"  mean={pt_s['mean_ms']:.2f}ms  p50={pt_s['p50_ms']:.2f}ms  "
              f"p95={pt_s['p95_ms']:.2f}ms  p99={pt_s['p99_ms']:.2f}ms  "
              f"fps={pt_s['fps']:.1f}")
        print(f"  Output shape      : {pt_out.shape}, dtype={pt_out.dtype}")
        print(f"  Output stats      : min={pt_out.min().item():.6f}  max={pt_out.max().item():.6f}  mean={pt_out.mean().item():.6f}")
    else:
        print("[1/3] PyTorch baseline skipped (--skip_baseline)")

    # ── TRT engine ───────────────────────────────────────────────────────────
    print(f"[2/3] TensorRT {args.precision.upper()} inference …")
    trt_out, trt_s, vram = run_trt_benchmark(
        args.engine, dummy_inputs, args, args.warmup, args.iters, args.batch
    )
    print(f"  mean={trt_s['mean_ms']:.2f}ms  p50={trt_s['p50_ms']:.2f}ms  "
          f"p95={trt_s['p95_ms']:.2f}ms  p99={trt_s['p99_ms']:.2f}ms  "
          f"fps={trt_s['fps']:.1f}")
    print(f"  Output shape      : {trt_out.shape}, dtype={trt_out.dtype}")
    print(f"  Output stats      : min={trt_out.min().item():.6f}  max={trt_out.max().item():.6f}  mean={trt_out.mean().item():.6f}")
    if pt_s:
        print(f"  Speedup vs FP32 : {pt_s['mean_ms']/trt_s['mean_ms']:.2f}×")

    # ── numerical verification ────────────────────────────────────────────────
    print("[3/3] Numerical verification …")
    num = {}
    if pt_out is not None:
        num = cosine_check(pt_out, trt_out)
        thresh = PRECISION_THRESHOLDS.get(args.precision, PRECISION_THRESHOLDS["fp16"])
        cos_ok = num["cosine"] >= thresh["cos_min"]
        rel_ok = num["rel_l2"] <= thresh["rel_max"]
        print(f"  Cosine similarity : {num['cosine']:.6f}  "
              f"({'PASS ✓' if cos_ok else 'WARN ⚠'} ≥{thresh['cos_min']})")
        print(f"  Relative L2 error : {num['rel_l2']:.4f}   "
              f"({'PASS ✓' if rel_ok else 'WARN ⚠'} ≤{thresh['rel_max']})")
        print(f"  Max absolute error: {num['max_abs_err']:.6f}")
        print(f"  All close: trt_out vs pt_out: {torch.allclose(trt_out, pt_out, rtol=1e-4, atol=1e-5)}")
        print(f"  Max diff: {(trt_out - pt_out).abs().max().item():.6f}")
    else:
        print("  Skipped (no baseline output)")

    # ── save outputs ───────────────────────────────────────────────────────────
    metrics = {
        "timestamp":   ts,
        "engine":      args.engine,
        "precision":   args.precision,
        "batch":       args.batch,
        "warmup":      args.warmup,
        "iters":       args.iters,
        "pytorch_fp32": pt_s,
        "tensorrt":    trt_s,
        "speedup_mean": round(pt_s["mean_ms"] / trt_s["mean_ms"], 3) if pt_s else None,
        "vram":        vram,
        "numerical":   num,
    }

    write_json(json_path, metrics)
    write_markdown(md_path, args.precision, pt_s, trt_s, num, vram, args, manifest)

    print()
    print("Benchmark complete.")
    print(f"  JSON    : {json_path}")
    print(f"  Report  : {md_path}")


if __name__ == "__main__":
    main()
