#!/usr/bin/env python3
"""
benchmark_pytorch_vs_onnx.py
====================================================================
Benchmarks PyTorch model vs. ONNX Runtime inference latency and precision.
Loads the checkpoint weights to instantiate the PyTorch FLOAT model.
Performs a numerical sanity check comparing PyTorch model.fmt.forward_with_cfv 
against ONNX Runtime, and runs latency profiling with GPU synchronization.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import onnxruntime as ort

# Ensure the root folder and tensorrt folder are in the Python search path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
tensorrt_dir = os.path.join(ROOT, "tensorrt")
if tensorrt_dir not in sys.path:
    sys.path.insert(0, tensorrt_dir)

import _fmt_utils
from _fmt_utils import load_fmt_wrapper, build_dummy_inputs, add_model_args, perf_stats, cosine_check


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime for FLOAT FMT")
    
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/float.pth",
        help="Path to PyTorch checkpoint file",
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        default="accelerate_dev/float_fmt.onnx",
        help="Path to the exported ONNX model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warm-up iterations",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark profiling runs",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX execution provider (cuda or cpu)",
    )
    
    # Base model configuration args
    parser = add_model_args(parser)
    args = parser.parse_args()
    
    # Initialize rank and ngpus as required by FlowMatchingTransformer device placements
    args.rank = 0 if torch.cuda.is_available() else "cpu"
    args.ngpus = 1

    # Verify files
    if not os.path.isfile(args.ckpt_path):
        print(f"[Error] PyTorch checkpoint not found at: {args.ckpt_path}")
        sys.exit(1)
    if not os.path.isfile(args.onnx_model):
        print(f"[Error] ONNX model not found at: {args.onnx_model}. Please run export_onnx.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and args.provider == "cuda" else "cpu")
    print(f"Benchmarking on device: {device}")

    # 1. Load PyTorch model
    print(f"Loading PyTorch model wrapper...")
    wrapper = load_fmt_wrapper(args, device)
    wrapper.eval()

    # 2. Load ONNX Runtime Session
    print(f"Loading ONNX model: {args.onnx_model}")
    providers = [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }),
        "CPUExecutionProvider"
    ] if device.type == "cuda" else ["CPUExecutionProvider"]
    
    try:
        ort_session = ort.InferenceSession(args.onnx_model, providers=providers)
    except Exception as e:
        print(f"[Error] Failed to initialize ONNX Runtime session: {e}")
        sys.exit(1)

    # 3. Generate identical inputs for validation and benchmarking
    print("\nGenerating identical inputs...")
    dummy_inputs = build_dummy_inputs(args, device, batch=args.batch_size)
    t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch = dummy_inputs

    # Prepare inputs for ONNX (NumPy arrays)
    feed_dict = {
        "t": t_torch.cpu().numpy(),
        "x": x_torch.cpu().numpy(),
        "wa": wa_torch.cpu().numpy(),
        "wr": wr_torch.cpu().numpy(),
        "we": we_torch.cpu().numpy(),
        "prev_x": prev_x_torch.cpu().numpy(),
        "prev_wa": prev_wa_torch.cpu().numpy()
    }

    # 4. Numerical Sanity Check
    print("\n--- Running Sanity Check ---")
    
    # Run PyTorch forward pass (using model.fmt.forward_with_cfv directly as requested)
    # The default cfg scale values match the FMTWrapper defaults: a=2.0, r=1.0, e=1.0
    with torch.no_grad():
        pytorch_out = wrapper.fmt.forward_with_cfv(
            t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch,
            a_cfg_scale=2.0, r_cfg_scale=1.0, e_cfg_scale=1.0
        )
        
    # Run ONNX Runtime forward pass
    ort_outputs = ort_session.run(None, feed_dict)
    onnx_out = torch.from_numpy(ort_outputs[0]).to(device)

    # Calculate discrepancy using cosine_check helper
    metrics = cosine_check(pytorch_out, onnx_out)
    print(f"PyTorch Output Shape : {list(pytorch_out.shape)}")
    print(f"ONNX Output Shape    : {list(onnx_out.shape)}")
    print(f"Maximum Absolute Diff: {metrics['max_abs_err']:.6e}")
    print(f"L2 Discrepancy (Rel) : {metrics['rel_l2']:.6e}")
    print(f"Cosine Similarity    : {metrics['cosine']:.6f}")

    if metrics['max_abs_err'] < 1e-4:
        print("Sanity Check Passed! Outputs match closely.")
    else:
        print("Sanity Check Warning: Outputs have noticeable divergence.")

    # 5. Benchmark PyTorch Latency
    print(f"\n--- Benchmarking PyTorch ({args.runs} runs, {args.warmup} warmups) ---")
    pytorch_latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(args.warmup):
            _ = wrapper.fmt.forward_with_cfv(
                t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch,
                a_cfg_scale=2.0, r_cfg_scale=1.0, e_cfg_scale=1.0
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timing loop
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(args.runs):
                start_event.record()
                _ = wrapper.fmt.forward_with_cfv(
                    t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch,
                    a_cfg_scale=2.0, r_cfg_scale=1.0, e_cfg_scale=1.0
                )
                end_event.record()
                torch.cuda.synchronize()
                pytorch_latencies.append(start_event.elapsed_time(end_event))  # ms
        else:
            for _ in range(args.runs):
                t0 = time.perf_counter()
                _ = wrapper.fmt.forward_with_cfv(
                    t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch,
                    a_cfg_scale=2.0, r_cfg_scale=1.0, e_cfg_scale=1.0
                )
                pytorch_latencies.append((time.perf_counter() - t0) * 1000)  # ms

    pytorch_stats = perf_stats(pytorch_latencies)
    print(f"PyTorch Latency (ms):")
    print(f"  Mean   : {pytorch_stats['mean_ms']:.3f}")
    print(f"  Median : {pytorch_stats['median_ms']:.3f}")
    print(f"  95th%  : {pytorch_stats['p95_ms']:.3f}")

    # 6. Benchmark ONNX Runtime Latency
    print(f"\n--- Benchmarking ONNX Runtime ({args.runs} runs, {args.warmup} warmups) ---")
    ort_latencies = []

    # Warmup
    for _ in range(args.warmup):
        _ = ort_session.run(None, feed_dict)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timing loop
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, feed_dict)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ort_latencies.append((time.perf_counter() - t0) * 1000)  # ms

    ort_stats = perf_stats(ort_latencies)
    print(f"ONNX Runtime Latency (ms):")
    print(f"  Mean   : {ort_stats['mean_ms']:.3f}")
    print(f"  Median : {ort_stats['median_ms']:.3f}")
    print(f"  95th%  : {ort_stats['p95_ms']:.3f}")

    # 7. Speedup Summary
    speedup = pytorch_stats['mean_ms'] / ort_stats['mean_ms']
    print("\n--- Summary ---")
    print(f"PyTorch Mean Latency       : {pytorch_stats['mean_ms']:.3f} ms")
    print(f"ONNX Runtime Mean Latency  : {ort_stats['mean_ms']:.3f} ms")
    print(f"ONNX speedup factor        : {speedup:.2f}x")


if __name__ == "__main__":
    main()
