#!/usr/bin/env python3
"""
benchmark_pytorch_vs_onnx.py
====================================================================
Benchmarks PyTorch model vs. ONNX Runtime inference latency and precision.
Includes a numerical sanity check and uses GPU-synchronized events for accurate PyTorch timing.
"""

import os
import argparse
import time
import torch
import torchvision.models as models
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX Runtime")
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="accelerate_dev/resnet50.onnx",
        help="Path to the exported ONNX model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmark inputs",
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
    args = parser.parse_args()

    # Check for ONNX model file
    if not os.path.isfile(args.onnx_model):
        print(f"[Error] ONNX model file not found at: {args.onnx_model}")
        print("Please export your model first using export_onnx.py.")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() and args.provider == "cuda" else "cpu")
    print(f"Benchmarking on device: {device}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load PyTorch and ONNX Runtime Models
    # ─────────────────────────────────────────────────────────────────────────
    print("Loading PyTorch model (ResNet50 placeholder)...")
    try:
        weights = models.ResNet50_Weights.DEFAULT
        pytorch_model = models.resnet50(weights=weights)
    except Exception as e:
        print(f"Warning: Failed to load pre-trained weights ({e}). Initializing random ResNet50.")
        pytorch_model = models.resnet50()

    pytorch_model.eval().to(device)

    print(f"Loading ONNX model for ORT inference from: {args.onnx_model}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    try:
        ort_session = ort.InferenceSession(args.onnx_model, providers=providers)
    except Exception as e:
        print(f"[Error] Failed to initialize ONNX Runtime session: {e}")
        raise SystemExit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Sanity Check (Numerical Agreement)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Running Sanity Check ---")
    # Generate identical input tensor
    # ResNet-50 input shape [Batch, Channels, Height, Width]
    x_torch = torch.randn(args.batch_size, 3, 224, 224, device=device)
    x_numpy = x_torch.cpu().numpy()

    # Forward PyTorch
    with torch.no_grad():
        pytorch_out = pytorch_model(x_torch)

    # Forward ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: x_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_out = torch.from_numpy(ort_outputs[0]).to(device)

    # Calculate discrepancy
    abs_diff = torch.abs(pytorch_out - onnx_out)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()

    print(f"PyTorch Output Shape : {list(pytorch_out.shape)}")
    print(f"ONNX Output Shape    : {list(onnx_out.shape)}")
    print(f"Maximum Absolute Diff: {max_diff:.6e}")
    print(f"Mean Absolute Diff   : {mean_diff:.6e}")
    if max_diff < 1e-4:
        print("Sanity Check Passed! Outputs match closely.")
    else:
        print("Sanity Check Warning: Outputs have noticeable divergence (Precision/Op mapping discrepancy).")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Latency Benchmark: PyTorch
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n--- Benchmarking PyTorch ({args.runs} runs, {args.warmup} warmups) ---")
    pytorch_latencies = []

    # Warm-up phase
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = pytorch_model(x_torch)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Profiling loop
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(args.runs):
                start_event.record()
                _ = pytorch_model(x_torch)
                end_event.record()
                torch.cuda.synchronize()
                pytorch_latencies.append(start_event.elapsed_time(end_event))  # returns milliseconds
        else:
            for _ in range(args.runs):
                t0 = time.perf_counter()
                _ = pytorch_model(x_torch)
                pytorch_latencies.append((time.perf_counter() - t0) * 1000)  # ms

    # Calculate statistics
    pytorch_latencies = np.array(pytorch_latencies)
    pytorch_mean = np.mean(pytorch_latencies)
    pytorch_median = np.percentile(pytorch_latencies, 50)
    pytorch_p95 = np.percentile(pytorch_latencies, 95)
    print(f"PyTorch Latency:")
    print(f"  Mean   : {pytorch_mean:.3f} ms")
    print(f"  Median : {pytorch_median:.3f} ms")
    print(f"  95th%  : {pytorch_p95:.3f} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Latency Benchmark: ONNX Runtime
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n--- Benchmarking ONNX Runtime ({args.runs} runs, {args.warmup} warmups) ---")
    ort_latencies = []

    # Warm-up phase
    for _ in range(args.warmup):
        _ = ort_session.run(None, ort_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Profiling loop
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        # Note: If CUDA provider is async, ORT session syncs unless bindings are used.
        # But to ensure exact CPU/GPU execution synchronization, we synchronize cuda if applicable.
        if device.type == "cuda":
            torch.cuda.synchronize()
        ort_latencies.append((time.perf_counter() - t0) * 1000)  # ms

    # Calculate statistics
    ort_latencies = np.array(ort_latencies)
    ort_mean = np.mean(ort_latencies)
    ort_median = np.percentile(ort_latencies, 50)
    ort_p95 = np.percentile(ort_latencies, 95)
    print(f"ONNX Runtime Latency:")
    print(f"  Mean   : {ort_mean:.3f} ms")
    print(f"  Median : {ort_median:.3f} ms")
    print(f"  95th%  : {ort_p95:.3f} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Speedup Summary
    # ─────────────────────────────────────────────────────────────────────────
    speedup = pytorch_mean / ort_mean
    print("\n--- Summary ---")
    print(f"PyTorch Mean Latency       : {pytorch_mean:.3f} ms")
    print(f"ONNX Runtime Mean Latency  : {ort_mean:.3f} ms")
    print(f"ONNX speedup factor        : {speedup:.2f}x")


if __name__ == "__main__":
    main()
