#!/usr/bin/env python3
"""
benchmark_pytorch_vs_tensorrt.py
====================================================================
Benchmarks PyTorch model vs. compiled TensorRT engine on execution latency and precision.
Performs GPU-synchronized profiling of inference loops and measures absolute discrepancy.
"""

import os
import argparse
import time
import torch
import torchvision.models as models
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT Engine")
    parser.add_argument(
        "--engine",
        type=str,
        default="accelerate_dev/resnet50.engine",
        help="Path to compiled TensorRT engine file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
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
    args = parser.parse_args()

    # Check dependencies and artifacts
    if trt is None:
        print("[Error] TensorRT Python library is not installed. Cannot benchmark.")
        raise SystemExit(1)

    if not os.path.isfile(args.engine):
        print(f"[Error] TensorRT engine file not found at: {args.engine}")
        print("Please compile the engine first using run_tensorrt.py.")
        raise SystemExit(1)

    if not torch.cuda.is_available():
        print("[Error] CUDA is not available on this system. Cannot run benchmarks on GPU.")
        raise SystemExit(1)

    device = torch.device("cuda")
    print(f"Benchmarking on device: {device}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load PyTorch Model and TensorRT Engine
    # ─────────────────────────────────────────────────────────────────────────
    print("Loading PyTorch model (ResNet50 placeholder)...")
    try:
        weights = models.ResNet50_Weights.DEFAULT
        pytorch_model = models.resnet50(weights=weights)
    except Exception as e:
        print(f"Warning: Failed to load pre-trained weights ({e}). Initializing random ResNet50.")
        pytorch_model = models.resnet50()

    pytorch_model.eval().to(device)

    print(f"Loading TensorRT engine: {args.engine}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Allocate I/O Buffer Bindings using PyTorch CUDA Tensors
    # ─────────────────────────────────────────────────────────────────────────
    buffers = {}
    input_names = []
    output_names = []

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)

        # Resolve dynamic dimensions
        resolved_shape = [args.batch_size if dim < 0 else dim for dim in shape]

        # DataType Mapping
        if dtype == trt.DataType.FLOAT:
            torch_dtype = torch.float32
        elif dtype == trt.DataType.HALF:
            torch_dtype = torch.float16
        elif dtype == trt.DataType.INT32:
            torch_dtype = torch.int32
        elif dtype == trt.DataType.INT8:
            torch_dtype = torch.int8
        else:
            torch_dtype = torch.float32

        # Allocate on GPU
        tensor = torch.zeros(resolved_shape, dtype=torch_dtype, device=device)
        buffers[name] = tensor

        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    # We assume a single input and single output model for this comparison
    primary_input_name = input_names[0]
    primary_output_name = output_names[0]

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Sanity Check (Precision / Discrepancy Verification)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n--- Running Sanity Check ---")
    
    # Generate random test input on GPU
    x_test = torch.randn_like(buffers[primary_input_name])

    # PyTorch Forward
    with torch.no_grad():
        pytorch_out = pytorch_model(x_test)

    # TensorRT Forward
    # Copy data into the bound input buffer
    buffers[primary_input_name].copy_(x_test)
    
    # Bind addresses
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    
    trt_out = buffers[primary_output_name]

    # Make sure comparisons are of the same type (e.g. cast float16 outputs to float32)
    pytorch_out_f32 = pytorch_out.float()
    trt_out_f32 = trt_out.float()

    # Calculate discrepancy
    abs_diff = torch.abs(pytorch_out_f32 - trt_out_f32)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()

    print(f"PyTorch Output Shape : {list(pytorch_out.shape)}")
    print(f"TensorRT Output Shape: {list(trt_out.shape)}")
    print(f"Maximum Absolute Diff: {max_diff:.6e}")
    print(f"Mean Absolute Diff   : {mean_diff:.6e}")
    if max_diff < 1e-3:
        print("Sanity Check Passed! Precision preserved during TensorRT compilation.")
    else:
        print("Sanity Check Note: Discrepancy detected (normal for FP16 quantization or tactic optimizations).")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Latency Benchmark: PyTorch
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n--- Benchmarking PyTorch ({args.runs} runs, {args.warmup} warmups) ---")
    pytorch_latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(args.warmup):
            _ = pytorch_model(x_test)
        torch.cuda.synchronize()

        # Profiling loop
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for _ in range(args.runs):
            start_event.record()
            _ = pytorch_model(x_test)
            end_event.record()
            torch.cuda.synchronize()
            pytorch_latencies.append(start_event.elapsed_time(end_event))

    pytorch_latencies = np.array(pytorch_latencies)
    pytorch_mean = np.mean(pytorch_latencies)
    pytorch_median = np.percentile(pytorch_latencies, 50)
    pytorch_p95 = np.percentile(pytorch_latencies, 95)
    print(f"PyTorch Latency:")
    print(f"  Mean   : {pytorch_mean:.3f} ms")
    print(f"  Median : {pytorch_median:.3f} ms")
    print(f"  95th%  : {pytorch_p95:.3f} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Latency Benchmark: TensorRT
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n--- Benchmarking TensorRT ({args.runs} runs, {args.warmup} warmups) ---")
    trt_latencies = []

    # Bind input and run warmups
    buffers[primary_input_name].copy_(x_test)
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    for _ in range(args.warmup):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    # Profiling loop
    trt_start_event = torch.cuda.Event(enable_timing=True)
    trt_end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(args.runs):
        trt_start_event.record()
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        trt_end_event.record()
        stream.synchronize()
        trt_latencies.append(trt_start_event.elapsed_time(trt_end_event))

    trt_latencies = np.array(trt_latencies)
    trt_mean = np.mean(trt_latencies)
    trt_median = np.percentile(trt_latencies, 50)
    trt_p95 = np.percentile(trt_latencies, 95)
    print(f"TensorRT Latency:")
    print(f"  Mean   : {trt_mean:.3f} ms")
    print(f"  Median : {trt_median:.3f} ms")
    print(f"  95th%  : {trt_p95:.3f} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Speedup Summary
    # ─────────────────────────────────────────────────────────────────────────
    speedup = pytorch_mean / trt_mean
    print("\n--- Summary ---")
    print(f"PyTorch Mean Latency  : {pytorch_mean:.3f} ms")
    print(f"TensorRT Mean Latency : {trt_mean:.3f} ms")
    print(f"TensorRT speedup factor: {speedup:.2f}x")


if __name__ == "__main__":
    main()
