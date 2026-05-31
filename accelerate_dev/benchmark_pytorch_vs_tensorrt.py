#!/usr/bin/env python3
"""
benchmark_pytorch_vs_tensorrt.py
====================================================================
Benchmarks PyTorch model vs. compiled TensorRT engine on execution latency and precision.
Loads the checkpoint weights to instantiate the PyTorch FLOAT model.
Loads the compiled TensorRT engine, allocates buffers, performs a correctness sanity check,
and runs a latency benchmark using GPU-synchronized events.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

try:
    import tensorrt as trt
except ImportError:
    trt = None

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
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT Engine for FLOAT FMT")
    
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/float.pth",
        help="Path to PyTorch checkpoint file",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="accelerate_dev/float_fmt.engine",
        help="Path to compiled TensorRT engine file",
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
    
    # Base model configuration args
    parser = add_model_args(parser)
    args = parser.parse_args()

    # Check dependencies and files
    if trt is None:
        print("[Error] TensorRT Python library is not installed. Cannot benchmark.")
        sys.exit(1)
    if not os.path.isfile(args.ckpt_path):
        print(f"[Error] PyTorch checkpoint not found at: {args.ckpt_path}")
        sys.exit(1)
    if not os.path.isfile(args.engine):
        print(f"[Error] TensorRT engine file not found at: {args.engine}. Please compile it first.")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("[Error] CUDA is not available on this system. Cannot run TensorRT benchmarks.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Benchmarking on device: {device}")

    # 1. Load PyTorch model
    print(f"Loading PyTorch model wrapper...")
    args.rank = 0
    args.ngpus = 1
    wrapper = load_fmt_wrapper(args, device)
    wrapper.eval()

    # 2. Load TensorRT Engine
    print(f"Loading TensorRT engine: {args.engine}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    print("Created TensorRT Execution Context.")

    # 3. Allocate I/O Buffer Bindings using PyTorch CUDA Tensors
    buffers = {}
    input_names = []
    output_names = []

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)

        # Resolve dynamic dimensions (-1)
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

    # 4. Generate identical inputs for validation
    print("\nGenerating identical inputs...")
    dummy_inputs = build_dummy_inputs(args, device, batch=args.batch_size)
    t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch = dummy_inputs

    # 5. Numerical Sanity Check
    print("\n--- Running Sanity Check ---")
    
    # Run PyTorch forward pass (using model.fmt.forward_with_cfv directly as requested)
    with torch.no_grad():
        pytorch_out = wrapper.fmt.forward_with_cfv(
            t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch,
            a_cfg_scale=2.0, r_cfg_scale=1.0, e_cfg_scale=1.0
        )

    # Run TensorRT forward pass
    # Copy data into the bound input buffers
    for name in input_names:
        if name == "t":
            buffers[name].copy_(t_torch)
        elif name == "x":
            buffers[name].copy_(x_torch)
        elif name == "wa":
            buffers[name].copy_(wa_torch)
        elif name == "wr":
            buffers[name].copy_(wr_torch)
        elif name == "we":
            buffers[name].copy_(we_torch)
        elif name == "prev_x":
            buffers[name].copy_(prev_x_torch)
        elif name == "prev_wa":
            buffers[name].copy_(prev_wa_torch)

    # Bind input shapes to the execution context for dynamic shape execution
    for name in input_names:
        context.set_input_shape(name, buffers[name].shape)

    # Bind addresses
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.Stream()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    
    trt_out = buffers[output_names[0]]

    # Compute discrepancy metrics (converting output to float32 for comparison)
    metrics = cosine_check(pytorch_out, trt_out.float())
    print(f"PyTorch Output Shape : {list(pytorch_out.shape)}")
    print(f"TensorRT Output Shape: {list(trt_out.shape)}")
    print(f"Maximum Absolute Diff: {metrics['max_abs_err']:.6e}")
    print(f"L2 Discrepancy (Rel) : {metrics['rel_l2']:.6e}")
    print(f"Cosine Similarity    : {metrics['cosine']:.6f}")

    if metrics['max_abs_err'] < 1e-3:
        print("Sanity Check Passed! Precision preserved during compilation.")
    else:
        print("Sanity Check Note: Discrepancy detected (normal for FP16 quantization or tactic optimizations).")

    # 6. Benchmark PyTorch Latency
    print(f"\n--- Benchmarking PyTorch ({args.runs} runs, {args.warmup} warmups) ---")
    pytorch_latencies = []

    with torch.no_grad():
        # Warmup
        for _ in range(args.warmup):
            _ = wrapper.fmt.forward_with_cfv(
                t_torch, x_torch, wa_torch, wr_torch, we_torch, prev_x_torch, prev_wa_torch,
                a_cfg_scale=2.0, r_cfg_scale=1.0, e_cfg_scale=1.0
            )
        torch.cuda.synchronize()

        # Timing loop
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

    pytorch_stats = perf_stats(pytorch_latencies)
    print(f"PyTorch Latency (ms):")
    print(f"  Mean   : {pytorch_stats['mean_ms']:.3f}")
    print(f"  Median : {pytorch_stats['median_ms']:.3f}")
    print(f"  95th%  : {pytorch_stats['p95_ms']:.3f}")

    # 7. Benchmark TensorRT Latency
    print(f"\n--- Benchmarking TensorRT ({args.runs} runs, {args.warmup} warmups) ---")
    trt_latencies = []

    # Warmup
    for _ in range(args.warmup):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    # Timing loop
    trt_start_event = torch.cuda.Event(enable_timing=True)
    trt_end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(args.runs):
        trt_start_event.record()
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        trt_end_event.record()
        stream.synchronize()
        trt_latencies.append(trt_start_event.elapsed_time(trt_end_event))  # ms

    trt_stats = perf_stats(trt_latencies)
    print(f"TensorRT Latency (ms):")
    print(f"  Mean   : {trt_stats['mean_ms']:.3f}")
    print(f"  Median : {trt_stats['median_ms']:.3f}")
    print(f"  95th%  : {trt_stats['p95_ms']:.3f}")

    # 8. Speedup Summary
    speedup = pytorch_stats['mean_ms'] / trt_stats['mean_ms']
    print("\n--- Summary ---")
    print(f"PyTorch Mean Latency  : {pytorch_stats['mean_ms']:.3f} ms")
    print(f"TensorRT Mean Latency : {trt_stats['mean_ms']:.3f} ms")
    print(f"TensorRT speedup factor: {speedup:.2f}x")


if __name__ == "__main__":
    main()
