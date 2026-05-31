#!/usr/bin/env python3
"""
run_tensorrt.py
====================================================================
Compiles or loads a TensorRT engine, saving/reading the engine file directly
within the 'accelerate_dev/' directory.

This script supports:
1. Building/Compiling an ONNX model to a TensorRT engine.
   - Option A: Using the 'trtexec' CLI tool via subprocess.
   - Option B: Using the TensorRT Python Builder API.
2. Loading and running inference on the compiled engine.
"""

import os
import argparse
import subprocess
import time
import torch
import numpy as np

# Try importing tensorrt. If not available, print a warning.
try:
    import tensorrt as trt
except ImportError:
    trt = None


def build_engine_with_api(onnx_path: str, engine_path: str, precision: str = "fp16") -> bool:
    """
    Option B: Compile ONNX model to a TensorRT engine using the Python API.
    """
    if trt is None:
        print("[Error] TensorRT Python library is not installed. Cannot use Python API builder.")
        return False

    print(f"Building engine using TensorRT Python API (Precision: {precision.upper()})...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # Configure network flags (explicit batch is required for ONNX parsing)
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, logger)

    # Read and parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[Error] Failed to parse ONNX file:")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    
    # Set workspace memory pool limit (e.g., 2 GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))

    # Configure quantization precision flags
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 mode enabled.")
        else:
            print("  [Warning] FP16 is not supported on this GPU platform. Falling back to FP32.")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  INT8 mode enabled. NOTE: This script uses random calibration data.")
            # [CUSTOMIZATION POINT] Implement a custom calibrator for accurate INT8 scale factors
            # e.g., config.int8_calibrator = MyInt8Calibrator(calibration_data)
        else:
            print("  [Warning] INT8 is not supported on this GPU platform.")
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("  Falling back to FP16.")
    else:
        print("  FP32 mode selected. No quantization flags set.")

    # [CUSTOMIZATION POINT] Define optimization profiles if using dynamic axes
    # profile = builder.create_optimization_profile()
    # profile.set_shape("input", min_shape=(1, 3, 224, 224), opt_shape=(1, 3, 224, 224), max_shape=(4, 3, 224, 224))
    # config.add_optimization_profile(profile)

    # Build and serialize engine
    print("  Building network... (This may take several minutes)")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[Error] Engine build failed.")
        return False
    
    print(f"  Engine build complete in {time.time() - t0:.2f} seconds.")

    # Save to disk
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"  Saved engine to: {engine_path}")
    return True


def build_engine_with_trtexec(onnx_path: str, engine_path: str, precision: str = "fp16") -> bool:
    """
    Option A: Compile ONNX model using the 'trtexec' CLI tool via subprocess.
    """
    print(f"Building engine using 'trtexec' command line (Precision: {precision.upper()})...")
    
    # Construct trtexec command
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--workspace=2048",  # 2 GB workspace
    ]

    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.extend(["--int8", "--fp16"]) # Enable FP16 fallback

    # [CUSTOMIZATION POINT] Add dynamic shapes flag to trtexec if needed
    # cmd.append("--minShapes=input:1x3x224x224")
    # cmd.append("--optShapes=input:1x3x224x224")
    # cmd.append("--maxShapes=input:4x3x224x224")

    print(f"  Running command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"  trtexec compilation completed in {time.time() - t0:.2f} seconds.")
        return True
    except FileNotFoundError:
        print("[Error] 'trtexec' command-line executable not found in PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print("[Error] trtexec compilation failed.")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return False


def run_engine_inference(engine_path: str, batch_size: int = 1):
    """
    Loads a compiled TensorRT engine, allocates PyTorch CUDA tensors, binds them,
    and runs a forward pass using the modern IExecutionContext execute_async_v3 API.
    """
    if trt is None:
        print("[Error] TensorRT Python library is not installed. Inference cannot be run.")
        return

    print(f"Loading engine from: {engine_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    print("Created TensorRT Execution Context.")

    # ─────────────────────────────────────────────────────────────────────────
    # Allocate I/O buffers using PyTorch CUDA Tensors (Zero-copy GPU pointers)
    # ─────────────────────────────────────────────────────────────────────────
    buffers = {}
    
    # Iterate through all input/output tensors to inspect shape/type and allocate
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)

        # Handle dynamic dimensions (e.g. batch size dimensions set to -1)
        resolved_shape = []
        for dim in shape:
            if dim < 0:
                resolved_shape.append(batch_size)
            else:
                resolved_shape.append(dim)

        # Map TensorRT DataType to PyTorch torch.dtype
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

        # Allocate tensor on GPU
        tensor = torch.zeros(resolved_shape, dtype=torch_dtype, device="cuda")
        buffers[name] = tensor
        
        mode_str = "Input" if mode == trt.TensorIOMode.INPUT else "Output"
        print(f"  {mode_str} Tensor Name='{name}', Shape={resolved_shape}, Dtype={torch_dtype}")

    # Identify primary inputs/outputs based on order or customization
    input_names = [name for name in buffers if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    output_names = [name for name in buffers if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]

    # Generate dummy input on GPU
    for name in input_names:
        buffers[name].normal_()  # Fill with random normal values in-place
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run Inference
    # ─────────────────────────────────────────────────────────────────────────
    # Bind CUDA data pointers of the PyTorch tensors to the execution context
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    # Create a CUDA stream for asynchronous execution
    stream = torch.cuda.Stream()

    print("Running forward pass via TensorRT Context...")
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    print("Inference complete!")

    # Print results
    for name in output_names:
        out_tensor = buffers[name]
        print(f"  Output Tensor: '{name}'")
        print(f"    Shape : {list(out_tensor.shape)}")
        print(f"    Mean  : {out_tensor.mean().item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Build and run TensorRT Engine")
    parser.add_argument(
        "--onnx",
        type=str,
        default="accelerate_dev/resnet50.onnx",
        help="Path to input ONNX model file",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="accelerate_dev/resnet50.engine",
        help="Path to save or load compiled engine file",
    )
    parser.add_argument(
        "--build-method",
        type=str,
        default="api",
        choices=["api", "trtexec"],
        help="Compile using Python API ('api') or trtexec command-line ('trtexec')",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="Inference precision configuration (fp32, fp16, int8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for running inference",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip engine compilation, just load and run existing engine file",
    )
    args = parser.parse_args()

    # Create directory for engine artifact if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.engine)), exist_ok=True)

    if not args.skip_build:
        if not os.path.isfile(args.onnx):
            print(f"[Error] Input ONNX file not found at: {args.onnx}. Please run export_onnx.py first.")
            raise SystemExit(1)

        success = False
        if args.build_method == "api":
            success = build_engine_with_api(args.onnx, args.engine, args.precision)
        else:
            success = build_engine_with_trtexec(args.onnx, args.engine, args.precision)

        if not success:
            print("[Error] Failed to build TensorRT engine.")
            raise SystemExit(1)

    # Load and run the built engine
    if not torch.cuda.is_available():
        print("[Error] CUDA is not available on this system. Cannot run TensorRT engine.")
        raise SystemExit(1)
        
    run_engine_inference(args.engine, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
