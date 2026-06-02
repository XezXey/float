#!/usr/bin/env python3
"""
run_onnxruntime.py
====================================================================
Initializes an ONNX Runtime inference session on the exported FLOAT FMT model,
generates dummy input arrays using NumPy, runs a forward pass, and prints the output shape.
Supports switching between CUDAExecutionProvider and CPUExecutionProvider.
"""

import os
import argparse
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Run FLOAT FMT inference using ONNX Runtime")
    parser.add_argument(
        "--model",
        type=str,
        default="accelerate_dev/float_fmt.onnx",
        help="Path to the exported ONNX model file",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="ONNX execution provider to use (cuda or cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generating dummy inputs",
    )
    
    # Model dimensions corresponding to base FLOAT options
    parser.add_argument("--n-curr", type=int, default=50, help="Current frame sequence length (L)")
    parser.add_argument("--n-prev", type=int, default=10, help="Previous frame context length (P)")
    parser.add_argument("--dim-w", type=int, default=512, help="Face motion dimension (dim_w)")
    parser.add_argument("--dim-a", type=int, default=512, help="Audio features dimension (dim_a)")
    parser.add_argument("--dim-e", type=int, default=7, help="Emotion features dimension (dim_e)")
    
    args = parser.parse_args()

    # 1. Setup execution providers
    if args.provider == "cuda":
        # Prefer CUDA, fallback to CPU
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider"
        ]
    else:
        providers = ["CPUExecutionProvider"]

    print(f"Loading ONNX model: {args.model}")
    if not os.path.isfile(args.model):
        print(f"Error: ONNX model file not found at '{args.model}'. Please run export_onnx.py first.")
        raise SystemExit(1)

    print(f"Attempting to initialize session with providers: {providers}")
    try:
        session = ort.InferenceSession(args.model, providers=providers)
    except Exception as e:
        print(f"Failed to create ONNX Runtime session: {e}")
        raise SystemExit(1)

    active_providers = session.get_providers()
    print(f"Active providers in session: {active_providers}")

    # 2. Inspect session inputs/outputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"\nModel Inputs ({len(inputs)}):")
    for i, inp in enumerate(inputs):
        print(f"  Input {i}: Name='{inp.name}', Type={inp.type}, Shape={inp.shape}")

    print(f"Model Outputs ({len(outputs)}):")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: Name='{out.name}', Type={out.type}, Shape={out.shape}")

    # 3. Generate dummy NumPy inputs matching FMTWrapper/forward_with_cfv signature
    print(f"\nGenerating dummy NumPy inputs (Batch size={args.batch_size})...")
    np.random.seed(42)
    
    # Signature: t, x, wa, wr, we, prev_x, prev_wa
    t_arr = np.array([0.5], dtype=np.float32)
    x_arr = np.random.randn(args.batch_size, args.n_curr, args.dim_w).astype(np.float32)
    wa_arr = np.random.randn(args.batch_size, args.n_curr, args.dim_a).astype(np.float32)
    wr_arr = np.random.randn(args.batch_size, args.dim_w).astype(np.float32)
    we_arr = np.random.randn(args.batch_size, 1, args.dim_e).astype(np.float32)
    prev_x_arr = np.random.randn(args.batch_size, args.n_prev, args.dim_w).astype(np.float32)
    prev_wa_arr = np.random.randn(args.batch_size, args.n_prev, args.dim_a).astype(np.float32)

    feed_dict = {
        "t": t_arr,
        "x": x_arr,
        "wa": wa_arr,
        "wr": wr_arr,
        "we": we_arr,
        "prev_x": prev_x_arr,
        "prev_wa": prev_wa_arr
    }

    # 4. Run forward pass
    print("Running forward pass via ONNX Runtime...")
    ort_outputs = session.run([outputs[0].name], feed_dict)
    
    output_array = ort_outputs[0]
    print(f"Inference complete!")
    print(f"  Output name : '{outputs[0].name}'")
    print(f"  Output shape: {output_array.shape}")
    print(f"  Output mean : {output_array.mean():.4f}")
    print(f"  Output std  : {output_array.std():.4f}")


if __name__ == "__main__":
    main()
