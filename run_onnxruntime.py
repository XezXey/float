#!/usr/bin/env python3
"""
run_onnxruntime.py
====================================================================
Initializes an ONNX Runtime inference session, runs a single forward pass,
and prints the output shape. Supports selecting execution providers (CUDA vs CPU).
"""

import os
import argparse
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using ONNX Runtime"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="accelerate_dev/resnet50.onnx",
        help="Path to the exported ONNX model file",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution provider to use (cuda or cpu)",
    )
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Configure Execution Providers
    # ─────────────────────────────────────────────────────────────────────────
    if args.provider == "cuda":
        # CUDAExecutionProvider is preferred, fallback to CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
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
        print("Please check that your ONNX model path and environment are correct.")
        raise SystemExit(1)

    # Print the execution provider that was actually selected
    active_providers = session.get_providers()
    print(f"Active providers in session: {active_providers}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Get Input and Output Node Information
    # ─────────────────────────────────────────────────────────────────────────
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"Model Inputs ({len(inputs)}):")
    for i, inp in enumerate(inputs):
        print(f"  Input {i}: Name='{inp.name}', Type={inp.type}, Shape={inp.shape}")

    print(f"Model Outputs ({len(outputs)}):")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: Name='{out.name}', Type={out.type}, Shape={out.shape}")

    primary_input = inputs[0]
    primary_output = outputs[0]

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Handle Dynamic Axes and Generate Dummy Inputs
    # ─────────────────────────────────────────────────────────────────────────
    # If the input shape has dynamic batch sizes (represented as strings or None/-1)
    # we replace them with a default batch size of 1.
    run_shape = []
    for dim in primary_input.shape:
        if isinstance(dim, str) or dim is None or dim < 0:
            run_shape.append(1)  # Default dynamic axis to 1
        else:
            run_shape.append(dim)

    print(f"Resolving input shape for run: {primary_input.shape} -> {run_shape}")
    dummy_input = np.random.randn(*run_shape).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Execute Forward Pass
    # ─────────────────────────────────────────────────────────────────────────
    print("Running forward pass via ONNX Runtime...")
    feed_dict = {primary_input.name: dummy_input}
    
    # Run the model (session.run returns a list of outputs)
    ort_outputs = session.run([primary_output.name], feed_dict)
    
    output_array = ort_outputs[0]
    print(f"Inference complete!")
    print(f"Output shape: {output_array.shape}")
    print(f"Output mean : {output_array.mean():.4f}")


if __name__ == "__main__":
    main()
