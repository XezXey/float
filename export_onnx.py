#!/usr/bin/env python3
"""
export_onnx.py
====================================================================
Exports a PyTorch model (pre-trained ResNet-50 as a placeholder) to ONNX format.
Saves the exported model directly inside the 'accelerate_dev/' directory.
Includes configurations for opset version, constant folding, input/output names,
and dynamic axes support.
"""

import os
import argparse
import torch
import torchvision.models as models


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="accelerate_dev",
        help="Directory to save the exported ONNX model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet50.onnx",
        help="Filename of the exported ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (e.g., 11, 14, 17)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dummy input",
    )
    parser.add_argument(
        "--disable-constant-folding",
        action="store_true",
        help="Disable constant folding optimization during export",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axis for batch dimension",
    )
    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.model_name)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load PyTorch model
    # ─────────────────────────────────────────────────────────────────────────
    # [CUSTOMIZATION POINT] Replace this block with your own model loading.
    # e.g.:
    # from models import MyModel
    # model = MyModel()
    # model.load_state_dict(torch.load("path/to/weights.pth", map_location="cpu"))
    print("Loading PyTorch model (ResNet50 placeholder)...")
    try:
        # Load pre-trained ResNet-50 weights
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception as e:
        print(f"Warning: Failed to load pre-trained weights ({e}). Initializing random ResNet50.")
        model = models.resnet50()

    # Put the model in evaluation mode (crucial for batchnorm, dropout, etc.)
    model.eval()

    # Move model to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model placed on device: {device}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Create Dummy Input
    # ─────────────────────────────────────────────────────────────────────────
    # [CUSTOMIZATION POINT] Modify the shape and data type to match your model's inputs.
    # ResNet-50 expects standard ImageNet layout: [Batch, Channels, Height, Width]
    dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=device)
    print(f"Dummy input shape: {list(dummy_input.shape)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Configure Input/Output Names and Dynamic Axes
    # ─────────────────────────────────────────────────────────────────────────
    # [CUSTOMIZATION POINT] Name inputs/outputs matching your architecture/serving requirements
    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if args.dynamic:
        # Define dynamic axes mapping name to index-to-dimension-name dict
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        print("Dynamic batch size dimension enabled.")

    do_constant_folding = not args.disable_constant_folding

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Perform ONNX Export
    # ─────────────────────────────────────────────────────────────────────────
    print(f"Exporting model to ONNX format (opset_version={args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Successfully exported and saved ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
