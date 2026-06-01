#!/usr/bin/env python3
"""
export_onnx.py
====================================================================
Loads the pre-trained FLOAT PyTorch model, wraps the FlowMatchingTransformer
using FMTWrapper (from tensorrt._fmt_utils), creates dummy inputs matching
the target shapes required by FMTWrapper/forward_with_cfv, and exports it to ONNX.
Saves the exported model inside the 'accelerate_dev/' directory.
"""

import os
import sys
import argparse
import torch

# Ensure the root folder and tensorrt folder are in the Python search path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _fmt_utils import load_fmt_wrapper, build_dummy_inputs, add_model_args


def main():
    parser = argparse.ArgumentParser(description="Export FLOAT FlowMatchingTransformer (FMT) to ONNX format")
    
    # Model and input options
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/float.pth",
        help="Path to the PyTorch checkpoint file (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="accelerate_dev",
        help="Directory to save the exported ONNX model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="float_fmt.onnx",
        help="Filename of the exported ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (e.g., 14, 16, 17)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the dummy inputs during export",
    )
    parser.add_argument(
        "--disable-constant-folding",
        action="store_true",
        help="Disable constant folding optimization during export",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Disable dynamic batch size dimensions",
    )
    
    # Add default options from the base FLOAT model arguments
    parser = add_model_args(parser)
    args = parser.parse_args()
    
    # Initialize rank and ngpus as required by FlowMatchingTransformer device placements
    args.rank = 0 if torch.cuda.is_available() else "cpu"
    args.ngpus = 1

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.model_name)

    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the FMT model wrapper
    print(f"Loading FMT model from checkpoint: {args.ckpt_path}")
    if not os.path.isfile(args.ckpt_path):
        print(f"Error: Checkpoint file not found at '{args.ckpt_path}'. Please make sure float.pth exists.")
        sys.exit(1)
        
    wrapper = load_fmt_wrapper(args, device)
    
    # 3. Create dummy inputs matching the FMTWrapper forward signature
    # Signature: forward(t, x, wa, wr, we, prev_x, prev_wa)
    print(f"Generating dummy inputs with batch size = {args.batch_size}...")
    dummy_inputs = build_dummy_inputs(args, device, batch=args.batch_size)
    
    t, x, wa, wr, we, prev_x, prev_wa, a_cfg_scale, e_cfg_scale = dummy_inputs
    print(f"Input shapes:")
    print(f"  t       : {list(t.shape)}")
    print(f"  x       : {list(x.shape)}")
    print(f"  wa      : {list(wa.shape)}")
    print(f"  wr      : {list(wr.shape)}")
    print(f"  we      : {list(we.shape)}")
    print(f"  prev_x  : {list(prev_x.shape)}")
    print(f"  prev_wa : {list(prev_wa.shape)}")
    print(f"  a_cfg_scale : {a_cfg_scale.item()}")
    print(f"  e_cfg_scale : {e_cfg_scale.item()}")

    # 4. Define input names, output names, and dynamic axes
    input_names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa", "a_cfg_scale", "e_cfg_scale"]
    output_names = ["output"]

    dynamic_axes = None
    if not args.no_dynamic:
        # We make the batch dimension dynamic for all tensor inputs and outputs.
        # Note: 't' represents the scalar timestep and has shape (1,), so we do not make it dynamic.
        dynamic_axes = {
            "x": {0: "batch_size"},
            "wa": {0: "batch_size"},
            "wr": {0: "batch_size"},
            "we": {0: "batch_size"},
            "prev_x": {0: "batch_size"},
            "prev_wa": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
        print("Dynamic batch axis enabled for all tensor inputs and outputs.")

    do_constant_folding = not args.disable_constant_folding

    # 5. Perform ONNX export
    print(f"Exporting model to ONNX format (opset_version={args.opset})...")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Successfully exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
