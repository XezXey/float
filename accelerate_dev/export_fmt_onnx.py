#!/usr/bin/env python3
"""
[#] 17 June 2026
====================================================================
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
import warnings
from rich.console import Console
console = Console()
print = console.print

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
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )
    
    # Add default options from the base FLOAT model arguments
    parser = add_model_args(parser)
    args = parser.parse_args()
    if not args.verbose:
        from torch.jit import TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=TracerWarning)
    
    # Initialize rank and ngpus as required by FlowMatchingTransformer device placements
    args.rank = 0 if torch.cuda.is_available() else "cpu"
    args.ngpus = 1

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cyan]\[#] Using device: {device}")

    # 2. Load the FMT model wrapper
    print(f"Loading FMT model from checkpoint: {args.ckpt_path}")
    if not os.path.isfile(args.ckpt_path):
        print(f"[red]\[!] Error: Checkpoint file not found at '{args.ckpt_path}'.")
        sys.exit(1)
        
    fmt_wrapper = load_fmt_wrapper(args, device)
    print(f"[green]\[#] Successfully loaded FLOAT model weights from: {args.ckpt_path}")
    
    # 3. Create dummy inputs matching the FMTWrapper forward signature
    # Signature: forward(t, x, wa, wr, we, prev_x, prev_wa)
    print(f"[cyan]\[#] Generating dummy inputs with batch size = {args.batch_size}...")
    dummy_inputs = build_dummy_inputs(args, device, batch=args.batch_size)
    
    t, x, wa, wr, we, prev_x, prev_wa, a_cfg_scale, e_cfg_scale = dummy_inputs
    print(f"[cyan]\[#] Input shapes:")
    print(f"[cyan]  t       : {list(t.shape)}")
    print(f"[cyan]  x       : {list(x.shape)}")
    print(f"[cyan]  wa      : {list(wa.shape)}")
    print(f"[cyan]  wr      : {list(wr.shape)}")
    print(f"[cyan]  we      : {list(we.shape)}")
    print(f"[cyan]  prev_x  : {list(prev_x.shape)}")
    print(f"[cyan]  prev_wa : {list(prev_wa.shape)}")
    print(f"[cyan]  a_cfg_scale : {a_cfg_scale.item()}")
    print(f"[cyan]  e_cfg_scale : {e_cfg_scale.item()}")

    # 4. Define input names, output names, and dynamic axes
    input_names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa", "a_cfg_scale", "e_cfg_scale"]
    output_names = ["output"]

    dynamic_axes = None
    # do_constant_folding = not args.disable_constant_folding

    # 5. Perform ONNX export
    print(f"[cyan]\[#] Exporting FLOAT's FMT into ONNX format (opset_version={args.opset})...")
    torch.onnx.export(
        fmt_wrapper,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print(f"[green]\[#] Successfully exported ONNX FMT to: {output_path}")


if __name__ == "__main__":
    print("[green]" + "=" * 100)
    print("[green]FLOAT's FMT ONNX Conversion...")
    print("[green]" + "=" * 100)
    main()
