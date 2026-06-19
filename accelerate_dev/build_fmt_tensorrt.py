#!/usr/bin/env python3
"""
[#] 17 June 2026
====================================================================
build_fmt_tensorrt.py
====================================================================
Builds a TensorRT engine given FLOAT's FMT ONNX model.
Supports FP32, FP16, TF32, and FP8 precision modes.
Configures dynamic batch profiles for all inputs.
"""

import tensorrt as trt
import argparse
import onnx
import os
import pyfiglet
from rich.console import Console
console = Console()
print = console.print

parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
parser.add_argument(
    "--input_onnx_path",
    type=str,
    default="accelerate_dev/float_fmt.onnx",
    help="Path to the ONNX model file",
)
parser.add_argument(
    "--output_engine_path",
    type=str,
    default="accelerate_dev/float_fmt.engine",
    help="Path to save the compiled TensorRT engine file",
)
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "tf32", "fp8"],
    default="fp32",
    help="Precision mode: fp32, fp16, tf32, or fp8 (default: fp32)",
)

args = parser.parse_args()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def inspect_onnx_model(onnx_path):
    #NOTE: Checking the ONNX model's input shapes and types
    print("=" * 100)
    print(f"\[#] Inspecting ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  Input: '{inp.name}', shape: {shape}, dtype: {inp.type.tensor_type.elem_type}")
    print("=" * 100)

def build_engine(input_onnx_path, output_engine_path=None, precision="fp32"):

    inspect_onnx_model(input_onnx_path)
    
    #NOTE: Create builder, network, and parser
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        # Precision flags config
        print(f"\[#] Configuring precision mode: {precision}")
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("[green]\[#] FP16 precision enabled")
            else:
                print("[yellow]\[!] FP16 warning: Platform does not support fast FP16 execution.")
        elif precision == "tf32":
            config.set_flag(trt.BuilderFlag.TF32)
            print("[green]\[#] TF32 precision enabled")
        elif precision == "fp8":
            config.set_flag(trt.BuilderFlag.FP8)
            # # Enable FP16 as fallback for layers that don't support FP8
            config.set_flag(trt.BuilderFlag.FP16)
            print("[green]\[#] FP8 precision enabled (with FP16 fallback)")
        else: # fp32
            # Clear default TF32 if strict FP32 is requested
            config.clear_flag(trt.BuilderFlag.TF32)
            print("[green]\[#] Strict FP32 precision enabled (TF32 disabled)")

        with open(input_onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[red] \[!] Parser error: {parser.get_error(i)}")
                return None

        print(f"[yellow]\[#] Using the static batch size = 1 (for autoregressive generation).")
        
        with console.status("[green] Building engine...", spinner="dots"):
            serialized = builder.build_serialized_network(network, config)

        if serialized is None:
            print(f"[red]\[!] ERROR: build failed")
            return None
        else:
            print(f"[green]\[#] Successfully built TensorRT engine from ONNX model: {input_onnx_path}")

        if output_engine_path:
            # Check if output_engine_path is a directory or ends with a slash
            if os.path.isdir(output_engine_path) or output_engine_path.endswith(('/', '\\')):
                onnx_base = os.path.splitext(os.path.basename(input_onnx_path))[0]
                engine_filename = f"{onnx_base}_{precision}.trt"
                save_engine_path = os.path.join(output_engine_path, engine_filename)
            else:
                # It is a filename. Check if user already specified '_<precision>' in the base filename
                dir_name = os.path.dirname(output_engine_path)
                file_name = os.path.basename(output_engine_path)
                base_name, ext = os.path.splitext(file_name)
                
                precision_suffix = f"_{precision}"
                if precision_suffix.lower() not in base_name.lower():
                    base_name = f"{base_name}_{precision}"
                    ext = ".trt"
                
                save_engine_path = os.path.join(dir_name, f"{base_name}{ext}")

            # Ensure parent directories exist
            parent_dir = os.path.dirname(save_engine_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(save_engine_path, "wb") as f:
                f.write(serialized)
            print(f"[green]\[#] Saved to {save_engine_path}")

        return serialized


def main():
    print("=" * 120)
    print(pyfiglet.figlet_format("TensorRT for FLOAT's FMT", width=150))
    print("=" * 120)
    print(f"[bold cyan]\[#] Building TensorRT engine from ONNX model: {args.input_onnx_path}[/bold cyan]")
    build_engine(
        input_onnx_path=args.input_onnx_path,
        output_engine_path=args.output_engine_path,
        precision=args.precision
    )

if __name__ == "__main__":
    main()