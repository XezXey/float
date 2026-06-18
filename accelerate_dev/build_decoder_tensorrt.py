#!/usr/bin/env python3
"""
[#] 17 June 2026
====================================================================
build_decoder_tensorrt.py
====================================================================
Builds a TensorRT engine given FLOAT's Decoder ONNX model.
Supports FP32, FP16, TF32, and FP8 precision modes.
Configures dynamic batch profiles for all inputs.
"""

import os
import argparse
import tensorrt as trt
import onnx
from rich.console import Console
console = Console()
print = console.print
    
parser = argparse.ArgumentParser(description="Build TensorRT engine FLOAT's Decoder ONNX model")
parser.add_argument(
    "--input_onnx_path",
    type=str,
    default="./onnx_models/float_decoder.onnx",
    help="Path to the decoder ONNX model file",
)
parser.add_argument(
    "--output_engine_path",
    type=str,
    default="./trt_models/float_decoder.trt",
    help="Path to save the compiled TensorRT engine file/folder",
)
parser.add_argument(
    "--precision",
    type=str,
    choices=["fp32", "fp16", "tf32", "fp8"],
    default="fp32",
    help="Precision mode: fp32, fp16, tf32, or fp8 (default: fp32)",
)
parser.add_argument(
    "--min_batch",
    type=int,
    default=1,
    help="Minimum batch size (default: 1)",
)
parser.add_argument(
    "--opt_batch",
    type=int,
    default=1,
    help="Optimal batch size (default: 1)",
)
parser.add_argument(
    "--max_batch",
    type=int,
    default=1,
    help="Maximum batch size (default: 1)",
)

args = parser.parse_args()
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def inspect_onnx_model(onnx_path):
    #NOTE: Checking the ONNX model's input shapes and types
    print("=" * 100)
    print(f"\[#] Inspecting ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)
    dynamic = False
    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(d.dim_param or -1)
                dynamic = True
        print(f"  Input: '{inp.name}', shape: {shape}, dtype: {inp.type.tensor_type.elem_type}")
    print("=" * 100)
    return dynamic

def build_engine(input_onnx_path, output_engine_path=None, precision="fp32",
                 min_batch=1, opt_batch=1, max_batch=1):

    is_dynamic = inspect_onnx_model(input_onnx_path)

    #NOTE: Create builder, network, and parser
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        #NOTE: Set memory pool limit: 4GB
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        #NOTE: Precision flags config
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
            config.set_flag(trt.BuilderFlag.FP16)
            print("[green]\[#] FP8 precision enabled (with FP16 fallback)")
        else: # fp32
            config.clear_flag(trt.BuilderFlag.TF32)
            print("[green]\[#] Strict FP32 precision enabled (TF32 disabled)")

        with open(input_onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[red] \[!] Parser error: {parser.get_error(i)}")
                return None

        if is_dynamic:
            print(f"[green]\[#] Configuring dynamic optimization profile (min={min_batch}, opt={opt_batch}, max={max_batch})")
            profile = builder.create_optimization_profile()
            B_min, B_opt, B_max = min_batch, opt_batch, max_batch

            profile.set_shape("wa",    (B_min, 512),            (B_opt, 512),            (B_max, 512))
            profile.set_shape("feat0", (B_min, 512, 8, 8),      (B_opt, 512, 8, 8),      (B_max, 512, 8, 8))
            profile.set_shape("feat1", (B_min, 512, 16, 16),    (B_opt, 512, 16, 16),    (B_max, 512, 16, 16))
            profile.set_shape("feat2", (B_min, 512, 32, 32),    (B_opt, 512, 32, 32),    (B_max, 512, 32, 32))
            profile.set_shape("feat3", (B_min, 256, 64, 64),    (B_opt, 256, 64, 64),    (B_max, 256, 64, 64))
            profile.set_shape("feat4", (B_min, 128, 128, 128),  (B_opt, 128, 128, 128),  (B_max, 128, 128, 128))
            profile.set_shape("feat5", (B_min, 64, 256, 256),   (B_opt, 64, 256, 256),   (B_max, 64, 256, 256))
            profile.set_shape("feat6", (B_min, 32, 512, 512),   (B_opt, 32, 512, 512),   (B_max, 32, 512, 512))

            config.add_optimization_profile(profile)
        else:
            print(f"[yellow]\[#] Model has fully static input shapes. Skipping optimization profile.")

        # print(f"[green]\[#] Building engine...")
        # serialized = builder.build_serialized_network(network, config)
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
                output_engine_path = os.path.join(output_engine_path, engine_filename)
            else:
                dir_name = os.path.dirname(output_engine_path)
                file_name = os.path.basename(output_engine_path)
                base_name, ext = os.path.splitext(file_name)
                
                precision_suffix = f"_{precision}"
                if precision_suffix.lower() not in base_name.lower():
                    base_name = f"{base_name}_{precision}"
                    ext = ".trt"
                
                engine_path = os.path.join(dir_name, f"{base_name}{ext}")

            # Ensure parent directories exist
            parent_dir = os.path.dirname(engine_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            with open(engine_path, "wb") as f:
                f.write(serialized)
            print(f"[green]\[#] Saved to {engine_path}")

        return serialized

def main():
    print("=" * 100)
    print(f"[bold cyan]\[#] Building TensorRT engine from ONNX model: {args.input_onnx_path}[/bold cyan]")
    build_engine(
        input_onnx_path=args.input_onnx_path,
        output_engine_path=args.output_engine_path,
        precision=args.precision,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
    )

if __name__ == "__main__":
    main()
