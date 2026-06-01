import tensorrt as trt
import argparse
import onnx
parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
parser.add_argument(
    "--onnx_path",
    type=str,
    default="accelerate_dev/float_fmt.onnx",
    help="Path to the ONNX model file",
)
parser.add_argument(
    "--engine_path",
    type=str,
    default="accelerate_dev/float_fmt.engine",
    help="Path to save the compiled TensorRT engine file",
)
args = parser.parse_args()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def inspect_onnx_model(onnx_path):
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param 
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  name: {inp.name}, shape: {shape}, dtype: {inp.type.tensor_type.elem_type}")

def build_engine(onnx_path, engine_path=None, fp16=True,
                 min_batch=1, opt_batch=1, max_batch=4):

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("Parser error:", parser.get_error(i))
                return None

        profile = builder.create_optimization_profile()

        # Fixed inputs (no dynamic dims — shape is already fully static)
        profile.set_shape("t",          (1,),               (1,),               (1,))
        profile.set_shape("a_cfg_scale",(1,),               (1,),               (1,))
        profile.set_shape("e_cfg_scale",(1,),               (1,),               (1,))

        # Dynamic batch inputs
        B_min, B_opt, B_max = min_batch, opt_batch, max_batch
        profile.set_shape("x",       (B_min, 50, 512),  (B_opt, 50, 512),  (B_max, 50, 512))
        profile.set_shape("wa",      (B_min, 50, 512),  (B_opt, 50, 512),  (B_max, 50, 512))
        profile.set_shape("wr",      (B_min, 512),      (B_opt, 512),      (B_max, 512))
        profile.set_shape("we",      (B_min, 1, 7),     (B_opt, 1, 7),     (B_max, 1, 7))
        profile.set_shape("prev_x",  (B_min, 10, 512),  (B_opt, 10, 512),  (B_max, 10, 512))
        profile.set_shape("prev_wa", (B_min, 10, 512),  (B_opt, 10, 512),  (B_max, 10, 512))

        config.add_optimization_profile(profile)

        print("Building engine...")
        serialized = builder.build_serialized_network(network, config)

        if serialized is None:
            print("ERROR: build failed")
            return None

        if engine_path:
            with open(engine_path, "wb") as f:
                f.write(serialized)
            print(f"Saved to {engine_path}")

        return serialized

inspect_onnx_model(args.onnx_path)
serialized_engine = build_engine(args.onnx_path, args.engine_path)