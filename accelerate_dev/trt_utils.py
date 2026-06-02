import tensorrt as trt
import torch
import pycuda.driver as cuda
# import pycuda.autoinit
# Use this instead of autoinit

import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInferencer:
    def __init__(self, engine_path):
        # Load engine
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.inputs = {}   # name -> {"host": ..., "device": ...}
        self.outputs = {}  # name -> {"host": ..., "device": ...}
        
        print("Engine I/O tensors:")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)  # may have -1 for dynamic
            print(f"  {'INPUT ' if mode == trt.TensorIOMode.INPUT else 'OUTPUT'} | {name:20s} | {str(dtype):10s} | {tuple(shape)}")

    def _alloc_buffer(self, shape, dtype):
        size = int(np.prod(shape))
        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        return {"host": host_mem, "device": device_mem, "shape": shape}

    def infer(self, batch_size: int, t, x, wa, wr, we, prev_x, prev_wa, a_cfg_scale, e_cfg_scale):
        B = batch_size

        # --- Define actual shapes for this call ---
        input_data = {
            "t":           (t,           (1,)),
            "x":           (x,           (B, 50, 512)),
            "wa":          (wa,          (B, 50, 512)),
            "wr":          (wr,          (B, 512)),
            "we":          (we,          (B, 1, 7)),
            "prev_x":      (prev_x,      (B, 10, 512)),
            "prev_wa":     (prev_wa,     (B, 10, 512)),
            "a_cfg_scale": (a_cfg_scale, (1,)),
            "e_cfg_scale": (e_cfg_scale, (1,)),
        }

        bindings = {}

        # --- Upload inputs ---
        for name, (data, shape) in input_data.items():
            arr = np.ascontiguousarray(data.ravel(), dtype=np.float32)
            buf = self._alloc_buffer(shape, np.float32)
            np.copyto(buf["host"], arr)
            cuda.memcpy_htod_async(buf["device"], buf["host"], self.stream)
            self.context.set_input_shape(name, shape)
            self.context.set_tensor_address(name, int(buf["device"]))
            bindings[name] = buf

        # --- Allocate outputs ---
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))  # resolved shape
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                buf = self._alloc_buffer(shape, dtype)
                self.context.set_tensor_address(name, int(buf["device"]))
                bindings[name] = buf

        # --- Run ---
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # --- Download outputs ---
        results = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                buf = bindings[name]
                cuda.memcpy_dtoh_async(buf["host"], buf["device"], self.stream)
        
        self.stream.synchronize()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                buf = bindings[name]
                results[name] = buf["host"].reshape(buf["shape"]).copy()

        return results

    def cleanup(self):
        self.context.pop()  # Clean up CUDA context when done
        del self.context
        del self.engine
        del self.stream
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        
        # 3. Pop PyCUDA context
        try:
            self.context.pop()
            print("[CUDA] Context popped cleanly.")
        except cuda.LogicError as e:
            print(f"[CUDA] Pop warning: {e}")
    
    def check_engine_precision(self):
        """Check and summarize precision of loaded TRT engine."""
        if self.engine is None:
            print("No engine loaded.")
            return

        print("\n========== ENGINE PRECISION INFO ==========")

        # --- I/O Tensor Info ---
        print("\n[I/O Tensors]")
        io_dtypes = set()
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            mode  = self.engine.get_tensor_mode(name)
            io_dtypes.add(dtype)
            print(f"  {str(mode):<30} {name:<30} dtype={dtype}  shape={shape}")

        # --- Layer Info ---
        print("\n[Layers]")
        inspector = self.engine.create_engine_inspector()
        import json
        engine_json = json.loads(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        )
        raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        layer_precisions = {}
        layers_list = engine_json.get("Layers", [])
        # for layer in layers_list:
        #     name      = layer.get("Name", "unknown")
        #     precision = layer.get("LayerPrecision", "unknown")
        #     layer_precisions[precision] = layer_precisions.get(precision, 0) + 1
        #     print(f"  {name:<50} precision={precision}")
        import re
        for layer in layers_list:
            # Check if the layer entry is a dictionary (Standard TRT behavior)
            if isinstance(layer, dict):
                name = layer.get("Name", "unknown")
                precision = layer.get("LayerPrecision", layer.get("Precision", "unknown"))
            
            # Check if the layer entry is a string (Your current edge-case)
            elif isinstance(layer, str):
                name = layer.strip()
                
                # Try to extract precision if it's written in the string (e.g., "Factoring...", "FP16", "INT8")
                # TensorRT often logs precision in uppercase within these reformatted strings
                precision_match = re.search(r'\b(FP32|FP16|INT8|TF32|Float|Half|Int8)\b', name, re.IGNORECASE)
                precision = precision_match.group(0).upper() if precision_match else "Unknown/Mixed"
                
                # Shorten the name a bit for clean printing if it's too long
                if len(name) > 67:
                    name = name[:64] + "..."
            else:
                continue
            # Print individual layer info
            print(f"{name:<70} | {precision:<10}")
            
            # Track the counts
            layer_precisions[precision] = layer_precisions.get(precision, 0) + 1

        # --- Summary ---
        print("\n[Summary]")
        dtype_map = {
            trt.DataType.FLOAT : "FP32",
            trt.DataType.HALF  : "FP16",
            trt.DataType.INT8  : "INT8",
            trt.DataType.BF16  : "BF16",
            trt.DataType.FP8   : "FP8",
        }
        print(f"  I/O tensor dtypes : {[dtype_map.get(d, str(d)) for d in io_dtypes]}")
        print(f"  Layer precision counts:")
        for precision, count in sorted(layer_precisions.items()):
            print(f"    {precision:<10}: {count} layers")

        print("===========================================\n")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Test TRTInferencer with a sample engine and inputs")
    parser.add_argument("--engine_path", type=str, default="accelerate_dev/float_fmt.engine", help="Path to the TensorRT engine file")
    args = parser.parse_args()
    
    cuda.init()
    context = cuda.Device(0).make_context() 
    # --- Init (once) ---
    inferencer = TRTInferencer(args.engine_path)
    inferencer.check_engine_precision()

    # --- Prepare inputs (example with batch_size=1) ---
    B = 1
    inputs = {
        "t":           np.array([0.5],               dtype=np.float32),
        "x":           np.random.rand(B, 50, 512).astype(np.float32),
        "wa":          np.random.rand(B, 50, 512).astype(np.float32),
        "wr":          np.random.rand(B, 512).astype(np.float32),
        "we":          np.random.rand(B, 1, 7).astype(np.float32),
        "prev_x":      np.random.rand(B, 10, 512).astype(np.float32),
        "prev_wa":     np.random.rand(B, 10, 512).astype(np.float32),
        "a_cfg_scale": np.array([1.0],               dtype=np.float32),
        "e_cfg_scale": np.array([1.0],               dtype=np.float32),
    }

    n = 100
    import time
    # --- Run inference ---
    ss = time.time()
    for _ in range(n):
        start_time = time.time()
        outputs = inferencer.infer(batch_size=B, **inputs)
        print(outputs["output"].shape)
        end_time = time.time()
        print(f"Inference time: {(end_time - start_time):.2f} ms")
    se = time.time()
    print(f"Total time: {(se - ss):.2f} s")

    # --- Use results ---
    for name, tensor in outputs.items():
        print(f"Output '{name}': shape={tensor.shape}, dtype={tensor.dtype}")
    context.pop()
    
