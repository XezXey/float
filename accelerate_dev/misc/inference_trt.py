import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
cuda.init()
# Use this instead of autoinit
context = cuda.Device(0).make_context() 

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
    
# --- Init (once) ---
inferencer = TRTInferencer("./trt_models/fmt_onnx_maskfill_addcfg.trt")

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

context.pop()  # Clean up CUDA context when done