#!/usr/bin/env python3
"""
tensorrt_inference_example.py
====================================================================
Production-ready template demonstrating how to load and run inference on a
compiled TensorRT engine (.engine/.trt) using the TensorRT Python API and PyTorch.
Includes the self-contained TRTPredictor class.
"""

import os
import torch
import numpy as np

# Try importing TensorRT
try:
    import tensorrt as trt
except ImportError:
    trt = None


class TRTPredictor:
    """
    TRTPredictor manages the lifetime of a TensorRT engine and execution context,
    handles device memory allocation via PyTorch CUDA tensors (ensuring safe, automatic deallocation),
    and executes asynchronous inference with GPU stream synchronization.
    """
    def __init__(self, engine_path: str = "accelerate_dev/resnet50.engine", batch_size: int = 1):
        """
        Loads the TensorRT engine and prepares execution buffers.
        
        Args:
            engine_path (str): Path to the compiled engine plan.
            batch_size (int): Batch size dimension to reserve for dynamic shapes.
        """
        self.engine_path = engine_path
        self.batch_size = batch_size
        
        if trt is None:
            raise ImportError("[TRTPredictor] TensorRT library is not installed.")

        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(
                f"[TRTPredictor] Engine file not found at '{self.engine_path}'. "
                "Please run run_tensorrt.py to compile it first."
            )

        print(f"[TRTPredictor] Loading TensorRT engine from: {self.engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(self.engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
        # ─────────────────────────────────────────────────────────────────────
        # Pre-allocate I/O Buffers using PyTorch CUDA Tensors
        # ─────────────────────────────────────────────────────────────────────
        # Note: Using PyTorch CUDA tensors provides high-performance zero-copy bindings
        # and leverages PyTorch's caching allocator to automatically free GPU memory
        # when the predictor instance is garbage collected.
        self.buffers = {}
        self.input_names = []
        self.output_names = []
        
        print("[TRTPredictor] Allocating GPU device buffers...")
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            
            # Replace dynamic batch/sequence dimensions (-1) with the target batch size
            resolved_shape = [self.batch_size if dim < 0 else dim for dim in shape]
            
            # Map TensorRT datatypes to PyTorch datatypes
            if dtype == trt.DataType.FLOAT:
                torch_dtype = torch.float32
            elif dtype == trt.DataType.HALF:
                torch_dtype = torch.float16
            elif dtype == trt.DataType.INT32:
                torch_dtype = torch.int32
            elif dtype == trt.DataType.INT8:
                torch_dtype = torch.int8
            else:
                torch_dtype = torch.float32
                
            # Allocate CUDA memory buffer using PyTorch
            tensor = torch.zeros(resolved_shape, dtype=torch_dtype, device="cuda")
            self.buffers[name] = tensor
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                print(f"  Input  -> '{name}': Shape={resolved_shape}, Type={torch_dtype}")
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)
                print(f"  Output -> '{name}': Shape={resolved_shape}, Type={torch_dtype}")

        # Assume single-input single-output for simple predictors
        self.primary_input_name = self.input_names[0]
        self.primary_output_name = self.output_names[0]

    def predict(self, input_tensor) -> np.ndarray:
        """
        Runs asynchronous inference on the loaded GPU engine.
        
        Args:
            input_tensor (numpy.ndarray or torch.Tensor): Input data to run.
            
        Returns:
            numpy.ndarray: The computed output array copied back to the host (CPU).
        """
        # Ensure input data is a PyTorch tensor on CPU/GPU or a NumPy array
        if isinstance(input_tensor, np.ndarray):
            # Convert NumPy array to PyTorch CUDA tensor
            inp_gpu = torch.from_numpy(input_tensor).to(device="cuda", non_blocking=True)
        elif hasattr(input_tensor, "device"):
            # Already a PyTorch tensor, cast and/or move to CUDA
            inp_gpu = input_tensor.to(device="cuda", dtype=self.buffers[self.primary_input_name].dtype)
        else:
            raise TypeError("input_tensor must be a numpy.ndarray or a torch.Tensor")

        # 1. Copy the input data into our pre-allocated device buffer
        self.buffers[self.primary_input_name].copy_(inp_gpu, non_blocking=True)

        # 2. Bind the CUDA memory pointers of the buffers to the context
        for name, tensor in self.buffers.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # 3. Execute inference asynchronously on the custom CUDA stream
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        # 4. Synchronize the stream to block until execution completes
        self.stream.synchronize()

        # 5. Extract output tensor and copy to CPU NumPy array
        output_array = self.buffers[self.primary_output_name].cpu().numpy()
        return output_array

    def __del__(self):
        """
        Explicit deallocation hook to destroy the context and runtime objects.
        Device buffers are automatically collected by PyTorch's memory manager.
        """
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine
        print("[TRTPredictor] Resources successfully released.")


# ─────────────────────────────────────────────────────────────────────────────
# Integration Example / Usage Demo
# ─────────────────────────────────────────────────────────────────────────────
def main():
    engine_path = "accelerate_dev/resnet50.engine"

    if trt is None:
        print("TensorRT not installed. Skipping demo run.")
        return

    if not os.path.isfile(engine_path):
        print(f"Demo engine file '{engine_path}' not found. Please compile your engine first using run_tensorrt.py.")
        return

    if not torch.cuda.is_available():
        print("CUDA is not available on this system. Cannot run TensorRT demo.")
        return

    # Instantiate the predictor
    predictor = TRTPredictor(engine_path=engine_path, batch_size=1)

    # [CUSTOMIZATION POINT] Preprocessing pipeline:
    # Load your real input image/feature representation here.
    # Convert it, apply normalization, and reshape to match.
    print("\nGenerating dummy input (representing a preprocessed image/tensor)...")
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Run predictor
    print("Running prediction via TRTPredictor...")
    outputs = predictor.predict(dummy_input)

    # [CUSTOMIZATION POINT] Postprocessing pipeline:
    # Decode the output scores (e.g. Softmax probability, bounding boxes, labels)
    print("Inference completed!")
    print(f"  Output shape : {outputs.shape}")
    print(f"  Output mean  : {outputs.mean():.4f}")

    top1_index = np.argmax(outputs, axis=1)
    print(f"  Predicted class index: {top1_index.tolist()}")


if __name__ == "__main__":
    main()
