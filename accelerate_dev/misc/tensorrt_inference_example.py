#!/usr/bin/env python3
"""
tensorrt_inference_example.py
====================================================================
Production-ready template demonstrating how to load and run inference on the
compiled FLOAT FMT TensorRT engine (.engine/.trt) using the TensorRT Python API and PyTorch.
Contains the self-contained TRTPredictor class that adapts buffer allocation,
memory copying, and streaming implementations from the reference code.
"""

import os
import sys
import time
import numpy as np
import torch

try:
    import tensorrt as trt
except ImportError:
    trt = None


class TRTPredictor:
    """
    TRTPredictor manages the lifetime of a TensorRT engine and execution context,
    pre-allocates GPU buffers using PyTorch CUDA tensors, and executes asynchronous
    inference on a custom CUDA stream with zero-copy bindings.
    """
    def __init__(self, engine_path: str = "accelerate_dev/float_fmt.engine", batch_size: int = 1,
                 n_curr: int = 50, n_prev: int = 10, dim_w: int = 512, dim_a: int = 512, dim_e: int = 7):
        """
        Loads the TensorRT engine and prepares execution buffers.
        
        Args:
            engine_path (str): Path to the compiled engine plan.
            batch_size (int): Batch size dimension to reserve.
            n_curr (int): Current frame sequence length (L).
            n_prev (int): Previous frame context length (P).
            dim_w (int): Face motion dimension.
            dim_a (int): Audio features dimension.
            dim_e (int): Emotion features dimension.
        """
        self.engine_path = engine_path
        self.batch_size = batch_size
        self.n_curr = n_curr
        self.n_prev = n_prev
        self.n_out = n_prev + n_curr
        self.dim_w = dim_w
        self.dim_a = dim_a
        self.dim_e = dim_e

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
        print("[TRTPredictor] Allocating GPU device buffers...")
        self.buffers = {
            # inputs
            "t":             torch.zeros(1,                    dtype=torch.float32, device="cuda"),
            "x":             torch.zeros(batch_size, n_curr, dim_w, dtype=torch.float32, device="cuda"),
            "wa":            torch.zeros(batch_size, n_curr, dim_a, dtype=torch.float32, device="cuda"),
            "wr":            torch.zeros(batch_size,        dim_w, dtype=torch.float32, device="cuda"),
            "we":            torch.zeros(batch_size, 1,     dim_e, dtype=torch.float32, device="cuda"),
            "prev_x":        torch.zeros(batch_size, n_prev, dim_w, dtype=torch.float32, device="cuda"),
            "prev_wa":       torch.zeros(batch_size, n_prev, dim_a, dtype=torch.float32, device="cuda"),
            # output: Shape is (B, n_prev + n_curr, dim_w)
            "output":        torch.zeros(batch_size, self.n_out, dim_w, dtype=torch.float32, device="cuda"),
        }

        # Print layout information
        for name, tensor in self.buffers.items():
            role = "Output" if name == "output" else "Input"
            print(f"  {role} -> '{name}': Shape={list(tensor.shape)}, Type={tensor.dtype}")

    def predict(self, t, x, wa, wr, we, prev_x, prev_wa) -> torch.Tensor:
        """
        Runs asynchronous inference on the loaded GPU engine.
        Copies data to preallocated buffers, binds pointers, sets input shapes,
        and triggers execution.
        
        Args:
            t, x, wa, wr, we, prev_x, prev_wa: PyTorch tensors or NumPy arrays matching
                                               the input specifications of FMTWrapper.
                                               Can reside on CPU or GPU.
                                               
        Returns:
            torch.Tensor: The computed motion latent output tensor (B, P+L, 512) on CUDA.
        """
        def _to_buf(name: str, src):
            if isinstance(src, torch.Tensor):
                self.buffers[name].copy_(src.to(dtype=torch.float32, device="cuda"), non_blocking=True)
            elif isinstance(src, np.ndarray):
                src_torch = torch.from_numpy(src).to(dtype=torch.float32, device="cuda")
                self.buffers[name].copy_(src_torch, non_blocking=True)
            else:
                # Handle scalar case for t
                if name == "t":
                    self.buffers["t"].fill_(float(src))
                else:
                    raise TypeError(f"Input '{name}' must be a numpy.ndarray or a torch.Tensor")

        # 1. Copy data into our pre-allocated device buffers (non-blocking)
        _to_buf("t", t)
        _to_buf("x", x)
        _to_buf("wa", wa)
        _to_buf("wr", wr)
        _to_buf("we", we)
        _to_buf("prev_x", prev_x)
        _to_buf("prev_wa", prev_wa)

        # 2. Set dynamic input shapes in context
        input_names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
        for name in input_names:
            self.context.set_input_shape(name, self.buffers[name].shape)

        # 3. Bind the CUDA memory pointers of the buffers to the context
        for name, tensor in self.buffers.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # 4. Execute inference asynchronously on our custom CUDA stream
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        # 5. Synchronize the stream to block until execution completes
        self.stream.synchronize()

        # 6. Return the output CUDA tensor
        return self.buffers["output"]

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
    engine_path = "accelerate_dev/float_fmt.engine"

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

    # Generate dummy input tensors (can be on CPU or GPU)
    print("\nGenerating dummy inputs...")
    t = 0.5
    x = torch.randn(1, 50, 512)
    wa = torch.randn(1, 50, 512)
    wr = torch.randn(1, 512)
    we = torch.randn(1, 1, 7)
    prev_x = torch.zeros(1, 10, 512)
    prev_wa = torch.zeros(1, 10, 512)

    # Run predictor
    print("Running prediction via TRTPredictor...")
    outputs = predictor.predict(t, x, wa, wr, we, prev_x, prev_wa)

    print("Inference completed!")
    print(f"  Output type  : {type(outputs)}")
    print(f"  Output device: {outputs.device}")
    print(f"  Output shape : {list(outputs.shape)}")
    print(f"  Output mean  : {outputs.mean().item():.4f}")


if __name__ == "__main__":
    main()
