#!/usr/bin/env python3
"""
onnx_inference_example.py
====================================================================
Production-ready template illustrating how to load and run inference on the
exported FLOAT FMT ONNX model using ONNX Runtime.
Contains the self-contained ONNXPredictor class, which mimics the expected
input/output mapping of the FlowMatchingTransformer forward_with_cfv execution pathway.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
import torch
import argparse

parser = argparse.ArgumentParser(description="ONNX Inference Example for FLOAT FMT Model")
parser.add_argument(
    "--model_path",
    type=str,
    default="accelerate_dev/float_fmt.onnx",
    help="Path to the ONNX model file exported by export_onnx.py",
)


class ONNXPredictor:
    """
    ONNXPredictor encapsulates an ONNX Runtime inference session for the FLOAT FMT model.
    It manages execution providers and handles converting PyTorch tensors or NumPy arrays
    to the formats expected by ONNX Runtime, returning outputs aligned with forward_with_cfv.
    """
    def __init__(self, model_path: str = "accelerate_dev/float_fmt.onnx", provider: str = "cuda"):
        """
        Initializes the InferenceSession.
        
        Args:
            model_path (str): Path to the ONNX model file.
            provider (str): Execution provider target: 'cuda' or 'cpu'.
        """
        self.model_path = model_path
        
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"ONNX model file not found at '{self.model_path}'. "
                "Please run export_onnx.py to export it first."
            )
            
        # Configure execution providers
        if provider == "cuda":
            self.providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider"
            ]
        else:
            self.providers = ["CPUExecutionProvider"]
            
        print(f"[ONNXPredictor] Initializing session with providers: {self.providers}")
        self.session_options = ort.SessionOptions()
        self.session_options.log_severity_level = 3  # Suppress INFO and WARNING logs from ONNX Runtime, 0 = VERBOSE, 1 = INFO, 2 = WARNING, 3 = ERROR
        self.session = ort.InferenceSession(self.model_path, providers=self.providers, sess_options=self.session_options)

        
        # Log which provider was actually chosen by ONNX Runtime
        active_providers = self.session.get_providers()
        print(f"[ONNXPredictor] Session successfully created. Active providers: {active_providers}")
        
        # Get metadata
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        
        # Save input names for quick access during run
        self.input_names = [inp.name for inp in self.inputs]
        self.output_name = self.outputs[0].name
        
        print(f"[ONNXPredictor] Inputs expected: {self.input_names}")
        print(f"[ONNXPredictor] Output name: '{self.output_name}'")

    def predict(self, t, x, wa, wr, we, prev_x, prev_wa):
        """
        Executes a forward pass mimicking model.fmt.forward_with_cfv.
        
        Args:
            t       : scalar or (1,) tensor of ODE timestep
            x       : (B, L, 512) motion latent
            wa      : (B, L, 512) audio features
            wr      : (B, 512) reference motion latent
            we      : (B, 1, 7) emotion latent
            prev_x  : (B, P, 512) previous motion context
            prev_wa : (B, P, 512) previous audio context
            
        Returns:
            torch.Tensor or np.ndarray: Output vector field shape (B, P+L, 512)
            Returns a torch.Tensor on the device of input 'x' if input is a PyTorch tensor,
            otherwise returns a NumPy array.
        """
        is_torch = isinstance(x, torch.Tensor)
        device = x.device if is_torch else None

        def to_numpy(tensor, name):
            if isinstance(tensor, np.ndarray):
                return tensor.astype(np.float32)
            elif isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy().astype(np.float32)
            else:
                raise TypeError(f"Input '{name}' must be a numpy.ndarray or a torch.Tensor")

        # 1. Convert all inputs to float32 NumPy arrays for ONNX Runtime
        feed_dict = {
            "t": to_numpy(t, "t"),
            "x": to_numpy(x, "x"),
            "wa": to_numpy(wa, "wa"),
            "wr": to_numpy(wr, "wr"),
            "we": to_numpy(we, "we"),
            "prev_x": to_numpy(prev_x, "prev_x"),
            "prev_wa": to_numpy(prev_wa, "prev_wa")
        }

        # 2. Run session
        ort_outputs = self.session.run([self.output_name], feed_dict)
        output_numpy = ort_outputs[0]

        # 3. Cast output to match the input framework
        if is_torch:
            return torch.from_numpy(output_numpy).to(device=device)
        return output_numpy


# ─────────────────────────────────────────────────────────────────────────────
# Integration Example / Usage Demo
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parser.parse_args()
    model_path = args.model_path

    if not os.path.isfile(model_path):
        print(f"ONNX file '{model_path}' not found. Please run export_onnx.py first.")
        return

    # Initialize predictor
    predictor = ONNXPredictor(model_path=model_path, provider="cuda")

    # Generate dummy input arrays (using PyTorch tensors to test the auto-conversion)
    print("\nGenerating dummy inputs as PyTorch tensors...")
    
    t = torch.tensor([0.5])
    x = torch.randn(1, 50, 512, device="cuda")
    wa = torch.randn(1, 50, 512, device="cuda")
    wr = torch.randn(1, 512, device="cuda")
    we = torch.randn(1, 1, 7, device="cuda")
    prev_x = torch.randn(1, 10, 512, device="cuda")
    prev_wa = torch.randn(1, 10, 512, device="cuda")

    # Run predictor
    print("Running prediction via ONNXPredictor...")
    outputs = predictor.predict(t, x, wa, wr, we, prev_x, prev_wa)

    # Decode and print output shape and device
    print("Inference completed!")
    print(f"  Output type  : {type(outputs)}")
    if isinstance(outputs, torch.Tensor):
        print(f"  Output device: {outputs.device}")
    print(f"  Output shape : {list(outputs.shape)}")
    print(f"  Output mean  : {outputs.mean().item():.4f}")


if __name__ == "__main__":
    main()
