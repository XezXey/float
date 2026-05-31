#!/usr/bin/env python3
"""
onnx_inference_example.py
====================================================================
Production-ready template illustrating how to load and run inference on an
exported ONNX model using ONNX Runtime. Contains the self-contained ONNXPredictor class.
"""

import os
import numpy as np
import onnxruntime as ort


class ONNXPredictor:
    """
    ONNXPredictor encapsulates an ONNX Runtime inference session, manages
    execution providers, and coordinates running forward passes on input data.
    """
    def __init__(self, model_path: str = "accelerate_dev/resnet50.onnx", provider: str = "cuda"):
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
            
        # Select execution providers
        if provider == "cuda":
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]
            
        print(f"[ONNXPredictor] Initializing session with providers: {self.providers}")
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        
        # Log which provider was actually chosen by ONNX Runtime
        active_providers = self.session.get_providers()
        print(f"[ONNXPredictor] Session successfully created. Active providers: {active_providers}")
        
        # Get input and output metadata
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        
        self.input_name = self.inputs[0].name
        self.output_name = self.outputs[0].name
        self.expected_shape = self.inputs[0].shape
        
        print(f"[ONNXPredictor] Configured: Input Name='{self.input_name}' -> Output Name='{self.output_name}'")
        print(f"[ONNXPredictor] Expected input shape: {self.expected_shape}")

    def predict(self, input_data) -> np.ndarray:
        """
        Executes a forward pass on the ONNX model.
        
        Args:
            input_data (numpy.ndarray or torch.Tensor): The input array or tensor.
            
        Returns:
            numpy.ndarray: The raw inference outputs from the model.
        """
        # Convert PyTorch tensor to NumPy if necessary
        # (ONNX Runtime session.run expects raw NumPy inputs)
        if hasattr(input_data, "detach") and hasattr(input_data, "cpu") and hasattr(input_data, "numpy"):
            # Move to CPU, detach gradient, and convert to numpy array
            input_numpy = input_data.detach().cpu().numpy()
        elif isinstance(input_data, np.ndarray):
            input_numpy = input_data
        else:
            raise TypeError("input_data must be a numpy.ndarray or a torch.Tensor")
            
        # Ensure correct float32 precision
        if input_numpy.dtype != np.float32:
            input_numpy = input_numpy.astype(np.float32)
            
        # Construct ONNX execution dict
        feed_dict = {self.input_name: input_numpy}
        
        # Run forward pass (returns a list of outputs)
        ort_outputs = self.session.run([self.output_name], feed_dict)
        
        return ort_outputs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Integration Example / Usage Demo
# ─────────────────────────────────────────────────────────────────────────────
def main():
    model_path = "accelerate_dev/resnet50.onnx"
    
    if not os.path.isfile(model_path):
        print(f"Demo file '{model_path}' not found. Please export your model first using export_onnx.py.")
        return

    # Initialize the predictor (configured to use CUDA if available)
    predictor = ONNXPredictor(model_path=model_path, provider="cuda")

    # [CUSTOMIZATION POINT] Preprocessing pipeline:
    # Load your real input image, audio wave, etc. here.
    # Convert it, resize, normalize, and shape it to match: [Batch, Channels, Height, Width].
    print("\nGenerating dummy input (representing a preprocessed image/tensor)...")
    
    # We create a dummy input matching ResNet-50's expectation [1, 3, 224, 224]
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Perform inference
    print("Running prediction via ONNXPredictor...")
    outputs = predictor.predict(dummy_input)

    # [CUSTOMIZATION POINT] Postprocessing pipeline:
    # Insert code here to decode outputs (e.g. Softmax, argmax for class label, scale outputs, etc.)
    print(f"Outputs generated successfully!")
    print(f"  Shape : {outputs.shape}")
    print(f"  Mean  : {outputs.mean():.4f}")
    
    # Example post-process (argmax)
    top1_index = np.argmax(outputs, axis=1)
    print(f"  Predicted class index: {top1_index.tolist()}")


if __name__ == "__main__":
    main()
