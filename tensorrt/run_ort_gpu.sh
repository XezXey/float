#!/bin/bash
# A robust launcher to guarantee ONNX Runtime binds to the GPU without crashing

# 1. Activate the correct environment (prevents running the wrong python binary)
source /conda_envs/FLOAT_TRT_3.9/bin/activate

# 2. Inject the correct CUDA 12 and cuDNN 9 paths to prevent "No such file" fallbacks
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/conda_envs/FLOAT_TRT_3.9/lib/python3.9/site-packages/nvidia/cublas/lib:/conda_envs/FLOAT_TRT_3.9/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 3. Run the inference script
echo "Starting ONNX Runtime on GPU..."
python3 tensorrt/ort_infer_example.py --onnx ./checkpoints/onnx_models/fmt.onnx
