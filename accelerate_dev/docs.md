Before do

Installing requirements

(uv) pip install -r ./accelerate_dev/requirements_tensorrt.txt

Testing on 

NVCC
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:10:22_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0

GPU 
Nvidia geforece 3090 24GB Driver Version: 560.35.03      CUDA Version: 12.6 

python 3.9.25


Main step
1. Convert .pt checkpoints into ONNX

2. Convert ONNX into TensorRT engine


Step 1: Convert .pt checkpoints into ONNX
The code consists of 2 separate ONNX conversion
    - FLOAT's decoder
    - FLOAT's FMT (transformers)

to run for decoder: 

python ./accelerate_dev/export_decoder_onnx.py --output_dir <outputh_dir> --model_name <output_onnx_model_name>

example:

1. Converting FLOAT's decoder (.pt) into ONNX:
python ./accelerate_dev/export_decoder_onnx.py --output_dir ./test_before_released/decoder/ --model_name float_decoder.onnx 


Step 2: Convert ONNX model into TensorRT
The code consists of 2 separate tensorRT building
    - FLOAT's decoder
    - FLOAT's FMT (transformers)

to run for decoder: 
python ./accelerate_dev/build_decoder_tensorrt.py 
--onnx_path <path_to_onnx>
./test_before_released/decoder/float_decoder.onnx 
--engine_path <output_dir>
./test_before_released/decoder/ 
--precision <precision>

available precision choices: fp32, fp16, tf32, fp8

Note: For FP8, some layers that cant be quantized into FP8 would automatically fallbakcs into FP16 by TensorRT's builder

example:
1. Building TensorRT engine from FLOAT's decoder onnx model.
python ./accelerate_dev/build_decoder_tensorrt.py 
--onnx_path ./test_before_released/decoder/float_decoder.onnx 
--engine_path ./test_before_released/decoder/ 
--precision fp32

2. Building TensorRT engine from FLOAT's FMT onnx model





