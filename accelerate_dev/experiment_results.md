# TensorRT Precision Combinations Benchmark Report

This report summarizes the performance metrics of all 9 precision combinations (FMT vs Decoder) for both the default configuration and the expressive emotion configuration.

### Configuration: default

| FMT Precision | Decoder Precision | Sampling Time (s) | Decoding Time (s) | Decoding FPS | Inference Time (s) | Inference FPS | Total Time (s) | Total FPS | Success |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | FP32 | 0.70 | 2.48 | 539.63 | 3.55 | 105.84 | 10.57 | 35.57 | ✅ |
| FP32 | TF32 | 0.71 | 2.44 | 532.70 | 3.51 | 107.00 | 10.50 | 35.81 | ✅ |
| FP32 | FP16 | 0.70 | 1.77 | 538.08 | 2.83 | 132.69 | 9.73 | 38.64 | ✅ |
| TF32 | FP32 | 0.59 | 2.48 | 639.08 | 3.52 | 106.97 | 10.51 | 35.79 | ✅ |
| TF32 | TF32 | 0.59 | 2.42 | 637.55 | 3.41 | 110.34 | 10.44 | 36.02 | ✅ |
| TF32 | FP16 | 0.60 | 1.77 | 629.36 | 2.75 | 136.69 | 9.61 | 39.13 | ✅ |
| FP16 | FP32 | 0.52 | 2.47 | 717.25 | 3.36 | 111.98 | 10.29 | 36.55 | ✅ |
| FP16 | TF32 | 0.52 | 2.42 | 716.36 | 3.31 | 113.66 | 10.23 | 36.76 | ✅ |
| FP16 | FP16 | 0.51 | 1.79 | 740.89 | 2.66 | 141.56 | 10.37 | 36.26 | ✅ |

### Configuration: emo=sad

| FMT Precision | Decoder Precision | Sampling Time (s) | Decoding Time (s) | Decoding FPS | Inference Time (s) | Inference FPS | Total Time (s) | Total FPS | Success |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | FP32 | 0.53 | 2.48 | 712.63 | 3.37 | 111.45 | 10.38 | 36.24 | ✅ |
| FP32 | TF32 | 0.55 | 2.44 | 685.61 | 3.38 | 111.40 | 10.66 | 35.28 | ✅ |
| FP32 | FP16 | 0.52 | 1.76 | 724.72 | 2.78 | 135.26 | 9.65 | 38.96 | ✅ |
| TF32 | FP32 | 0.45 | 2.47 | 826.92 | 3.29 | 114.30 | 10.11 | 37.19 | ✅ |
| TF32 | TF32 | 0.47 | 2.42 | 801.43 | 3.25 | 115.71 | 10.07 | 37.35 | ✅ |
| TF32 | FP16 | 0.42 | 1.77 | 896.01 | 2.57 | 146.15 | 9.50 | 39.58 | ✅ |
| FP16 | FP32 | 0.37 | 2.48 | 1016.31 | 3.22 | 116.76 | 10.06 | 37.37 | ✅ |
| FP16 | TF32 | 0.35 | 2.42 | 1064.88 | 3.15 | 119.54 | 10.16 | 37.00 | ✅ |
| FP16 | FP16 | 0.36 | 1.78 | 1041.25 | 2.50 | 150.23 | 9.45 | 39.77 | ✅ |
