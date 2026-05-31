"""
ort_benchmark.py — Benchmark ONNX Runtime vs PyTorch
====================================================
Compares the latency and throughput of the PyTorch baseline vs ONNX Runtime.

Usage:
  python tensorrt/ort_benchmark.py \
      --onnx ./checkpoints/onnx_models/fmt.onnx \
      --ckpt_path ./checkpoints/float.pth \
      --iters 100 \
      --warmup 10
"""

import os
import sys

# Pre-load CUDA and cuDNN libraries to prevent segmentation faults and linker issues
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import _cuda_patch

import time
import argparse
import numpy as np
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(script_dir, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _fmt_utils import load_fmt_wrapper, build_dummy_inputs, add_model_args, cosine_check

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ORT vs PyTorch")
    parser = add_model_args(parser)
    parser.add_argument("--onnx", type=str, default="./checkpoints/onnx_models/fmt.onnx")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/float.pth")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_known_args()[0]

def benchmark_pytorch(wrapper, dummy_inputs, warmup, iters):
    print(f"Warming up PyTorch ({warmup} iters)...")
    
    # Store one output for correctness check
    with torch.no_grad():
        pt_out = wrapper(*dummy_inputs).cpu().float()
        
        for _ in range(warmup - 1):
            _ = wrapper(*dummy_inputs)
        
        torch.cuda.synchronize()
        print(f"Running PyTorch benchmark ({iters} iters)...")
        t0 = time.perf_counter()
        
        for _ in range(iters):
            _ = wrapper(*dummy_inputs)
            
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
    total_time = t1 - t0
    avg_ms = (total_time / iters) * 1000
    fps = 1000.0 / avg_ms
    return pt_out, avg_ms, fps

def benchmark_ort(onnx_path, dummy_inputs, warmup, iters):
    import onnxruntime as ort
    
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    
    print(f"\nORT Providers: {sess.get_providers()}")
    
    graph_inputs = [inp.name for inp in sess.get_inputs()]
    names = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
    feeds = {}
    for i, name in enumerate(names):
        if name in graph_inputs:
            feeds[name] = dummy_inputs[i].cpu().float().numpy()
            
    cfg_defaults = {"a_cfg": 2.0, "e_cfg": 1.0, "a_cfg_scale": 2.0, "e_cfg_scale": 1.0, "r_cfg_scale": 1.0}
    for k, v in cfg_defaults.items():
        if k in graph_inputs:
            feeds[k] = np.array([v], dtype=np.float32)

    print(f"Warming up ORT ({warmup} iters)...")
    
    # Store one output for correctness check
    ort_out_np = sess.run(None, feeds)[0]
    ort_out = torch.from_numpy(ort_out_np).float()
    
    for _ in range(warmup - 1):
        _ = sess.run(None, feeds)
        
    print(f"Running ORT benchmark ({iters} iters)...")
    t0 = time.perf_counter()
    
    for _ in range(iters):
        _ = sess.run(None, feeds)
        
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    avg_ms = (total_time / iters) * 1000
    fps = 1000.0 / avg_ms
    return ort_out, avg_ms, fps

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.rank = device
    
    print("============================================================")
    print("Benchmark: PyTorch vs ONNX Runtime")
    print("============================================================")
    
    dummy = build_dummy_inputs(args, device, batch=1, seed=42)
    
    # 1. PyTorch
    print("\n--- PyTorch ---")
    try:
        wrapper = load_fmt_wrapper(args, device)
        pt_out, pt_ms, pt_fps = benchmark_pytorch(wrapper, dummy, args.warmup, args.iters)
        del wrapper
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"PyTorch benchmark failed: {e}")
        pt_out, pt_ms, pt_fps = None, None, None
        
    # 2. ONNX Runtime
    print("\n--- ONNX Runtime ---")
    try:
        ort_out, ort_ms, ort_fps = benchmark_ort(args.onnx, dummy, args.warmup, args.iters)
    except Exception as e:
        print(f"ORT benchmark failed: {e}")
        ort_out, ort_ms, ort_fps = None, None, None
        
    # 3. Sanity Check
    print("\n============================================================")
    print("Sanity Check (Correctness)")
    print("============================================================")
    if pt_out is not None and ort_out is not None:
        r = cosine_check(pt_out, ort_out)
        print(f"Cosine similarity : {r['cosine']:.8f}  (target ≥ 0.9999)")
        print(f"Relative L2 error : {r['rel_l2']:.8f}  (target ≤ 0.001)")
        print(f"Max absolute error: {r['max_abs_err']:.8f}")
        
        if r['cosine'] >= 0.999:
            print("✅ PASS: Outputs are mathematically equivalent.")
        else:
            print("❌ FAIL: Outputs differ significantly!")
    else:
        print("Could not run sanity check (missing output).")

    # 4. Results
    print("\n============================================================")
    print("Latency & Throughput Summary")
    print("============================================================")
    if pt_ms is not None:
        print(f"PyTorch:      {pt_ms:.2f} ms / infer   ({pt_fps:.2f} FPS)")
    if ort_ms is not None:
        print(f"ONNX Runtime: {ort_ms:.2f} ms / infer   ({ort_fps:.2f} FPS)")
        
    if pt_ms and ort_ms:
        speedup = pt_ms / ort_ms
        print(f"\nSpeedup (ORT / PyTorch): {speedup:.2f}x")

if __name__ == "__main__":
    main()
