"""
ort_infer_example.py — Load an ONNX model and run FMT inference via ONNX Runtime
================================================================================
Self-contained example showing how to use a pre-exported .onnx file
for new inputs using ONNX Runtime.

Requirements:
    pip uninstall onnxruntime -y
    pip install onnxruntime-gpu numpy

Usage:
    python tensorrt/ort_infer_example.py \\
        --onnx ./checkpoints/onnx_models/fmt.onnx
"""

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: Patch LD_LIBRARY_PATH BEFORE importing onnxruntime.
#
# onnxruntime-gpu dlopen()s libcublasLt.so.12 and libcudnn.so.9 at import
# time. If the OS linker can't find them, the process segfaults or raises a
# fatal FAIL before any Python error handler can catch it. We fix this by
# injecting the correct library paths into the process environment and the
# runtime linker search list at the very top of the script — before any
# onnxruntime symbol is touched.
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

# Pre-load CUDA and cuDNN libraries to prevent segmentation faults and linker issues
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import _cuda_patch


# ─────────────────────────────────────────────────────────────────────────────
# Normal imports (onnxruntime is safe to load now)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is not installed.\n"
        "To run on GPU, please install it via:\n"
        "  pip uninstall onnxruntime -y && pip install onnxruntime-gpu"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must match the shapes the engine was built with)
# ─────────────────────────────────────────────────────────────────────────────
NUM_CURR_FRAMES = 50    # wav2vec_sec=2.0 × fps=25
NUM_PREV_FRAMES = 10    # attention context
DIM_MOTION      = 512   # dim_w
DIM_AUDIO       = 512   # dim_a
DIM_EMOTION     = 7     # number of emotion classes
NUM_OUT_FRAMES  = NUM_PREV_FRAMES + NUM_CURR_FRAMES   # = 60


def load_ort_session(onnx_path: str):
    """
    Load an ONNX model into an ORT InferenceSession.
    Prefers CUDAExecutionProvider; gracefully falls back to CPU.
    """
    print(f"Loading ONNX model from: {onnx_path}")

    # Use get_available_providers() — only lists providers that successfully
    # initialised their native libraries (not all_providers which can crash).
    available = ort.get_available_providers()
    print(f"ORT available providers: {available}")

    if "CUDAExecutionProvider" in available:
        print("Using CUDAExecutionProvider (GPU) ✓")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        print("WARNING: CUDA EP not available — running on CPU.")
        print("  Check that LD_LIBRARY_PATH includes libcublasLt.so.12 and libcudnn.so.9")
        providers = ["CPUExecutionProvider"]

    so = ort.SessionOptions()
    so.log_severity_level = 3    # suppress ORT info spam
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    # Report the providers that are actually active after session creation
    active = sess.get_providers()
    print(f"Active providers     : {active}")
    if active[0] != "CUDAExecutionProvider":
        print("  ⚠  GPU not active — inference is running on CPU!")

    input_names = [inp.name for inp in sess.get_inputs()]
    print(f"Graph input names    : {input_names}")

    return sess, input_names


def demo(onnx_path: str):
    # ── 1. Load session ───────────────────────────────────────────────────────
    sess, expected_inputs = load_ort_session(onnx_path)

    # ── 2. Build dummy inputs ─────────────────────────────────────────────────
    B = 1
    inputs = {
        "t":       np.array([0.5], dtype=np.float32),
        "x":       np.random.randn(B, NUM_CURR_FRAMES, DIM_MOTION).astype(np.float32),
        "wa":      np.random.randn(B, NUM_CURR_FRAMES, DIM_AUDIO).astype(np.float32),
        "wr":      np.random.randn(B, DIM_MOTION).astype(np.float32),
        "we":      np.random.randn(B, 1, DIM_EMOTION).astype(np.float32),
        "prev_x":  np.zeros((B, NUM_PREV_FRAMES, DIM_MOTION), dtype=np.float32),
        "prev_wa": np.zeros((B, NUM_PREV_FRAMES, DIM_AUDIO), dtype=np.float32),
    }

    # Handle models exported with --cfg_as_inputs
    cfg_defaults = {
        "a_cfg_scale": 2.0, "e_cfg_scale": 1.0, "r_cfg_scale": 1.0,
        "a_cfg": 2.0, "e_cfg": 1.0,
    }
    for k, v in cfg_defaults.items():
        if k in expected_inputs:
            inputs[k] = np.array([v], dtype=np.float32)

    missing = [n for n in expected_inputs if n not in inputs]
    if missing:
        raise ValueError(f"Missing inputs for the model: {missing}")

    feeds = {k: inputs[k] for k in expected_inputs}

    # ── 3. Warm up ────────────────────────────────────────────────────────────
    print("\nWarming up …")
    for _ in range(5):
        sess.run(None, feeds)

    # ── 4. Single timed inference call ────────────────────────────────────────
    print("Running inference …")
    t0 = time.perf_counter()
    outputs = sess.run(None, feeds)
    output = outputs[0]
    ms = (time.perf_counter() - t0) * 1000

    # ── 5. Use the output ─────────────────────────────────────────────────────
    vector_field = output[:, NUM_PREV_FRAMES:, :]   # (1, 50, 512)
    next_prev_x  = output[:, -NUM_PREV_FRAMES:, :]  # (1, 10, 512)

    print(f"\nResults:")
    print(f"  Latency          : {ms:.2f} ms  ({1000/ms:.1f} FPS)")
    print(f"  Output shape     : {output.shape}")
    print(f"  Vector field     : {vector_field.shape}")
    print(f"  Next prev_x      : {next_prev_x.shape}")
    print(f"  Output norm      : {np.linalg.norm(output):.4f}")

    # ── 6. Latency profile ────────────────────────────────────────────────────
    lats = []
    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        lats.append((time.perf_counter() - t0) * 1000)

    a = np.array(lats)
    print(f"\nLatency over 50 iters:")
    print(f"  mean={a.mean():.2f}ms  p50={np.percentile(a,50):.2f}ms  "
          f"p95={np.percentile(a,95):.2f}ms  p99={np.percentile(a,99):.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Runtime FMT inference example")
    parser.add_argument(
        "--onnx", type=str,
        default="./checkpoints/onnx_models/fmt.onnx",
        help="Path to the .onnx file",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"ONNX model not found: {args.onnx}")
        raise SystemExit(1)

    demo(args.onnx)
