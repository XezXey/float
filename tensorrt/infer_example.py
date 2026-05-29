"""
infer_example.py — Load a compiled TRT engine and run FMT inference
====================================================================
Self-contained example showing how to use a pre-built .engine file
for new inputs. No model checkpoint or ONNX file needed at runtime.

Requirements:
    pip install tensorrt torch

Usage:
    python tensorrt/infer_example.py \
        --engine ./checkpoints/trt_engines/fmt_fp16.engine

Swap out the random tensors at the bottom with your real encoder outputs.
"""

import argparse
import time
import torch
import tensorrt as trt


# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must match the shapes the engine was built with)
# ─────────────────────────────────────────────────────────────────────────────
NUM_CURR_FRAMES = 50    # wav2vec_sec=2.0 × fps=25
NUM_PREV_FRAMES = 10    # attention context
DIM_MOTION      = 512   # dim_w
DIM_AUDIO       = 512   # dim_a
DIM_EMOTION     = 7     # number of emotion classes

# Output is prev + curr frames concatenated (how FMT.forward works)
NUM_OUT_FRAMES  = NUM_PREV_FRAMES + NUM_CURR_FRAMES   # = 60


# ─────────────────────────────────────────────────────────────────────────────
# Engine loader  (call once at startup)
# ─────────────────────────────────────────────────────────────────────────────
def load_engine(engine_path: str):
    """
    Deserialise a TensorRT engine from disk.
    Returns (engine, context) — keep both alive for the lifetime of your app.
    """
    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    print(f"Engine loaded: {engine_path}")
    print(f"  Layers : {engine.num_layers}")
    return engine, context


# ─────────────────────────────────────────────────────────────────────────────
# Allocate I/O buffers  (call once per session — reuse across calls)
# ─────────────────────────────────────────────────────────────────────────────
def allocate_buffers(batch: int = 1):
    """
    Pre-allocate GPU tensors for all engine inputs and the output.
    Returns a dict of named torch.Tensor on CUDA.

    Keep these tensors alive and just fill them with new data each call —
    avoids repeated GPU allocation overhead.
    """
    B, L, P, D, A, E = batch, NUM_CURR_FRAMES, NUM_PREV_FRAMES, DIM_MOTION, DIM_AUDIO, DIM_EMOTION

    buffers = {
        # ── inputs ────────────────────────────────────────────────────────────
        "t":             torch.zeros(1,    dtype=torch.float32, device="cuda"),
        "x":             torch.zeros(B, L, D, dtype=torch.float32, device="cuda"),
        "wa":            torch.zeros(B, L, A, dtype=torch.float32, device="cuda"),
        "wr":            torch.zeros(B,    D, dtype=torch.float32, device="cuda"),
        "we":            torch.zeros(B, 1, E, dtype=torch.float32, device="cuda"),
        "prev_x":        torch.zeros(B, P, D, dtype=torch.float32, device="cuda"),
        "prev_wa":       torch.zeros(B, P, A, dtype=torch.float32, device="cuda"),
        # ── output ────────────────────────────────────────────────────────────
        # Shape is (B, NUM_PREV + NUM_CURR, DIM_MOTION) = (1, 60, 512)
        # because FMT.forward() prepends prev_x to x before the transformer.
        "motion_latent": torch.zeros(B, NUM_OUT_FRAMES, D, dtype=torch.float32, device="cuda"),
    }
    return buffers


# ─────────────────────────────────────────────────────────────────────────────
# Single inference call
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(context, buffers: dict, stream: torch.cuda.Stream) -> torch.Tensor:
    """
    Fill `buffers` with your input data before calling this function,
    then read the result from buffers["motion_latent"] afterwards.

    Args:
        context  : TensorRT IExecutionContext (from load_engine)
        buffers  : dict of named CUDA tensors (from allocate_buffers)
        stream   : a torch.cuda.Stream (create one and reuse it)

    Returns:
        motion_latent : (B, 60, 512) float32 CUDA tensor
                        Slice [:, NUM_PREV_FRAMES:, :] for the current-frame
                        predictions only, or use all 60 frames as context
                        for the next ODE step.
    """
    # Bind every tensor's GPU memory address to the engine
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())

    # Async execute — no CPU↔GPU copy needed, everything is already on device
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    return buffers["motion_latent"]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — fill buffers and infer in one call
# ─────────────────────────────────────────────────────────────────────────────
def infer(
    context,
    buffers: dict,
    stream: torch.cuda.Stream,
    *,
    t: float,
    x:       torch.Tensor,   # (B, 50, 512)  motion latent
    wa:      torch.Tensor,   # (B, 50, 512)  audio features
    wr:      torch.Tensor,   # (B, 512)      reference motion
    we:      torch.Tensor,   # (B, 1, 7)     emotion
    prev_x:  torch.Tensor,   # (B, 10, 512)  previous motion context
    prev_wa: torch.Tensor,   # (B, 10, 512)  previous audio context
) -> torch.Tensor:
    """
    High-level helper: copies your inputs into the pre-allocated buffers
    and returns the output tensor.

    All input tensors can be on CPU or CUDA — they will be moved/cast
    to float32 CUDA automatically.
    """
    def _to_buf(dst: torch.Tensor, src: torch.Tensor):
        dst.copy_(src.to(dtype=torch.float32, device="cuda"), non_blocking=True)

    buffers["t"].fill_(t)
    _to_buf(buffers["x"],       x)
    _to_buf(buffers["wa"],      wa)
    _to_buf(buffers["wr"],      wr)
    _to_buf(buffers["we"],      we)
    _to_buf(buffers["prev_x"],  prev_x)
    _to_buf(buffers["prev_wa"], prev_wa)

    return run_inference(context, buffers, stream)


# ─────────────────────────────────────────────────────────────────────────────
# Demo — swap these random tensors with your real encoder outputs
# ─────────────────────────────────────────────────────────────────────────────
def demo(engine_path: str):
    # ── 1. Load engine (once at startup) ─────────────────────────────────────
    engine, context = load_engine(engine_path)

    # ── 2. Allocate buffers (once per session) ────────────────────────────────
    buffers = allocate_buffers(batch=1)
    stream  = torch.cuda.Stream()

    # ── 3. Build your real inputs ─────────────────────────────────────────────
    #
    # In a real pipeline these come from:
    #   x       ← motion autoencoder encoder output for the current window
    #   wa      ← wav2vec2 audio features for the current window
    #   wr      ← motion autoencoder encoder output for the reference frame
    #   we      ← Audio2Emotion classifier output
    #   prev_x  ← x from the previous ODE window (or zeros for the first window)
    #   prev_wa ← wa from the previous ODE window (or zeros for the first window)
    #   t       ← ODE solver timestep, sampled by your ODE integrator (e.g. torchdiffeq)
    #
    # Here we use random tensors for demonstration:
    B = 1
    x       = torch.randn(B, NUM_CURR_FRAMES, DIM_MOTION)
    wa      = torch.randn(B, NUM_CURR_FRAMES, DIM_AUDIO)
    wr      = torch.randn(B,                  DIM_MOTION)
    we      = torch.randn(B, 1,               DIM_EMOTION)
    prev_x  = torch.zeros(B, NUM_PREV_FRAMES, DIM_MOTION)   # zeros for first window
    prev_wa = torch.zeros(B, NUM_PREV_FRAMES, DIM_AUDIO)
    t_val   = 0.5   # ODE timestep ∈ [0, 1]

    # ── 4. Warm up (first call compiles CUDA kernels) ─────────────────────────
    print("\nWarming up …")
    for _ in range(5):
        infer(context, buffers, stream,
              t=t_val, x=x, wa=wa, wr=wr, we=we, prev_x=prev_x, prev_wa=prev_wa)

    # ── 5. Run one inference call ─────────────────────────────────────────────
    print("Running inference …")
    t0     = time.perf_counter()
    output = infer(context, buffers, stream,
                   t=t_val, x=x, wa=wa, wr=wr, we=we, prev_x=prev_x, prev_wa=prev_wa)
    ms     = (time.perf_counter() - t0) * 1000

    # ── 6. Use the output ─────────────────────────────────────────────────────
    #
    # output shape: (1, 60, 512)
    #   - Dim 1 contains NUM_PREV_FRAMES (10) + NUM_CURR_FRAMES (50) = 60 frames
    #   - This matches FMT.forward() which prepends prev_x before the transformer
    #
    # For the ODE solver's vector field, you typically want the current-frame slice:
    vector_field = output[:, NUM_PREV_FRAMES:, :]   # (1, 50, 512)

    # The full output can also serve as the next window's prev_x context:
    next_prev_x = output[:, -NUM_PREV_FRAMES:, :]   # (1, 10, 512)  last 10 frames

    print(f"\nResults:")
    print(f"  Latency          : {ms:.2f} ms  ({1000/ms:.1f} FPS)")
    print(f"  Output shape     : {tuple(output.shape)}")
    print(f"  Vector field     : {tuple(vector_field.shape)}  (pass to motion decoder)")
    print(f"  Next prev_x      : {tuple(next_prev_x.shape)}  (carry forward to next step)")
    print(f"  Output norm      : {output.norm():.4f}")
    print(f"  Output range     : [{output.min():.4f}, {output.max():.4f}]")

    # ── 7. Quick latency profile ──────────────────────────────────────────────
    import numpy as np
    lats = []
    for _ in range(50):
        t0 = time.perf_counter()
        infer(context, buffers, stream,
              t=t_val, x=x, wa=wa, wr=wr, we=we, prev_x=prev_x, prev_wa=prev_wa)
        lats.append((time.perf_counter() - t0) * 1000)
    a = np.array(lats)
    print(f"\nLatency over 50 iters:")
    print(f"  mean={a.mean():.2f}ms  p50={np.percentile(a,50):.2f}ms  "
          f"p95={np.percentile(a,95):.2f}ms  p99={np.percentile(a,99):.2f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRT FMT inference example")
    parser.add_argument(
        "--engine", type=str,
        default="./checkpoints/trt_engines/fmt_fp16.engine",
        help="Path to the compiled TensorRT .engine file",
    )
    args = parser.parse_args()

    if not __import__("os").path.isfile(args.engine):
        print(f"Engine not found: {args.engine}")
        print("Build one first:")
        print("  ./tensorrt/run_pipeline.sh --precision fp16 --skip_onnx")
        raise SystemExit(1)

    demo(args.engine)
