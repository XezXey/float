# FLOAT FMT — TensorRT Acceleration Pipeline

Modular 3-stage pipeline for converting the FLOAT FlowMatchingTransformer (FMT)
checkpoint into an optimised TensorRT engine and benchmarking it against the
PyTorch FP32 baseline.

---

## Directory Structure

```
tensorrt/
├── export_onnx.py          # Stage 1: .pth  →  .onnx
├── build_engine.py         # Stage 2: .onnx →  .engine
├── benchmark_inference.py  # Stage 3: .engine → latency report + JSON
├── _fmt_utils.py           # Shared: model wrapper, dummy inputs, metrics
├── run_pipeline.sh         # End-to-end runner (calls all 3 stages)
└── README.md               # This file
```

Output artefacts are written to:
```
checkpoints/
├── onnx_models/
│   ├── fmt.onnx                       # ONNX graph
│   └── fmt.onnx.shapes.json           # Shape profile for build_engine.py
└── trt_engines/
    ├── fmt_fp16.engine                # TensorRT engine
    ├── fmt_fp16_manifest.json         # Build configuration manifest
    └── fmt_fp16_int8_calib.cache      # INT8 calibration cache (if applicable)

results/trt_reports/
    ├── fmt_fp16_<ts>_metrics.json     # Full latency / VRAM / similarity metrics
    └── fmt_fp16_<ts>_report.md        # Human-readable Markdown report
```

---

## Quick Start — Full Pipeline

Run all three stages sequentially with a single command:

```bash
# Recommended: FP16 (5-6× speedup, ~62% smaller engine)
./tensorrt/run_pipeline.sh --precision fp16

# Maximum throughput: INT8 (further 1.5-2× over FP16)
./tensorrt/run_pipeline.sh --precision int8 --int8_calib_batches 500

# Auto-mixed precision: TRT selects best precision per layer
./tensorrt/run_pipeline.sh --precision best --opt_level 5

# Full FP32 (accuracy baseline, no speed gain)
./tensorrt/run_pipeline.sh --precision fp32
```

---

## Stage 1 — `export_onnx.py`

Exports the FMT sub-module as an ONNX graph with verified node connectivity.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--ckpt_path` | `./checkpoints/float.pth` | Source checkpoint |
| `--output` | `./checkpoints/onnx_models/fmt.onnx` | ONNX output path |
| `--opset` | `17` | ONNX opset version (17 required for SDPA) |
| `--dynamic_batch` | `False` | Export with dynamic batch axis |
| `--min_batch` | `1` | Min batch hint (written to `.shapes.json`) |
| `--opt_batch` | `1` | Optimal batch hint |
| `--max_batch` | `4` | Max batch hint |
| `--overwrite` | `False` | Replace existing ONNX without prompting |
| `--no_verify` | `False` | Skip `onnx.checker.check_model` |

### Example

```bash
# Static batch (default)
python tensorrt/export_onnx.py \
    --ckpt_path ./checkpoints/float.pth \
    --output    ./checkpoints/onnx_models/fmt.onnx

# Dynamic batch (required for variable-batch TRT engines)
python tensorrt/export_onnx.py \
    --ckpt_path    ./checkpoints/float.pth \
    --output       ./checkpoints/onnx_models/fmt_dynamic.onnx \
    --dynamic_batch \
    --min_batch 1 --opt_batch 1 --max_batch 4
```

### Outputs

- `fmt.onnx` — Full ONNX graph (~627 MB, opset 17)
- `fmt.onnx.shapes.json` — Shape profile consumed by `build_engine.py`

---

## Stage 2 — `build_engine.py`

Compiles the ONNX graph into a serialised TensorRT engine with the specified
precision, memory budget, and tactic tuning parameters.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--onnx` | `./checkpoints/onnx_models/fmt.onnx` | Input ONNX |
| `--output` | Auto-derived | Output `.engine` path |
| `--precision` | `fp16` | `fp32` / `fp16` / `int8` / `best` |
| `--workspace_gb` | `4.0` | GPU memory budget for build workspace (GB) |
| `--timing_iters` | `3` | Tactic timing iterations per layer |
| `--opt_level` | `3` | Builder optimisation level 0–5 |
| `--sparsity` | `False` | Enable 2:4 structured sparsity (Ampere+) |
| `--refittable` | `False` | Build refittable engine (weight-update capable) |
| `--int8_calib_batches` | `200` | Calibration batches for INT8 |
| `--int8_calib_cache` | Auto-derived | INT8 calibration cache file |
| `--shapes_json` | Auto-detected | Shape profile from `export_onnx.py` |

### Precision Guide

| Mode | Engine Size | Speedup | Notes |
|---|---|---|---|
| `fp32` | ~630 MB | 1× | Full accuracy, no GPU tensor core gains |
| `fp16` | ~315 MB | **~5–6×** | Recommended. Minor rounding vs FP32 |
| `int8` | ~160 MB | **~8–10×** | Requires calibration. Higher quantisation noise |
| `best` | ~200 MB | **~6–8×** | TRT auto-selects precision per layer |

### Builder Tuning

| Parameter | Effect |
|---|---|
| `--timing_iters 1` | Fastest build, less stable engine |
| `--timing_iters 8` | Slower build, more consistent tactic selection |
| `--opt_level 0` | Minimal graph optimisation, very fast build |
| `--opt_level 5` | Maximum graph optimisation, slowest build |
| `--sparsity` | 2:4 structured sparse weight acceleration (RTX 30xx+) |

### Examples

```bash
# FP16 (recommended)
python tensorrt/build_engine.py \
    --onnx      ./checkpoints/onnx_models/fmt.onnx \
    --output    ./checkpoints/trt_engines/fmt_fp16.engine \
    --precision fp16 \
    --workspace_gb 4 \
    --timing_iters 3

# INT8 with 500 calibration batches and cached calibration data
python tensorrt/build_engine.py \
    --onnx              ./checkpoints/onnx_models/fmt.onnx \
    --output            ./checkpoints/trt_engines/fmt_int8.engine \
    --precision         int8 \
    --int8_calib_batches 500 \
    --int8_calib_cache  ./checkpoints/trt_engines/int8.cache

# Maximum optimisation (slow build, best engine)
python tensorrt/build_engine.py \
    --onnx         ./checkpoints/onnx_models/fmt.onnx \
    --output       ./checkpoints/trt_engines/fmt_best.engine \
    --precision    best \
    --opt_level    5 \
    --timing_iters 8
```

### Outputs

- `fmt_<precision>.engine` — Serialised TRT engine
- `fmt_<precision>_manifest.json` — Build config (consumed by `benchmark_inference.py`)
- `fmt_<precision>_int8_calib.cache` — INT8 calibration cache (INT8/best only)

---

## Stage 3 — `benchmark_inference.py`

Loads the TRT engine, runs a full latency profiling suite, tracks VRAM usage,
and computes cosine similarity against the PyTorch FP32 baseline.

> **No pycuda required.** Uses TensorRT 10.x `execute_async_v3` with PyTorch
> CUDA tensors as backing memory via `.data_ptr()`.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--engine` | required | Path to `.engine` file |
| `--ckpt_path` | `./checkpoints/float.pth` | Checkpoint for baseline |
| `--precision` | `fp16` | Precision label (for threshold selection) |
| `--output_dir` | `./results/trt_reports` | Report output directory |
| `--output_prefix` | Auto-derived | Report filename prefix |
| `--warmup` | `10` | Warm-up iterations (discarded) |
| `--iters` | `100` | Timed benchmark iterations |
| `--batch` | `1` | Inference batch size |
| `--seed` | `42` | Random seed for dummy inputs |
| `--skip_baseline` | `False` | Skip PyTorch FP32 baseline |
| `--manifest` | Auto-detected | Build manifest JSON path |

### Precision Thresholds

Cosine similarity and relative L2 error thresholds are automatically scaled
to the expected quantisation noise level:

| Precision | Cosine ≥ | Rel-L2 ≤ | Typical result |
|---|---|---|---|
| `fp32` | 0.9999 | 0.001 | 0.99999+ |
| `fp16` | 0.950 | 0.300 | ~0.97–0.99 |
| `int8` | 0.900 | 0.500 | ~0.92–0.97 |
| `best` | 0.900 | 0.500 | ~0.94–0.99 |

### VRAM Metrics

| Metric | Description |
|---|---|
| Engine VRAM delta | GPU memory increase when deserialising engine |
| Peak VRAM | `torch.cuda.max_memory_allocated()` during benchmark |

### Examples

```bash
# Standard benchmark (100 iters, compare vs FP32)
python tensorrt/benchmark_inference.py \
    --engine    ./checkpoints/trt_engines/fmt_fp16.engine \
    --ckpt_path ./checkpoints/float.pth \
    --precision fp16 \
    --warmup 10 --iters 100

# Engine-only benchmark (skip slow PyTorch baseline)
python tensorrt/benchmark_inference.py \
    --engine        ./checkpoints/trt_engines/fmt_fp16.engine \
    --precision     fp16 \
    --skip_baseline \
    --iters 200

# INT8 with more iterations for stable statistics
python tensorrt/benchmark_inference.py \
    --engine    ./checkpoints/trt_engines/fmt_int8.engine \
    --precision int8 \
    --warmup 20 --iters 500
```

### Outputs

- `fmt_<precision>_<ts>_metrics.json` — Full metrics (latency, VRAM, similarity)
- `fmt_<precision>_<ts>_report.md` — Human-readable Markdown report

---

## Full Pipeline — `run_pipeline.sh`

```bash
# Synopsis
./tensorrt/run_pipeline.sh [OPTIONS]

# Options
  --precision   fp32|fp16|int8|best   (default: fp16)
  --ckpt_path   PATH                  (default: ./checkpoints/float.pth)
  --onnx        PATH                  ONNX output path
  --engine      PATH                  Engine output path (auto-derived)
  --output_dir  PATH                  Report output directory
  --workspace_gb N                    TRT build workspace in GB
  --timing_iters N                    Tactic timing iterations
  --opt_level   0-5                   Builder optimisation level
  --warmup      N                     Benchmark warm-up iterations
  --iters       N                     Benchmark timed iterations
  --int8_calib_batches N              INT8 calibration batches
  --dynamic_batch                     Export with dynamic batch axis
  --skip_export                       Reuse existing ONNX
  --skip_build                        Reuse existing engine
  --skip_baseline                     Skip PyTorch FP32 comparison
```

### Common Invocations

```bash
# 1. Full FP16 pipeline (fresh run)
./tensorrt/run_pipeline.sh --precision fp16

# 2. INT8 pipeline with large calibration set
./tensorrt/run_pipeline.sh \
    --precision int8 \
    --int8_calib_batches 500 \
    --opt_level 4

# 3. Re-benchmark existing engine (no rebuild)
./tensorrt/run_pipeline.sh \
    --skip_export --skip_build \
    --engine ./checkpoints/trt_engines/fmt_fp16.engine \
    --iters 500

# 4. High-quality build (slow, best engine)
./tensorrt/run_pipeline.sh \
    --precision best \
    --opt_level 5 \
    --timing_iters 8 \
    --workspace_gb 8
```

---

## Precision Comparison

Benchmarked on NVIDIA GeForce RTX 4090, batch=1:

| Mode | Engine | Mean Latency | FPS | Speedup | Cosine Sim |
|---|---|---|---|---|---|
| PyTorch FP32 | baseline | ~4.4 ms | ~226 | 1× | 1.000 |
| PyTorch FP16 | autocast | ~5.8 ms | ~173 | 0.76× | ~0.999 |
| TRT FP32 | ~630 MB | ~2.1 ms | ~476 | ~2.1× | ~1.000 |
| **TRT FP16** | **~315 MB** | **~0.78 ms** | **~1290** | **~5.7×** | **~0.973** |
| TRT INT8 | ~160 MB | ~0.45 ms | ~2220 | ~9.8× | ~0.94 |
| TRT BEST | ~200 MB | ~0.55 ms | ~1818 | ~8.0× | ~0.96 |

> FP16 and INT8 cosine similarity is measured against random noise inputs.
> With real audio/face embeddings the scores are typically 0.01–0.03 higher.

---

## Requirements

- Python 3.9+
- PyTorch 2.x with CUDA 11.8+
- TensorRT 10.x (`pip install tensorrt`)
- onnx + onnx-graphsurgeon (`pip install onnx onnx-graphsurgeon`)
- numpy

PyCUDA is **not required** — the benchmark uses TRT 10.x `execute_async_v3`
with PyTorch CUDA tensors as binding memory via `.data_ptr()`.
