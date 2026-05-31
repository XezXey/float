#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — FLOAT FMT TensorRT End-to-End Pipeline
# =============================================================================
# Runs all three stages sequentially:
#   Stage 1 → export_onnx.py       (.pth → .onnx)
#   Stage 2 → build_engine.py      (.onnx → .engine)
#   Stage 3 → benchmark_inference.py (.engine → report)
#
# Usage:
#   ./tensorrt/run_pipeline.sh [OPTIONS]
#
# Options:
#   --precision   fp32|fp16|int8|best  (default: fp16)
#   --ckpt_path   /path/to/float.pth   (default: ./checkpoints/float.pth)
#   --onnx        /path/to/output.onnx (default: ./checkpoints/onnx_models/fmt.onnx)
#   --engine      /path/to/output.engine (default: auto-derived)
#   --output_dir  /path/to/reports     (default: ./results/trt_reports)
#   --workspace_gb  N                  (default: 4)
#   --timing_iters  N                  (default: 3)
#   --opt_level     0-5                (default: 3)
#   --warmup        N                  (default: 10)
#   --iters         N                  (default: 100)
#   --dynamic_batch                    (flag: export with dynamic batch axis)
#   --skip_export                      (flag: reuse existing .onnx)
#   --skip_build                       (flag: reuse existing .engine; requires --engine)
#   --skip_baseline                    (flag: skip PyTorch FP32 comparison)
#   --int8_calib_batches N             (default: 200)
#
# Examples:
#   # FP16 pipeline (recommended)
#   ./tensorrt/run_pipeline.sh --precision fp16
#
#   # INT8 pipeline with more calibration batches
#   ./tensorrt/run_pipeline.sh --precision int8 --int8_calib_batches 500
#
#   # Reuse existing ONNX + engine, just re-benchmark
#   ./tensorrt/run_pipeline.sh --skip_export --skip_build \
#       --engine ./checkpoints/trt_engines/fmt_fp16.engine
#
#   # Best precision mode, high optimisation level
#   ./tensorrt/run_pipeline.sh --precision best --opt_level 5 --timing_iters 5
# =============================================================================
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────────
PRECISION="fp16"
CKPT_PATH="./checkpoints/float.pth"
ONNX_PATH="./checkpoints/onnx_models/fmt.onnx"
ENGINE_PATH=""
OUTPUT_DIR="./results/trt_reports"
WORKSPACE_GB="4"
TIMING_ITERS="3"
OPT_LEVEL="3"
WARMUP="10"
ITERS="100"
CALIB_BATCHES="200"
SKIP_EXPORT=""
SKIP_BUILD=""
SKIP_BASELINE=""
DYNAMIC_BATCH=""
ATTN_FP32_FALLBACK=""
DISABLE_CUDNN_TACTIC=""
OBEY_PRECISION=""
EXTRA_ARGS=""

# ── parse CLI ─────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --precision)        PRECISION="$2";        shift 2 ;;
        --ckpt_path)        CKPT_PATH="$2";        shift 2 ;;
        --onnx)             ONNX_PATH="$2";        shift 2 ;;
        --engine)           ENGINE_PATH="$2";      shift 2 ;;
        --output_dir)       OUTPUT_DIR="$2";       shift 2 ;;
        --workspace_gb)     WORKSPACE_GB="$2";     shift 2 ;;
        --timing_iters)     TIMING_ITERS="$2";     shift 2 ;;
        --opt_level)        OPT_LEVEL="$2";        shift 2 ;;
        --warmup)           WARMUP="$2";           shift 2 ;;
        --iters)            ITERS="$2";            shift 2 ;;
        --int8_calib_batches) CALIB_BATCHES="$2"; shift 2 ;;
        --skip_export)      SKIP_EXPORT="1";       shift ;;
        --skip_build)       SKIP_BUILD="1";        shift ;;
        --skip_baseline)    SKIP_BASELINE="1";     shift ;;
        --dynamic_batch)    DYNAMIC_BATCH="1";     shift ;;
        --attn_fp32_fallback)    ATTN_FP32_FALLBACK="1"; shift ;;
        --disable_cudnn_tactic)  DISABLE_CUDNN_TACTIC="1"; shift ;;
        --obey_precision)        OBEY_PRECISION="1"; shift ;;
        *)                  EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ── validate precision ──────────────────────────────────────────────────────────────
if [[ ! "$PRECISION" =~ ^(fp32|fp16|int8|best)$ ]]; then
    echo "[ERROR] Invalid --precision '$PRECISION'. Must be one of: fp32 fp16 int8 best" >&2
    exit 1
fi

# ── resolve engine path ───────────────────────────────────────────────────────────
if [[ -z "$ENGINE_PATH" ]]; then
    ENGINE_DIR="$(dirname "$ONNX_PATH"/../trt_engines)"
    mkdir -p "$ENGINE_DIR"
    ENGINE_PATH="${ENGINE_DIR}/fmt_${PRECISION}.engine"
fi

# ── header ─────────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  FLOAT FMT — TensorRT Pipeline  (${PRECISION^^})                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo "  Checkpoint  : $CKPT_PATH"
echo "  ONNX output : $ONNX_PATH"
echo "  Engine      : $ENGINE_PATH"
echo "  Reports     : $OUTPUT_DIR"
echo ""

START_TIME=$(date +%s)

# ── STAGE 1: ONNX Export ───────────────────────────────────────────────────────────
echo "╭──────────────────────────────────────────────────────────╮"
echo "│  STAGE 1 — ONNX Export                                  │"
echo "╰──────────────────────────────────────────────────────────╯"

if [[ -n "$SKIP_EXPORT" ]]; then
    echo "  [SKIP] --skip_export set. Using existing ONNX: $ONNX_PATH"
    if [[ ! -f "$ONNX_PATH" ]]; then
        echo "  [ERROR] ONNX not found: $ONNX_PATH" >&2
        exit 1
    fi
else
    EXPORT_ARGS="--ckpt_path $CKPT_PATH --output $ONNX_PATH"
    [[ -n "$DYNAMIC_BATCH" ]] && EXPORT_ARGS="$EXPORT_ARGS --dynamic_batch"

    python3 "$(dirname "$0")/export_onnx.py" \
        $EXPORT_ARGS $EXTRA_ARGS
fi

echo ""

# ── STAGE 2: Build TRT Engine ────────────────────────────────────────────────────────
echo "╭──────────────────────────────────────────────────────────╮"
echo "│  STAGE 2 — TensorRT Engine Build  [${PRECISION^^}]                 │"
echo "╰──────────────────────────────────────────────────────────╯"

if [[ -n "$SKIP_BUILD" ]]; then
    echo "  [SKIP] --skip_build set. Using existing engine: $ENGINE_PATH"
    if [[ ! -f "$ENGINE_PATH" ]]; then
        echo "  [ERROR] Engine not found: $ENGINE_PATH" >&2
        exit 1
    fi
else
    BUILD_FLAGS=""
    [[ -n "$ATTN_FP32_FALLBACK"   ]] && BUILD_FLAGS="$BUILD_FLAGS --attn_fp32_fallback"
    [[ -n "$DISABLE_CUDNN_TACTIC" ]] && BUILD_FLAGS="$BUILD_FLAGS --disable_cudnn_tactic"
    [[ -n "$OBEY_PRECISION"       ]] && BUILD_FLAGS="$BUILD_FLAGS --obey_precision"

    python3 "$(dirname "$0")/build_engine.py" \
        --onnx          "$ONNX_PATH"      \
        --output        "$ENGINE_PATH"    \
        --precision     "$PRECISION"      \
        --workspace_gb  "$WORKSPACE_GB"   \
        --timing_iters  "$TIMING_ITERS"   \
        --opt_level     "$OPT_LEVEL"      \
        --int8_calib_batches "$CALIB_BATCHES" \
        --ckpt_path     "$CKPT_PATH"      \
        $BUILD_FLAGS $EXTRA_ARGS
fi

echo ""

# ── STAGE 3: Benchmark ──────────────────────────────────────────────────────────────────
echo "╭──────────────────────────────────────────────────────────╮"
echo "│  STAGE 3 — Inference Benchmark                           │"
echo "╰──────────────────────────────────────────────────────────╯"

BENCH_ARGS="--engine $ENGINE_PATH --ckpt_path $CKPT_PATH --precision $PRECISION
  --warmup $WARMUP --iters $ITERS --output_dir $OUTPUT_DIR"
[[ -n "$SKIP_BASELINE" ]] && BENCH_ARGS="$BENCH_ARGS --skip_baseline"

python3 "$(dirname "$0")/benchmark_inference.py" \
    $BENCH_ARGS $EXTRA_ARGS

echo ""

# ── summary ──────────────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE                                         ║"
echo "║  Precision : ${PRECISION^^}                                           ║"
echo "║  Engine    : $ENGINE_PATH"
echo "║  Reports   : $OUTPUT_DIR"
echo "║  Total time: ${ELAPSED}s"
echo "╚═══════════════════════════════════════════════════════════╝"
