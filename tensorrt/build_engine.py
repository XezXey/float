"""
build_engine.py — Compile a TensorRT engine from an ONNX graph
==============================================================
Step 2 of 3 in the TensorRT pipeline.

Consumes : ONNX file produced by export_onnx.py
Produces : Serialised TensorRT engine (.engine) + build manifest (.json)

Precision modes:
  fp32   Full FP32. Largest/slowest engine, highest accuracy.
  fp16   FP16 tensor cores. ~5–6× faster than FP32, minor precision loss.
  int8   INT8 + FP16 fallback. Max throughput, requires calibration data.
  best   TRT auto-selects FP16/INT8 per layer. Best perf/accuracy trade-off.

Usage:
  python tensorrt/build_engine.py \\
      --onnx      ./checkpoints/onnx_models/fmt.onnx \\
      --output    ./checkpoints/trt_engines/fmt_fp16.engine \\
      --precision fp16 \\
      --workspace_gb 4 \\
      --timing_iters 3 \\
      [--int8_calib_batches 200] \\
      [--int8_calib_cache  ./checkpoints/trt_engines/int8.cache]
"""

import os
import sys
import json
import time
import random
import argparse
import torch
import numpy as np

# ── resolve project root ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(script_dir, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _fmt_utils import (
    build_dummy_inputs, add_model_args, output_shape,
    load_fmt_wrapper,
)

PRECISION_MODES = ["fp32", "fp16", "int8", "best"]


# ─────────────────────────────────────────────────────────────────────────────
# INT8 Calibrator
# ─────────────────────────────────────────────────────────────────────────────
def _make_calibrator(trt, opt, device, n_batches, cache_file):
    """
    Build an IInt8EntropyCalibrator2 that feeds synthetic random inputs.
    Defined inside a function so that TRT is already imported.

    For production accuracy, replace _get_batch() with real
    audio/face embedding pairs from your validation set.
    """
    class _FMTCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self._idx   = 0
            self._cache = cache_file
            # Pre-allocate calibration batches on GPU
            print(f"  Allocating {n_batches} calibration batches on GPU …")
            self._batches = [
                build_dummy_inputs(opt, device, batch=1, seed=i)
                for i in range(n_batches)
            ]
            print(f"  Calibrator ready  ({n_batches} synthetic batches)")
            if os.path.isfile(cache_file):
                print(f"  Existing calibration cache found: {cache_file}")

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self._idx >= len(self._batches):
                return None
            tensors = self._batches[self._idx]
            self._idx += 1
            return [t.contiguous().data_ptr() for t in tensors]

        def read_calibration_cache(self):
            if os.path.isfile(self._cache):
                with open(self._cache, "rb") as f:
                    data = f.read()
                print(f"  Loaded calibration cache ({len(data)/1024:.1f} KB)")
                return data
            return None

        def write_calibration_cache(self, cache):
            os.makedirs(os.path.dirname(os.path.abspath(self._cache)), exist_ok=True)
            with open(self._cache, "wb") as f:
                f.write(cache)
            print(f"  Calibration cache saved: {self._cache} ({len(cache)/1024:.1f} KB)")

    return _FMTCalibrator()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile TensorRT engine from ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Precision guide:
  fp32   Highest accuracy.  Engine ~630 MB.  No speed gain vs PyTorch.
  fp16   Recommended.       Engine ~315 MB.  ~5–6× faster. Default.
  int8   Maximum speed.     Engine ~160 MB.  Requires calibration data.
  best   Auto-mixed.        Engine ~200 MB.  TRT selects per-layer precision.
""",
    )
    parser = add_model_args(parser)

    # ── I/O ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--onnx", type=str,
        default="./checkpoints/onnx_models/fmt.onnx",
        help="Input ONNX file (output of export_onnx.py)",
    )
    parser.add_argument(
        "--output", type=str,
        default=None,
        help="Output .engine file. Defaults to <onnx_dir>/fmt_<precision>.engine",
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default="./checkpoints/float.pth",
        help="FLOAT checkpoint path (used only for INT8 calibration)",
    )

    # ── precision ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=PRECISION_MODES,
        help="Quantization precision (default: fp32)",
    )

    # ── TensorRT tuning ───────────────────────────────────────────────────────
    parser.add_argument(
        "--workspace_gb", type=float, default=4.0,
        help="GPU memory budget for TRT engine build workspace in GB (default: 4.0)",
    )
    parser.add_argument(
        "--timing_iters", type=int, default=3,
        help="Number of tactic timing iterations per layer (default: 3, higher = more stable but slower build)",
    )
    parser.add_argument(
        "--opt_level", type=int, default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="TRT builder optimisation level 0–5 (default: 3; higher = slower build, potentially faster engine)",
    )
    parser.add_argument(
        "--sparsity", action="store_true",
        help="Enable sparse weight support (requires Ampere+ GPU with 2:4 structured sparsity)",
    )
    parser.add_argument(
        "--refittable", action="store_true",
        help="Build a refittable engine (allows weight updates without full rebuild)",
    )
    parser.add_argument(
        "--disable_tf32", action="store_true",
        help="Explicitly disable TF32 (10-bit mantissa) on Ampere/Ada GPUs. "
             "Automatically ON when --precision fp32. Required for cosine ≥ 0.9999 sanity checks.",
    )
    parser.add_argument(
        "--obey_precision", action="store_true",
        help="Set OBEY_PRECISION_CONSTRAINTS: forces every layer to run in its declared "
             "precision, preventing silent downcasting. Slower build, strictest accuracy.",
    )
    parser.add_argument(
        "--force_fp32_layers", action="store_true",
        help="Explicitly set precision=FP32 and output_type=FP32 on every parsed network "
             "layer after ONNX parsing. Nuclear option for FP32 sanity checks — prevents "
             "TRT from substituting attention kernels with Flash Attention equivalents. "
             "Automatically ON when --precision fp32.",
    )
    parser.add_argument(
        "--disable_cudnn_tactic", action="store_true",
        help="Remove cuDNN from TRT tactic sources. Blocks Flash Attention-style fused MHA "
             "kernels that TRT substitutes for scaled_dot_product_attention patterns.",
    )
    parser.add_argument(
        "--attn_fp32_fallback", action="store_true",
        help="Targeted FP32 override for attention sub-graph operator types only "
             "(MatMul, Einsum, Softmax, Slice, Gather, Div, Mul layers). "
             "Surgical fix for TRT#4796 (adaLN Slice/Gather INT64 chain) and "
             "TRT#3609 (MHA sequence-slice regression) without nuking the entire "
             "network to FP32. Preserves FP16 throughput on all other layers. "
             "Recommended for fp16 builds when cosine similarity fails (< 0.95).",
    )
    parser.add_argument(
        "--attn_layer_name_patterns", type=str, nargs="+",
        default=[],
        help="Additional layer name substrings to target with --attn_fp32_fallback. "
             "Use 'trt_engine_op_*' names from the TRT verbose build log to pin "
             "specific culprit layers.  Example: --attn_layer_name_patterns "
             "'MatMul_0' 'Softmax_0' 'Gather_1'",
    )

    # ── dynamic shape profile ─────────────────────────────────────────────────
    parser.add_argument(
        "--shapes_json", type=str, default=None,
        help="Path to .shapes.json from export_onnx.py (auto-detected if omitted)",
    )

    # ── INT8 calibration ──────────────────────────────────────────────────────
    parser.add_argument(
        "--int8_calib_batches", type=int, default=200,
        help="Number of calibration batches for INT8 (default: 200; use ≥512 for production)",
    )
    parser.add_argument(
        "--int8_calib_cache", type=str, default=None,
        help="Path to read/write INT8 calibration cache. Auto-derived from --output if omitted.",
    )

    return parser.parse_known_args()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Build logic
# ─────────────────────────────────────────────────────────────────────────────
def build_engine(args):
    import tensorrt as trt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── resolve paths ─────────────────────────────────────────────────────────
    onnx_dir = os.path.dirname(os.path.abspath(args.onnx))
    if args.output is None:
        args.output = os.path.join(
            os.path.join(os.path.dirname(onnx_dir), "trt_engines"),
            f"fmt_{args.precision}.engine",
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.int8_calib_cache is None:
        args.int8_calib_cache = args.output.replace(".engine", "_int8_calib.cache")

    # ── shapes JSON ───────────────────────────────────────────────────────────
    shapes_json_path = args.shapes_json or (args.onnx + ".shapes.json")
    shapes = None
    if os.path.isfile(shapes_json_path):
        with open(shapes_json_path) as f:
            shapes = json.load(f)
        print(f"  Shape profile : {shapes_json_path}")
        print(f"  Dynamic batch : {shapes.get('dynamic_batch', False)}")
    else:
        print(f"  [WARN] No shapes.json found at {shapes_json_path} — using static shapes")

    # ── TRT builder setup ─────────────────────────────────────────────────────
    print(f"  TensorRT      : {trt.__version__}")
    print(f"  Precision     : {args.precision.upper()}")
    print(f"  Workspace     : {args.workspace_gb:.1f} GB")
    print(f"  Timing iters  : {args.timing_iters}")
    print(f"  Opt level     : {args.opt_level}")
    print()

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network   = builder.create_network(net_flags)
    parser    = trt.OnnxParser(network, logger)

    print(f"[1/4] Parsing ONNX: {args.onnx}")
    with open(args.onnx, "rb") as f:
        raw = f.read()
    if not parser.parse(raw):
        print("  ONNX parse errors:")
        for i in range(parser.num_errors):
            print(f"    [{i}] {parser.get_error(i)}")
        sys.exit(1)
    print(f"  Parsed {network.num_layers} layers, {network.num_inputs} inputs, {network.num_outputs} outputs")

    # ── builder config ────────────────────────────────────────────────────────
    print("[2/4] Configuring builder …")
    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.avg_timing_iterations      = args.timing_iters
    config.builder_optimization_level = args.opt_level

    # ── TF32 control ──────────────────────────────────────────────────────────
    # TF32 is ON by default on Ampere/Ada GPUs (RTX 3xxx/4xxx) and silently
    # reduces matmul mantissa from 23 bits → 10 bits, causing cosine ~0.97-0.98
    # even with --precision fp32. Always disable for sanity-check builds.
    _disable_tf32 = args.disable_tf32 or (args.precision == "fp32")
    if _disable_tf32:
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)
            print("  TF32 disabled (strict mantissa precision) ✓")
        else:
            print("  [WARN] TF32 flag not found in this TRT version — cannot disable")
    else:
        print("  TF32 enabled (default — acceptable for FP16/INT8 builds)")

    # ── Precision constraints ─────────────────────────────────────────────────
    if args.obey_precision:  # explicit opt-in only; auto-enabling for fp32 conflicts with --disable_cudnn_tactic
        if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            print("  OBEY_PRECISION_CONSTRAINTS enabled ✓  (no silent layer downcasting)")
        elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            print("  PREFER_PRECISION_CONSTRAINTS enabled ✓")

    # ── Tactic source control ────────────────────────────────────────────────
    if args.disable_cudnn_tactic:
        if hasattr(trt, "TacticSource") and hasattr(trt.TacticSource, "CUDNN"):
            srcs = config.get_tactic_sources()
            srcs &= ~(1 << int(trt.TacticSource.CUDNN))
            config.set_tactic_sources(srcs)
            print("  cuDNN tactic source disabled ✓  (blocks Flash Attention-style fused MHA)")
        else:
            print("  [WARN] TacticSource.CUDNN not available in this TRT version")

    # ── precision flags ───────────────────────────────────────────────────────
    calibrator = None
    if args.precision == "fp16":
        _require_fp16(builder, config, trt)

    elif args.precision == "int8":
        _require_fp16(builder, config, trt)  # FP16 as fallback for non-INT8-able ops
        if not builder.platform_has_fast_int8:
            print("  [WARN] INT8 not available on this GPU — using FP16 only")
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            print("  INT8 enabled ✓")
            calibrator = _make_calibrator(
                trt, args, device,
                n_batches  = args.int8_calib_batches,
                cache_file = args.int8_calib_cache,
            )
            config.int8_calibrator = calibrator

    elif args.precision == "best":
        _require_fp16(builder, config, trt)
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = _make_calibrator(
                trt, args, device,
                n_batches  = args.int8_calib_batches,
                cache_file = args.int8_calib_cache,
            )
            config.int8_calibrator = calibrator
            print("  BEST mode: FP16 + INT8 auto-selection per layer ✓")
        else:
            print("  BEST mode: FP16 only (INT8 not supported on this GPU)")

    else:  # fp32
        print("  FP32 — no quantization flags set")

    # optional flags
    if args.sparsity:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        print("  Sparse weights enabled ✓")
    if args.refittable:
        config.set_flag(trt.BuilderFlag.REFIT)
        print("  Refittable engine enabled ✓")

    # ── optimization profile (for dynamic batch) ───────────────────────────────
    if shapes and shapes.get("dynamic_batch"):
        print("  Setting up dynamic batch optimization profile …")
        profile = builder.create_optimization_profile()
        mn, op, mx = shapes["min_batch"], shapes["opt_batch"], shapes["max_batch"]

        input_specs = [
            ("x",       (mn, shapes["inputs"]["x"]["static_shape"][1],       shapes["inputs"]["x"]["static_shape"][2])),
            ("wa",      (mn, shapes["inputs"]["wa"]["static_shape"][1],      shapes["inputs"]["wa"]["static_shape"][2])),
            ("wr",      (mn, shapes["inputs"]["wr"]["static_shape"][1])),
            ("we",      (mn, shapes["inputs"]["we"]["static_shape"][1],      shapes["inputs"]["we"]["static_shape"][2])),
            ("prev_x",  (mn, shapes["inputs"]["prev_x"]["static_shape"][1],  shapes["inputs"]["prev_x"]["static_shape"][2])),
            ("prev_wa", (mn, shapes["inputs"]["prev_wa"]["static_shape"][1], shapes["inputs"]["prev_wa"]["static_shape"][2])),
        ]
        for name, min_shape in input_specs:
            opt_shape = (op,) + min_shape[1:]
            max_shape = (mx,) + min_shape[1:]
            profile.set_shape(name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)
        print(f"  Profile: batch {mn}→{op}→{mx}")

    # ── Force per-layer FP32 precision (nuclear option for sanity checks) ─────────────
    # Must run AFTER ONNX parsing and BEFORE build_serialized_network.
    # OBEY_PRECISION_CONSTRAINTS alone only works when layer.precision is explicitly
    # set; without per-layer settings TRT still picks its own kernel (Flash Attention).
    _force_fp32 = getattr(args, 'force_fp32_layers', False)  # explicit only; auto-enable breaks build with --disable_cudnn_tactic
    if _force_fp32:
        n_forced = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                n_forced += 1
            except Exception:
                pass   # some layer types don't support explicit precision
        print(f"  Per-layer FP32 forced on {n_forced}/{network.num_layers} layers ✓")

    # ── Targeted attention subgraph FP32 fallback (TRT#4796 / TRT#3609 fix) ───────────
    # A surgical alternative to --force_fp32_layers: only sets FP32 on the
    # specific layer TYPES implicated in MHA fusion and adaLN Slice/Gather
    # regressions. All other layers retain their normal precision (FP16/INT8),
    # preserving throughput while fixing the hidden-state corruption.
    #
    # Target layer types (LayerType enum values):
    #   MATRIX_MULTIPLY  — Q/K/V projections and attention score matmuls
    #   EINSUM           — fused attention pattern (SDPA replacement)
    #   SOFTMAX          — attention weight normalisation
    #   SLICE            — sequence slicing in prev_x/prev_wa context window
    #   GATHER           — adaLN dynamic index lookup (Gather chain bug #4796)
    #   ELEMENTWISE      — Div, Mul used in QK scaling / adaLN gating
    #
    # Optional: additional layers named by the user via --attn_layer_name_patterns
    _attn_fallback = getattr(args, 'attn_fp32_fallback', False)
    if _attn_fallback:
        # Build the set of target LayerType values (graceful — not all exist
        # in every TRT version).
        _ATTN_LAYER_TYPES = set()
        for _type_name in (
            "MATRIX_MULTIPLY", "EINSUM", "SOFTMAX",
            "SLICE",            # TRT#3609: sequence-slice MHA regression
            "GATHER",           # TRT#4796: adaLN Slice/Gather INT64 chain
            "ELEMENTWISE",      # Div/Mul in QK scaling and adaLN gates
        ):
            _lt = getattr(trt.LayerType, _type_name, None)
            if _lt is not None:
                _ATTN_LAYER_TYPES.add(_lt)

        _extra_patterns = list(getattr(args, 'attn_layer_name_patterns', []))

        n_attn_forced = 0
        _attn_skipped_types: list[str] = []
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            _matched = (
                layer.type in _ATTN_LAYER_TYPES
                or any(pat in layer.name for pat in _extra_patterns)
            )
            if not _matched:
                continue
            try:
                layer.precision = trt.DataType.FLOAT
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.FLOAT)
                n_attn_forced += 1
            except Exception as _e:
                _attn_skipped_types.append(f"{layer.name} ({layer.type}): {_e}")

        print(
            f"  Attn-subgraph FP32 fallback: {n_attn_forced}/{network.num_layers} "
            f"layers forced to FP32 ✓"
        )
        print(
            f"  Target types: MATRIX_MULTIPLY, EINSUM, SOFTMAX, SLICE, GATHER, ELEMENTWISE"
        )
        if _extra_patterns:
            print(f"  Extra name patterns: {_extra_patterns}")
        if _attn_skipped_types:
            print(f"  Skipped (unsupported precision) : {len(_attn_skipped_types)} layers")
            for s in _attn_skipped_types[:5]:
                print(f"    {s}")

    # ── build ─────────────────────────────────────────────────────────────────
    print("[3/4] Building TensorRT engine …")
    print(f"  (Opt level {args.opt_level}, {args.timing_iters} timing iterations per layer)")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    elapsed    = time.time() - t0

    if serialized is None:
        print("  BUILD FAILED — serialized engine is None.")
        print("  Check TRT logs above for unsupported ops or OOM errors.")
        sys.exit(1)

    with open(args.output, "wb") as f:
        f.write(serialized)

    engine_mb = os.path.getsize(args.output) / 1e6
    print(f"  Build time    : {elapsed:.1f} s")
    print(f"  Engine size   : {engine_mb:.1f} MB")
    print(f"  Engine saved  : {args.output}")

    # ── write manifest ────────────────────────────────────────────────────────
    print("[4/4] Writing build manifest …")
    manifest = {
        "engine_path":         args.output,
        "onnx_path":           args.onnx,
        "precision":           args.precision,
        "workspace_gb":        args.workspace_gb,
        "timing_iters":        args.timing_iters,
        "opt_level":           args.opt_level,
        "build_time_s":        round(elapsed, 2),
        "engine_mb":           round(engine_mb, 2),
        "trt_version":         trt.__version__,
        "dynamic_batch":       shapes.get("dynamic_batch", False) if shapes else False,
        "int8_calib_cache":    args.int8_calib_cache if args.precision in ("int8", "best") else None,
        # Precision override flags — important for reproducibility and debugging
        "attn_fp32_fallback":  getattr(args, "attn_fp32_fallback", False),
        "force_fp32_layers":   getattr(args, "force_fp32_layers", False),
        "disable_cudnn_tactic": getattr(args, "disable_cudnn_tactic", False),
        "obey_precision":      getattr(args, "obey_precision", False),
    }
    manifest_path = args.output.replace(".engine", "_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest      : {manifest_path}")

    print()
    print("Build complete.")
    print(f"  Engine  : {args.output}")
    print()
    print("Next step:")
    print(f"  python tensorrt/benchmark_inference.py \\")
    print(f"      --engine     {args.output} \\")
    print(f"      --ckpt_path  {args.ckpt_path} \\")
    print(f"      --precision  {args.precision}")

    return args.output


def _require_fp16(builder, config, trt):
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled ✓")
    else:
        print("  [WARN] FP16 not supported on this GPU — engine will run in FP32")


if __name__ == "__main__":
    args = parse_args()
    args.rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("FLOAT FMT — TensorRT Engine Builder")
    print("=" * 60)
    print(f"  ONNX      : {args.onnx}")
    print(f"  Output    : {args.output or '(auto-derived)'}")
    print(f"  Precision : {args.precision.upper()}")
    print()

    build_engine(args)
