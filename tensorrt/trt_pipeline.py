"""
trt_pipeline.py — DEPRECATED
=============================
This monolithic pipeline has been refactored into separate, modular scripts:

  Stage 1: tensorrt/export_onnx.py
  Stage 2: tensorrt/build_engine.py
  Stage 3: tensorrt/benchmark_inference.py
  Runner:  tensorrt/run_pipeline.sh

See tensorrt/README.md for full documentation and usage examples.

To reproduce the last benchmarked run (FP16, RTX 4090):
  ./tensorrt/run_pipeline.sh --precision fp16 --skip_export \\
      --engine ./checkpoints/trt_engines/fmt_fp16_20260528_162205.engine

This file is kept for reference only and will not be maintained.
"""
raise SystemExit(
    "\n\ntrt_pipeline.py is deprecated.\n"
    "Use the modular pipeline instead:\n"
    "  ./tensorrt/run_pipeline.sh --precision fp16\n"
    "See tensorrt/README.md for details.\n"
)
