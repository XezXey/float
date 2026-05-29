"""
export_onnx.py — Export FLOAT FMT checkpoint to ONNX
=====================================================
Step 1 of 3 in the TensorRT pipeline.

Outputs:
  <output>            ONNX graph file (default: checkpoints/onnx_models/fmt.onnx)
  <output>.shapes.json  Input/output shape profile for build_engine.py

Usage:
  python tensorrt/export_onnx.py \\
      --ckpt_path  ./checkpoints/float.pth \\
      --output     ./checkpoints/onnx_models/fmt.onnx \\
      --opset      17 \\
      [--dynamic_batch]          # Export with dynamic batch axis
      [--min_batch 1]            # Minimum batch for TRT profile
      [--opt_batch 1]            # Optimal batch for TRT profile
      [--max_batch 4]            # Maximum batch for TRT profile
"""

import os
import sys
import json
import argparse
import torch

# ── resolve project root ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(script_dir, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from _fmt_utils import (
    FMTWrapper, load_fmt_wrapper, build_dummy_inputs,
    output_shape, add_model_args,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export FLOAT FMT to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser = add_model_args(parser)

    # ── core I/O ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ckpt_path", type=str,
        default="./checkpoints/float.pth",
        help="Path to FLOAT checkpoint (.pth)",
    )
    parser.add_argument(
        "--output", type=str,
        default="./checkpoints/onnx_models/fmt.onnx",
        help="Destination ONNX file path",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17 — required for scaled_dot_product_attention)",
    )

    # ── shape options ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--dynamic_batch", action="store_true",
        help="Export with a dynamic batch axis. Needed for variable-batch TRT engines.",
    )
    parser.add_argument("--min_batch", type=int, default=1,
                        help="Min batch size hint (written to .shapes.json for build_engine.py)")
    parser.add_argument("--opt_batch", type=int, default=1,
                        help="Optimal batch size hint")
    parser.add_argument("--max_batch", type=int, default=4,
                        help="Max batch size hint")

    # ── export options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing ONNX file without prompting",
    )
    parser.add_argument(
        "--no_constant_folding", action="store_true",
        help="Disable constant folding during ONNX export",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Run onnx.checker.check_model after export (default: True)",
    )
    parser.add_argument(
        "--no_verify", action="store_false", dest="verify",
        help="Skip ONNX graph verification",
    )
    parser.add_argument(
        "--input_seed", type=int, default=42,
        help="Random seed for dummy inputs used during tracing",
    )

    return parser.parse_known_args()[0]


def main():
    args = parse_args()
    args.rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device    = args.rank

    print("=" * 60)
    print("FLOAT FMT — ONNX Export")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.ckpt_path}")
    print(f"  Output     : {args.output}")
    print(f"  Opset      : {args.opset}")
    print(f"  Dynamic    : {args.dynamic_batch}")
    print()

    # ── guard existing file ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if os.path.isfile(args.output) and not args.overwrite:
        print(f"  [INFO] ONNX already exists: {args.output}")
        print(f"         Pass --overwrite to replace it.")
        _write_shapes_json(args)
        return

    # ── load model ────────────────────────────────────────────────────────────
    print("[1/4] Loading model …")
    wrapper      = load_fmt_wrapper(args, device)
    dummy_inputs = build_dummy_inputs(args, device, batch=1, seed=args.input_seed)

    # ── build export kwargs ───────────────────────────────────────────────────
    input_names  = ["t", "x", "wa", "wr", "we", "prev_x", "prev_wa"]
    output_names = ["motion_latent"]

    dynamic_axes = None
    if args.dynamic_batch:
        # Batch dimension is dim-0 for all tensors except 't' (shape (1,))
        dynamic_axes = {
            name: {0: "batch_size"}
            for name in ["x", "wa", "wr", "we", "prev_x", "prev_wa", "motion_latent"]
        }
        print("  Dynamic axes: batch_size on all non-time tensors")

    export_kwargs = dict(
        export_params        = True,
        opset_version        = args.opset,
        do_constant_folding  = not args.no_constant_folding,
        input_names          = input_names,
        output_names         = output_names,
        dynamo               = False,   # use torch.onnx legacy tracer
    )
    if dynamic_axes:
        export_kwargs["dynamic_axes"] = dynamic_axes

    # ── trace & export ────────────────────────────────────────────────────────
    print("[2/4] Tracing model (this may take ~10s) …")
    t_in, x_in, wa_in, wr_in, we_in, px_in, pwa_in = dummy_inputs
    print(f"  Input shapes:")
    for name, t in zip(input_names, dummy_inputs):
        print(f"    {name:8s}: {tuple(t.shape)}")

    with torch.no_grad():
        # dry-run to confirm forward pass works
        out_ref = wrapper(*dummy_inputs)
    print(f"  Output shape : {tuple(out_ref.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            args.output,
            **export_kwargs,
        )

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[3/4] ONNX saved: {args.output}  ({size_mb:.1f} MB)")

    # ── verify ────────────────────────────────────────────────────────────────
    if args.verify:
        print("[4/4] Verifying ONNX graph …")
        try:
            import onnx
            model = onnx.load(args.output)
            onnx.checker.check_model(model)
            print("  onnx.checker.check_model: PASSED ✓")

            # also run shape inference
            import onnx.shape_inference
            inferred = onnx.shape_inference.infer_shapes(model)
            output_vi = inferred.graph.output[0]
            print(f"  Inferred output shape: {[d.dim_value for d in output_vi.type.tensor_type.shape.dim]}")
        except ImportError:
            print("  [WARN] 'onnx' package not installed — skipping verification")
        except Exception as e:
            print(f"  [FAIL] ONNX checker error: {e}")
            sys.exit(1)
    else:
        print("[4/4] Verification skipped (--no_verify)")

    # ── write shape profile ───────────────────────────────────────────────────
    _write_shapes_json(args, out_ref)

    print()
    print("Export complete.")
    print(f"  ONNX        : {args.output}")
    print(f"  Shape JSON  : {args.output}.shapes.json")
    print()
    print("Next step:")
    prec_example = "fp16"
    print(f"  python tensorrt/build_engine.py \\")
    print(f"      --onnx   {args.output} \\")
    print(f"      --output ./checkpoints/trt_engines/fmt_{prec_example}.engine \\")
    print(f"      --precision {prec_example}")


def _write_shapes_json(args, out_ref=None):
    """Write a shape-profile JSON consumed by build_engine.py."""
    n_curr = int(args.wav2vec_sec * args.fps)
    n_prev = args.num_prev_frames
    dim_w  = args.dim_w
    dim_a  = args.dim_a
    dim_e  = args.dim_e
    n_out  = n_prev + n_curr

    profile = {
        "dynamic_batch": args.dynamic_batch,
        "min_batch":     args.min_batch,
        "opt_batch":     args.opt_batch,
        "max_batch":     args.max_batch,
        "inputs": {
            "t":       {"static_shape": [1],                      "dtype": "float32"},
            "x":       {"static_shape": [1, n_curr, dim_w],       "dtype": "float32"},
            "wa":      {"static_shape": [1, n_curr, dim_a],       "dtype": "float32"},
            "wr":      {"static_shape": [1, dim_w],               "dtype": "float32"},
            "we":      {"static_shape": [1, 1, dim_e],            "dtype": "float32"},
            "prev_x":  {"static_shape": [1, n_prev, dim_w],      "dtype": "float32"},
            "prev_wa": {"static_shape": [1, n_prev, dim_a],      "dtype": "float32"},
        },
        "output": {
            "motion_latent": {"static_shape": [1, n_out, dim_w], "dtype": "float32"},
        },
    }

    json_path = args.output + ".shapes.json"
    with open(json_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  Shape JSON  : {json_path}")


if __name__ == "__main__":
    main()