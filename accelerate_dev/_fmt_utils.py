"""
_fmt_utils.py — Shared FLOAT FMT utilities
==========================================
Imported by export_onnx.py, build_engine.py, and benchmark_inference.py.
Provides: FMTWrapper, model loading, dummy input generation, stats helpers.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.float_with_onnxruntime.FLOAT import FLOAT
from options.base_options import BaseOptions


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────
class FMTWrapper(torch.nn.Module):
    """
    Wraps the FlowMatchingTransformer (FMT) with CFG baked in.

    Two modes, selected at construction time:

    cfg_as_inputs=False  (default, backward-compatible)
        a_cfg=2.0 and e_cfg=1.0 are baked as graph constants.
        forward(t, x, wa, wr, we, prev_x, prev_wa)

    cfg_as_inputs=True  (for dynamic CFG scale export)
        a_cfg and e_cfg are scalar tensor inputs passed at runtime.
        forward(t, x, wa, wr, we, prev_x, prev_wa, a_cfg, e_cfg)
        Export with --cfg_as_inputs flag; call infer(..., a_cfg=2.5, e_cfg=0.8).

    Output shape: (B, num_prev + num_curr, dim_w)  e.g. (1, 60, 512)
    """

    def __init__(self, fmt, cfg_as_inputs: bool = False):
        super().__init__()
        self.fmt           = fmt
        self.cfg_as_inputs = cfg_as_inputs

    def forward(self, t, x, wa, wr, we, prev_x, prev_wa, a_cfg=None, e_cfg=None):
        """
        Args:
            t        : (1,)        ODE timestep in [0, 1]
            x        : (B, L, 512) motion latent (current window)
            wa       : (B, L, 512) audio features (current window)
            wr       : (B, 512)    reference motion latent
            we       : (B, 1, 7)   emotion latent
            prev_x   : (B, P, 512) previous motion context
            prev_wa  : (B, P, 512) previous audio context
            a_cfg    : (1,) tensor audio CFG scale   [only when cfg_as_inputs=True]
            e_cfg    : (1,) tensor emotion CFG scale [only when cfg_as_inputs=True]
        Returns:
            Tensor of shape (B, P+L, 512) — predicted vector field
        """
        null_wa = torch.zeros_like(wa)
        null_we = torch.zeros_like(we)

        audio_cat   = torch.cat([null_wa, wa,     wa      ], dim=0)
        ref_cat     = torch.cat([wr,      wr,     wr      ], dim=0)
        emotion_cat = torch.cat([null_we, we,     null_we ], dim=0)
        x_cat       = torch.cat([x,       x,      x       ], dim=0)
        px_cat      = torch.cat([prev_x,  prev_x, prev_x  ], dim=0)
        pwa_cat     = torch.cat([prev_wa, prev_wa,prev_wa  ], dim=0)

        out = self.fmt.forward(
            t, x_cat, audio_cat, ref_cat, emotion_cat,
            px_cat, pwa_cat, train=False
        )  # (3B, P+L, 512)

        uncond, all_cond, audio_uncond = torch.chunk(out, 3, dim=0)

        if self.cfg_as_inputs:
            # a_cfg / e_cfg are (1,) scalar tensors — dynamic at runtime
            return uncond + a_cfg * (audio_uncond - uncond) + e_cfg * (all_cond - audio_uncond)
        else:
            # Constants baked into graph — original behaviour, no extra inputs
            return uncond + 2.0 * (audio_uncond - uncond) + 1.0 * (all_cond - audio_uncond)



# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_fmt_wrapper(opt, device: torch.device) -> FMTWrapper:
    """
    Load the FLOAT checkpoint and return an eval-mode FMTWrapper.

    Only the FMT (FlowMatchingTransformer) sub-module is wrapped;
    the motion autoencoder and audio/emotion encoders are loaded into
    the FLOAT model but not exported.
    """
    print(f"  Loading FLOAT model from: {opt.ckpt_path}")
    model = FLOAT(opt).to(device)
    model.eval()

    state_dict = torch.load(opt.ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = [], []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])
            elif "wav2vec2" in name:
                pass  # wav2vec2 weights loaded separately by HuggingFace
            else:
                missing.append(name)

    if missing:
        print(f"  [WARN] {len(missing)} params not in checkpoint: {missing[:3]}{'...' if len(missing) > 3 else ''}")
    print(f"  Checkpoint loaded ({os.path.getsize(opt.ckpt_path)/1e6:.1f} MB)")

    wrapper = FMTWrapper(model.fmt).to(device)
    wrapper.eval()
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Dummy inputs
# ─────────────────────────────────────────────────────────────────────────────
def build_dummy_inputs(opt, device: torch.device, batch: int = 1, seed: int = 42):
    """
    Build random dummy inputs matching FMTWrapper.forward() signature.

    Returns a tuple:
        (t, x, wa, wr, we, prev_x, prev_wa, a_cfg_scale, r_cfg_scale, e_cfg_scale)

    Shapes (batch=1):
        t       : (1,)
        x       : (1, 50, 512)
        wa      : (1, 50, 512)
        wr      : (1, 512)
        we      : (1, 1, 7)
        prev_x  : (1, 10, 512)
        prev_wa : (1, 10, 512)
        a_cfg_scale : (1,)
        r_cfg_scale : (1,)
        e_cfg_scale : (1,)
    """
    torch.manual_seed(seed)
    n_curr = int(opt.wav2vec_sec * opt.fps)   # default: 50
    n_prev = opt.num_prev_frames               # default: 10
    dim_w  = opt.dim_w                         # 512
    dim_a  = opt.dim_a                         # 512
    dim_e  = opt.dim_e                         # 7

    return (
        torch.tensor([0.5], device=device, dtype=torch.float32),
        torch.randn(batch, n_curr, dim_w, device=device),
        torch.randn(batch, n_curr, dim_a, device=device),
        torch.randn(batch,        dim_w,  device=device),
        torch.randn(batch, 1,     dim_e,  device=device),
        torch.randn(batch, n_prev, dim_w, device=device),
        torch.randn(batch, n_prev, dim_a, device=device),
        # torch.tensor([2.0], device=device, dtype=torch.float32),  # a_cfg_scale
        # torch.tensor([1.0], device=device, dtype=torch.float32),  # r_cfg_scale
        # torch.tensor([1.0], device=device, dtype=torch.float32),  # e_cfg_scale
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output shape helper
# ─────────────────────────────────────────────────────────────────────────────
def output_shape(opt, batch: int = 1):
    """Return the expected FMTWrapper output shape for a given batch size."""
    n_total = opt.num_prev_frames + int(opt.wav2vec_sec * opt.fps)  # 60
    return (batch, n_total, opt.dim_w)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────
def perf_stats(latencies_ms: list) -> dict:
    """Compute performance statistics from a list of latency measurements (ms)."""
    a = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms":   float(np.mean(a)),
        "median_ms": float(np.median(a)),
        "p50_ms":    float(np.percentile(a, 50)),
        "p95_ms":    float(np.percentile(a, 95)),
        "p99_ms":    float(np.percentile(a, 99)),
        "min_ms":    float(np.min(a)),
        "max_ms":    float(np.max(a)),
        "std_ms":    float(np.std(a)),
        "fps":       float(1000.0 / np.mean(a)),
        "n_iters":   len(latencies_ms),
    }


def cosine_check(a: torch.Tensor, b: torch.Tensor) -> dict:
    """
    Compute cosine similarity and L2 error between two tensors.
    Both are flattened to 1-D before comparison.
    """
    af = a.contiguous().view(-1).float()
    bf = b.contiguous().view(-1).float()
    cos = F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    l2  = torch.norm(af - bf).item()
    rel = l2 / (torch.norm(af).item() + 1e-8)
    max_abs_err = (af - bf).abs().max().item()
    return {"cosine": cos, "l2": l2, "rel_l2": rel, "max_abs_err": max_abs_err}


# Precision-tuned pass/warn thresholds (cosine_sim, rel_l2_error)
PRECISION_THRESHOLDS = {
    "fp32": {"cos_min": 0.9999, "rel_max": 0.001,  "label": "near-lossless"},
    "fp16": {"cos_min": 0.950,  "rel_max": 0.300,  "label": "minor FP16 rounding"},
    "int8": {"cos_min": 0.900,  "rel_max": 0.500,  "label": "INT8 quantization noise"},
    "best": {"cos_min": 0.900,  "rel_max": 0.500,  "label": "mixed INT8/FP16 noise"},
}


# ─────────────────────────────────────────────────────────────────────────────
# argparse helper
# ─────────────────────────────────────────────────────────────────────────────
def add_model_args(parser):
    """Attach BaseOptions model/audio args to an existing ArgumentParser."""
    return BaseOptions().initialize(parser)
