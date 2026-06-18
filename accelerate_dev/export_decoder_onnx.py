#!/usr/bin/env python3
"""
export_decoder_onnx.py
====================================================================
Loads the pre-trained FLOAT PyTorch model, wraps the Synthesis decoder (dec)
using DecoderWrapper, creates dummy inputs matching the target shapes,
and exports it to ONNX.
"""

import os
import sys
import argparse
import torch
import warnings
from rich.console import Console
console = Console()
print = console.print

# Ensure the root folder is in the Python search path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from generate import InferenceOptions
from models.float_with_onnxruntime.FLOAT_ONNX import FLOAT


class DecoderWrapper(torch.nn.Module):
    """
    Wraps the Decoder network to accept flattened features as individual inputs.
    This guarantees clean tracing and ONNX export without dealing with lists of tensors.
    """
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, wa, feat0, feat1, feat2, feat3, feat4, feat5, feat6):
        feats = [feat0, feat1, feat2, feat3, feat4, feat5, feat6]
        img, _ = self.dec(wa, alpha=None, feats=feats)
        return img

def main():
    parser = argparse.ArgumentParser(description="Export FLOAT's Decoder to ONNX format")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/float.pth",
        help="Path to the PyTorch checkpoint file (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="onnx_models",
        help="Directory to save the exported ONNX model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="float_decoder.onnx",
        help="Filename of the exported ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the dummy inputs during export",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )
    
    args, extra_args = parser.parse_known_args()
    if not args.verbose:
        from torch.jit import TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=TracerWarning)

    # Build default options
    sys.argv = [sys.argv[0]]
    opt = InferenceOptions().parse()
    opt.ckpt_path = args.ckpt_path
    opt.rank = 0
    opt.ngpus = 1

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cyan]\[#] Using device: {device}")

    # Load FLOAT model and copy state weights
    print(f"[cyan]\[#] Loading FLOAT model from: {opt.ckpt_path}")
    if not os.path.isfile(opt.ckpt_path):
        print(f"[red]\[!] Error: Checkpoint file not found at '{opt.ckpt_path}'.")
        sys.exit(1)

    model = FLOAT(opt).to(device)
    model.eval()

    state_dict = torch.load(opt.ckpt_path, map_location=device, weights_only=True)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict:
                param.copy_(state_dict[name])
    
    print(f"[green]\[#] Successfully loaded FLOAT model weights from: {opt.ckpt_path}")

    #NOTE: Modifying ModulatedConv2d to avoid dynamic tracing issues with groups/batch dimensions
    from models.float_with_onnxruntime.styledecoder import ModulatedConv2d
    from torch.nn import functional as F

    def patched_forward(self, input, style):
        batch = 1  # Static batch size to resolve weight shapes
        _, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out

    ModulatedConv2d.forward = patched_forward

    # Wrap the decoder
    dec_wrapper = DecoderWrapper(model.motion_autoencoder.dec).to(device)
    dec_wrapper.eval()

    # Generate dummy inputs for the decoder
    print(f"[cyan]\[#] Generating dummy inputs...")
    B = args.batch_size
    wa = torch.randn(B, 512, device=device)
    feat0 = torch.randn(B, 512, 8, 8, device=device)
    feat1 = torch.randn(B, 512, 16, 16, device=device)
    feat2 = torch.randn(B, 512, 32, 32, device=device)
    feat3 = torch.randn(B, 256, 64, 64, device=device)
    feat4 = torch.randn(B, 128, 128, 128, device=device)
    feat5 = torch.randn(B, 64, 256, 256, device=device)
    feat6 = torch.randn(B, 32, 512, 512, device=device)

    dummy_inputs = (wa, feat0, feat1, feat2, feat3, feat4, feat5, feat6)

    input_names = ["wa", "feat0", "feat1", "feat2", "feat3", "feat4", "feat5", "feat6"]
    output_names = ["output"]

    # The synthesis network (StyleGAN2-based) uses modulated convolutions with dynamic group-convolutions.
    # To export it to ONNX, we must use a fixed batch size of 1.
    # Since the decoding pipeline (decode_latent_into_image) processes chunk-by-chunk (with batch=1),
    # a static batch size of 1 is perfectly suited and avoids shape-tracing errors.
    dynamic_axes = None

    print(f"[cyan]\[#] Exporting FLOAT's Decoder into ONNX format (opset_version={args.opset})...")
    torch.onnx.export(
        dec_wrapper,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"[green]\[#] Successfully exported ONNX decoder to: {output_path}")


if __name__ == "__main__":
    main()
