"""
Export individual components of FLOAT model to ONNX.
"""
import os
import torch
import torch_tensorrt
import sys
import argparse
from torch.export.dynamic_shapes import Dim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.float.FLOAT import FLOAT
from options.base_options import BaseOptions

def load_weight(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    with torch.no_grad():
        for model_name, model_param in model.named_parameters():
            if model_name in state_dict:
                model_param.copy_(state_dict[model_name].to(device))
            elif "wav2vec2" in model_name: 
                pass
            else:
                print(f"[Warning] {model_name} not found in state_dict.")

class FMTWrapper(torch.nn.Module):
    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
    def forward(self, t, x, wa, wr, we, prev_x, prev_wa, a_cfg, r_cfg, e_cfg):
        return self.fmt.forward_with_cfv_onnx_compat(t, x, wa, wr, we, prev_x, prev_wa, a_cfg, r_cfg, e_cfg)

def export_models_to_onnx(opt, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.rank = device
    
    # Initialize the FLOAT model
    model = FLOAT(opt).to(device)
    model.eval()
    
    # Load weights
    load_weight(model, opt.ckpt_path, device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export FMT (Flow Matching Transformer)
    print("Exporting FMT...")
    fmt_wrapper = FMTWrapper(model.fmt).to(device)
    fmt_wrapper.eval()
    
    B = 1
    num_frames = 50 # opt.wav2vec_sec * opt.fps (2 * 25)
    dim_a = opt.dim_a   # audio
    dim_e = opt.dim_e   # emotion
    dim_h = opt.dim_h   # hidden size
    dim_m = opt.dim_m   # model - #N blocks 
    dim_w = opt.dim_w   # 
    num_prev = opt.num_prev_frames
    
    dummy_t = torch.tensor([0.0], device=device)
    dummy_x = torch.randn(B, num_frames, dim_w, device=device)
    dummy_wa = torch.randn(B, num_frames, dim_a, device=device)
    dummy_wr = torch.randn(B, dim_w, device=device)
    dummy_we = torch.randn(B, 1, dim_e, device=device)
    dummy_prev_x = torch.randn(B, num_prev, dim_w, device=device)
    dummy_prev_wa = torch.randn(B, num_prev, dim_a, device=device)
    dummy_a_cfg = torch.tensor(2.0, device=device)
    dummy_r_cfg = torch.tensor(1.0, device=device)
    dummy_e_cfg = torch.tensor(1.0, device=device)
    # dummy_a_cfg = 2.0
    # dummy_r_cfg = 1.0
    # dummy_e_cfg = 1.0
    print("Dummy inputs created for FMT export.")
    print("Input shapes:")
    print("t:", dummy_t.shape, dummy_t.device)
    print("x:", dummy_x.shape, dummy_x.device)
    print("wa:", dummy_wa.shape, dummy_wa.device)
    print("wr:", dummy_wr.shape, dummy_wr.device)
    print("we:", dummy_we.shape, dummy_we.device)
    print("prev_x:", dummy_prev_x.shape, dummy_prev_x.device)
    print("prev_wa:", dummy_prev_wa.shape, dummy_prev_wa.device)
    # print("a_cfg:", dummy_a_cfg.shape, dummy_a_cfg.device)
    # print("r_cfg:", dummy_r_cfg.shape, dummy_r_cfg.device)
    # print("e_cfg:", dummy_e_cfg.shape, dummy_e_cfg.device)
    print("fmt.device:", next(model.fmt.parameters()).device)
    print("fmt_wrapper.device:", next(fmt_wrapper.parameters()).device)
    print(fmt_wrapper.forward(dummy_t, dummy_x, dummy_wa, dummy_wr, dummy_we, dummy_prev_x, dummy_prev_wa, dummy_a_cfg, dummy_r_cfg, dummy_e_cfg).shape)
    # 2. Compile directly to TensorRT (Bypassing manual ONNX export)
    inputs = [dummy_t, dummy_x, dummy_wa, dummy_wr, dummy_we, dummy_prev_x, dummy_prev_wa, dummy_a_cfg, dummy_r_cfg, dummy_e_cfg]
    trt_model = torch_tensorrt.compile(
        fmt_wrapper, 
        ir="torch_compile", # Uses PyTorch's native Dynamo backend
        inputs=inputs, 
        enabled_precisions={torch.float16}
    )
    outputs = trt_model(*inputs)
    exit()
    

    torch.onnx.export(
        fmt_wrapper, 
        (dummy_t, dummy_x, dummy_wa, dummy_wr, dummy_we, dummy_prev_x, dummy_prev_wa, dummy_a_cfg, dummy_r_cfg, dummy_e_cfg),
        os.path.join(output_dir, "fmt.onnx"),
        export_params=True,
        opset_version=17,              # Keep opset high to support the internal attention blocks
        # do_constant_folding=True,
        input_names=['t', 'x', 'wa', 'wr', 'we', 'prev_x', 'prev_wa', 'a_cfg', 'r_cfg', 'e_cfg'],
        output_names=['out'],
        dynamic_axes={
            'x': {0: 'batch_size'},
            'wa': {0: 'batch_size'},
            'wr': {0: 'batch_size'},
            'we': {0: 'batch_size'},
            'prev_x': {0: 'batch_size'},
            'prev_wa': {0: 'batch_size'},
            'out': {0: 'batch_size'}
        }
    )
    exit()

    torch.onnx.export(
        fmt_wrapper, 
        (dummy_t, dummy_x, dummy_wa, dummy_wr, dummy_we, dummy_prev_x, dummy_prev_wa, dummy_a_cfg, dummy_r_cfg, dummy_e_cfg),
        os.path.join(output_dir, "fmt.onnx"),
        export_params=True,
        input_names=['t', 'x', 'wa', 'wr', 'we', 'prev_x', 'prev_wa', 'a_cfg', 'r_cfg', 'e_cfg'],
        # output_names=['out'],
        # dynamic_shapes={
        #     't': Dim.AUTO,#(min=1, max=10),
        #     'x': Dim.AUTO,#(min=1, max=10),
        #     'wa': Dim.AUTO,#(min=1, max=10),
        #     'wr': Dim.AUTO,#(min=1, max=10),
        #     'we': Dim.AUTO,#(min=1, max=10),
        #     'prev_x': Dim.AUTO,#(min=1, max=10),
        #     'prev_wa': Dim.AUTO,#(min=1, max=10),
        #     # 'out': Dim.AUTO,#(min=1, max=10) 
        #     'a_cfg': Dim.AUTO,#(min=1, max=10),
        #     'r_cfg': Dim.AUTO,#(min=1, max=10),
        #     'e_cfg': Dim.AUTO,#(min=1, max=10)
        # },
        dynamo=True
    )
    print("FMT exported successfully!")

    # To fully utilize TRT and ONNX, other components like motion_autoencoder.enc & .dec,
    # and audio_encoder / emotion_encoder can also be exported similarly.
    
    print("You can extend this script to export other components like motion_autoencoder.dec, etc.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    base_options = BaseOptions()
    parser = base_options.initialize(parser)
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to checkpont (float.pth)')
    parser.add_argument('--output_dir', type=str, default='./onnx_models', help='Directory to save ONNX models')
    
    opt, _ = parser.parse_known_args()
    export_models_to_onnx(opt, opt.output_dir) 