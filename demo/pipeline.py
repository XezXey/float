import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math
import time
import argparse
import pycuda.driver as cuda

# Set CUDA_VISIBLE_DEVICES to 1 to match the reference command
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Import models/float from main repo first
import models
import models.float

# Extend search path for distillation components
distil_models_float_path = "/home/mint/Dev/SCBx-TalkingHead/SCB-AI_talking_head_distillation/models/float"
if distil_models_float_path not in models.float.__path__:
    models.float.__path__.append(distil_models_float_path)

distil_path = "/home/mint/Dev/SCBx-TalkingHead/SCB-AI_talking_head_distillation"
if distil_path not in sys.path:
    sys.path.append(distil_path)

from models.float.FLOAT_distil import FLOAT as StudentFLOAT
from scripts.distil_component import load_distilled_weight
from scripts.data_processor import DataProcessor as StudentDataProcessor
from generate import InferenceOptions as DistilInferenceOptions
from accelerate_dev.trt_utils import TRTInferencer

class StudentFLOATWithTRTDecoder(StudentFLOAT):
    def __init__(self, opt, trt_decoder_path=None):
        super().__init__(opt)
        self.trt_decoder_path = trt_decoder_path
        self.context = None
        self.trt_dec_inferencer = None
        
        if trt_decoder_path is not None:
            print("#" * 100)
            print(f"[#] Initializing TensorRT Decoder session with model: {trt_decoder_path}")
            self.init_trt_decoder(trt_decoder_path)
            print("#" * 100)

    def init_trt_decoder(self, trt_decoder_path: str):
        try:
            cuda.init()
        except Exception:
            pass
        self.context = cuda.Device(0).make_context()
        self.trt_dec_inferencer = TRTInferencer(trt_decoder_path)
        self.trt_dec_stream = torch.cuda.Stream()
        self.warmup_trt_decoder()
        if self.context is not None:
            self.context.pop()

    @torch.no_grad()
    def warmup_trt_decoder(self):
        print("[#TENSORRT] Warming up TensorRT Decoder engine with dummy inputs...")
        B = 1
        wa = torch.randn(B, 512).to(device=self.opt.rank, dtype=torch.float32)
        feat0 = torch.randn(B, 512, 8, 8).to(device=self.opt.rank, dtype=torch.float32)
        feat1 = torch.randn(B, 512, 16, 16).to(device=self.opt.rank, dtype=torch.float32)
        feat2 = torch.randn(B, 512, 32, 32).to(device=self.opt.rank, dtype=torch.float32)
        feat3 = torch.randn(B, 256, 64, 64).to(device=self.opt.rank, dtype=torch.float32)
        feat4 = torch.randn(B, 128, 128, 128).to(device=self.opt.rank, dtype=torch.float32)
        feat5 = torch.randn(B, 64, 256, 256).to(device=self.opt.rank, dtype=torch.float32)
        feat6 = torch.randn(B, 32, 512, 512).to(device=self.opt.rank, dtype=torch.float32)
        feats = [feat0, feat1, feat2, feat3, feat4, feat5, feat6]
        
        # Ensure context is pushed for warmup
        if self.context is not None:
            self.context.push()
        try:
            for _ in range(10):
                _ = self.forward_decoder_tensorrt(wa, feats)
        finally:
            if self.context is not None:
                self.context.pop()
        print("[#TENSORRT] Decoder warmup completed.")

    @torch.no_grad()
    def forward_decoder_tensorrt(self, wa, feats):
        wa_gpu = wa.to(device=self.opt.rank, dtype=torch.float32).contiguous()
        feats_gpu = [feat.to(device=self.opt.rank, dtype=torch.float32).contiguous() for feat in feats]
        output_tensor = torch.empty((1, 3, 512, 512), dtype=torch.float32, device=self.opt.rank)
        
        self.trt_dec_inferencer.context.set_input_shape("wa", wa_gpu.shape)
        for i, feat_gpu in enumerate(feats_gpu):
            self.trt_dec_inferencer.context.set_input_shape(f"feat{i}", feat_gpu.shape)
            
        self.trt_dec_inferencer.context.set_tensor_address("wa", wa_gpu.data_ptr())
        for i, feat_gpu in enumerate(feats_gpu):
            self.trt_dec_inferencer.context.set_tensor_address(f"feat{i}", feat_gpu.data_ptr())
        self.trt_dec_inferencer.context.set_tensor_address("output", output_tensor.data_ptr())
        
        current_stream = torch.cuda.current_stream()
        self.trt_dec_stream.wait_stream(current_stream)
        self.trt_dec_inferencer.context.execute_async_v3(stream_handle=self.trt_dec_stream.cuda_stream)
        current_stream.wait_stream(self.trt_dec_stream)
        return output_tensor

    @torch.no_grad()
    def decode_latent_into_image(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> dict:
        T = r_d.shape[1]
        d_hat = []
        
        # Ensure context is pushed for TensorRT execution
        if self.context is not None:
            self.context.push()
        try:
            for t in range(T):
                s_r_d_t = s_r + r_d[:, t]
                if self.trt_dec_inferencer is not None:
                    img_t = self.forward_decoder_tensorrt(s_r_d_t, s_r_feats)
                else:
                    img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats)
                d_hat.append(img_t)
        finally:
            if self.context is not None:
                self.context.pop()
                
        d_hat = torch.stack(d_hat, dim=1).squeeze()
        return {'d_hat': d_hat}

class StudentInferenceOptions(DistilInferenceOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--trt_decoder_path",
                default=None, type=str, help="Path to the TensorRT decoder engine file (optional)")
        parser.add_argument('--use_FMT_weight', action='store_true')
        parser.add_argument('--inference_mode', type=str, choices=['single', 'multiple'], default='single')
        parser.add_argument('--emotion_lambda', type=float, default=1.0)
        parser.add_argument('--motion_lambda', type=float, default=1.0)
        parser.add_argument('--audio_lambda', type=float, default=1.0)
        parser.add_argument('--batch_size', type=int, default=1)
        return parser

class TalkingHeadPipeline:
    def __init__(self, ckpt_path, trt_decoder_path, a_cfg_scale=2.0, e_cfg_scale=5.5):
        # Programmatically setup default options
        parser = argparse.ArgumentParser()
        opt_builder = StudentInferenceOptions()
        parser = opt_builder.initialize(parser)
        self.opt = parser.parse_args(args=[])
        
        # Override paths and scale parameters
        self.opt.ckpt_path = ckpt_path
        self.opt.trt_decoder_path = trt_decoder_path
        self.opt.a_cfg_scale = a_cfg_scale
        self.opt.e_cfg_scale = e_cfg_scale
        self.opt.rank = 0  # Device 0 under CUDA_VISIBLE_DEVICES=1 (maps to GPU 1)
        self.opt.ngpus = 1
        
        print(f"[Pipeline] Initializing pipeline with rank={self.opt.rank}")
        torch.cuda.empty_cache()
        
        # Instantiate student model
        self.model = StudentFLOATWithTRTDecoder(self.opt, trt_decoder_path=trt_decoder_path)
        
        print(f"[Pipeline] Loading weights from {ckpt_path}")
        load_distilled_weight(self.model, ckpt_path, self.opt.rank)
        
        self.model.to(self.opt.rank)
        self.model.eval()
        
        # Instantiate processor
        self.data_processor = StudentDataProcessor(self.opt)
        
        # Dimension shortcuts
        self.dim_w = self.opt.dim_w
        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)  # 50 frames
        self.num_prev_frames = int(self.opt.num_prev_frames)  # 10 frames
        
        print("[Pipeline] Initialization completed successfully.")

    def init_avatar(self, image_path):
        """
        Load and preprocess the reference portrait once.
        Returns:
            dict containing identity states: s_r, r_s, s_r_feats
        """
        print(f"[Pipeline] Initializing avatar from image: {image_path}")
        # Load and preprocess
        s_np = self.data_processor.default_img_loader(image_path)
        s_cropped = self.data_processor.process_img(s_np)
        s_tensor = self.data_processor.transform(image=s_cropped)['image'].unsqueeze(0).to(self.opt.rank)
        
        with torch.no_grad():
            s_r, r_s_lambda, s_r_feats = self.model.encode_image_into_latent(s_tensor)
            r_s = self.model.motion_autoencoder.dec.direction(r_s_lambda)
            
        avatar_data = {
            's_r': s_r.cpu(),  # Keep on CPU until needed to save GPU memory or pass easily
            'r_s': r_s.cpu(),
            's_r_feats': [feat.cpu() for feat in s_r_feats]
        }
        print("[Pipeline] Avatar initialized successfully.")
        return avatar_data

    def process_audio_chunk(self, audio_samples):
        """
        Convert raw float32 audio samples (16kHz) to model-ready features.
        Args:
            audio_samples: 1D numpy array of audio samples (normalized float32).
        Returns:
            dict: {
                'audio_features': wa_t (torch.Tensor of shape [1, T, 512]),
                'raw_audio_tensor': a (torch.Tensor of shape [1, num_samples])
            }
        """
        sampling_rate = self.opt.sampling_rate
        # Run wav2vec preprocessor
        a = self.data_processor.wav2vec_preprocessor(
            audio_samples, 
            sampling_rate=sampling_rate, 
            return_tensors='pt'
        ).input_values[0].unsqueeze(0).to(self.opt.rank)
        
        # Calculate matching seq_len
        T = math.ceil(a.shape[-1] * self.opt.fps / sampling_rate)
        
        with torch.no_grad():
            # Pad to multiple of seq_len sampling alignment if not matching
            if a.shape[1] % int(T * sampling_rate / self.opt.fps) != 0:
                pad_len = int(T * sampling_rate / self.opt.fps) - a.shape[1]
                a = F.pad(a, (0, pad_len), mode='replicate')
            
            # Extract and project wav2vec features
            wa = self.model.audio_encoder.inference(a, seq_len=T)
            
        return {
            'audio_features': wa,          # [1, T, 512] on GPU
            'raw_audio_tensor': a          # [1, num_samples] on GPU
        }

    def inference_step(self, audio_features, hidden_states, previous_frames=None, avatar_data=None, emo='S2E', seed=42):
        """
        One forward pass through Student model + TensorRT decoder.
        Args:
            audio_features: torch.Tensor of shape [1, 50, 512] (on GPU)
            hidden_states: dict containing 'prev_x_t' [1, 10, 512] and 'prev_wa_t' [1, 10, 512]
            previous_frames: optional, ignored in pure latent loop but kept in signature
            avatar_data: preprocessed identity features (s_r, r_s, s_r_feats)
            emo: str ('angry', 'happy', 'neutral', 'sad', 'surprise', or 'S2E' to predict from audio)
            seed: int for generator manual seed
        Returns:
            tuple: (frames, updated_hidden_states)
                frames: torch.Tensor of shape [50, 3, 512, 512]
                updated_hidden_states: dict containing new prev_x_t and prev_wa_t
        """
        if avatar_data is None:
            raise ValueError("avatar_data is required for inference_step")
            
        B = audio_features.shape[0]
        device = self.opt.rank
        
        # Unpack avatar data onto GPU
        s_r = avatar_data['s_r'].to(device)
        r_s = avatar_data['r_s'].to(device)
        s_r_feats = [feat.to(device) for feat in avatar_data['s_r_feats']]
        
        # Retrieve or initialize hidden states
        if hidden_states is None or 'prev_x_t' not in hidden_states or 'prev_wa_t' not in hidden_states:
            prev_x_t = torch.zeros(B, self.num_prev_frames, self.dim_w, device=device)
            prev_wa_t = torch.zeros(B, self.num_prev_frames, 512, device=device)
        else:
            prev_x_t = hidden_states['prev_x_t'].to(device)
            prev_wa_t = hidden_states['prev_wa_t'].to(device)
        
        print("Input to model [pipeline.py]:")
        print(prev_x_t.shape, prev_x_t.mean(), prev_x_t)

        # Pad audio features if shorter than 50 frames
        wa_t = audio_features
        if wa_t.shape[1] < self.num_frames_for_clip:
            wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')
            
        # 1. Determine Emotion embedding we
        emo_idx = self.model.emotion_encoder.label2id.get(str(emo).lower(), None)
        if emo_idx is None:
            # S2E mode: require a raw audio feature tensor
            # Since ser classification expects a, we pad or reconstruct it
            # We can create a dummy 2s silent audio if none is provided
            # Or we can predict emotion if a raw audio tensor is passed in hidden_states or avatar_data
            # In our system, process_audio_chunk returns raw_audio_tensor.
            raw_a = hidden_states.get('raw_audio_tensor', None)
            if raw_a is not None:
                we = self.model.emotion_encoder.predict_emotion(raw_a.to(device)).unsqueeze(1)
            else:
                # Fallback to neutral one-hot
                neutral_idx = self.model.emotion_encoder.label2id.get('neutral', 4)
                we = F.one_hot(torch.tensor(neutral_idx, device=device), num_classes=self.opt.dim_e).unsqueeze(0).unsqueeze(0)
        else:
            we = F.one_hot(torch.tensor(emo_idx, device=device), num_classes=self.opt.dim_e).unsqueeze(0).unsqueeze(0)
            
        # 2. Sample random Gaussian noise x0
        g = torch.Generator(device)
        g.manual_seed(seed)
        x0 = torch.randn(B, self.num_frames_for_clip, self.dim_w, device=device, generator=g)
        
        # 3. FMT Forward Pass
        t_zero = torch.zeros(B, device=device)
        with torch.no_grad():
            v_pred = self.model.fmt.forward(
                t=t_zero,
                x=x0,
                wa=wa_t,
                wr=r_s,
                we=we,
                prev_x=prev_x_t,
                prev_wa=prev_wa_t,
                a_cfg_scale=self.opt.a_cfg_scale,
                r_cfg_scale=self.opt.r_cfg_scale,
                e_cfg_scale=self.opt.e_cfg_scale,
                train=False
            )
            
            # Predict next latents
            sample_t = x0 + v_pred[:, self.num_prev_frames:]
            
            # 4. Decode latents into images via TensorRT / motion autoencoder
            # (Note: context push/pop is handled inside decode_latent_into_image wrapper)
            res = self.model.decode_latent_into_image(s_r=s_r, s_r_feats=s_r_feats, r_d=sample_t)
            frames = res['d_hat']
            
            # Carry forward states for autoregression
            next_prev_x = sample_t[:, -self.num_prev_frames:]
            next_prev_wa = wa_t[:, -self.num_prev_frames:]
            
        print("pipeline.py")
        print(sample_t[:, -self.num_prev_frames:].shape, torch.mean(sample_t[:, -self.num_prev_frames:]),  sample_t[:, -self.num_prev_frames:])
        next_hidden_states = {
            'prev_x_t': next_prev_x,
            'prev_wa_t': next_prev_wa
        }
        
        return frames, next_hidden_states

def stream_output(frames):
    """
    Format output frames for client streaming.
    Converts a tensor [50, 3, 512, 512] on GPU to a numpy array [50, 512, 512, 3] of uint8.
    """
    # Swap axes [T, C, H, W] -> [T, H, W, C]
    vid = frames.permute(0, 2, 3, 1)
    # Clamp and convert
    vid = vid.detach().clamp(-1, 1).cpu()
    vid = ((vid + 1) / 2 * 255).type(torch.uint8).numpy()
    return vid
