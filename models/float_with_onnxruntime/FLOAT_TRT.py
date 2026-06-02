import os
import time
import tqdm

import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort


from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float_with_onnxruntime.generator import Generator
from models.float_with_onnxruntime.FMT import FlowMatchingTransformer

from accelerate_dev.trt_utils import TRTInferencer
import pycuda.driver as cuda

######## Main Phase 2 model ########		
class FLOAT(BaseModel):
	def __init__(self, opt, trt_model_path: str = None, trt_decoder_path: str = None):
		super().__init__()
		self.opt = opt

		self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(self.opt.num_prev_frames)

		# motion latent auto-encoder
		self.motion_autoencoder = Generator(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
		self.motion_autoencoder.requires_grad_(False)

		# condition encoders
		self.audio_encoder 		= AudioEncoder(opt)
		self.emotion_encoder	= Audio2Emotion(opt)

		# FMT; Flow Matching Transformer
		self.fmt = FlowMatchingTransformer(opt)
		
		# ODE options
		self.odeint_kwargs = {
			'atol': self.opt.ode_atol,
			'rtol': self.opt.ode_rtol,
			'method': self.opt.torchdiffeq_ode_method
		}
		self.context = None
  
		# ONNX Runtime session for forward_with_cfv
		if trt_model_path is None:
			print("[#] No TensorRT model path provided. Using PyTorch implementation. (For accelerate_dev/trt_export.py)")
		else:
			print("#"*100)
			print(f"[#] Initializing TensorRT session with model: {trt_model_path}")
			self.init_trt_engine(trt_model_path)
			print("#"*100)

		# TensorRT Decoder setup
		if trt_decoder_path is None:
			print("[#] No TensorRT decoder path provided. Using PyTorch decoder implementation.")
			self.trt_dec_inferencer = None
		else:
			print("#"*100)
			print(f"[#] Initializing TensorRT Decoder session with model: {trt_decoder_path}")
			self.init_trt_decoder(trt_decoder_path)
			print("#"*100)
   
	def init_trt_engine(self, onnx_model_path: str):
			self.onnx_model_path = onnx_model_path
			cuda.init()
			self.context = cuda.Device(0).make_context()
			self.trt_inferencer = TRTInferencer(self.onnx_model_path)
			self.trt_stream = torch.cuda.Stream()
			self.warmup_trt_engine()

	def init_trt_decoder(self, trt_decoder_path: str):
			self.trt_decoder_path = trt_decoder_path
			try:
				cuda.init()
			except Exception:
				pass
			if not hasattr(self, 'context') or self.context is None:
				self.context = cuda.Device(0).make_context()
			self.trt_dec_inferencer = TRTInferencer(self.trt_decoder_path)
			self.trt_dec_stream = torch.cuda.Stream()
			self.warmup_trt_decoder()

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
		for _ in range(10):
			_ = self.forward_decoder_tensorrt(wa, feats)
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
	def warmup_trt_engine(self):
		print("[#TENSORRT] Warming up TensorRT engine with dummy inputs...")
		# --- Prepare inputs (example with batch_size=1) ---
		B = 1
		inputs = {
			"t":           np.array([0.5],               dtype=np.float32),
			"x":           np.random.rand(B, 50, 512).astype(np.float32),
			"wa":          np.random.rand(B, 50, 512).astype(np.float32),
			"wr":          np.random.rand(B, 512).astype(np.float32),
			"we":          np.random.rand(B, 1, 7).astype(np.float32),
			"prev_x":      np.random.rand(B, 10, 512).astype(np.float32),
			"prev_wa":     np.random.rand(B, 10, 512).astype(np.float32),
			"a_cfg_scale": np.array([1.0],               dtype=np.float32),
			"e_cfg_scale": np.array([1.0],               dtype=np.float32),
		}
  
		start_time = time.time()
		for i in tqdm.tqdm(range(100)):
			_ = self.trt_inferencer.infer(batch_size=B, **inputs)
		end_time = time.time()
		print(f"[#TENSORRT] Warmup completed in {end_time - start_time:.2f} seconds.")
 
 
	######## Motion Encoder - Decoder ########
	@torch.no_grad()
	def encode_image_into_latent(self, x: torch.Tensor) -> list:
		x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
		x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
		return x_r, x_r_lambda, x_r_feats

	@torch.no_grad()
	def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
		x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
		r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
		return r_x

	@torch.no_grad()
	def decode_latent_into_image(self, s_r: torch.Tensor , s_r_feats: list, r_d: torch.Tensor) -> dict:
		T = r_d.shape[1]
		d_hat = []
		for t in range(T):
			s_r_d_t = s_r + r_d[:, t]
			if getattr(self, 'trt_dec_inferencer', None) is not None:
				img_t = self.forward_decoder_tensorrt(s_r_d_t, s_r_feats)
			else:
				img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha = None, feats = s_r_feats)
			d_hat.append(img_t)
		d_hat = torch.stack(d_hat, dim=1).squeeze()
		return {'d_hat': d_hat}


	######## Motion Sampling and Inference ########
	@torch.no_grad()
	def sample(
		self,
		data: dict,
		a_cfg_scale: float = 1.0,
		r_cfg_scale: float = 1.0,
		e_cfg_scale: float = 1.0,
		emo: str = None,
		nfe: int = 10,
		seed: int = None,
	) -> torch.Tensor:

		r_s, a = data['r_s'], data['a']
		B = a.shape[0]

		# make time 
		time = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)
		
		# encoding audio first with whole audio
		a = a.to(self.opt.rank)
		T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
		wa = self.audio_encoder.inference(a, seq_len=T)

		# encoding emotion first
		emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
		if emo_idx is None:
			we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
		else:
			we = F.one_hot(torch.tensor(emo_idx, device = a.device), num_classes = self.opt.dim_e).unsqueeze(0).unsqueeze(0)

		sample = []
		# sampleing chunk by chunk
		for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
			if self.opt.fix_noise_seed:
				seed = self.opt.seed if seed is None else seed	
				g = torch.Generator(self.opt.rank)
				g.manual_seed(seed)
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank, generator = g)
			else:
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)

			if t == 0: # should define the previous
				prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
				prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
			else:
				prev_x_t = sample_t[:, -self.num_prev_frames:]
				prev_wa_t = wa_t[:, -self.num_prev_frames:]
			
			wa_t = wa[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]

			if wa_t.shape[1] < self.num_frames_for_clip: # padding by replicate
				wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')

			def sample_chunk(tt, zt):
				if getattr(self, 'trt_inferencer', None) is not None:
					out = self.forward_with_cfv_tensorrt(
							t 			= tt.unsqueeze(0),
							x 			= zt,
							wa 			= wa_t, 			 
							wr 			= r_s,
							we 			= we, 
							prev_x 		= prev_x_t, 	
							prev_wa 	= prev_wa_t,
							a_cfg_scale = a_cfg_scale,
							r_cfg_scale = r_cfg_scale,
							e_cfg_scale = e_cfg_scale,
						)
				else:
					out = self.fmt.forward_with_cfv(
							t 			= tt.unsqueeze(0),
							x 			= zt,
							wa 			= wa_t, 			 
							wr 			= r_s,
							we 			= we, 
							prev_x 		= prev_x_t, 	
							prev_wa 	= prev_wa_t,
							a_cfg_scale = a_cfg_scale,
							r_cfg_scale = r_cfg_scale,
							e_cfg_scale = e_cfg_scale,
						)
				out_current = out[:, self.num_prev_frames:]
				return out_current

			# solve ODE
			trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
			sample_t = trajectory_t[-1]
			sample.append(sample_t)
		sample = torch.cat(sample, dim=1)[:, :T]
		return sample

	@torch.no_grad()
	def forward_with_cfv_tensorrt(
		self,
		t,
		x,
		wa,
		wr,
		we,
		prev_x,
		prev_wa,
		a_cfg_scale,
		r_cfg_scale,
		e_cfg_scale,
	):
		# Wrap scalars as CUDA tensors if necessary
		if not isinstance(a_cfg_scale, torch.Tensor):
			a_cfg_scale = torch.tensor([a_cfg_scale], dtype=torch.float32, device=x.device)
		if not isinstance(e_cfg_scale, torch.Tensor):
			e_cfg_scale = torch.tensor([e_cfg_scale], dtype=torch.float32, device=x.device)

		# Ensure inputs are contiguous float32 GPU tensors
		t_gpu = t.to(device=x.device, dtype=torch.float32).contiguous()
		x_gpu = x.to(device=x.device, dtype=torch.float32).contiguous()
		wa_gpu = wa.to(device=x.device, dtype=torch.float32).contiguous()
		wr_gpu = wr.to(device=x.device, dtype=torch.float32).contiguous()
		we_gpu = we.to(device=x.device, dtype=torch.float32).contiguous()
		prev_x_gpu = prev_x.to(device=x.device, dtype=torch.float32).contiguous()
		prev_wa_gpu = prev_wa.to(device=x.device, dtype=torch.float32).contiguous()
		a_cfg_gpu = a_cfg_scale.to(device=x.device, dtype=torch.float32).contiguous()
		e_cfg_gpu = e_cfg_scale.to(device=x.device, dtype=torch.float32).contiguous()

		# Determine the output shape
		B = x.shape[0]
		out_len = prev_x.shape[1] + x.shape[1]
		dim_w = x.shape[2]
		out_shape = (B, out_len, dim_w)

		# Pre-allocate output directly on GPU
		output_tensor = torch.empty(out_shape, dtype=torch.float32, device=x.device)

		# Bind input dimensions
		self.trt_inferencer.context.set_input_shape("t", t_gpu.shape)
		self.trt_inferencer.context.set_input_shape("x", x_gpu.shape)
		self.trt_inferencer.context.set_input_shape("wa", wa_gpu.shape)
		self.trt_inferencer.context.set_input_shape("wr", wr_gpu.shape)
		self.trt_inferencer.context.set_input_shape("we", we_gpu.shape)
		self.trt_inferencer.context.set_input_shape("prev_x", prev_x_gpu.shape)
		self.trt_inferencer.context.set_input_shape("prev_wa", prev_wa_gpu.shape)
		self.trt_inferencer.context.set_input_shape("a_cfg_scale", a_cfg_gpu.shape)
		self.trt_inferencer.context.set_input_shape("e_cfg_scale", e_cfg_gpu.shape)

		# Bind raw memory addresses
		self.trt_inferencer.context.set_tensor_address("t", t_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("x", x_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("wa", wa_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("wr", wr_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("we", we_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("prev_x", prev_x_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("prev_wa", prev_wa_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("a_cfg_scale", a_cfg_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("e_cfg_scale", e_cfg_gpu.data_ptr())
		self.trt_inferencer.context.set_tensor_address("output", output_tensor.data_ptr())

		# Enqueue execution asynchronously on a dedicated, non-default stream to avoid performance and synchronization issues
		current_stream = torch.cuda.current_stream()
		self.trt_stream.wait_stream(current_stream)  # Wait for PyTorch inputs to be ready
		self.trt_inferencer.context.execute_async_v3(stream_handle=self.trt_stream.cuda_stream)
		current_stream.wait_stream(self.trt_stream)  # Wait for TensorRT outputs to be ready before PyTorch consumes them

		return output_tensor

	@torch.no_grad()
	def forward_with_cfv_onnxruntime(
		self,
		t,
		x,
		wa,
		wr,
		we,
		prev_x,
		prev_wa,
		a_cfg_scale,
		r_cfg_scale,
		e_cfg_scale
	):
		feed_dict = {
			"t": t.cpu().numpy().astype(np.float32),
			"x": x.cpu().numpy().astype(np.float32),
			"wa": wa.cpu().numpy().astype(np.float32),
			"wr": wr.cpu().numpy().astype(np.float32),
			"we": we.cpu().numpy().astype(np.float32),
			"prev_x": prev_x.cpu().numpy().astype(np.float32),
			"prev_wa": prev_wa.cpu().numpy().astype(np.float32),
			"a_cfg_scale": np.array([a_cfg_scale], dtype=np.float32),
			"e_cfg_scale": np.array([e_cfg_scale], dtype=np.float32)
		}
		ort_outputs = self.fmt_onnx_session.run([self.fmt_onnx_output_name], feed_dict)
		output_numpy = ort_outputs[0]
		return torch.from_numpy(output_numpy).to(device=x.device)

	@torch.no_grad()
	def inference(
		self,
		data: dict,
		a_cfg_scale = None,
		r_cfg_scale = None,
		e_cfg_scale = None,
		emo			= None,
		nfe			= 10,
		seed		= None,
	) -> dict:

		s, a = data['s'], data['a']
		s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
		if 's_r' in data:
			r_s = self.encode_identity_into_motion(s_r)
		else:
			r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
		data['r_s'] = r_s

		# set conditions
		if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
		if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
		if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale

		import time
		start = time.time()
		sample = self.sample(data, a_cfg_scale = a_cfg_scale, r_cfg_scale = r_cfg_scale, e_cfg_scale = e_cfg_scale, emo = emo, nfe = nfe, seed = seed)
		end = time.time()
		print(f"[#TENSORRT]> Sampling completed in {end - start:.2f} seconds.")
		dec_start = time.time()
		data_out = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)
		dec_end = time.time()
		print(f"[#TENSORRT]> Decoding completed in {dec_end - dec_start:.2f} seconds.")
		print(f"[#TENSORRT]> Achieved FPS = {data_out['d_hat'].shape[0] / (end - start):.2f} frames/sec.")
		print(f"[#TENSORRT]> Video's shapes: {data_out['d_hat'].shape}")
		return data_out




################ Condition Encoders ################
class AudioEncoder(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.only_last_features = opt.only_last_features
		
		self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(opt.num_prev_frames)

		self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only = True)
		self.wav2vec2.feature_extractor._freeze_parameters()

		for name, param in self.wav2vec2.named_parameters():
			param.requires_grad = False

		audio_input_dim = 768 if opt.only_last_features else 12 * 768

		self.audio_projection = nn.Sequential(
			nn.Linear(audio_input_dim, opt.dim_w),
			nn.LayerNorm(opt.dim_w),
			nn.SiLU()
			)

	def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
		if self.only_last_features:
			a = a.last_hidden_state
		else:
			a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
			a = a.reshape(a.shape[0], a.shape[1], -1)
		return a

	def forward(self, a:torch.Tensor, prev_a:torch.Tensor = None) -> torch.Tensor:
		if prev_a is not None:
			a = torch.cat([prev_a, a], dim = 1)
			if a.shape[1] % int( (self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip + self.num_prev_frames)
		else:
			if a.shape[1] % int( self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
	
		return self.audio_projection(a) # frame by frame

	@torch.no_grad()
	def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
			a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
		a = self.get_wav2vec2_feature(a, seq_len=seq_len)
		return self.audio_projection(a)



class Audio2Emotion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(opt.audio2emotion_path, local_files_only=True)
        self.wav2vec2_for_emotion.eval()
        
		# seven labels
        self.id2label = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
						4: "neutral", 5: "sad", 6: "surprise"}

        self.label2id = {v: k for k, v in self.id2label.items()}

    @torch.no_grad()
    def predict_emotion(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
        logits = self.wav2vec2_for_emotion.forward(a).logits
        return F.softmax(logits, dim=1) 	# scores

#######################################################