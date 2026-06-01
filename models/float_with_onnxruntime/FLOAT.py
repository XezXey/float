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

######## Main Phase 2 model ########		
class FLOAT(BaseModel):
	def __init__(self, opt, onnx_model_path: str = None, onnx_provider: str = "cuda"):
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
  
		# ONNX Runtime session for forward_with_cfv
		if onnx_model_path is None:
			print("[#] No ONNX model path provided. Using PyTorch implementation. (For accelerate_dev/onnx_export.py)")
		else:
			print("#"*100)
			print(f"[#] Initializing ONNX Runtime session with model: {onnx_model_path} and provider: {onnx_provider}")
			self.init_onnx_runtime(onnx_model_path, onnx_provider)
			print("#"*100)
   
	def init_onnx_runtime(self, onnx_model_path: str, onnx_provider: str):
			self.onnx_model_path = onnx_model_path
			
			if not os.path.isfile(self.onnx_model_path):
				raise FileNotFoundError(
					f"ONNX model file not found at '{self.onnx_model_path}'. "
					"Please run export_onnx.py to export it first."
				)
				
			# Configure execution providers
			if onnx_provider == "cuda":
				self.providers = [
        			("TensorrtExecutionProvider", {
        			    "device_id": 0,
        			    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
        			    "trt_fp16_enable": True,
        			    "trt_engine_cache_enable": True,
        			    "trt_engine_cache_path": "./trt_cache",
        			}),
					("CUDAExecutionProvider", {
						"device_id": 0,
						"arena_extend_strategy": "kNextPowerOfTwo",
						"gpu_mem_limit": 2 * 1024 * 1024 * 1024,
						"cudnn_conv_algo_search": "EXHAUSTIVE",
						"do_copy_in_default_stream": True,
					}),
					"CPUExecutionProvider"
				]
			else:
				self.providers = ["CPUExecutionProvider"]
				
			print(f"[ONNXPredictor] Initializing session with providers: {self.providers}")
			self.session_options = ort.SessionOptions()
			self.session_options.log_severity_level = 3  # Suppress INFO and WARNING logs from ONNX Runtime, 0 = VERBOSE, 1 = INFO, 2 = WARNING, 3 = ERROR
			self.fmt_onnx_session = ort.InferenceSession(self.onnx_model_path, providers=self.providers, sess_options=self.session_options)
			self.fmt_onnx_inputs = self.fmt_onnx_session.get_inputs()
			self.fmt_onnx_outputs = self.fmt_onnx_session.get_outputs()
			print(f"[ONNXPredictor] fmt_onnx_inputs: {[inp.name for inp in self.fmt_onnx_inputs]}")
			print(f"[ONNXPredictor] fmt_onnx_outputs: {[out.name for out in self.fmt_onnx_outputs]}")

			# Save input names for quick access during run
			self.fmt_onnx_input_names = [inp.name for inp in self.fmt_onnx_inputs]
			self.fmt_onnx_output_name = self.fmt_onnx_outputs[0].name

			# Log which provider was actually chosen by ONNX Runtime
			print("="*100)
			print(f"[ONNXPredictor] Session successfully created. Active providers: {self.fmt_onnx_session.get_providers()}")
			print(f"[ONNXPredictor] Inputs expected: {self.fmt_onnx_input_names}")
			print(f"[ONNXPredictor] Output name: '{self.fmt_onnx_output_name}'")
			print("="*100)
	
			self.warmup_onnx_runtime()
  
	@torch.no_grad()
	def warmup_onnx_runtime(self):
		print("[ONNXPredictor] Warming up ONNX Runtime with dummy inputs...")

		from accelerate_dev._fmt_utils import load_fmt_wrapper, build_dummy_inputs, add_model_args
		dummy_inputs = build_dummy_inputs(self.opt, device='cuda', batch=1)
		feed_dict = {
			"t": dummy_inputs[0].cpu().numpy().astype(np.float32),
			"x": dummy_inputs[1].cpu().numpy().astype(np.float32),
			"wa": dummy_inputs[2].cpu().numpy().astype(np.float32),
			"wr": dummy_inputs[3].cpu().numpy().astype(np.float32),
			"we": dummy_inputs[4].cpu().numpy().astype(np.float32),
			"prev_x": dummy_inputs[5].cpu().numpy().astype(np.float32),
			"prev_wa": dummy_inputs[6].cpu().numpy().astype(np.float32),
			"a_cfg_scale": dummy_inputs[7].cpu().numpy().astype(np.float32),
			"e_cfg_scale": dummy_inputs[8].cpu().numpy().astype(np.float32)
		}

		start_time = time.time()
		for i in tqdm.tqdm(range(100)):
			_ = self.fmt_onnx_session.run([self.fmt_onnx_output_name], feed_dict)
		end_time = time.time()
		print(f"[ONNXPredictor] Warmup completed in {end_time - start_time:.2f} seconds.")
 
 
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
		seed: int = None
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
				out = self.forward_with_cfv_onnxruntime(
						t 			= tt.unsqueeze(0),
						x 			= zt,
						wa 			= wa_t, 			 
						wr 			= r_s,
						we 			= we, 
						prev_x 		= prev_x_t, 	
						prev_wa 	= prev_wa_t,
						a_cfg_scale = a_cfg_scale,
						r_cfg_scale = r_cfg_scale,
						e_cfg_scale = e_cfg_scale
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
		print(f"[#ONNXRUNTIME]> Sampling completed in {end - start:.2f} seconds.")
		dec_start = time.time()
		data_out = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)
		dec_end = time.time()
		print(f"[#ONNXRUNTIME]> Decoding completed in {dec_end - dec_start:.2f} seconds.")
		print(f"[#ONNXRUNTIME]> Achieved FPS = {data_out['d_hat'].shape[0] / (end - start):.2f} frames/sec.")
		print(f"[#ONNXRUNTIME]> Video's shapes: {data_out['d_hat'].shape}")
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