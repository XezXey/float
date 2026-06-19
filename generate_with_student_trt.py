"""
	Inference Stage 2 for Distilled Student Model (w/ or w/o TensorRT Decoder)
"""

import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment
import torch.nn.functional as F
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch

import time
import math
from tqdm import tqdm
from pathlib import Path
import sys

# Import models/float from main repo first to initialize its path
import models
import models.float

# Extend the search path of models.float to search the distillation repo's models/float directory
distil_models_float_path = "/home/mint/Dev/SCBx-TalkingHead/SCB-AI_talking_head_distillation/models/float"
if distil_models_float_path not in models.float.__path__:
	models.float.__path__.append(distil_models_float_path)

# Add distillation repo root to sys.path for top-level modules like scripts and options
distil_path = "/home/mint/Dev/SCBx-TalkingHead/SCB-AI_talking_head_distillation"
if distil_path not in sys.path:
	sys.path.append(distil_path)

from models.float.FLOAT_distil import FLOAT as StudentFLOAT
from scripts.distil_component import load_distilled_weight
from scripts.data_processor import DataProcessor as StudentDataProcessor
from generate import InferenceOptions as DistilInferenceOptions

from accelerate_dev.trt_utils import TRTInferencer
import pycuda.driver as cuda

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
	def decode_latent_into_image(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> dict:
		T = r_d.shape[1]
		d_hat = []
		frame_decoding_times = []
		for t in range(T):
			s_r_d_t = s_r + r_d[:, t]
			torch.cuda.synchronize()
			start_frame = time.time()
			if self.trt_dec_inferencer is not None:
				img_t = self.forward_decoder_tensorrt(s_r_d_t, s_r_feats)
			else:
				img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats)
			torch.cuda.synchronize()
			frame_decoding_times.append(time.time() - start_frame)
			d_hat.append(img_t)
		d_hat = torch.stack(d_hat, dim=1).squeeze()

		# Group the frame decoding times chunk-wise
		num_frames_for_clip = self.num_frames_for_clip
		num_chunks = int(math.ceil(T / num_frames_for_clip))
		self.chunk_decoding_times = []
		for c in range(num_chunks):
			chunk_frames_times = frame_decoding_times[c * num_frames_for_clip : (c + 1) * num_frames_for_clip]
			self.chunk_decoding_times.append(sum(chunk_frames_times))

		return {'d_hat': d_hat}

	@torch.no_grad()
	def sample(
		self,
		data: dict,
		a_cfg_scale: float = 1.0,
		r_cfg_scale: float = 1.0,
		e_cfg_scale: float = 1.0,
		emo: str = None,
		seed: int = None
	) -> torch.Tensor:
		r_s, a = data['r_s'], data['a']
		B = a.shape[0]
		
		# encoding audio first with whole audio
		a = a.to(self.opt.rank)
		T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
		wa = self.audio_encoder.inference(a, seq_len=T)

		# encoding emotion first
		emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
		if emo_idx is None:
			we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
		else:
			we = F.one_hot(torch.tensor(emo_idx, device=a.device), num_classes=self.opt.dim_e).unsqueeze(0).unsqueeze(0)

		sample = []
		self.chunk_generation_times = []
		start_GenVideo_time = time.time()
		total_fmt_inference_time = 0.0
		# sampling chunk by chunk
		for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
			seed_val = seed if seed is not None else 42
			g = torch.Generator(self.opt.rank)
			g.manual_seed(seed_val)
			x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank, generator=g)

			if t == 0: # should define the previous
				prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
				prev_wa_t = torch.zeros(B, self.num_prev_frames, 512).to(self.opt.rank) # size 512 for audio feature
			else:
				prev_x_t = sample_t[:, -self.num_prev_frames:]
				prev_wa_t = wa_t[:, -self.num_prev_frames:]
			
			wa_t = wa[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]

			if wa_t.shape[1] < self.num_frames_for_clip: # padding by replicate
				wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')

			t_zero = torch.zeros(B, device=self.opt.rank)
			
			torch.cuda.synchronize()
			start_inference_time = time.time()
			# Replicate generate_mint.py inference logic: Call forward directly, passing lambda weights in self.opt.
			v_pred = self.fmt.forward(
						t           = t_zero,
						x           = x0,
						wa          = wa_t,              
						wr          = r_s,
						we          = we, 
						prev_x      = prev_x_t,     
						prev_wa     = prev_wa_t,
						a_cfg_scale = a_cfg_scale,
						r_cfg_scale = r_cfg_scale,
						e_cfg_scale = e_cfg_scale,
						train       = False
				)
			
			sample_t = x0 + v_pred[:, self.num_prev_frames:]
			print("Chunk shape: ", sample_t.shape)
			sample.append(sample_t)
			torch.cuda.synchronize()
			end_inference_time = time.time()
			gen_t_duration = end_inference_time - start_inference_time
			self.chunk_generation_times.append(gen_t_duration)
			total_fmt_inference_time += gen_t_duration
			print(f"[#STUDENT] Inference completed for clip {t} in {gen_t_duration:.3f} seconds.")
   
		print(f"Total sampling time for all clips: {time.time() - start_GenVideo_time:.6f} seconds.")
		sample = torch.cat(sample, dim=1)[:, :T]
		return sample

	@torch.no_grad()
	def inference(
		self,
		data: dict,
		a_cfg_scale = None,
		r_cfg_scale = None,
		e_cfg_scale = None,
		emo			= None,
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

		sample = self.sample(data, a_cfg_scale = a_cfg_scale, r_cfg_scale = r_cfg_scale, e_cfg_scale = e_cfg_scale, emo = emo, seed = seed)
		data_out = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)

		# Print chunk-wise timing summary
		print("\n" + "="*50)
		print("          CHUNK-WISE RUNTIME SUMMARY          ")
		print("="*50)
		total_chunks = len(self.chunk_generation_times)
		totals = []
		for c in range(total_chunks):
			gen_t = self.chunk_generation_times[c]
			dec_t = self.chunk_decoding_times[c] if c < len(self.chunk_decoding_times) else 0.0
			chunk_total = gen_t + dec_t
			totals.append(chunk_total)
			print(f"Chunk {c:02d}:")
			print(f"  - Generation Time: {gen_t:.4f} seconds")
			print(f"  - Decoding Time:   {dec_t:.4f} seconds")
			print(f"  - Total Time:      {chunk_total:.4f} seconds")
		print("-"*50)
		if total_chunks > 0:
			mean_gen = np.mean(self.chunk_generation_times)
			std_gen = np.std(self.chunk_generation_times)
			mean_dec = np.mean(self.chunk_decoding_times)
			std_dec = np.std(self.chunk_decoding_times)
			mean_tot = np.mean(totals)
			std_tot = np.std(totals)
			print(f"Statistics across {total_chunks} chunks:")
			print(f"  - Generation Time: {mean_gen:.4f} ± {std_gen:.4f} seconds")
			print(f"  - Decoding Time:   {mean_dec:.4f} ± {std_dec:.4f} seconds")
			print(f"  - Total Time:      {mean_tot:.4f} ± {std_tot:.4f} seconds")
		print("="*50 + "\n")

		return data_out


class InferenceAgent:
	def __init__(self, opt):
		torch.cuda.empty_cache()
		self.opt = opt
		self.rank = opt.rank
		
		# Load Model
		self.load_model()
		
		print(f"Loading distilled weights from {opt.ckpt_path} for student model inference.")
		load_distilled_weight(self.G, opt.ckpt_path, opt.rank)
		
		self.G.to(self.rank)
		self.G.eval()

		# Load Data Processor
		self.data_processor = StudentDataProcessor(opt)

	def load_model(self) -> None:
		trt_decoder_path = getattr(self.opt, 'trt_decoder_path', None)
		self.G = StudentFLOATWithTRTDecoder(self.opt, trt_decoder_path=trt_decoder_path)

	def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
		if video_path.lower().endswith('.mp4'):
			out_file_path = video_path
			os.makedirs(os.path.dirname(video_path), exist_ok=True)
		else:
			os.makedirs(video_path, exist_ok=True)
			out_name = f'seed={self.opt.seed}_{os.path.basename(self.opt.ref_path).split(".")[0]}_with_{os.path.basename(self.opt.aud_path).split(".")[0]}_elambda{self.opt.emotion_lambda}_alambda{self.opt.motion_lambda}.mp4'
			out_file_path = os.path.join(video_path, out_name)

		with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
			temp_filename = temp_video.name

		try:
			vid = vid_target_recon.permute(0, 2, 3, 1)
			vid = vid.detach().clamp(-1, 1).cpu()
			vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
			torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)			
			if audio_path is not None:
				print(f"Merging audio with video using FFmpeg into {out_file_path}...")
				with open(os.devnull, 'wb') as f:
					command = f"ffmpeg -i {temp_filename} -i {audio_path} -c:v copy -c:a aac {out_file_path} -y"
					subprocess.call(command, shell=True, stdout=f, stderr=f)
			else:
				print(f"Saving video without audio to {out_file_path}...")
				import shutil
				shutil.move(temp_filename, out_file_path)
		finally:
			if os.path.exists(temp_filename):
				os.remove(temp_filename)
		return out_file_path

	@torch.no_grad()
	def run_inference(
		self,
		res_video_path: str,
		ref_path: str,
		audio_path: str,
		a_cfg_scale: float	= 2.0,
		r_cfg_scale: float	= 1.0,
		e_cfg_scale: float	= 1.0,
		emo: str 			= 'S2E',
		no_crop: bool 		= False,
		seed: int			= 25,
		verbose: bool 		= False
	) -> tuple:

		data = self.data_processor.preprocess(ref_path=ref_path, audio_path=audio_path, no_crop = no_crop)
		if verbose: print(f"> [Done] Preprocess.")

		run_mode = "Student_PyTorch" if self.G.trt_dec_inferencer is None else "Student_TRT_Decoder"

		# inference
		start_inf = time.time()
		d_hat = self.G.inference(
			data 		= data,
			a_cfg_scale = a_cfg_scale,
			r_cfg_scale = r_cfg_scale,
			e_cfg_scale = e_cfg_scale,
			emo 		= emo,
			seed		= seed
		)['d_hat']
		end_inf = time.time()
		print(f"> [#{run_mode}] Inference completed () in {end_inf - start_inf:.2f} seconds.")
		if d_hat.dim() == 4:
			n_frames = d_hat.shape[0]
		else:
			n_frames = d_hat.shape[1] if d_hat.dim() == 5 else 1
		print(f"> [#{run_mode}] Inference FPS = {n_frames / (end_inf - start_inf):.2f} frames/sec.")

		start_save = time.time()
		res_video_path = self.save_video(d_hat, res_video_path, audio_path)
		end_save = time.time()
		print(f"> [#{run_mode}] Video saving completed in {end_save - start_save:.2f} seconds.")
		print(f"> [Done] result saved at {res_video_path}")
		return res_video_path, {'n_frames': n_frames, 'mode': run_mode}


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


if __name__ == '__main__':
	opt = StudentInferenceOptions().parse()
	opt.rank, opt.ngpus  = 0,1
	
	from models.utils import seed_everything
	if opt.seed_everything:
		seed_everything(opt.seed)
		
	agent = InferenceAgent(opt)
	os.makedirs(opt.res_dir, exist_ok = True)

	# -------------- input -------------
	ref_path 		= opt.ref_path
	aud_path 		= opt.aud_path
	# ----------------------------------

	if opt.res_video_path is None:
		video_name = os.path.splitext(os.path.basename(ref_path))[0]
		audio_name = os.path.splitext(os.path.basename(aud_path))[0]
		call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
		res_video_path = os.path.join(opt.res_dir, "student_%s-%s-%s-aud_lambda%s-emo_lambda%s-mot_lambda%s-%s.mp4" \
									% (call_time, video_name, audio_name, opt.audio_lambda, opt.emotion_lambda, opt.motion_lambda, opt.emo))
	else:
		res_video_path = opt.res_video_path

	try:
		start = time.time()
		_, misc = agent.run_inference(
			res_video_path,
			ref_path,
			aud_path,
			a_cfg_scale = opt.a_cfg_scale,
			r_cfg_scale = opt.r_cfg_scale,
			e_cfg_scale = opt.e_cfg_scale,
			emo 		= opt.emo,
			no_crop 	= opt.no_crop,
			seed 		= opt.seed
			)
		end = time.time()
		run_mode = misc.get('mode', 'UNKNOWN')
		print(f"> [#{run_mode}] Total execution (Preprocess + {run_mode} + Save) time: {end - start:.2f} seconds.")
		print(f"> [#{run_mode}] Total execution FPS = {misc['n_frames'] / (end - start):.2f} frames/sec.")
	finally:
		if getattr(agent.G, 'context', None) is not None:
			agent.G.context.pop()  # Clean up CUDA context after all done
