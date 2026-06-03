"""
	Inference Stage 2 for Distilled Student Model (w/ or w/o TensorRT Decoder)
"""

import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment
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
		for t in range(T):
			s_r_d_t = s_r + r_d[:, t]
			if self.trt_dec_inferencer is not None:
				img_t = self.forward_decoder_tensorrt(s_r_d_t, s_r_feats)
			else:
				img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats)
			d_hat.append(img_t)
		d_hat = torch.stack(d_hat, dim=1).squeeze()
		return {'d_hat': d_hat}


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
		os.makedirs(video_path, exist_ok=True)
		with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
			temp_filename = temp_video.name
			vid = vid_target_recon.permute(0, 2, 3, 1)
			vid = vid.detach().clamp(-1, 1).cpu()
			vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
			torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)			
			if audio_path is not None:
				print("FOUND AUDIO")
				
				with open(os.devnull, 'wb') as f:
					out_name = f'seed={self.opt.seed}_{os.path.basename(self.opt.ref_path).split(".")[0]}_with_{os.path.basename(self.opt.aud_path).split(".")[0]}.mp4'
					command =  "ffmpeg -i {} -i {} -c:v copy -c:a aac {}/{} -y".format(temp_filename, audio_path, video_path, out_name)
					subprocess.call(command, shell=True, stdout=f, stderr=f)
				if os.path.exists(video_path):
					os.remove(temp_filename)
			else:
				os.rename(temp_filename, video_path)
			return video_path

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
	) -> str:

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
		print(f"> [#{run_mode}] Inference FPS = {d_hat.shape[0] / (end_inf - start_inf):.2f} frames/sec.")

		start_save = time.time()
		res_video_path = self.save_video(d_hat, res_video_path, audio_path)
		end_save = time.time()
		print(f"> [#{run_mode}] Video saving completed in {end_save - start_save:.2f} seconds.")
		print(f"> [Done] result saved at {res_video_path}")
		return res_video_path, {'n_frames': d_hat.shape[0], 'mode': run_mode}


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
		res_video_path = os.path.join(opt.res_dir, "student_%s-%s-%s-acfg%s-ecfg%s-%s.mp4" \
									% (call_time, video_name, audio_name, opt.a_cfg_scale, opt.e_cfg_scale, opt.emo))
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
