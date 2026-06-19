"""
[#] 17 June 2026
====================================================================
generate_with_decoder_tensorrt.py
====================================================================
This scripts perform the inference of FLOAT pipeline using TensorRT engine applied to FLOAT's decoder only
- Able to run with/without TensorRT engine for FLOAT's decoder
 
Main purpose: Use to experiment the performance gain of TensorRT engine for FLOAT's decoder only.
"""

import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment, math
import numpy as np
import torch.nn.functional as F
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch
from torchdiffeq import odeint

import time
import rich
from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

import sys
sys.path.append('../../')
from models.utils import seed_everything
from models.float_with_onnxruntime.FLOAT_TRT import FLOAT
from options.base_options import BaseOptions


class FLOATWithTiming(FLOAT):
	@torch.no_grad()
	def decode_latent_into_image(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> dict:
		T = r_d.shape[1]
		d_hat = []
		frame_decoding_times = []
		for t in range(T):
			s_r_d_t = s_r + r_d[:, t]
			torch.cuda.synchronize()
			start_frame = time.time()
			if getattr(self, 'trt_dec_inferencer', None) is not None:
				img_t = self.forward_decoder_tensorrt(s_r_d_t, s_r_feats)
			else:
				img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha = None, feats = s_r_feats)
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
		nfe: int = 10,
		seed: int = None
	) -> torch.Tensor:
		r_s, a = data['r_s'], data['a']
		B = a.shape[0]

		# make time 
		time_steps = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)
		
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
		self.chunk_generation_times = []
		# sampleing chunk by chunk
		for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
			if self.opt.fix_noise_seed:
				seed_val = self.opt.seed if seed is None else seed	
				g = torch.Generator(self.opt.rank)
				g.manual_seed(seed_val)
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
			torch.cuda.synchronize()
			start_chunk_gen = time.time()
			trajectory_t = odeint(sample_chunk, x0, time_steps, **self.odeint_kwargs)
			sample_t = trajectory_t[-1]
			torch.cuda.synchronize()
			gen_t_duration = time.time() - start_chunk_gen
			self.chunk_generation_times.append(gen_t_duration)

			sample.append(sample_t)
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

		sample = self.sample(data, a_cfg_scale = a_cfg_scale, r_cfg_scale = r_cfg_scale, e_cfg_scale = e_cfg_scale, emo = emo, nfe = nfe, seed = seed)
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

class DataProcessor:
	def __init__(self, opt):
		self.opt = opt
		self.fps = opt.fps
		self.sampling_rate = opt.sampling_rate
		self.input_size = opt.input_size

		self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

		# wav2vec2 audio preprocessor
		self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)

		# image transform 
		self.transform = A.Compose([
				A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
				A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
				A_pytorch.ToTensorV2(),
			])

	@torch.no_grad()
	def process_img(self, img:np.ndarray) -> np.ndarray:
		mult = 360. / img.shape[0]

		resized_img = cv2.resize(img, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)        
		bboxes = self.fa.face_detector.detect_from_image(resized_img)
		bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
		bboxes = bboxes[0] # Just use first bbox

		bsy = int((bboxes[3] - bboxes[1]) / 2)
		bsx = int((bboxes[2] - bboxes[0]) / 2)
		my  = int((bboxes[1] + bboxes[3]) / 2)
		mx  = int((bboxes[0] + bboxes[2]) / 2)
		
		bs = int(max(bsy, bsx) * 1.6)
		img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
		my, mx  = my + bs, mx + bs  	# BBox center y, bbox center x
		
		crop_img = img[my - bs:my + bs,mx - bs:mx + bs]
		crop_img = cv2.resize(crop_img, dsize = (self.input_size, self.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
		return crop_img

	def default_img_loader(self, path) -> np.ndarray:
		img = cv2.imread(path)
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	def default_aud_loader(self, path: str) -> torch.Tensor:
		speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)
		return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]


	def preprocess(self, ref_path:str, audio_path:str, no_crop:bool) -> dict:
		s = self.default_img_loader(ref_path)
		if not no_crop:
			s = self.process_img(s)
		s = self.transform(image=s)['image'].unsqueeze(0)
		a = self.default_aud_loader(audio_path).unsqueeze(0)
		return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
	def __init__(self, opt):
		torch.cuda.empty_cache()
		self.opt = opt
		self.rank = opt.rank
		
		# Load Model
		self.load_model()
		self.load_weight(opt.ckpt_path, rank=self.rank)
		self.G.to(self.rank)
		self.G.eval()

		# Load Data Processor
		self.data_processor = DataProcessor(opt)

	def load_model(self) -> None:
		trt_model_path = getattr(self.opt, 'trt_model_path', None)
		trt_decoder_path = getattr(self.opt, 'trt_decoder_path', None)
		self.G = FLOATWithTiming(self.opt, trt_model_path=trt_model_path, trt_decoder_path=trt_decoder_path)

	def load_weight(self, checkpoint_path: str, rank: int) -> None:
		state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
		with torch.no_grad():
			for model_name, model_param in self.G.named_parameters():
				if model_name in state_dict:
					model_param.copy_(state_dict[model_name].to(rank))
				elif "wav2vec2" in model_name: pass
				else:
					print(f"! Warning; {model_name} not found in state_dict.")

		del state_dict
  
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
		nfe: int			= 10,
		no_crop: bool 		= False,
		seed: int			= 25,
		verbose: bool 		= False
	) -> str:

		data = self.data_processor.preprocess(ref_path, audio_path, no_crop = no_crop)
		if verbose: print(f"> [Done] Preprocess.")

		# Determine labeling for printouts
		run_mode = "PyTorch"
		if getattr(self.G, 'trt_inferencer', None) is not None and getattr(self.G, 'trt_dec_inferencer', None) is not None:
			run_mode = "TENSORRT"
		elif getattr(self.G, 'trt_inferencer', None) is not None:
			run_mode = "TENSORRT_FMT_ONLY"
		elif getattr(self.G, 'trt_dec_inferencer', None) is not None:
			run_mode = "TENSORRT_DECODER_ONLY"

		# inference
		start_inf = time.time()
		d_hat = self.G.inference(
			data 		= data,
			a_cfg_scale = a_cfg_scale,
			r_cfg_scale = r_cfg_scale,
			e_cfg_scale = e_cfg_scale,
			emo 		= emo,
			nfe			= nfe,
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


class InferenceOptions(BaseOptions):
	def __init__(self):
		super().__init__()

	def initialize(self, parser):
		super().initialize(parser)
		parser.add_argument("--trt_model_path",
				default=None, type=str, help="Path to the TensorRT model file for FMT (optional)")
		parser.add_argument("--trt_decoder_path",
				default=None, type=str, help="Path to the TensorRT decoder engine file (optional)")
		parser.add_argument("--ref_path",
				default=None, type=str,help='ref')
		parser.add_argument('--aud_path',
				default=None, type=str, help='audio')
		parser.add_argument('--emo',
				default=None, type=str, help='emotion', choices=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
		parser.add_argument('--no_crop',
				action = 'store_true', help = 'not using crop')
		parser.add_argument('--res_video_path',
				default=None, type=str, help='res video path')
		parser.add_argument('--ckpt_path',
				default="./checkpoints/float.pth", type=str, help='checkpoint path')
		parser.add_argument('--res_dir',
				default="./results", type=str, help='result dir')
		parser.add_argument('--seed_everything', default=False,
				action='store_true', help='seed everything for reproducibility')
		return parser


if __name__ == '__main__':
	opt = InferenceOptions().parse()
	opt.rank, opt.ngpus  = 0,1
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
		res_video_path = os.path.join(opt.res_dir, "%s-%s-%s-nfe%s-seed%s-acfg%s-ecfg%s-%s.mp4" \
									% (call_time, video_name, audio_name, opt.nfe, opt.seed, opt.a_cfg_scale, opt.e_cfg_scale, opt.emo))
	else:
		res_video_path = opt.res_video_path

	if opt.seed_everything:
		seed_everything(opt.seed)
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
			nfe			= opt.nfe,
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
