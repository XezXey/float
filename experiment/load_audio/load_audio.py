from transformers import Wav2Vec2FeatureExtractor
import torch as th
import librosa
import math

fps = 25
sampling_rate = 16000
wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained("/home/mint/Dev/SCBx-TalkingHead/float/checkpoints/wav2vec2-base-960h", local_files_only=True)
speech_array, sampling_rate = librosa.load("/home/mint/Dev/SCBx-TalkingHead/float/assets/aud-sample-vs-1.wav", sr = sampling_rate)
a = wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]
T = math.ceil(a.shape[-1] * fps / sampling_rate)
print(speech_array.shape, sampling_rate)
print(a.shape)
print("Audio length in seconds:", speech_array.shape[0] / sampling_rate)
print("Converted into #frames:", T)
