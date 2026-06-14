from transformers import Wav2Vec2FeatureExtractor
import soundfile as sf
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

# Create a new audio file (15-secs) with white noise
duration = 15  # seconds
num_samples = duration * sampling_rate
white_noise = th.randn(num_samples).numpy()
sf.write("/home/mint/Dev/SCBx-TalkingHead/float/assets/white_noise.wav", white_noise, sampling_rate)

# Create a new audio file (15-secs) with a sine wave at 440 Hz
frequency = 440  # Hz
t = th.linspace(0, duration, num_samples)
sine_wave = 0.5 * th.sin(2 * math.pi * frequency * t).numpy()
sf.write("/home/mint/Dev/SCBx-TalkingHead/float/assets/sine_wave.wav", sine_wave, sampling_rate)

# Create a new audio file (15-secs) with a muted sound (silence)
silence = th.zeros(num_samples).numpy()
sf.write("/home/mint/Dev/SCBx-TalkingHead/float/assets/silence.wav", silence, sampling_rate)

