# Integrating FLOAT Talking Head into a Production Pipeline

### 🚧 ⚠️ WARNING: Mockup Only

**Known Bug-like Behavior**
- Random lagging on playback (live output box) — plays back chunk-wise, similar to a streaming platform
- Long waiting queue after submitting a text prompt — unclear whether the queue is stuck or how to clear it (needs better handling/feedback)

**This is a mockup** left in this state purely **for demonstration purposes** of the integration.

---

This document explains the architectural pattern for integrating the **FLOAT Real-Time Talking Head** into a production system. It details the core functions of the talking head pipeline and describes how to orchestrate them in a low-latency environment.

**Contents**
- [1. Architectural Pattern: Core Pipeline vs. Runner](#1-architectural-pattern-core-pipeline-vs-runner)
- [2. Core Functions of TalkingHeadPipeline](#2-core-functions-of-talkingheadpipeline)
- [3. Production Implementation Guidelines](#3-production-implementation-guidelines)
- [▶ Run This Demo](#-run-this-demo)

---

## 1. Architectural Pattern: Core Pipeline vs. Runner

To build a high-performance system, the implementation is divided into two separate concerns:

| Concern | File | Responsibility |
|---|---|---|
| 🧩 **Pipeline Definition** | [`pipeline.py`](./pipeline.py) | Atomic data processing and model forward-pass tasks. No knowledge of HTTP, WebSockets, or multi-client routing — a stateless/state-driven engine that transforms audio + image inputs into video frames. |
| 🛰️ **Runner / Server** | [`server.py`](./server.py) | Network I/O, threads, client connections, Text-to-Speech (TTS) generation, concurrency buffers, and pipeline execution scheduling. |

---

## 2. Core Functions of [`TalkingHeadPipeline`](./pipeline.py#L173-L225)

The class [`TalkingHeadPipeline`](./pipeline.py#L173-L225) handles model initialization, weights mapping (supporting student models or teacher models, optionally with TensorRT decoders), portrait preprocessing, and frame generation.

### 2.1 ⚙️ Initialization — `__init__`
```python
def __init__(self, ckpt_path, trt_decoder_path, a_cfg_scale=2.0, e_cfg_scale=5.5, is_teacher=None)
```
**Purpose:** Loads checkpoints, initializes the PyTorch model, optionally binds the TensorRT engine for decoding, and prepares the data processor.

**Key Actions:**
- Automatically identifies whether it is loading a distilled student or a teacher model based on the checkpoint name.
- Instantiates either [`StudentFLOATWithTRTDecoder`](./pipeline.py#L36-L128) or [`TeacherFLOATWithTRTDecoder`](./pipeline.py#L130-L156).

### 2.2 🖼️ Preprocessing Portrait — `init_avatar`
```python
def init_avatar(self, image_path)
```
**Purpose:** Processes the reference portrait image once and extracts the structural keypoints and textures.

**Inputs:** `image_path` (absolute path to the reference image)

**Outputs:** a dictionary containing:
| Key | Description |
|---|---|
| `s_r` | Reference identity latent representation |
| `r_s` | Reference 3D face structure/direction coefficients |
| `s_r_feats` | Multi-scale spatial feature maps extracted from the portrait. Passed to the decoder on each step to preserve high-frequency texture details (skin, hair, background) in the output frames |

> 💡 **Integration Tip:** Call this function once per user session or when the user changes their avatar. Store the returned features in memory (CPU or GPU) to avoid repetitive preprocessing.

### 2.3 🎙️ Audio Feature Extraction — `process_audio_chunk`
```python
def process_audio_chunk(self, audio_samples)
```
**Purpose:** Transforms raw audio samples into spatial-temporal features compatible with the Wav2Vec2 encoder.

**Inputs:** `audio_samples` — a 1D NumPy array representing mono, 16kHz float32 audio

**Outputs:** a dictionary containing:
| Key | Description |
|---|---|
| `audio_features` | Projected audio features tensor `[1, T, 512]` on the GPU |
| `raw_audio_tensor` | Audio tensor representation used downstream |

> 💡 **Integration Tip:** The model expects audio in 2-second chunks (32,000 samples at 16kHz), which matches the sequence length of 50 frames (at 25 FPS). Split long TTS streams or live microphone input into 2-second packets before passing them to this function.

### 2.4 🔁 Autoregressive Talking Head Generation — `inference_step`
```python
def inference_step(self, audio_features, hidden_states, previous_frames=None, avatar_data=None, emo='S2E', seed=42)
```
**Purpose:** Generates 50 video frames corresponding to the 2-second audio chunk.

**Inputs:**
| Parameter | Description |
|---|---|
| `audio_features` | Projected Wav2Vec2 features `[1, 50, 512]` |
| `hidden_states` | Autoregressive history dict containing `prev_x_t` (`[1, 10, 512]`) and `prev_wa_t` (`[1, 10, 512]`) |
| `avatar_data` | Preprocessed portrait keypoint dictionary from `init_avatar` |
| `emo` | Emotion tag (e.g. `'neutral'`, `'happy'`, `'sad'`, or `'S2E'` to predict emotion automatically from the audio) |
| `seed` | Manual seed for Gaussian noise generator initialization |

**Outputs:** a tuple of `(frames, updated_hidden_states)`
| Value | Description |
|---|---|
| `frames` | PyTorch tensor of shape `[50, 3, 512, 512]` on the GPU |
| `updated_hidden_states` | New history dict containing the last 10 frames of predicted latents and audio features, to be passed to the subsequent step |

**FMT Flow:**
1. Compiles the emotion embedding `we` using one-hot representation or `predict_emotion`.
2. Generates initial Gaussian noise `x0` representing the motion space.
3. Executes the Flow Matching Trajectory (FMT) solver (either Euler ODE steps for Teacher or a distilled single-step prediction for Student) using the conditional context (`wa_t`, `r_s`, `we`, `prev_x_t`, `prev_wa_t`).
4. Decodes the predicted latent space into high-resolution images via the TensorRT decoder.

### 2.5 🎞️ Frame Conversion — `stream_output`
```python
def stream_output(frames)
```
**Purpose:** Translates normalized network outputs into a displayable video stream format.

**Inputs:** Normalized GPU tensor `[50, 3, 512, 512]` in the range `[-1, 1]`

**Outputs:** A `uint8` NumPy array of shape `[50, 512, 512, 3]` in RGB format

---

## 3. Production Implementation Guidelines

When integrating these functions into a real-time service like [`server.py`](./demo/server.py), apply the following patterns:

### 3.1 🧵 Dedicated GPU Producer Thread
Running PyTorch models and TensorRT engines inside async loops (like FastAPI endpoints or asyncio tasks) will block the single-threaded event loop, leading to timed-out connections.

- **Design:** Spawn a separate daemon thread ([`producer_thread_func`](./server.py#L123-L288)) that executes a synchronous `while True` loop.
- **Context Binding:** If using PyCUDA / TensorRT, explicitly push/pop the CUDA device context on the producer thread:
  ```python
  if pipeline.model.context is not None:
      pipeline.model.context.push()
  ```

### 3.2 🔄 Smooth Transitions (Autoregressive Context Tracking)
Because the model generates motion autoregressively, passing an empty history (`None`) to `inference_step` during active speech causes the model to jump-cut.

- **Continuous Speech:** Always feed the `updated_hidden_states` returned by step *N* directly as the input `hidden_states` for step *N+1*.
- **Idle State:** When the avatar is not speaking, feed simulated silence features (e.g. all zeros) through the pipeline to keep the avatar blinking and breathing naturally.
- **Transition Alignment:** When transitioning from **Idle → Speaking** or handling **Interruptions**:
  - Keep a sliding cache of the last few seconds of generated frames' states (e.g., `generated_chunks_history`).
  - When new speech starts, retrieve the exact history context corresponding to the last frame currently playing on the client and load it into the model. This guarantees smooth, jitter-free lipsync starts.

### 3.3 🔊 Audio-Video Synchronization
To prevent lag between lips and voice, package and stream audio and video as a single bundle:

- **WebSocket Streaming:** For each 2-second chunk, encode the 50 frames to JPEGs (Base64) and convert the 32,000 audio samples to a list of floats. Send them as a single JSON object.
- **Client Buffer:** On the frontend, push incoming packets to a playout queue. Let the browser's Web Audio clock schedule the audio playback, and use canvas updates synchronized to the audio context progress to draw the matching frames.

---

## ▶ Run This Demo

> **ℹ️ Note:** This mock-up demo runs with the **teacher model** along with the **TensorRT Decoder** (since the current student model needs retraining on silent audio). This version takes approximately 0.26 ms of inference time for 50 frames (1 chunk). The student model is expected to improve by around 0.05 ms, resulting in a final inference time of 0.21 ms.

```bash
python -m demo.server --is_teacher
```
