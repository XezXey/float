# Integrating FLOAT Talking Head into a Production Pipeline


### 🚧 ⚠️ WARNING: Mockup Only

**Known Bug-like Behavior**
- Random lagging on playback (live output box) — plays back chunk-wise, similar to a streaming platform
- Long waiting queue after submitting a text prompt — unclear whether the queue is stuck or how to clear it (needs better handling/feedback)

**This is a mockup** left in this state purely **for demonstration purposes** of the integration. 

---

This document explains the architectural pattern for integrating the **FLOAT Real-Time Talking Head** into a production system. It details the core functions of the talking head pipeline and describes how to orchestrate them in a low-latency environment.


---

## 1. Architectural Pattern: Core Pipeline vs. Runner

To build a high-performance system, the implementation is divided into two separate concerns:

1. **The Pipeline Definition ([pipeline.py](./pipeline.py))**:
   Defines the atomic data processing and model forward-pass tasks. It has no knowledge of HTTP protocols, WebSockets, or multi-client routing. It simply acts as a stateless/state-driven engine for transforming audio and image inputs into video frames.
2. **The Runner/Server ([server.py](./server.py))**:
   Manages network I/O, threads, client connections, Text-to-Speech (TTS) generation, concurrency buffers, and schedules execution of the pipeline.

---

## 2. Core Functions of [TalkingHeadPipeline](./pipeline.py#L173-L225)

The class [TalkingHeadPipeline](file:///home/mint/Dev/SCBx-TalkingHead/float/demo/pipeline.py#L173-L225) handles model initialization, weights mapping (supporting student models or teacher models, optionally with TensorRT decoders), portrait preprocessing, and frame generation.

### 2.1 Initialization: `__init__`
```python
def __init__(self, ckpt_path, trt_decoder_path, a_cfg_scale=2.0, e_cfg_scale=5.5, is_teacher=None)
```
- **Purpose**: Loads checkpoints, initializes the PyTorch model, optionally binds the TensorRT engine for decoding, and prepares the data processor.
- **Key Actions**:
  - Automatically identifies whether it is loading a distilled student or a teacher model based on the checkpoint name.
  - Instantiates either [StudentFLOATWithTRTDecoder](file:///home/mint/Dev/SCBx-TalkingHead/float/demo/pipeline.py#L36-L128) or [TeacherFLOATWithTRTDecoder](file:///home/mint/Dev/SCBx-TalkingHead/float/demo/pipeline.py#L130-L156).
  - Moves weights onto the GPU (`cuda:0` / rank target device) and sets the model to evaluation mode.

### 2.2 Preprocessing Portrait: `init_avatar`
```python
def init_avatar(self, image_path)
```
- **Purpose**: Processes the reference portrait image once and extracts the structural keypoints and textures.
- **Inputs**: `image_path` (absolute path to the reference image).
- **Outputs**: A dictionary containing:
  - `s_r`: Reference identity latent representation.
  - `r_s`: Reference 3D face structure/direction coefficients.
  - `s_r_feats`: Multi-scale spatial feature maps extracted from the portrait. These are passed to the decoder on each step to preserve high-frequency texture details (skin, hair, background) in the output frames.
- **Integration Tip**: Call this function once per user session or when the user changes their avatar. Store the returned features in memory (CPU or GPU) to avoid repetitive preprocessing.

### 2.3 Audio Feature Extraction: `process_audio_chunk`
```python
def process_audio_chunk(self, audio_samples)
```
- **Purpose**: Transforms raw audio samples into spatial-temporal features compatible with the Wav2Vec2 encoder.
- **Inputs**: `audio_samples` (a 1D NumPy array representing mono, 16kHz float32 audio).
- **Outputs**: A dictionary containing:
  - `audio_features`: Projected audio features tensor `[1, T, 512]` on the GPU.
  - `raw_audio_tensor`: Audio tensor representation used downstream.
- **Integration Tip**: The model expects audio in 2-second chunks (32,000 samples at 16kHz), which matches the sequence length of 50 frames (at 25 FPS). Split long TTS streams or live microphone input into 2-second packets before passing them to this function.

### 2.4 Autoregressive Synthesis: `inference_step`
```python
def inference_step(self, audio_features, hidden_states, previous_frames=None, avatar_data=None, emo='S2E', seed=42)
```
- **Purpose**: Generates 50 video frames corresponding to the 2-second audio chunk.
- **Inputs**:
  - `audio_features`: The projected Wav2Vec2 features `[1, 50, 512]`.
  - `hidden_states`: Autoregressive history dictionary containing `prev_x_t` (`[1, 10, 512]`) and `prev_wa_t` (`[1, 10, 512]`).
  - `avatar_data`: The preprocessed portrait keypoint dictionary from `init_avatar`.
  - `emo`: Emotion tag (e.g. `'neutral'`, `'happy'`, `'sad'`, or `'S2E'` to predict emotion automatically from the audio).
  - `seed`: Manual seed for Gaussian noise generator initialization.
- **Outputs**: A tuple of `(frames, updated_hidden_states)`:
  - `frames`: A PyTorch tensor of shape `[50, 3, 512, 512]` on the GPU.
  - `updated_hidden_states`: The new history dictionary containing the last 10 frames of predicted latents and audio features, to be passed to the subsequent step.
- **FMT Flow**:
  1. Compiles the emotion embedding `we` using one-hot representation or `predict_emotion`.
  2. Generates initial Gaussian noise `x0` representing the motion space.
  3. Executes the Flow Matching Trajectory (FMT) solver (either Euler ODE steps for Teacher or a distilled single-step prediction for Student) using the conditional context (`wa_t`, `r_s`, `we`, `prev_x_t`, `prev_wa_t`).
  4. Decodes the predicted latent space into high-resolution images via the TensorRT decoder.

### 2.5 Frame Conversion: `stream_output`
```python
def stream_output(frames)
```
- **Purpose**: Translates normalized network outputs into a displayable video stream format.
- **Inputs**: Normalized GPU tensor `[50, 3, 512, 512]` in the range `[-1, 1]`.
- **Outputs**: A `uint8` NumPy array of shape `[50, 512, 512, 3]` in RGB format.

---

## 3. Production Implementation Guidelines

When integrating these functions into a real-time service like [server.py](file:///home/mint/Dev/SCBx-TalkingHead/float/demo/server.py), apply the following patterns:

### 3.1 Dedicated GPU Producer Thread
Running PyTorch models and TensorRT engines inside async loops (like FastAPI endpoints or asyncio tasks) will block the single-threaded event loop, leading to timed-out connections.
- **Design**: Spawn a separate daemon thread ([producer_thread_func](file:///home/mint/Dev/SCBx-TalkingHead/float/demo/server.py#L123-L288)) that executes a synchronous `while True` loop.
- **Context Binding**: If using PyCUDA / TensorRT, explicitly push/pop the CUDA device context on the producer thread:
  ```python
  if pipeline.model.context is not None:
      pipeline.model.context.push()
  ```
- **Throttling**: To avoid wasting GPU resources, verify that a client is actually connected before processing the next step. If no client is listening, put the thread to sleep.

### 3.2 Smooth Transitions (Autoregressive Context Tracking)
Because the model generates motion autoregressively, passing an empty history (`None`) to `inference_step` during active speech causes the model to jump-cut.
- **Continuous Speech**: Always feed the `updated_hidden_states` returned by step $N$ directly as the input `hidden_states` for step $N+1$.
- **Idle State**: When the avatar is not speaking, feed simulated silence features (e.g. all zeros) through the pipeline to keep the avatar blinking and breathing naturally.
- **Transition Alignment**: When transitioning from **Idle $\to$ Speaking** or handling **Interruptions**:
  - Keep a sliding cache of the last few seconds of generated frames' states (e.g., `generated_chunks_history`).
  - When new speech starts, retrieve the exact history context corresponding to the last frame currently playing on the client and load it into the model. This guarantees smooth, jitter-free lipsync starts.

### 3.3 Audio-Video Synchronization
To prevent lag between lips and voice, package and stream audio and video as a single bundle:
- **WebSocket Streaming**: For each 2-second chunk, encode the 50 frames to JPEGs (Base64) and convert the 32,000 audio samples to a list of floats. Send them as a single JSON object.
- **Client Buffer**: On the frontend, push incoming packets to a playout queue. Let the browser's Web Audio clock schedule the audio playback, and use canvas updates synchronized to the audio context progress to draw the matching frames.


---

### Run this demo

> **ℹ️ Note:** We provided this mock up demo with the **teacher model** (since the current student model needs retraining on silent audio), we provide inference code using the **teacher model (FLOAT's FMT)** along with the **TensorRT Decoder**.

```
python -m demo.server --is_teacher
```

---