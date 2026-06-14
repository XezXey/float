import os
import sys
import time
import asyncio
import threading
import queue
import collections
import shutil
import tempfile
import numpy as np
import cv2
import torch
from gtts import gTTS
import librosa
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Add current workspace to path to allow importing demo modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.pipeline import TalkingHeadPipeline, stream_output

# Create FastAPI app
app = FastAPI(title="FLOAT Real-Time Talking Head Streaming Demo")

# Bounded frame buffer (thread-safe, maxsize = 250 frames = 10s of video)
class FrameBuffer:
    def __init__(self, maxsize=250):
        self.queue = collections.deque(maxlen=maxsize)
        self.lock = threading.Lock()
        
    def put(self, item):
        with self.lock:
            self.queue.append(item)
            
    def get(self):
        with self.lock:
            if len(self.queue) > 0:
                return self.queue.popleft()
            return None
            
    def clear(self):
        with self.lock:
            self.queue.clear()
            
    def __len__(self):
        with self.lock:
            return len(self.queue)

frame_buffer = FrameBuffer(maxsize=250)

# Thread-safe queue for pending TTS audio chunks
pending_audio_queue = queue.Queue()

# Global pipeline and state
pipeline = None
avatar_data = None
active_connections = set()
active_texts = {}

# Setup templates
templates = Jinja2Templates(directory="demo/templates")

def encode_frame_to_base64(frame_rgb):
    """Encodes an RGB frame to a Base64-encoded JPEG string."""
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, jpeg = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    import base64
    return base64.b64encode(jpeg.tobytes()).decode('utf-8')

def text_to_speech_samples(text: str, sampling_rate=16000) -> np.ndarray:
    """Generates audio samples for TTS using gTTS with robust local fallback."""
    try:
        print(f"[TTS] Generating speech for text: '{text}' using gTTS...")
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
            temp_path = fp.name
        tts.save(temp_path)
        
        # Load MP3 file and resample to 16000Hz
        samples, sr = librosa.load(temp_path, sr=sampling_rate)
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return samples
    except Exception as e:
        print(f"[TTS Warning] gTTS generation failed or was blocked: {e}. Using local WAV fallback.")
        fallback_path = "assets/aud-sample-vs-1.wav"
        if os.path.exists(fallback_path):
            samples, sr = librosa.load(fallback_path, sr=sampling_rate)
            # Trim or pad to fit a moderate speech duration if needed (e.g. 5 seconds)
            return samples[:int(sampling_rate * 8.0)]
        else:
            print("[TTS Error] Fallback audio file assets/aud-sample-vs-1.wav not found. Generating synthetic tone.")
            # Generate synthetic beep (2 seconds of a sine wave)
            t = np.linspace(0, 2.0, int(sampling_rate * 2.0), endpoint=False)
            samples = 0.3 * np.sin(2 * np.pi * 440 * t)
            return samples.astype(np.float32)

def producer_thread_func():
    """Background daemon thread running the autoregressive inference loop."""
    global avatar_data, pipeline, hidden_states
    
    print("[Producer] Waiting for pipeline initialization...")
    while pipeline is None:
        time.sleep(0.5)
        
    print("[Producer] Pipeline initialized. Binding PyCUDA context to thread...")
    if pipeline.model.context is not None:
        pipeline.model.context.push()
        print("[Producer] PyCUDA context successfully bound.")
        
    hidden_states = None
    inference_times = []
    
    # Pre-cache idle features (silence) to avoid calling wav2vec repeatedly when idle
    idle_samples = np.random.normal(0, 1e-4, 32000).astype(np.float32)
    idle_features = pipeline.process_audio_chunk(idle_samples)
    idle_wa = idle_features['audio_features'].to(pipeline.opt.rank)
    
    print("[Producer] Autoregressive GPU loop running.")
    
    while True:
        # 1. Throttling: check if any clients are connected
        if len(active_connections) == 0:
            time.sleep(0.01)  # Checked more frequently (10ms)
            continue
            
        # 2. Wait for avatar image to be initialized
        if avatar_data is None:
            time.sleep(0.1)
            continue
            
        # 3. Choose between active TTS audio vs. idle breathing
        is_active = not pending_audio_queue.empty()
        
        text_id = "idle"
        chunk_idx = 0
        total_chunks = 1
        
        start_inf = time.time()  # Start timing inference step (FMT + TRT Decoder)
        
        if is_active:
            try:
                # Pop next active speech chunk
                chunk = pending_audio_queue.get_nowait()
                wa_t = chunk['audio_features'].to(pipeline.opt.rank)
                audio_samples = chunk['raw_audio_samples']
                raw_a = chunk['raw_audio_tensor'].to(pipeline.opt.rank)
                emotion = chunk.get('emotion', 'neutral')
                text_id = chunk.get('text_id', 'speech')
                chunk_idx = chunk.get('chunk_idx', 0)
                total_chunks = chunk.get('total_chunks', 1)
                
                # Make sure hidden states has raw audio tensor (used for Ser emotion prediction)
                if hidden_states is None:
                    hidden_states = {}
                hidden_states['raw_audio_tensor'] = raw_a
                
                # Forward pass
                frames, hidden_states = pipeline.inference_step(
                    audio_features=wa_t,
                    hidden_states=hidden_states,
                    avatar_data=avatar_data,
                    emo=emotion
                )
            except queue.Empty:
                is_active = False
            except Exception as e:
                print(f"[Producer Error] Active step failed: {e}")
                is_active = False
                
        if not is_active:
            # Idle state: feed silence and zero audio
            try:
                if hidden_states is None:
                    hidden_states = {}
                hidden_states['raw_audio_tensor'] = torch.zeros(1, 32000, device=pipeline.opt.rank)
                
                frames, hidden_states = pipeline.inference_step(
                    audio_features=idle_wa,
                    hidden_states=hidden_states,
                    avatar_data=avatar_data,
                    emo='neutral'
                )
                audio_samples = idle_samples
                text_id = "idle"
                chunk_idx = 0
                total_chunks = 1
            except Exception as e:
                print(f"[Producer Error] Idle step failed: {e}")
                time.sleep(1.0)
                continue
                
        inf_time = time.time() - start_inf  # Stop timing inference step
        
        # Accumulate rolling average of inference times (e.g. last 50 chunks)
        inference_times.append(inf_time)
        if len(inference_times) > 50:
            inference_times.pop(0)
        mean_inf = sum(inference_times) / len(inference_times)
                
        # 4. Convert frames to uint8 RGB numpy array [50, 512, 512, 3]
        vid = stream_output(frames)
        
        # 5. Pack each frame with its corresponding audio slice (640 samples per frame)
        # and append to the bounded frame buffer
        for i in range(50):
            frame_rgb = vid[i]
            audio_slice = audio_samples[i * 640 : (i + 1) * 640]
            
            # Encode frame to Base64 in background thread to keep WebSocket thread lightweight
            base64_frame = encode_frame_to_base64(frame_rgb)
            
            frame_buffer.put((base64_frame, audio_slice, text_id, chunk_idx, total_chunks, inf_time, mean_inf))
            
        # Throttling: If the consumer is running slow and buffer starts piling up,
        # pause the producer thread temporarily to prevent GPU compute waste
        while len(frame_buffer) > 150:
            time.sleep(0.005)  # Respond faster to buffer consumption (5ms)

# Pydantic models
class TextRequest(BaseModel):
    text: str
    emotion: str = "neutral"

@app.on_event("startup")
def startup_event():
    global pipeline
    print("[Server] Loading Talking Head Pipeline on device cuda:1 (via env mappings)...")
    
    ckpt_path = os.getenv(
        "CKPT_PATH", 
        "/data2/mint/SCB-RealTimeTalkingHead/checkpoint_distil/2026-04-29_DistilFLOAT_ECFG-10.0_LR-2e-05_BS-16_TrainSize-9960_FixSeed-True_EmotionLambda-1.0_MotionLambda-1.0_AudioLambda-1.0-VelLoss-FMTWeight/student_fmt_best_earlystop.pt"
    )
    trt_decoder_path = os.getenv("TRT_DECODER_PATH", "trt_models/float_decoder_fp16.trt")
    
    pipeline = TalkingHeadPipeline(
        ckpt_path=ckpt_path,
        trt_decoder_path=trt_decoder_path,
        a_cfg_scale=2.0,
        e_cfg_scale=5.5
    )
    
    # Start background producer daemon thread
    threading.Thread(target=producer_thread_func, daemon=True).start()
    print("[Server] Startup sequence completed. Ready for connections.")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serves the main Web UI page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/init_avatar")
async def init_avatar(file: UploadFile = File(...)):
    """Receives and preprocesses the reference portrait image."""
    global avatar_data
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[Server] Uploaded portrait saved to {file_path}. Processing landmarks...")
        
        # Run init_avatar in the default thread pool to avoid blocking ASGI loop
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, pipeline.init_avatar, file_path)
        
        avatar_data = data
        try:
            os.remove(file_path)
        except OSError:
            pass
            
        print("[Server] Avatar initialized successfully.")
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        print(f"[Server Error] init_avatar failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit_text")
async def submit_text(req: TextRequest):
    """Processes incoming text input, runs TTS, and queues audio features for the producer loop."""
    if avatar_data is None:
        raise HTTPException(status_code=400, detail="Avatar not initialized. Please upload a portrait first.")
        
    try:
        loop = asyncio.get_event_loop()
        
        # 1. Assign unique text_id and map text content
        text_id = f"txt_{int(time.time() * 1000)}"
        active_texts[text_id] = req.text
        
        # 2. FEATURE 1: Interrupt currently playing frames/audio immediately
        print(f"[Server] Interrupting current playbacks for prompt: '{req.text}'")
        # Clear queues
        frame_buffer.clear()
        while not pending_audio_queue.empty():
            try:
                pending_audio_queue.get_nowait()
            except queue.Empty:
                break
                
        # Send WebSocket control message to all active clients to tell them to flush immediately
        for ws in list(active_connections):
            try:
                await ws.send_json({"type": "control", "action": "interrupt"})
            except Exception as e:
                print(f"[WebSocket Error] Failed to send interrupt: {e}")
        
        # 3. Run TTS in executor (Measure timing here)
        start_tts = time.time()
        samples = await loop.run_in_executor(None, text_to_speech_samples, req.text)
        tts_time = time.time() - start_tts
        
        # 4. Pad samples to match 2-second chunks (32000 samples)
        chunk_size = 32000
        num_samples = len(samples)
        if num_samples % chunk_size != 0:
            pad_len = chunk_size - (num_samples % chunk_size)
            samples = np.pad(samples, (0, pad_len), mode='constant', constant_values=0)
            
        duration_sec = len(samples) / 16000.0
        print(f"[Server] Synthesized speech duration: {duration_sec:.2f} seconds. Segmenting chunks...")
        
        # 5. Extract features block-by-block (running wav2vec2 encoder in executor)
        def segment_and_extract(audio_samples):
            chunks = []
            num_chunks = len(audio_samples) // chunk_size
            for idx in range(num_chunks):
                chunk_samples = audio_samples[idx * chunk_size : (idx + 1) * chunk_size]
                processed = pipeline.process_audio_chunk(chunk_samples)
                chunks.append({
                    'audio_features': processed['audio_features'].cpu(),
                    'raw_audio_samples': chunk_samples,
                    'raw_audio_tensor': processed['raw_audio_tensor'].cpu(),
                    'emotion': req.emotion,
                    'text_id': text_id,
                    'chunk_idx': idx,
                    'total_chunks': num_chunks
                })
            return chunks
            
        chunks = await loop.run_in_executor(None, segment_and_extract, samples)
        
        # 6. Queue chunks to pending input
        for chunk in chunks:
            pending_audio_queue.put(chunk)
            
        return {
            "status": "success", 
            "text_id": text_id, 
            "duration_sec": duration_sec, 
            "chunks": len(chunks),
            "tts_time_ms": int(tts_time * 1000)
        }
    except Exception as e:
        print(f"[Server Error] submit_text failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/stream")
async def stream_websocket(websocket: WebSocket):
    """WebSocket endpoint pushing video frames and audio chunks to client at playback rate."""
    await websocket.accept()
    active_connections.add(websocket)
    print(f"[WebSocket] Client connected. Total active connections: {len(active_connections)}")
    
    frame_interval = 1.0 / 25.0  # 40ms per frame for 25 FPS
    
    try:
        while True:
            start_time = time.time()
            
            # Fetch from bounded frame buffer
            item = frame_buffer.get()
            
            if item is not None:
                base64_frame, audio_slice, text_id, chunk_idx, total_chunks, inf_time, mean_inf = item
                # Send frame JPEG + audio floats to client
                await websocket.send_json({
                    "image": base64_frame,
                    "audio": audio_slice.tolist() if isinstance(audio_slice, np.ndarray) else audio_slice,
                    "text_id": text_id,
                    "text": active_texts.get(text_id, ""),
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks,
                    "inf_time_ms": int(inf_time * 1000),
                    "mean_inf_ms": int(mean_inf * 1000)
                })
            else:
                # Buffer empty (e.g. initial buffering), pause briefly (1ms)
                await asyncio.sleep(0.001)
                continue
                
            # Sleep to maintain 25 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            await asyncio.sleep(sleep_time)
            
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    finally:
        active_connections.discard(websocket)
        print(f"[WebSocket] Cleaned connection. Total active: {len(active_connections)}")

if __name__ == "__main__":
    import uvicorn
    # Set CWD to float root so paths locate correctly
    uvicorn.run(app, host="0.0.0.0", port=4747)
