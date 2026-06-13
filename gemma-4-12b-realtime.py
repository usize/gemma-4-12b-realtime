#!/usr/bin/env python3
"""
gemma_realtime_server.py

Standalone /v1/realtime WebSocket server using Gemma 4 12B Unified.
Speaks the OpenAI Realtime protocol — point your Reachy Mini conversation
app at ws://localhost:8765/v1/realtime

First run downloads ~24 GB to ~/.cache/huggingface/

Dependencies:
    pip install fastapi uvicorn torch transformers accelerate numpy

Optional (TTS):
    pip install kokoro

Run:
    python gemma_realtime_server.py
    python gemma_realtime_server.py --host 0.0.0.0 --port 8765
    python gemma_realtime_server.py --device cpu
    python gemma_realtime_server.py --debug
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import sys
import time
import uuid
from typing import Any

from PIL import Image

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration
import uvicorn

# ── Optional TTS ──────────────────────────────────────────────────────────────
try:
    from kokoro import KPipeline as KokoroPipeline
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False

# ── Silero VAD ────────────────────────────────────────────────────────────────
try:
    _vad_model, _vad_utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False, trust_repo=True
    )
    (_get_speech_timestamps, _, _read_audio, *_) = _vad_utils
    _VAD_AVAILABLE = True
except Exception as e:
    _VAD_AVAILABLE = False
    _vad_load_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Gemma 4 12B /v1/realtime server")
parser.add_argument("--model",  default="google/gemma-4-12B-it")
parser.add_argument("--device", default="mps",
                    help="mps | cpu | cuda  (mps recommended on Apple Silicon)")
parser.add_argument("--host",   default="0.0.0.0")
parser.add_argument("--port",   type=int, default=8765)
parser.add_argument("--max-history-turns", type=int, default=20)
parser.add_argument("--vad-threshold",     type=float, default=0.4)
parser.add_argument("--vad-silence-ms",    type=int,   default=700)
parser.add_argument("--max-audio-secs",    type=int,   default=28)
# How long after sending audio output to suppress VAD (avoid mic echo pickup).
# Tune this to match your robot's speaker-to-mic delay + TTS duration estimate.
parser.add_argument("--echo-suppress-ms",  type=int,   default=300,
                    help="Suppress VAD for this many ms after sending audio response")
parser.add_argument("--tts-pitch-factor",  type=float, default=1,
                    help="Multiply TTS pitch by this factor without changing speed (1.0=off, 1.35≈160 Hz from ~120 Hz base)")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug logging (shows per-chunk VAD probabilities)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gemma-realtime")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID           = args.model
DEVICE             = args.device
SAMPLE_RATE        = 16_000   # VAD + model input rate
KOKORO_SR          = 24_000   # Kokoro native generation rate
TTS_SAMPLE_RATE    = 16_000   # Playback rate (app output_sample_rate=16000; must match)
MIC_GAIN           = 8.0      # ReSpeaker peaks ~0.04; bring to normal levels
MAX_AUDIO_SECS     = args.max_audio_secs
MAX_NEW_TOKENS     = 512
VAD_THRESHOLD      = args.vad_threshold
VAD_SILENCE_MS     = args.vad_silence_ms
MAX_HISTORY_TURNS  = args.max_history_turns
ECHO_SUPPRESS_MS   = args.echo_suppress_ms
TTS_PITCH_FACTOR   = args.tts_pitch_factor

# ─────────────────────────────────────────────────────────────────────────────
# Startup checks
# ─────────────────────────────────────────────────────────────────────────────
if not _VAD_AVAILABLE:
    log.error("Silero VAD failed to load: %s", _vad_load_error)
    log.error("Try: pip install silero-vad")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load models before the server starts accepting connections
# ─────────────────────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Loading Gemma 4 12B Unified from %s", MODEL_ID)
log.info("Device: %s | First run downloads ~24 GB — please wait …", DEVICE)
log.info("=" * 60)

_t0 = time.monotonic()
_processor = AutoProcessor.from_pretrained(MODEL_ID)
_model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
).eval()
log.info("Model ready in %.1f s", time.monotonic() - _t0)

if _KOKORO_AVAILABLE:
    log.info("Loading Kokoro TTS …")
    _tts = KokoroPipeline(lang_code="a")
    _kokoro_sr = getattr(_tts, "sample_rate", None) or getattr(_tts, "sr", None) or TTS_SAMPLE_RATE
    log.info("Kokoro ready (sample_rate=%d, TTS_SAMPLE_RATE=%d).", _kokoro_sr, TTS_SAMPLE_RATE)
else:
    log.warning("=" * 60)
    log.warning("KOKORO NOT INSTALLED — NO AUDIO OUTPUT WILL BE SENT")
    log.warning("Fix: pip install kokoro")
    log.warning("=" * 60)
    _tts = None

# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def pcm16_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    n_out = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _pitch_shift(audio: np.ndarray, factor: float) -> np.ndarray:
    """Shift pitch by `factor` via single-pass resample.

    factor > 1 → higher pitch and proportionally shorter duration.
    factor < 1 → lower pitch and longer duration.
    Duration change is small for modest factors (e.g. 1.2× → 17% shorter) and
    inaudible in conversational speech.
    """
    if abs(factor - 1.0) < 0.01:
        return audio
    n_out = max(1, int(round(len(audio) / factor)))
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _trim_silence(audio: np.ndarray, threshold: float = 0.02, pad_ms: int = 80,
                   sr: int = TTS_SAMPLE_RATE) -> np.ndarray:
    """Remove leading/trailing silence; keep a short pad so the end doesn't clip."""
    mask = np.abs(audio) > threshold
    if not mask.any():
        return audio
    first = int(np.argmax(mask))
    last = int(len(mask) - np.argmax(mask[::-1]))
    pad = int(pad_ms * sr / 1000)
    return audio[max(0, first):min(len(audio), last + pad)]


def run_tts(text: str) -> bytes:
    if _tts is None:
        return b""
    try:
        chunks = [
            a for _, _, a in _tts(text, voice="af_heart", speed=1.0)
            if a is not None
        ]
        if not chunks:
            return b""
        audio = _trim_silence(np.concatenate(chunks), sr=KOKORO_SR)
        audio = resample_linear(audio, KOKORO_SR, TTS_SAMPLE_RATE)  # 24→16 kHz
        audio = _pitch_shift(audio, TTS_PITCH_FACTOR)
        log.info("TTS: %.2fs of audio at %d Hz (pitch×%.2f)",
                 len(audio) / TTS_SAMPLE_RATE, TTS_SAMPLE_RATE, TTS_PITCH_FACTOR)
        return float32_to_pcm16(audio)
    except Exception as exc:
        log.warning("TTS error: %s", exc)
        return b""

# ─────────────────────────────────────────────────────────────────────────────
# VAD accumulator
# ─────────────────────────────────────────────────────────────────────────────

class VADAccumulator:
    """
    Buffers small incoming PCM chunks (often 160 samples / 10 ms from the
    OpenAI Realtime protocol) and runs Silero VAD on full 512-sample windows.
    Commits an utterance after VAD_SILENCE_MS of trailing silence.
    """

    CHUNK = 512  # Silero hard requirement: 512 samples at 16 kHz

    def __init__(self, session_id: str) -> None:
        self._id = session_id
        self._speech_frames: list[np.ndarray] = []
        self._has_speech = False
        self._silence_since: float | None = None
        self._pending: np.ndarray = np.array([], dtype=np.float32)

    def push(self, pcm_bytes: bytes, src_sr: int = SAMPLE_RATE) -> np.ndarray | None:
        audio = pcm16_to_float32(pcm_bytes)
        if src_sr != SAMPLE_RATE:
            audio = resample_linear(audio, src_sr, SAMPLE_RATE)
        audio = np.clip(audio * MIC_GAIN, -1.0, 1.0)

        self._pending = np.concatenate([self._pending, audio])

        committed: np.ndarray | None = None

        while len(self._pending) >= self.CHUNK:
            chunk = self._pending[: self.CHUNK]
            self._pending = self._pending[self.CHUNK:]

            prob = _vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()

            if prob > 0.1:
                log.debug("[%s][VAD] prob=%.3f has_speech=%s", self._id, prob, self._has_speech)
            if prob > 0.3 and not self._has_speech:
                log.info("[%s][VAD] near-threshold speech detected: prob=%.3f", self._id, prob)

            if prob >= VAD_THRESHOLD:
                self._has_speech = True
                self._silence_since = None
                self._speech_frames.append(chunk)
            elif self._has_speech:
                self._speech_frames.append(chunk)
                if self._silence_since is None:
                    self._silence_since = time.monotonic()
                elapsed_ms = (time.monotonic() - self._silence_since) * 1000
                if elapsed_ms >= VAD_SILENCE_MS:
                    committed = self._commit()
                    break

        return committed

    def flush(self) -> np.ndarray | None:
        return self._commit() if self._speech_frames else None

    def reset(self) -> None:
        """Discard all buffered audio — called during echo suppression window."""
        self._speech_frames.clear()
        self._pending = np.array([], dtype=np.float32)
        self._has_speech = False
        self._silence_since = None

    def _commit(self) -> np.ndarray:
        utt = np.concatenate(self._speech_frames)
        cap = MAX_AUDIO_SECS * SAMPLE_RATE
        if len(utt) > cap:
            log.warning("[%s] utterance capped from %.1fs to %ds",
                        self._id, len(utt) / SAMPLE_RATE, MAX_AUDIO_SECS)
            utt = utt[-cap:]
        self._speech_frames.clear()
        self._has_speech = False
        self._silence_since = None
        log.info("[%s] VAD committed utterance: %.2f s", self._id, len(utt) / SAMPLE_RATE)
        return utt

# ─────────────────────────────────────────────────────────────────────────────
# LLM inference
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = ""


def infer(audio: np.ndarray, history: list[dict], system_prompt: str = SYSTEM_PROMPT,
          image: "Image.Image | None" = None) -> str:
    # audio: float32, 16 kHz, normalized to [-1, 1] — matches Gemma 4 spec exactly
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    user_content: list[dict] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "audio", "audio": audio})
    messages.append({"role": "user", "content": user_content})

    inputs = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(DEVICE)

    # Cast float tensors to match model dtype (bfloat16); int tensors stay as-is.
    model_dtype = next(_model.parameters()).dtype
    inputs = {
        k: v.to(dtype=model_dtype) if v.is_floating_point() else v
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[-1]
    log.info("[infer] audio %.2fs image=%s → tokens: %d  features: %s",
             len(audio) / SAMPLE_RATE,
             f"{image.width}x{image.height}" if image is not None else "none",
             input_len,
             tuple(inputs["input_features"].shape) if "input_features" in inputs else "none")

    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    return _processor.decode(out[0][input_len:], skip_special_tokens=True).strip()

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Realtime protocol helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evt(kind: str, **kw: Any) -> str:
    return json.dumps({"type": kind, "event_id": str(uuid.uuid4()), **kw})

# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.id = str(uuid.uuid4())[:8]
        self.vad = VADAccumulator(self.id)
        self.history: list[dict] = []
        self.input_sr: int = 24_000
        self.system_prompt: str = SYSTEM_PROMPT

        # inference state
        self._busy = False          # True while inference + TTS is running
        self._pending_utt: np.ndarray | None = None  # most recent queued utt
        self._suppress_until: float = 0.0  # epoch time before which VAD commits are ignored
        self.latest_image: Image.Image | None = None  # most recent frame from app camera tool
        self._camera_ready: asyncio.Event = asyncio.Event()

    async def send(self, kind: str, **kw: Any) -> None:
        await self.ws.send_text(_evt(kind, **kw))

    async def _request_camera_frame(self, timeout: float = 0.7) -> None:
        """Ask the app to capture a frame via the camera tool and wait for it."""
        self._camera_ready.clear()
        cam_id = uuid.uuid4().hex[:8]
        await self.send(
            "response.function_call_arguments.done",
            name="camera",
            call_id=f"cam-{cam_id}",
            item_id=f"item-cam-{cam_id}",
            response_id=f"resp-cam-{cam_id}",
            output_index=0,
            arguments='{"question": "What do you see right now?"}',
        )
        try:
            await asyncio.wait_for(self._camera_ready.wait(), timeout=timeout)
            log.info("[%s] camera frame ready", self.id)
        except asyncio.TimeoutError:
            log.info("[%s] camera probe timed out (%.1fs) — proceeding without frame", self.id, timeout)

    def _echo_suppressed(self) -> bool:
        return time.monotonic() < self._suppress_until

    async def on_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("[%s] invalid JSON received", self.id)
            return

        t = msg.get("type", "")

        if t == "session.update":
            cfg = msg.get("session", {})
            log.debug("[%s] session.update keys: %s", self.id, list(cfg.keys()))
            instructions = cfg.get("instructions", "").strip()
            if instructions:
                self.system_prompt = instructions
                log.info("[%s] using app system prompt (%d chars)", self.id, len(instructions))

            # Support both OpenAI flat style and HF nested style:
            #   Flat: session.input_audio_format / session.input_audio_sample_rate
            #   HF:   session.audio.input.format = {"type": "audio/pcm", "rate": null|N}
            audio_in_fmt = ((cfg.get("audio") or {}).get("input") or {}).get("format") or {}
            fmt = cfg.get("input_audio_format") or (audio_in_fmt.get("type") if isinstance(audio_in_fmt, dict) else None) or "pcm16"

            _MISSING = object()
            if fmt in ("g711_ulaw", "g711_alaw"):
                self.input_sr = 8_000
            else:
                old_sr = cfg.get("input_audio_sample_rate")
                new_rate = audio_in_fmt.get("rate", _MISSING) if isinstance(audio_in_fmt, dict) else _MISSING
                if old_sr is not None:
                    self.input_sr = int(old_sr)
                elif isinstance(new_rate, (int, float)):
                    self.input_sr = int(new_rate)
                else:
                    # Explicit null (HF native) or absent → match our VAD/model rate
                    self.input_sr = SAMPLE_RATE  # 16000
            log.info("[%s] input format=%s sr=%d", self.id, fmt, self.input_sr)

        elif t == "input_audio_buffer.append":
            b64 = msg.get("audio", "")
            if not b64:
                return

            raw_bytes = base64.b64decode(b64)
            if not hasattr(self, "_first_audio_logged"):
                self._first_audio_logged = True
                log.info("[%s] first audio chunk received (%d bytes) — VAD active", self.id, len(raw_bytes))
            log.debug("[%s] audio chunk: %d bytes", self.id, len(raw_bytes))

            # during echo suppression, drain the VAD buffer without committing
            if self._echo_suppressed():
                self.vad.reset()
                return

            utt = self.vad.push(raw_bytes, self.input_sr)
            if utt is not None:
                await self._enqueue(utt)

        elif t in ("input_audio_buffer.commit", "response.create"):
            log.info("[%s] manual commit: %s", self.id, t)
            if not self._echo_suppressed():
                utt = self.vad.flush()
                if utt is not None:
                    await self._enqueue(utt)

        elif t == "conversation.item.create":
            item = msg.get("item", {})
            content = item.get("content", [])

            # Extract image (data URI from camera tool)
            for c in content:
                if c.get("type") == "input_image":
                    url = c.get("image_url", "")
                    if url.startswith("data:"):
                        try:
                            _, b64 = url.split(",", 1)
                            self.latest_image = Image.open(io.BytesIO(base64.b64decode(b64)))
                            log.info("[%s] camera frame received (%dx%d)",
                                     self.id, self.latest_image.width, self.latest_image.height)
                        except Exception as exc:
                            log.warning("[%s] failed to decode image: %s", self.id, exc)
                        else:
                            self._camera_ready.set()

            # Extract text turns (function_call_output etc.)
            text = " ".join(
                c.get("text", "")
                for c in content
                if c.get("type") == "text"
            )
            if text:
                role = item.get("role", "user")
                self.history.append({"role": role, "content": text})
                log.info("[%s] injected %s turn: %s", self.id, role, text[:60])

        else:
            log.debug("[%s] unhandled: %s", self.id, t)

    async def _enqueue(self, utt: np.ndarray) -> None:
        """
        Drop-and-replace queuing: if inference is already running, store the
        newest utterance and let the running task pick it up when done.
        This prevents queue pile-up from echo pickup.
        """
        self._pending_utt = utt
        if not self._busy:
            self._busy = True
            asyncio.ensure_future(self._respond_loop())

    async def _respond_loop(self) -> None:
        """
        Drain pending utterances one at a time.  If a new one arrives while
        we're generating, we process it immediately after — but we never queue
        more than one deep, so echo bursts don't pile up.
        """
        while self._pending_utt is not None:
            utt = self._pending_utt
            self._pending_utt = None
            await self._respond(utt)
        self._busy = False

    async def _respond(self, audio: np.ndarray) -> None:
        await self.send("input_audio_buffer.speech_started")
        await self.send("input_audio_buffer.speech_stopped")
        await self.send("response.created")

        await self._request_camera_frame()

        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, infer, audio, list(self.history), self.system_prompt, self.latest_image
        )
        log.info("[%s] inference %.1fs → %s", self.id, time.monotonic() - t0, text[:100])

        if not text:
            await self.send("response.audio.done")
            await self.send("response.done")
            return

        self.history.append({"role": "assistant", "content": text})
        if len(self.history) > MAX_HISTORY_TURNS:
            self.history = self.history[-MAX_HISTORY_TURNS:]

        await self.send("response.output_audio_transcript.delta", delta=text)
        await self.send("response.output_audio_transcript.done", transcript=text)

        pcm = await loop.run_in_executor(None, run_tts, text)
        if pcm:
            # activate echo suppression before we start sending audio back
            tts_duration_ms = len(pcm) / 2 / TTS_SAMPLE_RATE * 1000
            suppress_ms = tts_duration_ms + ECHO_SUPPRESS_MS
            self._suppress_until = time.monotonic() + suppress_ms / 1000
            self.vad.reset()
            log.info("[%s] TTS %.0f ms of audio, echo suppression %.0f ms",
                     self.id, tts_duration_ms, suppress_ms)

            chunk_size = 4096
            for i in range(0, len(pcm), chunk_size):
                await self.send(
                    "response.output_audio.delta",
                    delta=base64.b64encode(pcm[i: i + chunk_size]).decode(),
                )

        await self.send("response.output_audio.done")
        await self.send("response.done")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Gemma 4 Realtime", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "tts": "kokoro" if _tts else "none",
        "vad": "silero",
    }


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket) -> None:
    await ws.accept()
    session = Session(ws)
    log.info("Connected  → session %s", session.id)

    await session.send(
        "session.created",
        session={
            "id": session.id,
            "object": "realtime.session",
            "model": MODEL_ID,
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "input_audio_sample_rate": 16000,
            "output_audio_format": "pcm16",
            "output_audio_sample_rate": 16000,
            "turn_detection": {"type": "server_vad"},
        },
    )

    try:
        while True:
            await session.on_message(await ws.receive_text())
    except WebSocketDisconnect:
        log.info("Disconnected → session %s", session.id)
    except Exception as exc:
        log.exception("Session %s crashed: %s", session.id, exc)


if __name__ == "__main__":
    log.info("Starting server  ws://%s:%d/v1/realtime", args.host, args.port)
    log.info("Health check:    http://%s:%d/health", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
