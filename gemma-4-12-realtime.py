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
    python gemma_realtime_server.py --device cpu   # if MPS causes issues
    python gemma_realtime_server.py --debug        # verbose VAD logging
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
import uuid
from typing import Any

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
parser.add_argument("--vad-threshold",     type=float, default=0.5)
parser.add_argument("--vad-silence-ms",    type=int,   default=700)
parser.add_argument("--max-audio-secs",    type=int,   default=28)
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
# Constants (from parsed args)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID          = args.model
DEVICE            = args.device
SAMPLE_RATE       = 16_000          # Gemma expects 16 kHz mono float32
MAX_AUDIO_SECS    = args.max_audio_secs
MAX_NEW_TOKENS    = 512
VAD_THRESHOLD     = args.vad_threshold
VAD_SILENCE_MS    = args.vad_silence_ms
MAX_HISTORY_TURNS = args.max_history_turns

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
    log.info("Kokoro ready.")
else:
    log.warning("Kokoro not installed — transcript-only mode (no audio output)")
    _tts = None

# ─────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ─────────────────────────────────────────────────────────────────────────────

def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Raw s16le bytes → float32 numpy array in [-1, 1]."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 array → raw s16le bytes."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resample — adequate quality for speech."""
    if orig_sr == target_sr:
        return audio
    n_out = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def run_tts(text: str) -> bytes:
    """Return s16le PCM from Kokoro, or empty bytes if unavailable."""
    if _tts is None:
        return b""
    try:
        chunks = [
            a for _, _, a in _tts(text, voice="af_heart", speed=1.0)
            if a is not None
        ]
        return float32_to_pcm16(np.concatenate(chunks)) if chunks else b""
    except Exception as exc:
        log.warning("TTS error: %s", exc)
        return b""

# ─────────────────────────────────────────────────────────────────────────────
# VAD accumulator — fixed to handle small incoming chunks correctly
# ─────────────────────────────────────────────────────────────────────────────

class VADAccumulator:
    """
    Buffers incoming PCM (which arrives in small chunks — often 160 samples /
    10 ms from the OpenAI Realtime protocol) and runs Silero VAD only once a
    full 512-sample (32 ms) window is available.  Commits an utterance after
    VAD_SILENCE_MS of trailing silence.
    """

    # Silero VAD hard requirement: exactly 512 samples at 16 kHz
    CHUNK = 512

    def __init__(self, session_id: str) -> None:
        self._id = session_id
        self._speech_frames: list[np.ndarray] = []
        self._has_speech = False
        self._silence_since: float | None = None
        # accumulates sub-512-sample chunks until we have a full window
        self._pending: np.ndarray = np.array([], dtype=np.float32)

    def push(self, pcm_bytes: bytes, src_sr: int = SAMPLE_RATE) -> np.ndarray | None:
        """
        Feed a raw PCM chunk.  Returns a committed utterance array (float32,
        16 kHz) when end-of-speech is detected, otherwise None.
        """
        audio = pcm16_to_float32(pcm_bytes)
        if src_sr != SAMPLE_RATE:
            audio = resample_linear(audio, src_sr, SAMPLE_RATE)

        # append to pending buffer
        self._pending = np.concatenate([self._pending, audio])

        committed: np.ndarray | None = None

        # drain full 512-sample chunks
        while len(self._pending) >= self.CHUNK:
            chunk = self._pending[: self.CHUNK]
            self._pending = self._pending[self.CHUNK :]

            prob = _vad_model(torch.from_numpy(chunk), SAMPLE_RATE).item()

            if args.debug and prob > 0.05:
                log.debug("[%s][VAD] prob=%.3f has_speech=%s", self._id, prob, self._has_speech)

            if prob >= VAD_THRESHOLD:
                self._has_speech = True
                self._silence_since = None
                self._speech_frames.append(chunk)

            elif self._has_speech:
                # still collecting — keep trailing silence in the buffer so
                # the model hears the natural end of the utterance
                self._speech_frames.append(chunk)
                if self._silence_since is None:
                    self._silence_since = time.monotonic()
                elapsed_ms = (time.monotonic() - self._silence_since) * 1000
                if elapsed_ms >= VAD_SILENCE_MS:
                    committed = self._commit()
                    break  # stop processing; the utterance is done

        return committed

    def flush(self) -> np.ndarray | None:
        """Force-commit whatever speech has been buffered (e.g. on client commit event)."""
        if self._speech_frames:
            return self._commit()
        return None

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

SYSTEM_PROMPT = (
    "You are Reachy Mini, a friendly desktop robot assistant. "
    "You are listening to audio from a microphone that may also pick up "
    "your own voice played back from the speaker — ignore any audio that "
    "sounds like yourself and focus only on the human speaking to you. "
    "If multiple humans are speaking, address each one naturally. "
    "Keep responses concise and conversational."
)


def infer(audio: np.ndarray, history: list[dict]) -> str:
    """Run Gemma 4 12B on an audio utterance and return the response text."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": [
            {"type": "audio"},   # audio placeholder — must come before text
            {"type": "text", "text": "Respond to what was just said."},
        ],
    })

    prompt = _processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,  # keeps latency low for conversational use
    )

    inputs = _processor(
        text=prompt,
        audio=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
        )

    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return _processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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
        # will be updated from session.update — default to 24kHz (OpenAI spec)
        self.input_sr: int = 24_000
        self._lock = asyncio.Lock()

    async def send(self, kind: str, **kw: Any) -> None:
        await self.ws.send_text(_evt(kind, **kw))

    async def on_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("[%s] invalid JSON received", self.id)
            return

        t = msg.get("type", "")

        # ── session configuration ─────────────────────────────────────────
        if t == "session.update":
            cfg = msg.get("session", {})
            log.info("[%s] session.update: %s", self.id, json.dumps(cfg))

            fmt = cfg.get("input_audio_format", "pcm16")
            if fmt in ("g711_ulaw", "g711_alaw"):
                self.input_sr = 8_000
            else:
                # pcm16 in OpenAI Realtime is 24kHz by default; Reachy may
                # send 16kHz — the resampler handles either correctly
                self.input_sr = cfg.get("input_audio_sample_rate", 24_000)

            log.info("[%s] input format=%s sr=%d", self.id, fmt, self.input_sr)

        # ── incoming audio ────────────────────────────────────────────────
        elif t == "input_audio_buffer.append":
            b64 = msg.get("audio", "")
            if not b64:
                log.warning("[%s] empty audio payload", self.id)
                return

            raw_bytes = base64.b64decode(b64)
            log.debug("[%s] audio chunk: %d bytes (%.1f ms at %d Hz)",
                      self.id, len(raw_bytes),
                      len(raw_bytes) / 2 / self.input_sr * 1000,
                      self.input_sr)

            utt = self.vad.push(raw_bytes, self.input_sr)
            if utt is not None:
                asyncio.ensure_future(self._respond(utt))

        # ── manual commit / explicit response trigger ─────────────────────
        elif t in ("input_audio_buffer.commit", "response.create"):
            log.info("[%s] manual commit triggered by %s", self.id, t)
            utt = self.vad.flush()
            if utt is not None:
                asyncio.ensure_future(self._respond(utt))
            else:
                log.info("[%s] flush: no buffered speech to commit", self.id)

        # ── text turn injection (rare) ────────────────────────────────────
        elif t == "conversation.item.create":
            item = msg.get("item", {})
            text = " ".join(
                c.get("text", "")
                for c in item.get("content", [])
                if c.get("type") == "text"
            )
            if text:
                role = item.get("role", "user")
                self.history.append({"role": role, "content": text})
                log.info("[%s] injected %s turn: %s", self.id, role, text[:60])

        else:
            log.debug("[%s] unhandled message type: %s", self.id, t)

    async def _respond(self, audio: np.ndarray) -> None:
        """Run inference and stream the response back. Serialized per session."""
        async with self._lock:
            await self.send("input_audio_buffer.speech_started")
            await self.send("input_audio_buffer.speech_stopped")
            await self.send("response.created")

            t0 = time.monotonic()
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, infer, audio, list(self.history))
            log.info("[%s] inference %.1fs → %s", self.id, time.monotonic() - t0, text[:100])

            # update history (text only — audio blobs are not replayed)
            self.history.append({"role": "assistant", "content": text})
            if len(self.history) > MAX_HISTORY_TURNS:
                self.history = self.history[-MAX_HISTORY_TURNS:]

            # send transcript
            await self.send("response.audio_transcript.delta", delta=text)
            await self.send("response.audio_transcript.done", transcript=text)

            # TTS → PCM → stream in chunks
            pcm = await loop.run_in_executor(None, run_tts, text)
            if pcm:
                chunk_size = 4096  # ~128 ms at 16kHz s16le
                for i in range(0, len(pcm), chunk_size):
                    await self.send(
                        "response.audio.delta",
                        delta=base64.b64encode(pcm[i: i + chunk_size]).decode(),
                    )

            await self.send("response.audio.done")
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

    # announce session
    await session.send(
        "session.created",
        session={
            "id": session.id,
            "model": MODEL_ID,
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gemma4-unified"},
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

# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting server  ws://%s:%d/v1/realtime", args.host, args.port)
    log.info("Health check:    http://%s:%d/health", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
