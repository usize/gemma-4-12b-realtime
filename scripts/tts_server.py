#!/usr/bin/env python
"""Standalone TTS HTTP service — runs in the ML venv (.venv-ml).

Same rationale as the ASR service: mlx_audio needs hf-hub>=1.0 (incompatible with the
robot's reachy_mini), so TTS runs here and rlb calls it over HTTP. Loads Qwen3-TTS once
and keeps it warm.

    .venv-ml/bin/python scripts/tts_server.py --model <id> --voice aiden --port 8124

POST /speak  body = {"text": "...", "instruct"?: "...", "voice"?: "..."}
             -> WAV bytes (mono float->int16 at the model's native sample rate)
GET  /health -> {"ok": true, ...}
"""

from __future__ import annotations

import argparse
import io
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel


class SpeakRequest(BaseModel):
    text: str
    instruct: str | None = None
    voice: str | None = None
    speed: float | None = None


def _to_wav(audio: np.ndarray, sr: int) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2").tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16)
    return bio.getvalue()


def build_app(model_id: str, default_voice: str, default_instruct: str):
    from mlx_audio.tts.utils import load_model

    app = FastAPI(title="rlb-tts")
    model = load_model(model_id)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": model_id, "voice": default_voice}

    @app.post("/speak")
    async def speak(req: SpeakRequest) -> Response:  # async => runs in the MLX-init thread
        results = model.generate(
            text=req.text,
            voice=req.voice or default_voice,
            instruct=req.instruct or default_instruct,
            speed=req.speed or 1.0,
            lang_code="en",
            verbose=False,
        )
        # The model's audio sample rate (NOT audio_samples["samples-per-sec"], which is
        # a generation-throughput metric).
        sr = int(getattr(model, "sample_rate", 24000))
        chunks: list[np.ndarray] = []
        for r in results:
            chunks.append(np.asarray(r.audio, dtype=np.float32))
            sr = int(getattr(r, "sample_rate", sr))
        audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        wav = _to_wav(audio, sr)
        return Response(content=wav, media_type="audio/wav", headers={"X-Sample-Rate": str(sr)})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16")
    ap.add_argument("--voice", default="aiden")
    ap.add_argument("--instruct", default="Speak in a warm, androgynous, gently robotic "
                    "voice with subtle excitement and curiosity.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8124)
    args = ap.parse_args()
    uvicorn.run(build_app(args.model, args.voice, args.instruct), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
