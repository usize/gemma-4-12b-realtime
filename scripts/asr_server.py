#!/usr/bin/env python
"""Standalone ASR HTTP service — runs in the ML venv (.venv-ml).

Why a separate service: mlx_audio (MLX STT) requires huggingface-hub>=1.0, which is
incompatible with the robot's reachy_mini (pins ==0.34.4). So ASR lives in the ML venv
and rlb (main venv) calls it over HTTP, exactly like the LLM endpoint.

Self-contained on purpose (no `rlb` import) so it only needs mlx_audio + fastapi, both
present in .venv-ml. Loads the model once and keeps it warm.

    .venv-ml/bin/python scripts/asr_server.py --model <id> --port 8123

POST /transcribe   body = WAV bytes (16 kHz mono int16)  -> {"text": "..."}
GET  /health       -> {"ok": true, "model": "..."}
"""

from __future__ import annotations

import argparse
import tempfile

import uvicorn
from fastapi import FastAPI, Request


def _normalize(result) -> str:
    # whisper yields a generator of segments (.text); parakeet returns one AlignedResult.
    if hasattr(result, "text"):
        return result.text
    return "".join(getattr(s, "text", "") or "" for s in result)


def build_app(model_id: str, language: str):
    from mlx_audio.stt.utils import load_model

    app = FastAPI(title="rlb-asr")
    model = load_model(model_id)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "model": model_id}

    @app.post("/transcribe")
    async def transcribe(request: Request) -> dict:
        data = await request.body()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(data)
            f.flush()
            try:
                result = model.generate(f.name, language=language)
            except TypeError:
                result = model.generate(f.name)  # parakeet takes no `language`
        return {"text": _normalize(result).strip()}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    uvicorn.run(build_app(args.model, args.language), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
