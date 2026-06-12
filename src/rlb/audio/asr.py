"""Speech-to-text client (plan §9.1 fallback for Gemma 4's broken native audio).

The actual MLX model runs in the ML venv as an HTTP service (scripts/asr_server.py),
because mlx_audio needs hf-hub>=1.0 which conflicts with the robot's reachy_mini. This
client just POSTs a WAV to that service and returns the transcript, so the main venv
stays free of the MLX/transformers dependency tree.

Public surface is unchanged from the earlier in-process version (`transcribe_file`,
`transcribe_pcm`) so the orchestrator/tests don't care where transcription runs.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import httpx

from rlb.config import AsrConfig


class Transcriber:
    def __init__(self, cfg: AsrConfig, timeout_s: float = 30.0) -> None:
        self.cfg = cfg
        self._client = httpx.Client(base_url=cfg.base_url, timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            r = self._client.get("/health", timeout=2.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def transcribe_file(self, path: str | Path) -> str:
        """Transcribe a WAV file path to text via the ASR service."""
        return self._post(Path(path).read_bytes())

    def transcribe_pcm(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe raw int16 mono PCM (the bus.Utterance payload) to text."""
        return self._post(wav_bytes_from_pcm(pcm, sample_rate))

    def _post(self, wav: bytes) -> str:
        r = self._client.post(
            "/transcribe", content=wav, headers={"Content-Type": "audio/wav"}
        )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()


def wav_bytes_from_pcm(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw int16 mono PCM in a WAV container (also used for native-audio mode)."""
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return bio.getvalue()
