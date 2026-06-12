"""TTS client: POST text to the ML-venv TTS service, play through the robot speaker.

Synthesis (Qwen3-TTS) runs in the ML venv (scripts/tts_server.py); this client fetches
the WAV and plays it. The robot speaker runs at its own rate/channels, so we resample
and interleave to match `mini.media` and push via `push_audio_sample`.

`speed` is a client-side stopgap (simple resample → pitch shifts up) until a proper
pitch-preserving time-stretch lands; Qwen3-TTS ignores its own speed arg.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave

import httpx
import numpy as np

from rlb.config import TtsConfig


class TtsClient:
    def __init__(self, cfg: TtsConfig, robot_session=None, timeout_s: float = 60.0) -> None:
        self.cfg = cfg
        self.session = robot_session
        self._client = httpx.Client(base_url=cfg.base_url, timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            return self._client.get("/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def synthesize(self, text: str, instruct: str | None = None) -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sample_rate) for `text`."""
        r = self._client.post("/speak", json={"text": text, "instruct": instruct})
        r.raise_for_status()
        return _decode_wav(r.content)

    def prepare(self, text: str, out_sr: int, instruct: str | None = None) -> np.ndarray:
        """Synthesize and return mono float32 at `out_sr` (speed applied), ready to push.

        Lets the caller stream it to the robot speaker in chunks for interruptible
        (barge-in) playback, instead of the fire-and-forget `speak()`.
        """
        audio, sr = self.synthesize(text, instruct)
        if self.cfg.speed and self.cfg.speed != 1.0:
            audio = _resample(audio, sr, int(sr / self.cfg.speed))  # time-compress at sr
        return _resample(audio, sr, out_sr).astype(np.float32)

    def speak(self, text: str, instruct: str | None = None) -> float:
        """Synthesize and play `text`. Returns the audio duration in seconds."""
        audio, sr = self.synthesize(text, instruct)
        if self.cfg.speed and self.cfg.speed != 1.0:
            audio = _resample(audio, sr, int(sr / self.cfg.speed))  # time-compress
        duration = audio.size / sr
        if self.cfg.sink == "robot" and self.session is not None:
            self._play_robot(audio, sr)
        else:
            self._play_mac(audio, sr)
        return duration

    # ---- sinks --------------------------------------------------------------
    def _play_robot(self, audio: np.ndarray, sr: int) -> None:
        media = self.session.mini.media
        out_sr = media.get_output_audio_samplerate()
        # Push MONO at the output rate; push_audio_sample expands mono->stereo itself.
        data = (_resample(audio, sr, out_sr) if out_sr != sr else audio).astype(np.float32)
        try:
            media.start_playing()
        except Exception:  # noqa: BLE001 - some backends auto-open on push
            pass
        media.push_audio_sample(data)

    def _play_mac(self, audio: np.ndarray, sr: int) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(_encode_wav(audio, sr))
            path = f.name
        subprocess.run(["afplay", path], check=False)


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch == 2:
        a = a.reshape(-1, 2).mean(axis=1)  # downmix to mono
    return a, sr


def _encode_wav(audio: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return bio.getvalue()


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample (adequate for speech); also used to time-compress for speed."""
    if dst_sr == src_sr or audio.size == 0:
        return audio
    n_dst = int(round(audio.size * dst_sr / src_sr))
    x_old = np.linspace(0, 1, audio.size, endpoint=False)
    x_new = np.linspace(0, 1, n_dst, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
