"""Energy-based voice-activity detection with endpointing.

Dependency-free Phase-2 VAD (Silero is a later upgrade, plan §2.2). Calibrated against
the real ReSpeaker mic: ambient RMS ≈ 6e-4, speech RMS ≈ 1e-2, peak ≈ 0.04 (low gain).

It keeps a continuously-tracked ambient noise floor and triggers speech when the frame
RMS exceeds both `noise * start_factor` and an absolute floor (so a dead-silent room
doesn't make any tiny blip "speech"). A trailing-silence timer closes each utterance.

`push(frame)` takes fixed mono float32 frames; returns a closed utterance or None.
"""

from __future__ import annotations

import numpy as np


class EnergyVad:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        trailing_silence_ms: int = 700,
        min_utterance_ms: int = 250,
        start_factor: float = 3.5,    # speech when rms > noise*start_factor AND > abs_floor
        end_factor: float = 2.0,
        abs_floor: float = 0.0025,    # absolute RMS gate (between ambient and speech)
        warmup_ms: int = 300,         # estimate the floor before allowing triggers
    ) -> None:
        self.sr = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.trailing_silence = trailing_silence_ms / 1000.0
        self.min_utterance = min_utterance_ms / 1000.0
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.abs_floor = abs_floor
        self._frame_s = frame_ms / 1000.0

        self._noise = 0.003           # ambient RMS estimate (converges quickly)
        self._warmup_left = warmup_ms / 1000.0
        self._in_speech = False
        self._buf: list[np.ndarray] = []
        self._silence_s = 0.0

    def reset(self) -> None:
        self._in_speech = False
        self._buf.clear()
        self._silence_s = 0.0
        self._warmup_left = 0.3       # brief re-settle after a reset (e.g., post-TTS)

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-12)

        if not self._in_speech:
            # Track ambient continuously: fall fast, rise slow (robust to brief sounds).
            rate = 0.10 if rms < self._noise else 0.02
            self._noise = (1 - rate) * self._noise + rate * rms
            if self._warmup_left > 0:
                self._warmup_left -= self._frame_s
                return None
            if rms > self._noise * self.start_factor and rms > self.abs_floor:
                self._in_speech = True
                self._buf = [frame]
                self._silence_s = 0.0
            return None

        # In speech: accumulate; close on trailing silence.
        self._buf.append(frame)
        if rms < max(self._noise * self.end_factor, self.abs_floor * 0.6):
            self._silence_s += self._frame_s
        else:
            self._silence_s = 0.0

        if self._silence_s >= self.trailing_silence:
            utt = np.concatenate(self._buf)
            self.reset()
            speech_s = utt.size / self.sr - self.trailing_silence
            return utt if speech_s >= self.min_utterance else None
        return None
