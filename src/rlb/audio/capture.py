"""Robot microphone capture → framed mono audio → VAD → utterances.

Polls `mini.media.get_audio_sample()` (16 kHz, interleaved 2-ch float32 or int16 bytes),
downmixes to mono, slices into fixed frames, and feeds them to the VAD. Yields a closed
utterance (mono float32) whenever the VAD endpoints one.

A `listening` predicate gates capture so we don't transcribe our own TTS (the speaker
and mic are co-located — simplest echo avoidance: stop listening while speaking).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from rlb.audio.vad import EnergyVad


class MicCapture:
    def __init__(self, session, vad: EnergyVad) -> None:
        self.mini = session.mini
        self.vad = vad
        self.sr = vad.sr
        self.frame = vad.frame

    def _to_mono(self, sample) -> np.ndarray:
        if isinstance(sample, (bytes, bytearray)):
            a = np.frombuffer(sample, dtype="<i2").astype(np.float32) / 32768.0
        else:
            a = np.asarray(sample, dtype=np.float32)
        ch = self.mini.media.get_input_channels()
        if ch > 1 and a.size % ch == 0:
            a = a.reshape(-1, ch).mean(axis=1)
        return a

    def utterances(
        self, stop_event, listening: Callable[[], bool]
    ) -> Iterator[np.ndarray]:
        self.mini.media.start_recording()
        pending = np.zeros(0, np.float32)
        try:
            while not stop_event.is_set():
                sample = self.mini.media.get_audio_sample()
                if sample is None:
                    stop_event.wait(0.01)
                    continue
                if not listening():
                    # Drop audio + keep VAD clean while we're speaking.
                    pending = np.zeros(0, np.float32)
                    self.vad.reset()
                    continue
                pending = np.concatenate([pending, self._to_mono(sample)])
                while pending.size >= self.frame:
                    frame, pending = pending[: self.frame], pending[self.frame :]
                    utt = self.vad.push(frame)
                    if utt is not None:
                        yield utt
        finally:
            try:
                self.mini.media.stop_recording()
            except Exception:  # noqa: BLE001
                pass
