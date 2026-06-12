"""Real-time mic streaming + Silero VAD segmentation (the fastrtc-proven engine).

Two pieces that fix the rough edges of the first conversation attempt:

  * `MicStream` — a dedicated thread that continuously drains the robot mic into a
    bounded queue. Decoupling reading from processing is what prevents the
    "Audio input buffer overflowed" drops (the old single loop blocked the mic while
    running ASR/LLM/TTS). It applies a fixed gain because the ReSpeaker is low-level
    (speech peaks ~0.04).

  * `SileroSegmenter` — wraps Silero `VADIterator` (the same VAD fastrtc uses) to turn
    the stream into clean utterances with proper endpointing, instead of the old
    energy heuristic that fragmented speech.

Barge-in builds on these: the mic keeps draining during TTS, so a monitor can watch
`SileroSegmenter`/VAD for the user starting to talk and interrupt playback.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

SILERO_CHUNK = 512  # samples @ 16 kHz — Silero requires exactly this


class MicStream:
    def __init__(self, session, sample_rate: int = 16000, gain: float = 8.0,
                 max_queue: int = 400) -> None:
        self.mini = session.mini
        self.sr = sample_rate
        self.gain = gain
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.mini.media.start_recording()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self.mini.media.stop_recording()
        except Exception:  # noqa: BLE001
            pass

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_pending(self) -> None:
        """Discard queued audio (e.g. right after we finish speaking)."""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def _drain(self) -> None:
        ch = self.mini.media.get_input_channels()
        while not self._stop.is_set():
            s = self.mini.media.get_audio_sample()
            if s is None:
                self._stop.wait(0.004)
                continue
            if isinstance(s, (bytes, bytearray)):
                a = np.frombuffer(s, dtype="<i2").astype(np.float32) / 32768.0
            else:
                a = np.asarray(s, dtype=np.float32)
            if ch > 1 and a.size % ch == 0:
                a = a.reshape(-1, ch).mean(axis=1)
            a = np.clip(a * self.gain, -1.0, 1.0)
            try:
                self._q.put_nowait(a)
            except queue.Full:  # stay real-time: drop oldest
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put_nowait(a)


class SileroSegmenter:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5,
                 min_silence_ms: int = 700, speech_pad_ms: int = 200) -> None:
        from silero_vad import VADIterator, load_silero_vad

        self.sr = sample_rate
        self.model = load_silero_vad(onnx=True)
        self._vi = VADIterator(
            self.model, threshold=threshold, sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms,
        )
        self._pending = np.zeros(0, np.float32)
        self._buf: list[np.ndarray] = []
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._vi.reset_states()
        self._pending = np.zeros(0, np.float32)
        self._buf.clear()
        self._in_speech = False

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Feed mic audio; return any utterances that closed this call."""
        self._pending = np.concatenate([self._pending, frame])
        out: list[np.ndarray] = []
        while self._pending.size >= SILERO_CHUNK:
            chunk = self._pending[:SILERO_CHUNK]
            self._pending = self._pending[SILERO_CHUNK:]
            evt = self._vi(chunk)
            if self._in_speech:
                self._buf.append(chunk)
            if evt and "start" in evt:
                self._in_speech = True
                self._buf = [chunk]
            elif evt and "end" in evt and self._in_speech:
                self._in_speech = False
                if self._buf:
                    out.append(np.concatenate(self._buf))
                self._buf = []
        return out
