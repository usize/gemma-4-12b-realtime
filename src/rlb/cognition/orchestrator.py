"""The conversation orchestrator (plan §6.1) — the hands-free talk-to-it loop.

Audio path rebuilt on the fastrtc-proven engine (Silero VAD + a threaded mic drainer,
see rlb.audio.stream) after the first energy-VAD attempt fragmented speech and dropped
mic buffers. Flow:

    robot mic (threaded drain) → Silero segmentation → ASR service → Gemma (system +
    live state) → TTS → robot speaker,

with the MotionController in a thread so Reachy stays alive (breathing, idle gaze, and
speech-reactive bob while talking).

v1 conflict-free: MotionController solely owns head motion; echo avoidance gates the
segmenter while speaking. Barge-in (interrupt TTS when the user talks) is the next layer
and is why the mic keeps draining even while we speak.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time

import numpy as np

from rlb.audio.stream import MicStream, SileroSegmenter
from rlb.cognition.prompt import assemble_messages, clean_reply
from rlb.config import Config
from rlb.embodiment import MotionSkills, read_state, skills_schema
from rlb.inference import InferenceClient, image_part, text_part
from rlb.motion import MotionController
from rlb.search import find_and_center, search_tool
from rlb.vision import frame_jpeg

log = logging.getLogger("rlb.orchestrator")


class Orchestrator:
    def __init__(self, session, cfg: Config) -> None:
        self.session = session
        self.cfg = cfg
        from rlb.audio.asr import Transcriber
        from rlb.audio.tts import TtsClient

        self.tts = TtsClient(cfg.tts, robot_session=session)
        self.asr = Transcriber(cfg.audio.asr)
        self.motion = MotionController(session, cfg)
        self.skills = MotionSkills(self.motion)  # movement routed through the controller
        self.mic = MicStream(session, sample_rate=cfg.audio.sample_rate, gain=cfg.audio.mic_gain)
        self.segmenter = SileroSegmenter(
            sample_rate=cfg.audio.sample_rate,
            threshold=cfg.audio.vad.threshold,
            min_silence_ms=cfg.audio.vad.trailing_silence_ms,
            speech_pad_ms=cfg.audio.vad.speech_pad_ms,
        )
        # Separate, stricter VAD to catch the user talking over us (echo-sensitive).
        self._barge = (
            SileroSegmenter(
                sample_rate=cfg.audio.sample_rate,
                threshold=cfg.audio.vad.barge_threshold,
                min_silence_ms=150,
                speech_pad_ms=50,
            )
            if cfg.audio.barge_in
            else None
        )
        self.history: list[dict] = []
        self._speaking = threading.Event()

    # ---- one cognition turn -------------------------------------------------
    async def _respond(self, user_text: str, image_jpeg: bytes | None = None) -> str:
        """Agentic turn: the model may call movement skills, then speaks.

        Movement tool calls execute immediately (routed through the controller) and we
        loop back so the model produces a spoken reply. The last iteration drops tools so
        it must answer with words. Tool round-trips are NOT persisted to history.
        """
        # Report the controller's COMMANDED heading/head (stable, and the only source of
        # body_yaw — the SDK read leaves it None), so the model can reason about relative
        # turns ("more to the right") instead of guessing absolute angles blind.
        st = read_state(self.session.mini)
        st.body_yaw = self.motion.heading_deg
        st.yaw = self.motion.head_yaw_deg
        st.pitch = self.motion.head_pitch_deg
        state = st.line()
        parts = [text_part(user_text)]
        if image_jpeg:
            parts.append(image_part(image_jpeg))  # so it actually sees, not confabulates
        messages = assemble_messages(self.history, parts, state_line=state)
        tools = skills_schema() + [search_tool()]
        max_iters = 3
        reply = ""
        async with InferenceClient(self.cfg.inference) as ic:
            for i in range(max_iters):
                use_tools = tools if i < max_iters - 1 else None
                out = await ic.complete(messages, tools=use_tools, max_tokens=120, temperature=0.6)
                if use_tools and out.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": out.text or "",
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                            for tc in out.tool_calls
                        ],
                    })
                    for tc in out.tool_calls:
                        if tc.name == "find_and_face":
                            # An async perception loop (turn/look/center), not a one-shot
                            # move, so it's run here rather than via the sync dispatcher.
                            result = await find_and_center(
                                ic, self.motion, self.session.mini,
                                str(tc.arguments.get("target", "")), cfg=self.cfg)
                        else:
                            result = self.skills.execute(tc.name, tc.arguments)
                        log.info("skill %s(%s) -> %s", tc.name, tc.arguments, result)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    continue
                reply = clean_reply(out.text)
                break

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})
        if len(self.history) > 20:
            self.history = self.history[-20:]
        return reply

    def _speak(self, text: str) -> bool:
        """Speak in interruptible chunks; return True if the user barged in.

        Audio is pushed ~120 ms at a time, pacing playback to real time so we can watch
        the mic for the user talking over us and `stop_playing()` immediately. Driving
        speech-reactive motion throughout.
        """
        if not text:
            return False
        media = self.session.mini.media
        out_sr = media.get_output_audio_samplerate()
        data = self.tts.prepare(text, out_sr)

        self._speaking.set()
        self.motion.set_speech_level(0.7)
        self.mic.drain_pending()
        if self._barge:
            self._barge.reset()
        interrupted = False
        grace_s = 0.6  # always get the first words out before barge-in can fire
        try:
            with contextlib.suppress(Exception):
                media.start_playing()
            start = time.perf_counter()
            chunk = max(1, int(out_sr * 0.12))
            i = 0
            while i < len(data):
                media.push_audio_sample(data[i : i + chunk].astype(np.float32))
                i += chunk
                t_end = time.perf_counter() + chunk / out_sr
                while time.perf_counter() < t_end:
                    if self._barge is None:
                        time.sleep(0.01)
                        continue
                    frame = self.mic.read(timeout=0.02)
                    if frame is None:
                        continue
                    self._barge.push(frame)
                    if self._barge.in_speech and (time.perf_counter() - start) > grace_s:
                        interrupted = True
                        break
                if interrupted:
                    with contextlib.suppress(Exception):
                        media.stop_playing()
                    break
        finally:
            self.motion.set_speech_level(0.0)
            time.sleep(0.15)
            self.mic.drain_pending()
            self.segmenter.reset()
            self._speaking.clear()
        return interrupted

    # ---- the loop -----------------------------------------------------------
    def run(self, stop_event: threading.Event, greeting: str | None = None) -> None:
        self.session.mini.wake_up()
        motion_thread = threading.Thread(target=self.motion.run, args=(stop_event,), daemon=True)
        motion_thread.start()
        self.mic.start()
        log.info("conversation loop started; say something.")
        if greeting:
            self._speak(greeting)

        try:
            while not stop_event.is_set():
                frame = self.mic.read(timeout=0.1)
                if frame is None:
                    continue
                if self._speaking.is_set():
                    continue  # echo avoidance (barge-in will replace this)
                for utt in self.segmenter.push(frame):
                    self._handle_utterance(utt)
        finally:
            stop_event.set()
            self.mic.stop()
            self.tts.close()
            self.asr.close()
            motion_thread.join(timeout=2.0)

    def _handle_utterance(self, utt: np.ndarray) -> None:
        peak = float(np.abs(utt).max())
        if peak > 1e-4:
            utt = np.clip(utt * (0.6 / peak), -1.0, 1.0)
        pcm = (utt * 32767).astype("<i2").tobytes()
        user_text = self.asr.transcribe_pcm(pcm, self.cfg.audio.sample_rate)
        if not user_text or len(user_text.strip()) < 2:
            return
        log.info("you: %s", user_text)
        print(f"\n🧑 {user_text}")
        jpeg = frame_jpeg(self.session.mini, max_px=self.cfg.inference.max_image_px)
        reply = asyncio.run(self._respond(user_text, image_jpeg=jpeg))
        print(f"🤖 {reply}")
        self._speak(reply)
