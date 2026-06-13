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
import datetime
import json
import logging
import threading
import time

import numpy as np

from rlb.audio.stream import MicStream, SileroSegmenter
from rlb.cognition.prompt import (
    AMBIENT_DIRECTIVE,
    IDLE_REPLY,
    SYSTEM_PROMPT,
    assemble_messages,
    clean_reply,
    is_idle_reply,
    system_with_activity,
)
from rlb.config import Config
from rlb.embodiment import MotionSkills, read_state, skills_schema
from rlb.inference import InferenceClient, image_part, text_part
from rlb.motion import MotionController
from rlb.search import find_and_center, search_tool
from rlb.vision import FrameWatcher, encode_jpeg, frame_jpeg

log = logging.getLogger("rlb.orchestrator")


def activity_tool() -> dict:
    """Tool the model uses to start/stop a self-running activity (drives the heartbeat)."""
    return {
        "type": "function",
        "function": {
            "name": "set_activity",
            "description": (
                "Start or stop an ongoing activity you run on your own — a game like Simon "
                "Says, watching for something, following a moving object, etc. Pass a short "
                "goal to START (then you'll get a turn every few seconds to look and act "
                "without being spoken to). IMPORTANT: When you give a command or say something "
                "the user should do, always follow up on your next heartbeat tick to check if "
                "they complied (you'll see comparison snapshots). Pass an empty goal to STOP "
                "when it's over."),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string",
                             "description": "what you're doing, e.g. 'play Simon Says as the "
                                            "caller'; empty to stop"},
                },
                "required": ["goal"],
            },
        },
    }


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
        # Snapshot ring: keep last 3 timestamped image snapshots for context.
        self._snapshots: list[tuple[str, bytes]] = []
        self._MAX_SNAPSHOTS = 3
        self._context_token_est: int = 0
        # Heartbeat: an activity the model sets to keep acting on its own (see _tick).
        self._activity: str | None = None
        self._last_cognition = time.monotonic()
        self._last_user_input = time.monotonic()  # tracks last user utterance for dynamic heartbeat
        # Cheap pre-filter so an empty/static scene costs nothing (ego-motion compensated).
        self._watcher = (
            FrameWatcher(cfg.motion.camera_hfov_deg, cfg.motion.camera_vfov_deg,
                         threshold=cfg.cognition.heartbeat_motion_threshold)
            if cfg.cognition.heartbeat_diff else None
        )

    # ---- one cognition turn -------------------------------------------------
    async def _respond(self, user_text: str, image_jpeg: bytes | None = None,
                       is_heartbeat: bool = False) -> str:
        """Agentic turn: the model may call movement/activity skills, then speaks.

        Movement tool calls execute immediately (routed through the controller) and we
        loop back so the model produces a spoken reply. The last iteration drops tools so
        it must answer with words. Tool round-trips are NOT persisted to history; the
        caller persists the user/assistant pair.
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

        # Inject snapshot references for heartbeat ticks so the model can compare
        # "did the user comply since the last command?" against reference images.
        if is_heartbeat and self._snapshots:
            snap_refs = []
            for ts, snap_jpeg in self._snapshots:
                snap_refs.append(text_part(f"[snapshot @ {ts}]:"))
                snap_refs.append(image_part(snap_jpeg))
            parts = snap_refs + parts

        if image_jpeg:
            parts.append(text_part("[current image]:"))
            parts.append(image_part(image_jpeg))

        system = system_with_activity(self._activity) if self._activity else SYSTEM_PROMPT
        messages = assemble_messages(self.history, parts, state_line=state, system=system)
        tools = skills_schema() + [search_tool(), activity_tool()]
        max_iters = 3
        reply = ""
        async with InferenceClient(self.cfg.inference) as ic:
            for i in range(max_iters):
                use_tools = tools if i < max_iters - 1 else None
                out = await ic.complete(messages, tools=use_tools, temperature=0.6)
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
                        result = await self._dispatch(ic, tc)
                        log.info("skill %s(%s) -> %s", tc.name, tc.arguments, result)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    continue
                reply = clean_reply(out.text)
                break
        return reply

    async def _dispatch(self, ic, tc) -> str:
        """Run a single tool call, routing the non-motion ones that need orchestrator state."""
        if tc.name == "find_and_face":
            # An async perception loop (turn/look/center), not a one-shot move.
            return await find_and_center(
                ic, self.motion, self.session.mini,
                str(tc.arguments.get("target", "")), cfg=self.cfg)
        if tc.name == "set_activity":
            goal = str(tc.arguments.get("goal", "")).strip()
            self._activity = goal or None
            self._last_cognition = time.monotonic()  # don't tick until the interval passes
            log.info("activity %s", f"set: {goal}" if goal else "cleared")
            return f"activity set: {goal}" if goal else "activity cleared"
        return self.skills.execute(tc.name, tc.arguments)

    def _remember(self, user_text: str, reply: str, image_jpeg: bytes | None = None) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": reply})

        # Estimate tokens: ~1 per 4 chars for text, ~500 per image
        self._context_token_est += len(user_text) // 4 + len(reply) // 4
        if image_jpeg:
            self._context_token_est += 500

        # Save snapshot on each user turn (ring buffer of last N).
        if image_jpeg:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._snapshots.append((ts, image_jpeg))
            if len(self._snapshots) > self._MAX_SNAPSHOTS:
                self._snapshots.pop(0)

        if len(self.history) > 20:
            self.history = self.history[-20:]

    def _compact_if_needed(self) -> None:
        """Summarize history and free snapshots when context > 64k tokens."""
        if self._context_token_est < 64_000 or len(self.history) < 6:
            return
        recent = self.history[-4:]
        old = self.history[:-4]
        summary_parts = []
        in_user = False
        for msg in old:
            if msg["role"] == "user":
                summary_parts.append(f"You: {msg['content']}")
                in_user = True
            elif msg["role"] == "assistant" and in_user:
                summary_parts.append(f"Reachy: {msg['content']}")
                in_user = False
        summary_text = " ".join(summary_parts)
        self.history = [{"role": "system",
                         "content": f"(Earlier conversation summary: {summary_text})"}] + recent
        self._context_token_est = sum(len(m["content"]) // 4 for m in self.history)
        if len(self._snapshots) > 1:
            self._snapshots = [self._snapshots[-1]]

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
                if not self._speaking.is_set() and frame is not None:
                    for utt in self.segmenter.push(frame):
                        self._handle_utterance(utt)
                # Ambient heartbeat: whenever no one is talking, glance at the camera every
                # _heartbeat_interval_s() and react only if something needs it.
                if (not self._speaking.is_set()
                        and not self.segmenter.in_speech
                        and time.monotonic() - self._last_cognition >= self._heartbeat_interval_s()):
                    self._tick()
        finally:
            stop_event.set()
            self.mic.stop()
            self.tts.close()
            self.asr.close()
            motion_thread.join(timeout=2.0)

    def _heartbeat_interval_s(self) -> float:
        """Return dynamic heartbeat interval based on time since last user input."""
        elapsed = time.monotonic() - self._last_user_input
        if elapsed < self.cfg.cognition.heartbeat_fast_duration_s:
            return self.cfg.cognition.heartbeat_fast_s
        return self.cfg.cognition.heartbeat_idle_s

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
        self._compact_if_needed()
        reply = asyncio.run(self._respond(user_text, image_jpeg=jpeg))
        self._remember(user_text, reply, image_jpeg=jpeg)
        self._last_cognition = time.monotonic()
        self._last_user_input = time.monotonic()
        print(f"🤖 {reply}")
        self._speak(reply)

    def _tick(self) -> None:
        """Ambient heartbeat: glance at the camera and react only if something warrants it.

        Runs on a dynamic interval (fast after user input, slow when idle), in the main loop
        thread so cognition stays serialized with spoken turns. A '(wait)' / empty reply
        means 'nothing to react to' and is neither spoken nor remembered, so history stays
        to real exchanges and the robot doesn't chatter.

        When an activity is active, the scene-diff filter is bypassed so the model can
        check pending tasks even on a static frame. Recent image snapshots are injected
        into the prompt as references for comparison.
        """
        frame = self.session.mini.media.get_frame()
        # BYPASS scene-diff filter when an activity is active — need to check tasks
        skip_diff = self._activity is not None
        if not skip_diff and self._watcher is not None:
            changed, diff = self._watcher.changed(
                frame, self.motion.camera_yaw_deg, self.motion.camera_pitch_deg)
            if not changed:
                log.debug("heartbeat: no scene change (diff=%.3f), skipping", diff)
                self._last_cognition = time.monotonic()
                return
        jpeg = encode_jpeg(frame, max_px=self.cfg.inference.max_image_px)
        if jpeg is None:                       # dark/blocked frame — nothing to evaluate
            self._last_cognition = time.monotonic()
            return

        self._compact_if_needed()
        reply = asyncio.run(self._respond(AMBIENT_DIRECTIVE, image_jpeg=jpeg, is_heartbeat=True))
        if not is_idle_reply(reply):
            self._remember("(noticed something)", reply)
            print(f"🤖 {reply}")
            self._speak(reply)
        self._last_cognition = time.monotonic()  # next glance is heartbeat_s after this one
