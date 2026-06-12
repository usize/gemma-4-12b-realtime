"""The 100 Hz layered head controller (plan §4.2).

Composition each tick:
    head = breathing + crossfade(idle_scan, gaze, presence) + speech_reactive
then clamp to workspace, rate-limit (slew), and `set_target`. Gaze target and speech
level are set externally (by the perception/tts bus wiring); the loop itself never
blocks on anything slow.

Safety (plan §4.3): per-tick slew caps angular velocity; a soft watchdog logs loop
overruns; on stop we ease back to neutral via `goto_target`.
"""

from __future__ import annotations

import logging
import threading
import time

from rlb.config import Config
from rlb.motion.layers import (
    BreathingLayer,
    GazeLayer,
    IdleScanLayer,
    SpeechReactiveLayer,
    antenna_targets,
)
from rlb.motion.pose import HeadOffset, Limits, clamp_antennas, to_matrix

log = logging.getLogger("rlb.motion")


def _lerp(a: HeadOffset, b: HeadOffset, k: float) -> HeadOffset:
    return a.scaled(1.0 - k) + b.scaled(k)


class MotionController:
    def __init__(self, session, cfg: Config) -> None:
        self.session = session
        self.mini = session.mini
        self.hz = cfg.robot.control_hz
        layers = cfg.motion.layers or {}
        limits_cfg = cfg.motion.limits or {}

        br = layers.get("breathing", {})
        sp = layers.get("speech_reactive", {})
        gz = layers.get("gaze", {})
        idl = layers.get("idle_scan", {})

        self.breathing = BreathingLayer(amplitude=br.get("amplitude", 0.4))
        self.idle = IdleScanLayer(period_s=idl.get("period_s", 4.0))
        self.gaze = GazeLayer(tau_s=gz.get("smooth_tau_s", 0.25))
        self.speech = SpeechReactiveLayer(gain=sp.get("gain", 0.6))

        self.limits = Limits()
        # Map max angular velocity (deg/s) to a per-tick yaw step fraction.
        max_vel = limits_cfg.get("max_head_vel_dps", 180.0)
        self.limits.max_step_frac = min(0.2, (max_vel / self.hz) / self.limits.yaw)
        self._stall_s = limits_cfg.get("watchdog_stall_ms", 200) / 1000.0

        self._prev = HeadOffset()
        self._enabled_idle = idl.get("enabled", True)
        self._enabled_breath = br.get("enabled", True)

    # ---- external inputs (called from bus wiring) ----
    def set_gaze_target(self, off: HeadOffset | None) -> None:
        self.gaze.set_target(off)

    def set_speech_level(self, level: float) -> None:
        self.speech.set_level(level)

    # ---- the loop ----
    def run(self, stop_event: threading.Event) -> None:
        period = 1.0 / self.hz
        t0 = time.perf_counter()
        last = t0
        log.info("motion loop @ %.0f Hz (slew=%.3f/tick)", self.hz, self.limits.max_step_frac)
        try:
            while not stop_event.is_set():
                now = time.perf_counter()
                t = now - t0
                dt = now - last
                last = now
                if dt > self._stall_s:
                    log.warning("control loop overrun: %.0f ms", dt * 1000)

                head = self._compose(t, dt)
                antL, antR = clamp_antennas(*self._antennas())
                try:
                    self.mini.set_target(
                        head=to_matrix(head),
                        antennas=[antL, antR],
                        body_yaw=self._body_yaw,
                    )
                except Exception:  # noqa: BLE001 - never let one bad tick kill the loop
                    log.exception("set_target failed")

                # Maintain the rate, accounting for compute time.
                slack = period - (time.perf_counter() - now)
                if slack > 0:
                    stop_event.wait(slack)
        finally:
            self._safe_neutral()

    def _compose(self, t: float, dt: float) -> HeadOffset:
        breathing_h, self._br_ant = (
            self.breathing.update(t) if self._enabled_breath else (HeadOffset(), (0.0, 0.0))
        )
        idle_h = self.idle.update(t) if self._enabled_idle else HeadOffset()
        gaze_h, self._body_yaw, presence = self.gaze.update(t, dt)
        speech_h, self._sp_ant = self.speech.update(t, dt)

        head = breathing_h + _lerp(idle_h, gaze_h, presence) + speech_h
        head = self.limits.clamp(head)
        head = self.limits.slew(self._prev, head)
        self._prev = head
        return head

    def _antennas(self) -> tuple[float, float]:
        targets = antenna_targets(self._br_ant, self._sp_ant)
        return targets[0], targets[1]

    def _safe_neutral(self) -> None:
        try:
            self.mini.goto_target(head=to_matrix(HeadOffset()), body_yaw=0.0, duration=0.5)
            log.info("motion loop stopped; eased to neutral")
        except Exception:  # noqa: BLE001
            log.exception("failed to ease to neutral on stop")
