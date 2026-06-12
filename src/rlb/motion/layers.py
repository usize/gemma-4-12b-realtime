"""The four motion layers (plan §4.2). Each produces additive contributions the
controller composes: breathing + crossfade(idle_scan, gaze) + speech_reactive.

Layers are pure-ish state machines updated at the control rate; they hold only their
own internal state (filter positions, phases), never touch the robot directly.
"""

from __future__ import annotations

import math
import random

from rlb.motion.pose import ANTENNA_NEUTRAL, HeadOffset


class BreathingLayer:
    """Constant low-amplitude sinusoidal z/pitch — the idle 'is it alive?' base."""

    def __init__(self, amplitude: float = 0.4, period_s: float = 5.0) -> None:
        self.amp = amplitude
        self.w = 2 * math.pi / period_s

    def update(self, t: float) -> tuple[HeadOffset, tuple[float, float]]:
        s = math.sin(self.w * t)
        head = HeadOffset(z=0.004 * self.amp * s, pitch=1.2 * self.amp * math.sin(self.w * t + 0.5))
        # Gentle antenna sway in counter-phase.
        sway = 0.05 * self.amp * s
        return head, (sway, -sway)


class IdleScanLayer:
    """Slow curiosity wander when there's no gaze target (smooth value-noise)."""

    def __init__(self, period_s: float = 4.0) -> None:
        # Incommensurate frequencies => non-repeating smooth drift.
        base = 2 * math.pi / period_s
        self._fy = [base * 0.37, base * 0.61, base * 0.13]
        self._fp = [base * 0.29, base * 0.53, base * 0.17]
        self._py = [random.uniform(0, 2 * math.pi) for _ in self._fy]
        self._pp = [random.uniform(0, 2 * math.pi) for _ in self._fp]

    def update(self, t: float) -> HeadOffset:
        yaw = sum(math.sin(f * t + p) for f, p in zip(self._fy, self._py)) / len(self._fy)
        pitch = sum(math.sin(f * t + p) for f, p in zip(self._fp, self._pp)) / len(self._fp)
        return HeadOffset(yaw=14.0 * yaw, pitch=7.0 * pitch, z=0.003 * yaw)


class GazeLayer:
    """Critically-damped smooth pursuit toward a target offset, with body-yaw handoff.

    `set_target(off)` requests a look direction (None = no target). Each tick the head
    offset eases toward it; large yaw beyond `head_yaw_max` spills into body yaw so the
    body turns and the head re-centers — very lifelike (plan §4.2). Micro-fixations add
    tiny jitter so a held gaze never looks frozen.
    """

    def __init__(self, tau_s: float = 0.25, head_yaw_max: float = 22.0) -> None:
        self.omega = 1.0 / max(tau_s, 1e-3)
        self.head_yaw_max = head_yaw_max
        self._pos = HeadOffset()
        self._vel = [0.0] * 6
        self._target: HeadOffset | None = None
        self._presence = 0.0  # 0..1, eases in/out as targets appear/disappear
        self._body_yaw = 0.0
        self._micro_t = 0.0
        self._micro = (0.0, 0.0)

    def set_target(self, off: HeadOffset | None) -> None:
        self._target = off

    def update(self, t: float, dt: float) -> tuple[HeadOffset, float, float]:
        has = self._target is not None
        # Ease presence toward 1 when a target exists, else toward 0.
        self._presence += (float(has) - self._presence) * min(1.0, dt / 0.4)

        goal = self._target or HeadOffset()
        # Critically-damped 2nd-order filter per axis.
        g = goal.as_array(); p = self._pos.as_array()
        for i in range(6):
            accel = self.omega**2 * (g[i] - p[i]) - 2 * self.omega * self._vel[i]
            self._vel[i] += accel * dt
            p[i] += self._vel[i] * dt
        self._pos = HeadOffset.from_array(p)

        # Body-yaw handoff for large pan.
        excess = 0.0
        if abs(self._pos.yaw) > self.head_yaw_max:
            excess = self._pos.yaw - math.copysign(self.head_yaw_max, self._pos.yaw)
        self._body_yaw += (math.radians(excess) - self._body_yaw) * min(1.0, dt / 0.6)

        # Micro-fixation jitter (re-rolled every 1-3 s) while attending.
        self._micro_t -= dt
        if self._micro_t <= 0:
            self._micro = (random.uniform(-1.2, 1.2), random.uniform(-0.8, 0.8))
            self._micro_t = random.uniform(1.0, 3.0)
        head = HeadOffset(
            yaw=min(max(self._pos.yaw, -self.head_yaw_max), self.head_yaw_max) + self._micro[0] * self._presence,
            pitch=self._pos.pitch + self._micro[1] * self._presence,
            z=self._pos.z,
        )
        return head, self._body_yaw, self._presence


class BodyOrientation:
    """Body-first horizontal aim with a leading head (the user's chosen feel).

    `set_target_deg(h)` requests an absolute heading. Each tick the *body* eases toward it
    over `tau_s` (it's the slow, grounded DoF), while the *head* gets a transient lead
    proportional to the remaining error — so the head darts toward a new heading first,
    the body rotates to catch up, and the head re-centers as the error closes. Returns
    `(body_yaw_deg, head_lead_deg)`; the controller drives the body and adds the lead to
    the head offset.
    """

    def __init__(self, tau_s: float = 0.55, lead_gain: float = 0.6,
                 lead_max_deg: float = 18.0, limit_deg: float = 90.0) -> None:
        self.tau = max(tau_s, 1e-3)
        self.lead_gain = lead_gain
        self.lead_max = lead_max_deg
        self.limit = limit_deg
        self._target = 0.0
        self._pos = 0.0

    @property
    def pos(self) -> float:
        return self._pos

    def set_target_deg(self, deg: float) -> None:
        self._target = max(-self.limit, min(self.limit, deg))

    def update(self, dt: float) -> tuple[float, float]:
        err = self._target - self._pos
        self._pos += err * min(1.0, dt / self.tau)
        lead = max(-self.lead_max, min(self.lead_max, self.lead_gain * err))
        return self._pos, lead


class SpeechReactiveLayer:
    """Head bobs + antenna motion modulated by the TTS audio envelope while speaking."""

    def __init__(self, gain: float = 0.6) -> None:
        self.gain = gain
        self._level = 0.0

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))

    def update(self, t: float, dt: float) -> tuple[HeadOffset, tuple[float, float]]:
        # Decay so playback gaps relax smoothly.
        self._level += (0.0 - self._level) * min(1.0, dt / 0.15)
        l = self._level * self.gain
        bob = HeadOffset(pitch=-3.0 * l * (0.5 + 0.5 * math.sin(2 * math.pi * 6 * t)), z=0.003 * l)
        flick = 0.35 * l * math.sin(2 * math.pi * 5 * t)
        return bob, (flick, flick)


def antenna_targets(*offsets: tuple[float, float]) -> list[float]:
    """Sum antenna offsets (rad) onto the neutral pose -> [left, right]."""
    ln, rn = ANTENNA_NEUTRAL
    l = ln + sum(o[0] for o in offsets)
    r = rn + sum(o[1] for o in offsets)
    return [l, r]
