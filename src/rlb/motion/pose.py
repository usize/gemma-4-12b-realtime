"""Head-pose offsets and workspace limits for the motion blender.

Layers produce additive `HeadOffset` deltas (translation in metres, rotation in
degrees) around the neutral "looking forward" pose. The controller sums them, clamps
to `Limits`, and converts to the 4x4 the SDK wants via `create_head_pose`.

Limits are intentionally conservative (the SDK ships no explicit workspace constants);
they are validated in sim by `scripts/calibrate_gaze` / the motion smoke test, and the
controller additionally rate-limits per-tick change as a velocity guard (plan §4.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Antenna neutral (radians) — matches the SDK's init pose [-0.1745, 0.1745].
ANTENNA_NEUTRAL = (-0.1745, 0.1745)


@dataclass
class HeadOffset:
    x: float = 0.0      # metres
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0   # degrees
    pitch: float = 0.0
    yaw: float = 0.0

    def __add__(self, o: HeadOffset) -> HeadOffset:
        return HeadOffset(
            self.x + o.x, self.y + o.y, self.z + o.z,
            self.roll + o.roll, self.pitch + o.pitch, self.yaw + o.yaw,
        )

    def scaled(self, k: float) -> HeadOffset:
        return HeadOffset(self.x * k, self.y * k, self.z * k,
                          self.roll * k, self.pitch * k, self.yaw * k)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw], float)

    @classmethod
    def from_array(cls, a: np.ndarray) -> HeadOffset:
        return cls(*(float(v) for v in a))


@dataclass
class Limits:
    """Symmetric per-axis clamps and a max per-tick velocity (slew) guard."""

    x: float = 0.02       # ±2 cm
    y: float = 0.02
    z: float = 0.015      # ±1.5 cm
    roll: float = 15.0    # ±deg
    pitch: float = 20.0
    yaw: float = 25.0     # head pan; larger angles use body yaw
    # Max change per control tick (applied after clamp), as a fraction of full range.
    max_step_frac: float = 0.06

    def bounds(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw], float)

    def clamp(self, off: HeadOffset) -> HeadOffset:
        b = self.bounds()
        return HeadOffset.from_array(np.clip(off.as_array(), -b, b))

    def slew(self, prev: HeadOffset, target: HeadOffset) -> HeadOffset:
        """Rate-limit the move from prev->target to max_step_frac of full range/tick."""
        b = self.bounds()
        step = b * self.max_step_frac
        delta = np.clip(target.as_array() - prev.as_array(), -step, step)
        return HeadOffset.from_array(prev.as_array() + delta)


def to_matrix(off: HeadOffset) -> np.ndarray:
    """Convert a clamped offset to the SDK's 4x4 head pose (metres + degrees)."""
    from reachy_mini.utils import create_head_pose

    return create_head_pose(
        x=off.x, y=off.y, z=off.z,
        roll=off.roll, pitch=off.pitch, yaw=off.yaw,
        mm=False, degrees=True,
    )


def clamp_antennas(left: float, right: float, span: float = 0.9) -> tuple[float, float]:
    """Clamp antenna joint targets (rad) to a safe span around neutral."""
    ln, rn = ANTENNA_NEUTRAL
    return (
        float(np.clip(left, ln - span, ln + span)),
        float(np.clip(right, rn - span, rn + span)),
    )
