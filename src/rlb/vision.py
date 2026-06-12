"""Vision: grab a camera frame for the model (plan §3, Tier-A "one current frame").

The robot camera is 1920x1080 BGR uint8; we downscale to the configured latency budget
(`inference.max_image_px`, default 768 px longest edge) and JPEG-encode for the
multimodal message. This is the *slow understanding* path — one frame attached per turn
so the model can actually see, instead of confabulating.

`FrameWatcher` is the cheap pre-filter for the ambient heartbeat: it decides whether the
scene actually changed since the last glance, cancelling the head's OWN motion (idle scan,
breathing, gaze) so the robot only spends an LLM call when something in the world moved.
"""

from __future__ import annotations

import cv2
import numpy as np


def encode_jpeg(frame, max_px: int = 768, quality: int = 80,
                min_brightness: float = 10.0) -> bytes | None:
    """Downscale + JPEG-encode a BGR frame. None if missing or too dark."""
    if frame is None:
        return None
    frame = np.asarray(frame)
    if float(frame.mean()) < min_brightness:
        return None
    h, w = frame.shape[:2]
    scale = max_px / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def frame_jpeg(mini, max_px: int = 768, quality: int = 80, min_brightness: float = 10.0) -> bytes | None:
    """Return a downscaled JPEG of the current camera frame, or None if unavailable.

    Returns None for a too-dark frame (e.g. the head is tucked at rest, hiding the
    face-mounted camera) so the model isn't fed a black image to comment on.
    """
    return encode_jpeg(mini.media.get_frame(), max_px, quality, min_brightness)


def _gray(frame: np.ndarray, width: int) -> np.ndarray:
    """Downscale a BGR/gray frame to `width` px wide, single-channel float in [0,1]."""
    h, w = frame.shape[:2]
    height = max(1, round(width * h / w))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.float32) / 255.0


def _aligned_diff(prev: np.ndarray, cur: np.ndarray, dx: int, dy: int) -> float:
    """Mean abs difference of the region where `prev` shifted by (dx,dy) overlaps `cur`.

    cur[y,x] is compared to prev[y-dy, x-dx], i.e. prev shifted so its content lands where
    the camera's own motion carried it. Returns 1.0 (max) if there's no overlap (huge move
    → treat as 'changed', the safe default).
    """
    h, w = cur.shape
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    if x1 <= x0 or y1 <= y0:
        return 1.0
    cur_c = cur[y0:y1, x0:x1]
    prev_c = prev[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return float(np.mean(np.abs(cur_c - prev_c)))


class FrameWatcher:
    """Detects meaningful visual change between heartbeat glances, ego-motion compensated.

    Each glance, the head has usually panned/tilted (idle scan, breathing, gaze). Given the
    camera's yaw/pitch at each capture, we translate the previous frame by the matching
    pixel shift before diffing, so the robot's own motion cancels out and only world change
    (someone entering, a hand waving, an object moving) trips the threshold. Imperfect
    alignment only inflates the diff → an extra LLM call, never a missed event.
    """

    def __init__(self, hfov_deg: float, vfov_deg: float,
                 threshold: float = 0.06, width: int = 64) -> None:
        self.hfov = hfov_deg
        self.vfov = vfov_deg
        self.threshold = threshold
        self.width = width
        self._prev: tuple[np.ndarray, float, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def changed(self, frame, yaw_deg: float, pitch_deg: float) -> tuple[bool, float]:
        """Return (changed?, diff). The first frame, or any frame after a reset, is 'changed'."""
        if frame is None:
            return False, 0.0
        g = _gray(np.asarray(frame), self.width)
        if self._prev is None:
            self._prev = (g, yaw_deg, pitch_deg)
            return True, 1.0
        prev_g, prev_yaw, prev_pitch = self._prev
        h, w = g.shape
        # Ego-motion → pixel shift. Yaw left (+) moves world content right (+x); pitch down
        # (+) moves it up (−y, since y runs downward).
        dx = round((yaw_deg - prev_yaw) / self.hfov * w)
        dy = round(-(pitch_deg - prev_pitch) / self.vfov * h)
        diff = _aligned_diff(prev_g, g, dx, dy)
        self._prev = (g, yaw_deg, pitch_deg)
        return diff > self.threshold, diff
