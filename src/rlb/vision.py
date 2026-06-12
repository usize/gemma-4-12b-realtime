"""Vision: grab a camera frame for the model (plan §3, Tier-A "one current frame").

The robot camera is 1920x1080 BGR uint8; we downscale to the configured latency budget
(`inference.max_image_px`, default 768 px longest edge) and JPEG-encode for the
multimodal message. This is the *slow understanding* path — one frame attached per turn
so the model can actually see, instead of confabulating.
"""

from __future__ import annotations

import cv2
import numpy as np


def frame_jpeg(mini, max_px: int = 768, quality: int = 80, min_brightness: float = 10.0) -> bytes | None:
    """Return a downscaled JPEG of the current camera frame, or None if unavailable.

    Returns None for a too-dark frame (e.g. the head is tucked at rest, hiding the
    face-mounted camera) so the model isn't fed a black image to comment on.
    """
    frame = mini.media.get_frame()
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
