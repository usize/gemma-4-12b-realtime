"""Visual search-and-center: a closed perception loop layered on the body-first controller.

Not SLAM (no mapping) — visual servoing. When asked to find something that isn't in view,
the robot sweeps its body across the reachable yaw range, grabbing a camera frame at each
step and asking the model "is <target> here, and where?"; once seen, it nudges its heading
toward the target until the target is horizontally centered in the frame, then tilts the
head toward its vertical position.

Needs a multimodal inference endpoint — every step is driven by the camera frame. The yaw
range is the robot's physical body limit (±90°), so this scans the front arc, not a full
360. Reuses `point_at`'s (u,v)→azimuth math and the controller's body-first `turn`.
"""

from __future__ import annotations

import asyncio
import logging
import re

from rlb.embodiment import BODY_YAW_RANGE, HEAD_RANGE, _clamp
from rlb.inference import image_part, text_part
from rlb.vision import frame_jpeg

log = logging.getLogger("rlb.search")

_NUMS = re.compile(r"[-+]?\d*\.?\d+")

# Scanning prefers the head (fast, expressive) and only spills to the body for reach.
HEAD_SCAN_YAW = 25.0   # how far the head turns before the body takes over (deg)
SCAN_PITCH = 15.0      # head tilt for the down/up sweeps (deg); within head pitch range


def _center_out(rng: tuple[float, float], step: float) -> list[float]:
    """Headings ordered from centre outward: 0, +s, -s, +2s, -2s, … within `rng`.

    Center-out so things in front are found first (and it doesn't always sweep one way).
    """
    lo, hi = rng
    vals = [0.0]
    k = step
    while k <= max(abs(lo), abs(hi)) + 1e-6:
        if k <= hi:
            vals.append(round(k, 1))
        if -k >= lo:
            vals.append(round(-k, 1))
        k += step
    return vals


def _look_dir(controller, az: float, el: float) -> None:
    """Aim the gaze at (az, el) degrees using the HEAD first, body only for reach.

    The head takes up to ±HEAD_SCAN_YAW of the horizontal angle and all of the pitch; the
    body covers whatever horizontal angle is left. So small scans/corrections are pure head
    movement and only wide ones swing the body.
    """
    head_yaw = _clamp(az, -HEAD_SCAN_YAW, HEAD_SCAN_YAW)
    body = _clamp(az - head_yaw, *BODY_YAW_RANGE)
    controller.set_body_orientation(body)
    controller.set_head_posture(yaw=head_yaw, pitch=_clamp(el, *HEAD_RANGE["pitch"]))


def _effective_az(controller) -> float:
    """Where the camera is actually pointed horizontally = body heading + head yaw."""
    return controller.heading_deg + controller.head_yaw_deg


def search_tool() -> dict:
    """OpenAI tool schema for the search routine (handled by the orchestrator, not the
    sync skill dispatcher, because it runs an async perception loop)."""
    return {
        "type": "function",
        "function": {
            "name": "find_and_face",
            "description": ("Look around for something and turn to face it. Use when asked "
                            "to find, locate, or look at something that is not currently in "
                            "view. You sweep, look, and re-orient until it is centered."),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string",
                               "description": "what to look for, e.g. 'the red cup', 'a person'"},
                },
                "required": ["target"],
            },
        },
    }


async def _locate(ic, jpeg: bytes | None, target: str) -> tuple[float, float] | None:
    """Ask the model where `target` is in the frame. Returns (u, v) in 0..1 or None.

    Kept as a tiny, isolated query (not the conversation) with a rigid output format so a
    small model answers cleanly: two numbers if visible, the word 'no' if not.
    """
    if jpeg is None:
        return None
    prompt = (
        f"Look at the image. Is there {target} visible in it? If yes, reply with ONLY two "
        f"numbers separated by a space: the horizontal then vertical position of its centre, "
        f"each from 0 to 1 (left=0, right=1, top=0, bottom=1). If it is not visible, reply "
        f"with ONLY the word: no"
    )
    msgs = [{"role": "user", "content": [text_part(prompt), image_part(jpeg)]}]
    out = await ic.complete(msgs, max_tokens=48, temperature=0.0)
    txt = (out.text or "").strip().lower()
    nums = _NUMS.findall(txt)
    if "no" in txt and len(nums) < 2:
        return None
    if len(nums) < 2:
        return None
    u, v = float(nums[0]), float(nums[1])
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return None
    return u, v


async def find_and_center(ic, controller, mini, target: str, *, cfg,
                          center_eps: float = 0.10, settle_s: float = 0.7,
                          max_corrections: int = 6) -> str:
    """Sweep until `target` is seen, then servo the heading until it's centered.

    Returns a short status string the conversation model can speak.
    """
    target = (target or "it").strip()
    hfov = cfg.motion.camera_hfov_deg
    vfov = cfg.motion.camera_vfov_deg

    # Phase 1: three full side-to-side sweeps, one per head tilt — straight ahead, then
    # down, then up — so it finds things above and below it, not just at eye level. Each
    # sweep is centre-out and head-led (body only swings for the wide angles).
    az_vals = _center_out(BODY_YAW_RANGE, hfov * 0.6)
    gaze_points = [(az, el) for el in (0.0, SCAN_PITCH, -SCAN_PITCH) for az in az_vals]

    def jpeg():
        return frame_jpeg(mini, max_px=cfg.inference.max_image_px)

    found: tuple[float, float] | None = None
    for az, el in gaze_points:
        _look_dir(controller, az, el)
        await asyncio.sleep(settle_s)
        seen = await _locate(ic, jpeg(), target)
        log.info("search %r: gaze az=%.0f el=%.0f -> %s", target, az, el, seen)
        if seen is not None:
            found = seen
            break

    if found is None:
        _look_dir(controller, 0.0, 0.0)               # give up; face forward, level
        return f"I looked around but couldn't find {target}."

    # Phase 2: servo the gaze (head first) until the target is centered in the frame.
    u, v = found
    for _ in range(max_corrections):
        if abs(u - 0.5) <= center_eps and abs(v - 0.5) <= center_eps:
            break
        az = _effective_az(controller) - (u - 0.5) * hfov   # turn toward the target
        el = controller.head_pitch_deg + (v - 0.5) * vfov   # tilt toward it (down = +)
        _look_dir(controller, _clamp(az, -115.0, 115.0), el)
        await asyncio.sleep(settle_s)
        seen = await _locate(ic, jpeg(), target)
        if seen is None:
            break                                     # lost it; stop where we are
        u, v = seen

    if abs(u - 0.5) <= center_eps and abs(v - 0.5) <= center_eps:
        return f"Found {target} and turned to face it."
    return f"I can see {target} but couldn't get fully centred on it."
