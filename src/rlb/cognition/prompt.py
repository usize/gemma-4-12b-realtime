"""Prompt assembly with token-efficient live-state injection.

Design goals (from the user): the robot's current body pose should be visible to the
model every turn, but it must NOT bloat context or scatter through the history.

How we achieve that:
  * The *format legend* for the state line lives once in the system prompt (a stable
    prefix the server can prompt-cache).
  * The *live values* are injected into the current user turn only, at a fixed spot
    (the head of that turn, inside a <robot_state> tag). The current turn is never a
    cached prefix anyway, so this costs ~one short line and nothing more.
  * History stores plain turns with NO state lines, so old state never accumulates.

Net: a single ~30-token line that always appears in the same place and updates in
place, leaving the long system+history prefix cache-friendly.
"""

from __future__ import annotations

import re
from typing import Any

# Gemma 4 emits reasoning in channel blocks like "<|channel>thought\n<channel|>...",
# sometimes "*stage directions*", and the small model often echoes the proprioception
# line. None should be spoken.
_STAGE_DIRECTION = re.compile(r"\*[^*]*\*")
_STATE_ECHO = re.compile(r"head\s+(?:xyz_cm|rpy_deg)\b.*?antennas_deg=\([^)]*\)", re.I | re.S)


def clean_reply(text: str) -> str:
    """Strip Gemma's thinking/channel markup, any <...> tags, and stage directions.

    Returns "" for degenerate output (e.g. a "thought thought ..." loop) so the caller
    can skip speaking rather than voice garbage.
    """
    if "<channel|>" in text:
        # The spoken answer follows the final channel header.
        text = text.rsplit("<channel|>", 1)[-1]
    text = re.sub(r"<[^>]*>", "", text)      # drop ALL angle-bracket tags (channel/action/etc.)
    text = _STATE_ECHO.sub("", text)         # drop echoed proprioception line
    text = _STAGE_DIRECTION.sub("", text)    # drop *stage directions*
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    thoughts = sum(1 for w in words if w.strip(".,!?").lower() == "thought")
    if words and thoughts >= max(2, len(words) * 0.5):
        return ""
    return text

# Kept deliberately short: a long/complex system prompt makes the 4-bit model degenerate
# into a "<|channel>thought" loop. Brevity keeps replies clean and fast.
SYSTEM_PROMPT = (
    "You are Reachy, a small friendly desk robot with a movable head and two antennas. "
    "You see through a camera and hear through a microphone. Each turn includes a live "
    "camera image; describe only what you actually see in it and say so when you are "
    "unsure — never invent visual details. Reply in one or two short spoken sentences — "
    "no markdown, no lists, no stage directions. You can physically move, and moves stay "
    "until you change them: turn your body to face things, tilt your head to nod or glance, "
    "and use point_at to point at an object you see. Actually move when asked, never just "
    "claim you moved. A <robot_state> line gives your current pose; use it silently and "
    "never say it aloud."
)


def state_block(state_line: str) -> str:
    return f"<robot_state>{state_line}</robot_state>"


def assemble_messages(
    history: list[dict[str, Any]],
    user_parts: list[dict[str, Any]],
    *,
    state_line: str | None = None,
    system: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Build the message list for one turn.

    `history` is prior turns as [{role, content}] with content as plain strings (no
    state). `user_parts` is the current turn's content parts (text/image/audio). The
    live `state_line` is injected at the head of the current user turn only.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    messages.extend(history)

    parts = list(user_parts)
    if state_line:
        parts = [{"type": "text", "text": state_block(state_line)}, *parts]
    messages.append({"role": "user", "content": parts})
    return messages
