"""Embodiment: reading the robot's current pose and the skills that change it.

Two things the cognition layer needs (plan §6.2, Ring 0):
  * `RobotState` — a compact snapshot of where the body is right now, formatted into a
    single short line that gets injected into the prompt at a fixed spot (see
    cognition.prompt) so the model always knows its pose without burning tokens.
  * `EmbodimentSkills` — discrete actions that move individual degrees of freedom
    (head x/y/z + roll/pitch/yaw, body yaw, antennas), exposed both as Python methods
    and as OpenAI tool schemas + a dispatcher so the LLM can call them.

Units exposed to the model: translations in cm, all angles in degrees (more natural
for an LLM than metres/radians); we convert at the boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rlb.motion.pose import ANTENNA_NEUTRAL, HeadOffset

# Head workspace, in model-facing units (cm / degrees). Conservative; matches motion.Limits.
HEAD_RANGE = {
    "x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-1.5, 1.5),     # cm
    "roll": (-15.0, 15.0), "pitch": (-20.0, 20.0), "yaw": (-30.0, 30.0),  # deg
}
BODY_YAW_RANGE = (-90.0, 90.0)        # deg
ANTENNA_RANGE = (-50.0, 50.0)         # deg around neutral


@dataclass
class RobotState:
    x: float; y: float; z: float          # cm
    roll: float; pitch: float; yaw: float  # deg
    antenna_left: float; antenna_right: float  # deg, relative to neutral
    body_yaw: float | None = None          # deg

    def line(self) -> str:
        """One compact line for prompt injection (kept tiny + stable in shape)."""
        b = "" if self.body_yaw is None else f" body_yaw={self.body_yaw:+.0f}"
        return (
            f"head xyz_cm=({self.x:+.1f},{self.y:+.1f},{self.z:+.1f}) "
            f"rpy_deg=({self.roll:+.0f},{self.pitch:+.0f},{self.yaw:+.0f}){b} "
            f"antennas_deg=(L{self.antenna_left:+.0f},R{self.antenna_right:+.0f})"
        )


def read_state(mini) -> RobotState:
    """Snapshot the robot's current pose from the SDK."""
    from reachy_mini.utils import R

    m = np.asarray(mini.get_current_head_pose())
    tx, ty, tz = m[:3, 3]
    roll, pitch, yaw = R.from_matrix(m[:3, :3]).as_euler("xyz", degrees=True)

    al = ar = 0.0
    try:
        ant = mini.get_present_antenna_joint_positions()
        ln, rn = ANTENNA_NEUTRAL
        al = np.degrees(ant[0] - ln)
        ar = np.degrees(ant[1] - rn)
    except Exception:  # noqa: BLE001 - state read is best-effort
        pass

    return RobotState(
        x=tx * 100, y=ty * 100, z=tz * 100,
        roll=roll, pitch=pitch, yaw=yaw,
        antenna_left=al, antenna_right=ar,
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class EmbodimentSkills:
    """Ring-0 actions that move individual DoFs. Backed by goto_target for smoothness.

    Each setter takes only the axes you want to change (others hold current), so the
    model can say "tilt your head left 10 degrees" without restating the whole pose.
    """

    def __init__(self, session) -> None:
        self.session = session
        self.mini = session.mini

    # ---- skills -------------------------------------------------------------
    def set_head(self, *, x=None, y=None, z=None, roll=None, pitch=None, yaw=None,
                 duration: float = 0.6) -> str:
        """Move head DoFs (cm / degrees). Unspecified axes keep their current value."""
        cur = read_state(self.mini)
        vals = {
            "x": x if x is not None else cur.x,
            "y": y if y is not None else cur.y,
            "z": z if z is not None else cur.z,
            "roll": roll if roll is not None else cur.roll,
            "pitch": pitch if pitch is not None else cur.pitch,
            "yaw": yaw if yaw is not None else cur.yaw,
        }
        for k, (lo, hi) in HEAD_RANGE.items():
            vals[k] = _clamp(vals[k], lo, hi)
        from reachy_mini.utils import create_head_pose

        pose = create_head_pose(
            x=vals["x"] / 100, y=vals["y"] / 100, z=vals["z"] / 100,
            roll=vals["roll"], pitch=vals["pitch"], yaw=vals["yaw"], degrees=True,
        )
        self.mini.goto_target(head=pose, duration=duration)
        return f"head -> {vals}"

    def set_antennas(self, *, left=None, right=None, duration: float = 0.4) -> str:
        """Set antenna angles in degrees relative to neutral (unspecified hold)."""
        cur = read_state(self.mini)
        l = _clamp(left if left is not None else cur.antenna_left, *ANTENNA_RANGE)
        r = _clamp(right if right is not None else cur.antenna_right, *ANTENNA_RANGE)
        ln, rn = ANTENNA_NEUTRAL
        self.mini.goto_target(
            antennas=[ln + np.radians(l), rn + np.radians(r)], duration=duration
        )
        return f"antennas -> L{l:+.0f} R{r:+.0f}"

    def set_body_yaw(self, degrees: float, duration: float = 0.6) -> str:
        """Rotate the body to an absolute yaw (degrees)."""
        deg = _clamp(degrees, *BODY_YAW_RANGE)
        self.mini.goto_target(body_yaw=float(np.radians(deg)), duration=duration)
        return f"body_yaw -> {deg:+.0f}"

    def look_at(self, x: float, y: float, z: float, duration: float = 1.0) -> str:
        """Point the head at a 3D point in the robot's frame (metres)."""
        self.mini.look_at_world(x, y, z, duration=duration)
        return f"look_at -> ({x},{y},{z})"

    def reset_pose(self, duration: float = 0.6) -> str:
        """Return to the neutral 'looking straight ahead' pose."""
        from reachy_mini.utils import create_head_pose

        self.mini.goto_target(head=create_head_pose(), body_yaw=0.0, duration=duration)
        return "reset to neutral"

    # ---- LLM interface ------------------------------------------------------
    def execute(self, name: str, args: dict) -> str:
        """Dispatch a tool call by name. Returns a short result string for the model."""
        fn = getattr(self, name, None)
        if fn is None or name not in _SKILL_NAMES:
            return f"unknown skill {name!r}"
        return fn(**args)


_SKILL_NAMES = {"set_head", "set_antennas", "set_body_yaw", "look_at", "reset_pose"}


class MotionSkills:
    """Same skills as EmbodimentSkills, but routed through the live MotionController.

    Used inside the conversation loop, where the 100 Hz controller owns head motion: a
    skill sets a *commanded gesture* (top-priority layer held a few seconds) instead of
    calling goto_target directly, so they never fight. Model units (cm/deg) are converted
    to the controller's HeadOffset (m/deg).
    """

    def __init__(self, controller, hold_s: float = 4.0) -> None:
        self.c = controller
        self.hold_s = hold_s

    def _head_offset(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None) -> HeadOffset:
        g = lambda v, k: _clamp(v, *HEAD_RANGE[k]) if v is not None else 0.0  # noqa: E731
        return HeadOffset(
            x=g(x, "x") / 100, y=g(y, "y") / 100, z=g(z, "z") / 100,
            roll=g(roll, "roll"), pitch=g(pitch, "pitch"), yaw=g(yaw, "yaw"),
        )

    def set_head(self, *, x=None, y=None, z=None, roll=None, pitch=None, yaw=None) -> str:
        self.c.command(head=self._head_offset(x, y, z, roll, pitch, yaw), hold_s=self.hold_s)
        return "moving head"

    def set_antennas(self, *, left=None, right=None) -> str:
        l = _clamp(left if left is not None else 0.0, *ANTENNA_RANGE)
        r = _clamp(right if right is not None else 0.0, *ANTENNA_RANGE)
        self.c.command(antennas=(math.radians(l), math.radians(r)), hold_s=self.hold_s)
        return "moving antennas"

    def set_body_yaw(self, degrees: float) -> str:
        self.c.command(body_yaw_deg=_clamp(degrees, *BODY_YAW_RANGE), hold_s=self.hold_s)
        return "turning body"

    def look_at(self, x: float, y: float, z: float) -> str:
        # Robot frame: x forward, y left, z up -> head yaw/pitch.
        yaw = math.degrees(math.atan2(y, x))
        pitch = -math.degrees(math.atan2(z, math.hypot(x, y) + 1e-6))
        self.c.command(head=self._head_offset(yaw=yaw, pitch=pitch), hold_s=self.hold_s)
        return "looking there"

    def reset_pose(self) -> str:
        self.c.command(head=HeadOffset(), antennas=(0.0, 0.0), body_yaw_deg=0.0, hold_s=2.0)
        return "back to neutral"

    def execute(self, name: str, args: dict) -> str:
        if name not in _SKILL_NAMES:
            return f"unknown skill {name!r}"
        return getattr(self, name)(**args)


def skills_schema() -> list[dict]:
    """OpenAI tool schemas for the embodiment skills (Ring 0)."""
    def num(desc: str) -> dict:
        return {"type": "number", "description": desc}

    return [
        _tool("set_head", "Move the head. Provide only the axes to change; omitted axes hold. "
              "Translations in cm, rotations in degrees.", {
                  "x": num("left(-)/right(+) cm, [-2,2]"),
                  "y": num("back(-)/forward(+) cm, [-2,2]"),
                  "z": num("down(-)/up(+) cm, [-1.5,1.5]"),
                  "roll": num("tilt degrees, [-15,15]"),
                  "pitch": num("look down(+)/up(-) degrees, [-20,20]"),
                  "yaw": num("turn left(+)/right(-) degrees, [-30,30]"),
              }),
        _tool("set_antennas", "Set antenna angles in degrees relative to neutral, [-50,50].", {
            "left": num("left antenna degrees"),
            "right": num("right antenna degrees"),
        }),
        _tool("set_body_yaw", "Rotate the whole body to an absolute yaw in degrees, [-90,90].",
              {"degrees": num("absolute body yaw")}, required=["degrees"]),
        _tool("look_at", "Aim the head at a 3D point in the robot's frame (metres).", {
            "x": num("metres forward"), "y": num("metres left"), "z": num("metres up"),
        }, required=["x", "y", "z"]),
        _tool("reset_pose", "Return head and body to neutral.", {}),
    ]


def _tool(name: str, desc: str, props: dict, required: list | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required or []},
        },
    }
