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

import datetime
import math
from dataclasses import dataclass

import numpy as np

from rlb.motion.pose import ANTENNA_NEUTRAL

# Head workspace, in model-facing units (cm / degrees). Conservative; matches motion.Limits.
HEAD_RANGE = {
    "x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-1.5, 1.5),     # cm
    "roll": (-15.0, 15.0), "pitch": (-20.0, 20.0), "yaw": (-30.0, 30.0),  # deg
}
BODY_YAW_RANGE = (-90.0, 90.0)        # deg
ANTENNA_RANGE = (-50.0, 50.0)         # deg around neutral

# Camera/point fallbacks when no Config is available (otherwise MotionConfig wins).
DEFAULT_HFOV = 70.0
DEFAULT_VFOV = 42.0
DEFAULT_ANT_POINT = 35.0


def _fov(cfg) -> tuple[float, float, float]:
    if cfg is not None:
        m = cfg.motion
        return m.camera_hfov_deg, m.camera_vfov_deg, m.antenna_point_deg
    return DEFAULT_HFOV, DEFAULT_VFOV, DEFAULT_ANT_POINT


def _point_solution(u: float, v: float, *, hfov: float, vfov: float, ant_gain: float):
    """Map an object's image position to (az_off_deg, el_off_deg, (lean_l_deg, lean_r_deg)).

    `u`: 0=left edge .. 1=right edge; `v`: 0=top .. 1=bottom (image convention). Azimuth
    uses the head-yaw sign convention (left positive), elevation uses pitch (down
    positive). The antenna on the object's side leans forward and the other back, so the
    silhouette tilts toward the target — a single-DoF antenna can't pan, so this is the
    readable "gesturing toward it" lean rather than a literal pointer.
    """
    u = min(1.0, max(0.0, u))
    v = min(1.0, max(0.0, v))
    az_off = -(u - 0.5) * hfov          # object right (u>0.5) -> turn right (yaw negative)
    el_off = (v - 0.5) * vfov           # object low (v>0.5)  -> pitch down (positive)
    mag = ant_gain * min(1.0, abs(u - 0.5) * 2.0)
    lean = (-mag, mag) if u >= 0.5 else (mag, -mag)   # (left, right) degrees
    return az_off, el_off, lean


@dataclass
class RobotState:
    x: float; y: float; z: float          # cm
    roll: float; pitch: float; yaw: float  # deg
    antenna_left: float; antenna_right: float  # deg, relative to neutral
    body_yaw: float | None = None          # deg
    time_str: str = ""                     # human-readable current time

    def line(self) -> str:
        """One compact line for prompt injection (kept tiny + stable in shape)."""
        b = "" if self.body_yaw is None else f" body_yaw={self.body_yaw:+.0f}"
        t = f" now={self.time_str}" if self.time_str else ""
        return (
            f"head xyz_cm=({self.x:+.1f},{self.y:+.1f},{self.z:+.1f}) "
            f"rpy_deg=({self.roll:+.0f},{self.pitch:+.0f},{self.yaw:+.0f}){b} "
            f"antennas_deg=(L{self.antenna_left:+.0f},R{self.antenna_right:+.0f}){t}"
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
        time_str=datetime.datetime.now().strftime("%H:%M:%S %Z"),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class EmbodimentSkills:
    """Ring-0 actions that move individual DoFs. Backed by goto_target for smoothness.

    Each setter takes only the axes you want to change (others hold current), so the
    model can say "tilt your head left 10 degrees" without restating the whole pose.
    """

    def __init__(self, session, cfg=None) -> None:
        self.session = session
        self.mini = session.mini
        self.cfg = cfg

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

    def turn_body(self, degrees: float, duration: float = 0.6) -> str:
        """Turn relative to the current heading (left+/right-)."""
        cur = read_state(self.mini).body_yaw or 0.0
        return self.set_body_yaw(cur + degrees, duration=duration)

    def look_at(self, x: float, y: float, z: float, duration: float = 1.0) -> str:
        """Point the head at a 3D point in the robot's frame (metres)."""
        self.mini.look_at_world(x, y, z, duration=duration)
        return f"look_at -> ({x},{y},{z})"

    def point_at(self, u: float, v: float, duration: float = 0.8) -> str:
        """Turn to face an object seen at image position (u,v) and lean antennas toward it."""
        hfov, vfov, gain = _fov(self.cfg)
        az, el, (la, ra) = _point_solution(u, v, hfov=hfov, vfov=vfov, ant_gain=gain)
        cur = read_state(self.mini)
        self.set_body_yaw(az, duration=duration)
        self.set_head(pitch=_clamp(cur.pitch + el, *HEAD_RANGE["pitch"]), duration=duration)
        self.set_antennas(left=la, right=ra, duration=0.4)
        return f"point_at -> (u={u:.2f}, v={v:.2f})"

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


_SKILL_NAMES = {"set_head", "set_antennas", "set_body_yaw", "turn_body",
                "look_at", "point_at", "reset_pose"}


class MotionSkills:
    """Same skills as EmbodimentSkills, but routed through the live MotionController.

    Used inside the conversation loop, where the 100 Hz controller owns motion. Moves set
    the controller's *persistent posture* — they HOLD until changed or reset, with
    breathing/idle/gaze animating gently around them, instead of relaxing back. Horizontal
    aim is body-first: the head leads toward a new heading and the body catches up. Model
    units (cm/deg) convert to controller units (m/deg/rad).
    """

    def __init__(self, controller) -> None:
        self.c = controller

    def set_head(self, *, x=None, y=None, z=None, roll=None, pitch=None, yaw=None) -> str:
        def cv(v, k, scale=1.0):  # clamp to range, scale units; None = hold
            return None if v is None else _clamp(v, *HEAD_RANGE[k]) * scale
        self.c.set_head_posture(
            x=cv(x, "x", 0.01), y=cv(y, "y", 0.01), z=cv(z, "z", 0.01),  # cm -> m
            roll=cv(roll, "roll"), pitch=cv(pitch, "pitch"), yaw=cv(yaw, "yaw"),
        )
        return "head set"

    def set_antennas(self, *, left=None, right=None) -> str:
        self.c.set_antenna_posture(
            left_rad=None if left is None else math.radians(_clamp(left, *ANTENNA_RANGE)),
            right_rad=None if right is None else math.radians(_clamp(right, *ANTENNA_RANGE)),
        )
        return "antennas set"

    def set_body_yaw(self, degrees: float) -> str:
        deg = _clamp(degrees, *BODY_YAW_RANGE)
        self.c.set_body_orientation(deg)
        return f"facing {deg:+.0f}"

    def turn_body(self, degrees: float) -> str:
        """Turn relative to the current heading (left+/right-)."""
        deg = _clamp(self.c.heading_deg + degrees, *BODY_YAW_RANGE)
        self.c.set_body_orientation(deg)
        return f"turned to {deg:+.0f}"

    def look_at(self, x: float, y: float, z: float) -> str:
        # Robot frame: x forward, y left, z up. Horizontal -> body heading (body-first),
        # vertical -> head pitch.
        yaw = math.degrees(math.atan2(y, x))
        pitch = -math.degrees(math.atan2(z, math.hypot(x, y) + 1e-6))
        self.c.set_body_orientation(_clamp(yaw, *BODY_YAW_RANGE))
        self.c.set_head_posture(pitch=_clamp(pitch, *HEAD_RANGE["pitch"]))
        return "looking there"

    def point_at(self, u: float, v: float) -> str:
        hfov, vfov, gain = _fov(self.c.cfg)
        az, el, (la, ra) = _point_solution(u, v, hfov=hfov, vfov=vfov, ant_gain=gain)
        # Heading is relative to where the body currently points; pitch relative to the head.
        self.c.set_body_orientation(_clamp(self.c.heading_deg + az, *BODY_YAW_RANGE))
        self.c.set_head_posture(
            pitch=_clamp(self.c.head_pitch_deg + el, *HEAD_RANGE["pitch"]))
        self.c.set_antenna_posture(left_rad=math.radians(la), right_rad=math.radians(ra))
        return f"pointing (u={u:.2f}, v={v:.2f})"

    def reset_pose(self) -> str:
        self.c.reset_posture()
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
        _tool("set_head", "Tilt/nod/glance the head; holds. Only pass axes to change.", {
            "x": num("left(-)/right(+) cm"),
            "y": num("back(-)/forward(+) cm"),
            "z": num("down(-)/up(+) cm"),
            "roll": num("tilt deg"),
            "pitch": num("down(+)/up(-) deg"),
            "yaw": num("small turn left(+)/right(-) deg"),
        }),
        _tool("set_antennas", "Set antenna angles, degrees from neutral [-50,50]; holds.", {
            "left": num("left deg"), "right": num("right deg"),
        }),
        _tool("set_body_yaw", "Face an absolute body heading (deg [-90,90], 0=forward, "
              "left+/right-); holds.",
              {"degrees": num("absolute body yaw")}, required=["degrees"]),
        _tool("turn_body", "Turn relative to where you're facing now: positive=left, "
              "negative=right (deg). Use for 'turn more', 'a bit left'. Holds.",
              {"degrees": num("relative turn, left+/right-")}, required=["degrees"]),
        _tool("look_at", "Aim at a 3D point in the robot frame (metres); holds.", {
            "x": num("fwd m"), "y": num("left m"), "z": num("up m"),
        }, required=["x", "y", "z"]),
        _tool("point_at", "Point at an object in the camera image; holds.", {
            "u": num("image x, 0 left .. 1 right"),
            "v": num("image y, 0 top .. 1 bottom"),
        }, required=["u", "v"]),
        _tool("reset_pose", "Return to neutral, facing forward.", {}),
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
