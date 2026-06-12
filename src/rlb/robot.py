"""Connection layer around the reachy_mini SDK (v1.2.x).

Centralizes how every service connects so the policy lives in one place:

  * Real hardware: let the SDK spawn the vendor daemon over USB; full media backend.
  * MuJoCo sim:    we spawn `reachy-mini-daemon --sim --headless` ourselves, because
                   the SDK's auto-spawn omits `--headless`, and on macOS MuJoCo's
                   `launch_passive` viewer requires `mjpython` — headless avoids it.
                   Sim has no real camera/audio, so we connect with `no_media`.

`connect()` returns a `RobotSession` whose `.close()` restores a safe pose and tears
down any daemon we started. The motion service builds its 100 Hz loop on `mini.set_target`.
"""

from __future__ import annotations

import contextlib
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rlb.config import Config

DAEMON_BIN = "reachy-mini-daemon"
DAEMON_API_PORT = 8000  # daemon FastAPI/uvicorn port (Zenoh is on 7447)
DAEMON_PORT_TIMEOUT_S = 25.0  # wait for uvicorn to bind its socket
# Client connect timeout. Real hardware needs to absorb sequential motor init over
# serial (~10-20 s); sim is near-instant once the daemon is up.
CONNECT_TIMEOUT_REAL_S = 40.0
CONNECT_TIMEOUT_SIM_S = 15.0


@dataclass
class MediaInfo:
    in_samplerate: int
    in_channels: int
    out_samplerate: int
    out_channels: int
    has_doa: bool


@dataclass
class RobotSession:
    """A connected robot plus any daemon process we own. Always `close()` it."""

    mini: Any
    _daemon: subprocess.Popen | None = None
    is_sim: bool = False

    def enable(self) -> None:
        """Enable motor torque (and drop gravity-compensation) so commanded poses hold.

        Without this the daemon leaves motors compliant: set_target/goto_target return
        fine but the robot doesn't physically move. Best-effort across backends.
        """
        with contextlib.suppress(Exception):
            self.mini.enable_motors()
        with contextlib.suppress(Exception):
            self.mini.disable_gravity_compensation()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.mini.goto_sleep()
        if self._daemon is not None:
            self._daemon.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._daemon.wait(timeout=5)
            if self._daemon.poll() is None:
                self._daemon.kill()

    def __enter__(self) -> RobotSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _daemon_path() -> str:
    """Resolve the daemon executable (PATH, else alongside the active interpreter)."""
    found = shutil.which(DAEMON_BIN)
    if found:
        return found
    candidate = Path(sys.executable).with_name(DAEMON_BIN)
    if candidate.exists():
        return str(candidate)
    return DAEMON_BIN  # last resort; will raise a clear FileNotFoundError if missing


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _spawn_daemon(*, sim: bool, log_level: str) -> subprocess.Popen:
    """Start the daemon (headless sim or real-USB) and wait for its API port to bind.

    Port-bind means uvicorn is up; for real hardware, motor init may still be running —
    the client's connect timeout absorbs that. We pass --no-wake-up-on-start so wake
    behavior is driven explicitly by callers, not by daemon startup.
    """
    log_path = Path("/tmp/rlb-daemon.log")
    args = [_daemon_path()]
    if sim:
        args += ["--sim", "--headless"]
    args += ["--no-wake-up-on-start", "--log-level", log_level]
    proc = subprocess.Popen(
        args, stdout=log_path.open("w"), stderr=subprocess.STDOUT, start_new_session=True
    )
    deadline = time.time() + DAEMON_PORT_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"daemon exited early; see {log_path}")
        if _port_open(DAEMON_API_PORT):
            return proc
        time.sleep(0.3)
    proc.terminate()
    raise TimeoutError(f"daemon API port not up in {DAEMON_PORT_TIMEOUT_S}s; see {log_path}")


def connect(cfg: Config, *, use_sim: bool | None = None, manage_daemon: bool = True) -> RobotSession:
    """Connect to the robot (real or sim) and return a managed session.

    If `manage_daemon` and no daemon is already listening on :8000, we spawn one and the
    session owns it (kills it on close). If a daemon is already up we reuse it and do NOT
    take ownership — this is how long-lived services attach to a supervisor-run daemon.
    """
    from reachy_mini import ReachyMini

    sim = (cfg.robot.backend == "mujoco") if use_sim is None else use_sim

    daemon: subprocess.Popen | None = None
    if manage_daemon and not _port_open(DAEMON_API_PORT):
        daemon = _spawn_daemon(sim=sim, log_level=cfg.logging.level)

    mini = ReachyMini(
        spawn_daemon=False,
        use_sim=sim,
        media_backend="no_media" if sim else cfg.robot.media_backend,
        log_level=cfg.logging.level,
        timeout=CONNECT_TIMEOUT_SIM_S if sim else CONNECT_TIMEOUT_REAL_S,
    )
    session = RobotSession(mini=mini, _daemon=daemon, is_sim=sim)
    session.enable()  # torque on, else commanded poses won't physically move
    return session


def media_info(mini) -> MediaInfo:
    """Read media capabilities (samplerates/channels, DoA availability)."""
    doa = False
    with contextlib.suppress(Exception):
        doa = mini.media.get_DoA() is not None
    return MediaInfo(
        in_samplerate=mini.media.get_input_audio_samplerate(),
        in_channels=mini.media.get_input_channels(),
        out_samplerate=mini.media.get_output_audio_samplerate(),
        out_channels=mini.media.get_output_channels(),
        has_doa=doa,
    )


def neutral_pose() -> np.ndarray:
    """Identity head pose (4x4) — the safe 'looking straight ahead' target."""
    return np.eye(4, dtype=np.float64)


def smoke(cfg: Config, *, use_sim: bool) -> dict:
    """Phase-0 connection smoke test: connect, read state, do a tiny safe motion.

    In sim, media is absent by design, so frame/audio probes are skipped.
    """
    obs: dict = {"backend": "mujoco" if use_sim else "real"}
    with connect(cfg, use_sim=use_sim) as sess:
        mini = sess.mini
        mini.wake_up()
        obs["head_pose_shape"] = tuple(np.asarray(mini.get_current_head_pose()).shape)

        if not use_sim:
            info = media_info(mini)
            obs["media"] = info.__dict__
            frame = mini.media.get_frame()
            obs["frame_shape"] = None if frame is None else tuple(frame.shape)
        else:
            obs["media"] = "skipped (no media in sim)"

        # Tiny, slow, safe nod via task-space interpolation.
        target = neutral_pose()
        target[2, 3] += 0.01  # 1 cm up
        mini.goto_target(head=target, duration=0.5)
        time.sleep(0.6)
        mini.goto_target(head=neutral_pose(), duration=0.5)
        time.sleep(0.6)
        obs["motion_ok"] = True
    return obs
