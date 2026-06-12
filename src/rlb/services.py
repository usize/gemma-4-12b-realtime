"""Process supervisor for long-lived services (plan §7).

Each service is a named background process with a PID file and a log file under
`data/run/` and `data/logs/`. Start/stop/status are idempotent and never require
`pkill`. This is intentionally simple (PID files + signals, no daemon framework) so
it stays restartable and transparent.

Managed services today:
  * inference — the mlx-vlm OpenAI server (Gemma 4)
  * daemon    — the reachy_mini robot daemon (real or sim)

Phase 1+ services (motion/perception/...) register here as they land.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from rlb.config import Config


@dataclass
class Service:
    name: str
    argv: list[str]
    log_file: Path
    pid_file: Path
    ready_port: int | None = None  # if set, "started" means this port accepts connections
    ready_timeout_s: float = 30.0
    env: dict[str, str] = field(default_factory=dict)

    # ---- state helpers ----
    def pid(self) -> int | None:
        if not self.pid_file.exists():
            return None
        try:
            pid = int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            return None
        return pid if _alive(pid) else None

    def is_running(self) -> bool:
        return self.pid() is not None

    # ---- lifecycle ----
    def start(self) -> str:
        if self.is_running():
            return f"already running (pid {self.pid()})"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **self.env}
        # Ensure sibling console scripts (e.g. reachy-mini-daemon) are found.
        env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
        with self.log_file.open("a") as log:
            log.write(f"\n=== start {self.name} @ {time.strftime('%F %T')} ===\n")
            log.flush()
            proc = subprocess.Popen(
                self.argv, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
        self.pid_file.write_text(str(proc.pid))
        if self.ready_port is not None:
            if not self._wait_ready(proc):
                return f"started (pid {proc.pid}) but port {self.ready_port} not ready in {self.ready_timeout_s}s — check `rlb logs {self.name}`"
        return f"started (pid {proc.pid})"

    def _wait_ready(self, proc: subprocess.Popen) -> bool:
        deadline = time.time() + self.ready_timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                return False
            if _port_open(self.ready_port):
                return True
            time.sleep(0.3)
        return False

    def stop(self, timeout_s: float = 8.0) -> str:
        pid = self.pid()
        if pid is None:
            self.pid_file.unlink(missing_ok=True)
            return "not running"
        # Signal the whole process group (start_new_session => pgid == pid).
        _signal_group(pid, signal.SIGTERM)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not _alive(pid):
                break
            time.sleep(0.2)
        if _alive(pid):
            _signal_group(pid, signal.SIGKILL)
            time.sleep(0.3)
        self.pid_file.unlink(missing_ok=True)
        return f"stopped (pid {pid})"

    def status(self) -> dict:
        pid = self.pid()
        port_ok = _port_open(self.ready_port) if self.ready_port else None
        return {
            "name": self.name,
            "running": pid is not None,
            "pid": pid,
            "port": self.ready_port,
            "port_open": port_ok,
            "log": str(self.log_file),
        }


# --------------------------------------------------------------------------- #
# Process utilities
# --------------------------------------------------------------------------- #
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def _port_open(port: int | None, host: str = "127.0.0.1") -> bool:
    if port is None:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def build_services(cfg: Config) -> dict[str, Service]:
    """Construct the service registry from config."""
    run = cfg.repo_root / cfg.logging.data_dir / "run"
    logs = cfg.repo_root / cfg.logging.data_dir / "logs"

    parsed = urlparse(cfg.inference.base_url)
    infer_port = parsed.port or 8080
    ml_python = cfg.ml_python()  # MLX services run in the separate ML venv

    inference = Service(
        name="inference",
        argv=[
            ml_python, "-m", "mlx_vlm.server",
            "--model", cfg.inference.model,
            "--host", "127.0.0.1", "--port", str(infer_port),
            "--log-level", "INFO",
        ],
        log_file=logs / "inference.log",
        pid_file=run / "inference.pid",
        ready_port=infer_port,
        ready_timeout_s=180.0,  # first run downloads weights
    )

    asr = Service(
        name="asr",
        argv=[
            ml_python, str(cfg.repo_root / "scripts" / "asr_server.py"),
            "--model", cfg.audio.asr.model,
            "--language", cfg.audio.asr.language,
            "--host", cfg.audio.asr.host, "--port", str(cfg.audio.asr.port),
        ],
        log_file=logs / "asr.log",
        pid_file=run / "asr.pid",
        ready_port=cfg.audio.asr.port,
        ready_timeout_s=120.0,
    )

    tts = Service(
        name="tts",
        argv=[
            ml_python, str(cfg.repo_root / "scripts" / "tts_server.py"),
            "--model", cfg.tts.model, "--voice", cfg.tts.voice,
            "--instruct", cfg.tts.instruct,
            "--host", cfg.tts.host, "--port", str(cfg.tts.port),
        ],
        log_file=logs / "tts.log",
        pid_file=run / "tts.pid",
        ready_port=cfg.tts.port,
        ready_timeout_s=120.0,
    )

    sim = cfg.robot.backend == "mujoco"
    daemon_bin = shutil.which("reachy-mini-daemon") or "reachy-mini-daemon"
    daemon_argv = [daemon_bin]
    if sim:
        daemon_argv += ["--sim", "--headless"]
    daemon_argv += ["--no-wake-up-on-start", "--log-level", cfg.logging.level]
    daemon = Service(
        name="daemon",
        argv=daemon_argv,
        log_file=logs / "daemon.log",
        pid_file=run / "daemon.pid",
        ready_port=8000,
        ready_timeout_s=60.0,  # real hardware: motor init over serial
    )

    return {"inference": inference, "asr": asr, "tts": tts, "daemon": daemon}
