"""Typed configuration loaded from config.yaml, overridable by RLB_* env vars.

Precedence (low -> high): defaults in these models < config.yaml < environment.
Env override format: RLB_<SECTION>__<KEY>, e.g. RLB_INFERENCE__BASE_URL=http://box:8000/v1
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Repo root = two levels up from this file (src/rlb/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PREFIX = "RLB_"
ENV_NEST_SEP = "__"


class InferenceConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed-for-local"
    model: str = "gemma-4-12b-it"
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 5.0
    max_image_px: int = 768
    max_audio_seconds: float = 30.0
    stream: bool = True


class RobotConfig(BaseModel):
    media_backend: str = "default"
    backend: str = "real"  # real | mujoco
    control_hz: float = 100.0


class VadConfig(BaseModel):
    backend: str = "silero"  # silero (proven) | energy (legacy fallback)
    trailing_silence_ms: int = 700   # Silero min_silence_duration before endpointing
    min_utterance_ms: int = 250
    threshold: float = 0.5           # Silero speech probability threshold (0..1)
    speech_pad_ms: int = 200         # padding kept around detected speech
    barge_threshold: float = 0.8     # higher bar to call it a barge-in over our own voice
    # Legacy energy-VAD knobs (only used when backend == "energy").
    start_factor: float = 3.5
    end_factor: float = 2.0
    abs_floor: float = 0.0025


class AsrConfig(BaseModel):
    # MLX STT model. Used only when input_mode == "asr" (Gemma 4 native audio is
    # broken in mlx-vlm 0.6.3; see project memory / plan §9.1).
    # parakeet-tdt: self-contained MLX STT, fast, accurate, no filler hallucination
    # (whisper-large-v3-turbo MLX repo is weights-only / missing the processor).
    # Served as an HTTP service from the ML venv (mlx_audio needs hf-hub>=1.0, which
    # is incompatible with the robot's reachy_mini); rlb talks to it over HTTP.
    model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    language: str = "en"
    host: str = "127.0.0.1"
    port: int = 8123

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    source: str = "robot"  # robot | mac
    mic_gain: float = 8.0  # ReSpeaker is low-level (speech peaks ~0.04); boost for VAD/ASR
    barge_in: bool = True  # let the user interrupt while Reachy is speaking (echo-sensitive)
    # "asr": transcribe utterances to text, send Gemma text+image (current default).
    # "native": send raw audio to Gemma (flip back when mlx-vlm audio is fixed).
    input_mode: str = "asr"  # asr | native
    asr: AsrConfig = Field(default_factory=AsrConfig)
    vad: VadConfig = Field(default_factory=VadConfig)


class TtsConfig(BaseModel):
    # Qwen3-TTS (instruction-following, expressive) served from the ML venv over HTTP.
    backend: str = "qwen3-tts"
    model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
    voice: str = "aiden"  # user pick: androgynous-leaning base timbre
    instruct: str = (
        "Speak in a warm, androgynous, gently robotic voice with subtle excitement "
        "and curiosity."
    )
    speed: float = 1.44  # client-side playback speedup (model ignores its own --speed)
    sink: str = "robot"  # robot | mac
    host: str = "127.0.0.1"
    port: int = 8124

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class MotionConfig(BaseModel):
    layers: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)


class PerceptionConfig(BaseModel):
    enabled: bool = True
    face_backend: str = "mediapipe"  # mediapipe | yolo
    target_fps: int = 20
    scene_tick_s: float = 0.0


class CognitionConfig(BaseModel):
    rolling_summary_turns: int = 10
    memory_db: str = "data/memory.sqlite"
    max_tool_iters_per_turn: int = 6
    wake_mode: str = "always"  # always | wakeword | look_to_talk


class ToolsConfig(BaseModel):
    ring2_enabled: bool = False
    workspace_dir: str = "data/workspace"
    ring2_auto_approve: list[str] = Field(default_factory=list)


class BusConfig(BaseModel):
    transport: str = "zmq"  # zmq | inproc
    pub_endpoint: str = "tcp://127.0.0.1:5555"
    cmd_endpoint: str = "tcp://127.0.0.1:5556"


class MlConfig(BaseModel):
    # Separate virtualenv holding the MLX model stack (mlx-vlm, mlx_audio). It needs
    # hf-hub>=1.0, which conflicts with reachy_mini's hf-hub==0.34.4 — so the LLM and
    # ASR run there as HTTP services while the main venv stays robot-compatible.
    venv: str = ".venv-ml"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    data_dir: str = "data"


class Config(BaseModel):
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    cognition: CognitionConfig = Field(default_factory=CognitionConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    bus: BusConfig = Field(default_factory=BusConfig)
    ml: MlConfig = Field(default_factory=MlConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    def ml_python(self) -> str:
        """Path to the ML venv's python interpreter (for launching MLX services)."""
        return str(REPO_ROOT / self.ml.venv / "bin" / "python")

    def data_path(self, *parts: str) -> Path:
        """Resolve a path under the configured data dir, creating parents as needed."""
        p = (REPO_ROOT / self.logging.data_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    """Coerce an env-var string into bool/int/float when it cleanly parses."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _env_overrides() -> dict[str, Any]:
    """Collect RLB_SECTION__KEY env vars into a nested dict."""
    out: dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = key[len(ENV_PREFIX) :].lower().split(ENV_NEST_SEP)
        cursor = out
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(val)
    return out


def load_config(path: str | Path | None = None) -> Config:
    """Load config.yaml, deep-merge env overrides, and validate into a Config."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    merged = _deep_merge(raw, _env_overrides())
    return Config.model_validate(merged)
