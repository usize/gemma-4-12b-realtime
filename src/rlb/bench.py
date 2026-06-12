"""Latency benchmark for the inference endpoint (plan §0, §8 — calibrates everything).

Measures, per modality combo (text / +image / +audio):
  * TTFT  — time to first streamed token (the number that gates perceived response latency)
  * total — wall time to completion
  * tok/s — rough decode throughput (whitespace token estimate)

It degrades gracefully: if the endpoint is unreachable it says so and exits 0, so it
doubles as a connectivity check during bring-up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rlb.config import Config
from rlb.inference import InferenceClient, audio_part, image_part, text_part

# A 1x1 white JPEG (smallest valid-ish payload) and a tiny silent WAV header, so the
# bench can exercise the multimodal code paths without external asset files.
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    "08060607060508070707090909"
) + b"\x00" * 8 + bytes.fromhex("ffd9")
_TINY_WAV = (
    b"RIFF" + (36).to_bytes(4, "little") + b"WAVE"
    b"fmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
    + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
    + (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little") + b"data" + (0).to_bytes(4, "little")
)


@dataclass
class Trial:
    name: str
    ttft_s: float
    total_s: float
    tokens: int

    @property
    def tok_per_s(self) -> float:
        decode = max(self.total_s - self.ttft_s, 1e-6)
        return self.tokens / decode


async def _time_one(ic: InferenceClient, name: str, content: list[dict]) -> Trial:
    messages = [{"role": "user", "content": content}]
    start = time.perf_counter()
    ttft: float | None = None
    text = ""
    async for chunk in ic.stream(messages, max_tokens=128, temperature=0.2):
        if chunk.text and ttft is None:
            ttft = time.perf_counter() - start
        text += chunk.text
    total = time.perf_counter() - start
    return Trial(name, ttft or total, total, len(text.split()))


async def run_bench(cfg: Config, *, do_image: bool, do_audio: bool, console=None) -> list[Trial]:
    def out(msg: str) -> None:
        (console.print if console is not None else print)(msg)

    async with InferenceClient(cfg.inference) as ic:
        if not await ic.health():
            out(
                f"[yellow]inference endpoint unreachable at {cfg.inference.base_url}[/]"
                if console else f"inference endpoint unreachable at {cfg.inference.base_url}"
            )
            return []

        combos: list[tuple[str, list[dict]]] = [
            ("text", [text_part("Say hello in one short sentence.")]),
        ]
        if do_image:
            combos.append(
                ("text+image", [text_part("Describe this image in one sentence."),
                                image_part(_TINY_JPEG)])
            )
        if do_audio:
            combos.append(
                ("text+audio", [text_part("Transcribe and respond briefly."),
                                audio_part(_TINY_WAV)])
            )

        trials: list[Trial] = []
        for name, content in combos:
            trial = await _time_one(ic, name, content)
            trials.append(trial)
            out(
                f"{name:12s} TTFT={trial.ttft_s*1000:6.0f} ms  "
                f"total={trial.total_s:5.2f} s  ~{trial.tok_per_s:5.1f} tok/s"
            )
        return trials
