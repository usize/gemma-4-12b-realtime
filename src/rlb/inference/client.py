"""Async client for an OpenAI-compatible chat-completions endpoint (plan §1.2).

The endpoint location is a config value (`inference.base_url`); the same code targets
llama.cpp's server, LM Studio, vLLM, or an MLX server. The client:
  * assembles multimodal messages (text + JPEG image parts + WAV/PCM audio parts),
  * streams responses as `ChatChunk`s (token deltas + tool-call deltas) for low-latency TTS,
  * parses tool calls into `ToolCall`s,
  * applies one retry/timeout/health-check policy,
  * exposes latency-budget knobs via the InferenceConfig (max image px / audio seconds).

Multimodal part formats follow the OpenAI/vLLM convention:
  text  -> {"type": "text", "text": ...}
  image -> {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  audio -> {"type": "input_audio", "input_audio": {"data": <b64>, "format": "wav"}}
Open question (plan §9.1): confirm Gemma 4's exact audio part name on the chosen server.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from rlb.config import InferenceConfig

# A "content part" of a chat message (OpenAI multimodal content list element).
Part = dict[str, Any]


def text_part(text: str) -> Part:
    return {"type": "text", "text": text}


def image_part(jpeg_bytes: bytes) -> Part:
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def audio_part(wav_bytes: bytes, fmt: str = "wav") -> Part:
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatChunk:
    """One streamed delta. `text` accumulates spoken content; tool calls arrive whole."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class InferenceError(RuntimeError):
    pass


class InferenceClient:
    def __init__(self, cfg: InferenceConfig, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        timeout = httpx.Timeout(cfg.request_timeout_s, connect=cfg.connect_timeout_s)
        self._client = client or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> InferenceClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- health
    async def health(self) -> bool:
        """True if the endpoint lists models (cheap liveness probe)."""
        try:
            r = await self._client.get("/models", timeout=self.cfg.connect_timeout_s)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # ------------------------------------------------------------- non-stream
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatChunk:
        """One-shot completion. Aggregates a streamed response into a single chunk."""
        out = ChatChunk()
        async for chunk in self.stream(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        ):
            out.text += chunk.text
            out.tool_calls.extend(chunk.tool_calls)
            out.finish_reason = chunk.finish_reason or out.finish_reason
        return out

    # ----------------------------------------------------------------- stream
    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ChatChunk]:
        """Stream completion deltas as ChatChunks. Tool-call fragments are reassembled."""
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
        }
        if self.cfg.disable_thinking:
            # llama.cpp/Qwen3: suppress the hidden reasoning pass (saves the token budget
            # and latency that was otherwise returning empty content).
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Accumulate partial tool calls keyed by their streamed index.
        pending: dict[int, dict[str, Any]] = {}

        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise InferenceError(f"HTTP {resp.status_code}: {body[:500]}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    delta_chunk = _parse_sse_delta(data, pending)
                    if delta_chunk is not None:
                        yield delta_chunk
        except httpx.HTTPError as e:
            raise InferenceError(f"request failed: {e}") from e

        # Flush any completed tool calls assembled across deltas.
        finished = [_to_tool_call(tc) for tc in pending.values() if tc.get("name")]
        if finished:
            yield ChatChunk(tool_calls=finished, finish_reason="tool_calls")


def _parse_sse_delta(data: str, pending: dict[int, dict[str, Any]]) -> ChatChunk | None:
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = obj.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    delta = choice.get("delta") or {}
    finish = choice.get("finish_reason")

    # Reassemble tool-call fragments (name on first delta, arguments streamed).
    for tc in delta.get("tool_calls") or []:
        idx = tc.get("index", 0)
        slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["args"] += fn["arguments"]

    text = delta.get("content") or ""
    if text or finish:
        return ChatChunk(text=text, finish_reason=finish)
    return None


def _to_tool_call(slot: dict[str, Any]) -> ToolCall:
    try:
        args = json.loads(slot["args"]) if slot["args"] else {}
    except json.JSONDecodeError:
        args = {"_raw": slot["args"]}
    return ToolCall(id=slot.get("id") or slot["name"], name=slot["name"], arguments=args)
