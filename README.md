# Reachy Local Brain (`rlb`)

A fully-local embodied assistant for the **Reachy Mini Lite**, driven by **Gemma 4**
(multimodal: audio + vision + text) with continuous autonomous gaze, streaming TTS,
and a tool/coding harness for arbitrary actions.

Everything runs locally. The only network traffic is optional LAN traffic to a local
inference box. See [`reachy-local-brain-plan.md`](reachy-local-brain-plan.md) for the
full architecture and build phases.

## Architecture

Independent services communicate over a typed message bus (ZeroMQ pub/sub + msgpack):

```
motion (100 Hz blender) ◄── perception (camera+VAD) ──► cognition orchestrator ──► tool runtime
        ▲                                                      │
   TTS (Kokoro) ◄───────────────────────────────── inference client ──HTTP──► local/LAN Gemma 4
```

- **inference** — async client against an OpenAI-compatible endpoint (location is config).
- **bus** — pydantic message schemas + msgpack over ZeroMQ, with an in-process transport for tests.
- **motion / perception / tts / cognition** — the four runtime services (built across phases).

## Quickstart

```bash
uv pip install -e .            # core
uv pip install -e '.[sim]'     # + MuJoCo for hardware-free smoke tests
rlb doctor                     # environment + endpoint health check
rlb smoke --sim                # connect to the robot in MuJoCo and wiggle
```

Configuration lives in [`config.yaml`](config.yaml); any value is overridable by an
env var of the form `RLB_<SECTION>__<KEY>` (e.g. `RLB_INFERENCE__BASE_URL=...`).

## Status

Phase 0 (skeleton, bus, config, inference client) — in progress. See the plan's §8.
