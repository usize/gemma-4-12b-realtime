# Reachy Mini Local Brain — Implementation Plan

A fully-local embodied assistant for Reachy Mini Lite, driven by Gemma 4 12B (multimodal: audio + vision + text), with continuous autonomous gaze, streaming TTS, and a tool/coding harness for arbitrary actions.

**Target environment:** MacBook Pro M4 Max, 128 GB unified memory, macOS. Reachy Mini **Lite** (USB-tethered; the daemon runs on the Mac itself). Inference may optionally run on a second machine over the LAN — all model access goes through an OpenAI-compatible HTTP endpoint so the location is a config value, not an architectural decision.

---

## 0. Design principles

1. **Everything local.** No cloud APIs. The only network traffic is optional LAN traffic to a local inference box.
2. **The model is slow; the robot is fast.** Gemma inference runs at seconds-scale. Head motion must run at 50–100 Hz. Therefore: a fast reflexive motion layer that never blocks on the LLM, and a slow deliberative layer that *biases* the fast layer rather than driving servos directly.
3. **OpenAI-compatible endpoint abstraction.** One `INFERENCE_BASE_URL` env var. Works with llama.cpp server, LM Studio, vLLM on a Linux box, or MLX-served Gemma — interchangeable.
4. **Processes, not threads.** Each subsystem is its own process communicating over local transports (ZeroMQ or plain asyncio TCP/Unix sockets + msgpack). This matches the Reachy daemon's own client-server design and lets you restart the cognition layer without dropping the motion loop.
5. **Degrade gracefully.** If the model endpoint is down, the robot still idles, breathes, and tracks faces. If the camera dies, conversation still works.

---

## 1. Model layer

### 1.1 Gemma 4 12B (released June 2026)

Key facts the implementation should exploit:
- Encoder-free multimodal: text, images, audio (raw 16 kHz), and video project directly into the decoder. **Audio input is native — no separate Whisper/ASR stage is required.** It transcribes and *understands* audio in one pass (tone, ambient sound, etc.).
- Native tool calling and built-in thinking; vLLM has day-0 support with `gemma4` reasoning + tool parsers, served via OpenAI-compatible API. 256K context.
- Output is **text only** — TTS remains our job.
- A multi-token-prediction (MTP) drafter model ships alongside it for speculative decoding — use it if the serving stack supports it (vLLM does; check llama.cpp status at build time).

### 1.2 Serving options (pick via config, implement against the common API)

| Option | Where | Notes |
|---|---|---|
| **llama.cpp `llama-server`** | On the Mac | GGUF Q5_K_M (~8.5 GB weights, trivial in 128 GB). Verify multimodal (audio+image) support for Gemma 4 in llama.cpp at implementation time — it landed quickly for Gemma 3; confirm for 4. |
| **MLX / mlx-lm server** | On the Mac | Likely the best tokens/sec on Apple Silicon; check mlx-vlm for Gemma 4 audio support. |
| **vLLM** | Linux box over LAN | Reference path: official recipe at `recipes.vllm.ai/Google/gemma-4-12B-it`. Full tool-calling + reasoning parser support. Use this if a GPU box is available. |

**Implementation task:** a thin `InferenceClient` wrapper (httpx, async) that:
- sends multimodal messages (base64 WAV/PCM audio parts + JPEG image parts + text),
- handles streaming responses (SSE) and surfaces tokens as they arrive (needed for low-latency TTS),
- parses tool calls,
- has a single retry/timeout/health-check policy,
- exposes a `latency budget` knob (max image resolution, max audio clip length) so we can tune end-to-end response time empirically.

### 1.3 Latency-tiered model usage

- **Tier A (every turn):** user utterance audio clip + 1 current camera frame + rolling conversation summary → response + tool calls. Budget: first token < ~1.5 s.
- **Tier B (periodic, optional):** "scene awareness" tick — every N seconds of idle, send a downscaled frame with a tiny prompt ("anything noteworthy changed?") to give the robot ambient awareness. Strictly rate-limited and cancellable; off by default.

---

## 2. Audio input pipeline

Gemma's native audio ingestion replaces ASR, but we still need to decide *when* to send audio:

1. **Capture:** Reachy Mini Lite's microphones via the SDK media manager (`media_backend="default"`; LOCAL/GStreamer since daemon and client are on the same Mac), 16 kHz mono. Fallback: Mac's own mic via `sounddevice` if the robot mic path is flaky.
2. **VAD:** Silero VAD (tiny, CPU, well-proven) gates the stream into utterances. Endpointing: 600–800 ms of trailing silence closes an utterance.
3. **Barge-in:** if VAD triggers while TTS is playing, pause/duck playback, capture the utterance, and cancel the in-flight generation. (Echo caution: Reachy's speaker and mics are co-located — implement simple energy-based echo suppression first: ignore VAD triggers whose spectral profile matches the currently-playing TTS buffer, or just gate VAD by "TTS playing + no headset" with a push-to-interrupt hotword fallback. Iterate here; this is the hardest UX detail.)
4. **Wake behavior:** configurable — (a) always listening (default for a desk robot), (b) wake word via openWakeWord, (c) "look-to-talk": only treat speech as addressed to the robot when a face is detected facing the camera. Ship (a) first, add (c) — it's charming and cheap since perception already tracks faces.
5. The closed utterance is shipped to the cognition service as a WAV blob — no transcription step.

---

## 3. Vision pipeline

Two consumers with very different rates:

- **Fast perception (every frame, ~15–30 fps):** `mini.media.get_frame()` → MediaPipe face detection/mesh (or YOLO-face; the official `reachy_mini_conversation_app` supports both backends — copy their choice). Outputs: face bounding boxes, the "primary" face (largest/most central/most recent speaker), optional motion saliency (frame-diff based). Published on the bus as `PerceptionState` at frame rate.
- **Slow understanding (on demand):** the most recent frame, JPEG-encoded at a configurable max dimension (start at 768 px), attached to model calls. Also exposed to the model as a `look_and_describe` tool so it can *choose* to take a fresh look mid-conversation.

---

## 4. Motion: free-looking head instead of macros

### 4.1 Why not a VLA (decision + rationale to record in the repo)

SmolVLA / LeRobot policies are built for *manipulation* embodiments (SO-100/101 arms): multi-view images + proprio → gripper action chunks. Reachy Mini has no arms; its action space is 6-DoF head pose + body yaw + 2 antennas, and the desired behavior ("look around naturally, attend to people and motion") is well-served by a classical attention controller — better, cheaper, and lower-latency than fine-tuning a VLA from scratch with ~50 teleop episodes. **Phase 1–3 use an engineered attention system.** A learned gaze policy (LeRobot-format dataset of head-pose trajectories, small behavior-cloning model — not necessarily a full VLA) is kept as an optional Phase 5 experiment, with the data logger built in from day one so the dataset accrues for free.

### 4.2 Layered motion controller (the core of "alive")

Mirror the layered design in Pollen's conversation app (primary moves + blended wobble + head tracking), but own the implementation:

A single 100 Hz control loop calls `mini.set_target(...)` (the non-interpolated, high-frequency API). The commanded pose is a weighted blend of layers, top priority first:

1. **Commanded gestures** (from the LLM via tools): nod, shake, emotes, `look_at(x,y,z)`, dances. Executed as trajectories; while active they dominate.
2. **Gaze/attention layer:** a target selector + smooth pursuit controller.
   - Targets, scored every perception tick: detected faces (heavily weighted, sticky toward current speaker), motion saliency regions, sound direction if available, and LLM-suggested points of interest.
   - Smooth pursuit with a critically-damped 2nd-order filter (no robotic snapping), saccade behavior for large target jumps, micro-fixations (small random offsets every 1–3 s) so gaze never looks frozen.
   - **Curiosity/idle scan:** when no targets, slow Perlin-noise wander of gaze within safe limits, occasionally "checking" the last place a face was seen.
3. **Speech-reactive layer:** while TTS plays, modulate small head bobs/antenna motion from the audio envelope (the conversation app's "wobble" pattern).
4. **Breathing/idle base layer:** constant low-amplitude sinusoidal z/pitch motion.

All layers output pose deltas; the blender clamps to the SDK's safe workspace limits before `set_target`. Body yaw is used when the gaze target exceeds head pan range (turn body, recenter head — very lifelike).

### 4.3 Safety/limits

- Respect SDK joint/workspace limits; rate-limit set_target deltas (max angular velocity).
- Watchdog: if the control loop stalls > 200 ms, command a neutral pose via `goto_target` and log.
- A physical/keyboard kill switch: spacebar in the console / signal handler → torque-relaxed safe pose.

---

## 5. TTS

- **Primary: Kokoro-82M.** Runs CPU/MPS on Apple Silicon, very fast (RTF ≪ real time), Apache 2.0, good quality, supports French among its languages (nice-to-have). Sentence-level streaming: as the LLM streams tokens, split on sentence boundaries and synthesize each sentence while the next is generating — perceived latency = first sentence only.
- **Optional expressive voice: Chatterbox Turbo (MIT).** Slower but more emotional range + 10-second voice cloning; offer as a config switch (`TTS_BACKEND=kokoro|chatterbox`) behind the same interface.
- **Interface:** `TTSService.speak(text_stream) -> audio chunks` published to (a) the playback device (robot speaker via SDK media, fallback Mac speakers) and (b) the motion service's speech-reactive layer (envelope only).
- Playback must be cancellable mid-sentence for barge-in.

---

## 6. Cognition & the action harness

### 6.1 Orchestrator

An async event loop that owns conversation state:
- Receives utterance audio → builds the multimodal message (audio + current frame + system prompt + rolling summary + recent turns) → streams the response.
- Routes streamed text to TTS; routes tool calls to the tool runtime; loops tool results back to the model (standard agentic loop, capped iterations per turn).
- Maintains a **rolling summary**: raw turns kept for the last ~10 exchanges; older context compressed by an occasional summarization call. Persist to SQLite so the robot remembers across restarts.

### 6.2 Tool registry (what the model can do)

Declared via Gemma 4's native tool calling. Three rings, by blast radius:

**Ring 0 — embodiment (always on, auto-approved):**
`look_at(x,y,z)`, `track_person(on/off)`, `play_emotion(name)` (happy/curious/sad/confused antenna+head sets), `nod()`, `shake()`, `dance(name)`, `take_photo_and_describe()`, `set_voice(params)`.

**Ring 1 — desk utilities (auto-approved, read-mostly):**
`get_time()`, `timer(set/cancel)`, `remember(note)` / `recall(query)` (SQLite-backed), `read_clipboard()` (optional, off by default).

**Ring 2 — the coding harness (arbitrary actions, gated):**
- `run_python(code)` / `run_shell(cmd)` executed in a **sandbox**: a dedicated non-privileged workspace directory, subprocess with timeout, resource limits, and an allowlist/denylist policy file. No sudo, no rm -rf outside workspace, network policy configurable.
- `delegate_to_coder(task: str)` — shells out to **Claude Code in headless mode** (`claude -p "<task>" --output-format json`, with `--allowedTools` scoped to the workspace) for anything bigger than a snippet: "write me a script that graphs my CSV," "add a new dance to your own repertoire." This is the cleanest "voice → arbitrary action" path and it means the robot can extend itself: new tools dropped into `tools/` are hot-loaded into the registry on the next turn.
- **Approval policy:** Ring 2 calls are spoken aloud ("I'm going to run a script that does X — okay?") and require a verbal "yes" / console keypress, configurable to auto-approve per-tool once trusted. All Ring 2 executions logged verbatim.

### 6.3 System prompt sketch

Personality, embodiment description (what the body can/can't do), tool etiquette (prefer Ring 0 expressiveness liberally; ask before Ring 2), brevity norms for spoken output (1–3 sentences unless asked), and an instruction that responses are *spoken* — no markdown, no lists.

---

## 7. Process architecture

```
┌─────────────────────────── MacBook M4 Max ───────────────────────────┐
│                                                                       │
│  reachy_mini daemon (vendor, :8000)  ◄── USB ──►  Reachy Mini Lite    │
│        ▲ REST/WS + media                                              │
│        │                                                              │
│  ┌─────┴──────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────┐  │
│  │  motion    │◄──│  perception  │   │  cognition  │──►│ tool      │  │
│  │  service   │   │  service     │──►│ orchestrator│   │ runtime / │  │
│  │ (100Hz     │   │ (camera+VAD) │   │             │   │ sandbox + │  │
│  │  blender)  │◄──┼──────────────┼───│             │   │ claude -p │  │
│  └────────────┘   └──────────────┘   └──────┬──────┘   └───────────┘  │
│        ▲                                    │                         │
│  ┌─────┴──────┐                       ┌─────▼──────┐                  │
│  │ TTS service│◄──────────────────────│ Inference  │── HTTP ──► local │
│  │ (Kokoro)   │   text stream         │ client     │    or LAN model  │
│  └────────────┘                       └────────────┘    server        │
└───────────────────────────────────────────────────────────────────────┘
```

- **Bus:** ZeroMQ pub/sub for high-rate topics (`perception.state`, `tts.envelope`) + req/rep or simple HTTP for commands. Single `bus.py` module defining typed message schemas (pydantic + msgpack).
- **Supervisor:** one `make run` / `uv run rlb up` entrypoint that launches all services with structured logging (one color per service) and restarts crashed ones (motion service restart must re-home gracefully).
- **Repo layout:**

```
reachy-local-brain/
  pyproject.toml            # uv-managed
  config.yaml               # endpoints, backends, tunables — single source
  src/rlb/
    bus.py                  # schemas + transport
    inference/client.py
    audio/{capture.py,vad.py}
    vision/perception.py
    motion/{layers.py,controller.py,gestures.py}
    tts/{base.py,kokoro.py,chatterbox.py}
    cognition/{orchestrator.py,memory.py,prompt.py}
    tools/{registry.py,ring0_embodiment.py,ring1_utils.py,ring2_harness.py}
    sandbox/executor.py
    logging/datalogger.py   # LeRobot-format episode logger (Phase 5 prep)
  tests/                    # unit + a mujoco-sim integration suite
  scripts/{bench_latency.py,calibrate_gaze.py}
```

- **Simulation first:** the SDK's MuJoCo backend means the whole stack (minus real audio) is testable without hardware — CI runs the motion blender + a mocked cognition loop in sim.

---

## 8. Build phases (each independently demoable)

**Phase 0 — Skeleton & plumbing (1 session).** Repo, uv env, config, bus, supervisor. Reachy daemon connection smoke test (real + MuJoCo). InferenceClient against a running Gemma 4 endpoint: text-only round trip, then image, then audio clip. `bench_latency.py` reports tok/s and time-to-first-token for each modality combo — this calibrates everything downstream.

**Phase 1 — The robot feels alive (no LLM).** Perception service (camera → face detection). Motion service with all four layers; face tracking + idle curiosity scan + breathing. Tune until it passes the "leave it on your desk for an hour" vibe check.

**Phase 2 — Talk to it.** Audio capture + Silero VAD → utterance WAVs. Orchestrator: audio + frame → Gemma → streamed text → Kokoro sentence-streaming playback. Speech-reactive motion layer. Target: < 2.5 s from end-of-speech to first audio out; measure and iterate (frame size, audio length, MTP drafter, endpoint placement).

**Phase 3 — Embodied agency.** Tool calling: Ring 0 + Ring 1. Barge-in. Rolling summary + SQLite memory. `take_photo_and_describe`. Look-to-talk mode.

**Phase 4 — The coding harness.** Sandbox executor, Ring 2 tools, verbal approval flow, `delegate_to_coder` via headless Claude Code, hot-loadable `tools/` directory (the self-extension loop). Demo: "Reachy, write a script that pulls today's weather and tell me if I need a jacket, then add that as a permanent tool."

**Phase 5 (optional research) — Learned gaze.** The data logger has been recording (camera frames + head pose + audio events) in LeRobot dataset format since Phase 1. Train a small behavior-cloning gaze policy; A/B it against the engineered controller as an alternative layer-2 implementation.

---

## 9. Open questions to resolve during implementation (check, don't assume)

1. llama.cpp / MLX multimodal (audio) support status for Gemma 4 12B — if not ready, run vLLM on the LAN box, or temporarily front audio with whisper.cpp and feed text (keep the interface identical).
2. Exact Reachy Mini Lite mic configuration and whether sound-direction estimation is feasible (wireless version has more sensors; Lite may not support DOA — then drop sound-localization from the gaze scorer).
3. Echo/barge-in tuning with co-located speaker+mic (Section 2.3) — budget real experimentation time.
4. `set_target` max safe update rate and any daemon-side rate limiting — read `AGENTS.md` and `src/reachy_mini/reachy_mini.py` in the SDK repo first (the maintainers explicitly publish an AGENTS.md for coding agents — feed it to the harness alongside this plan).
5. Kokoro French voice quality, if bilingual operation is wanted later.

---

## 10. First instruction to the coding harness

> Read this plan and `https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md`. Execute Phase 0. Before writing code, verify: (a) reachy_mini SDK installs and connects in MuJoCo sim mode, (b) the configured Gemma 4 endpoint answers a text, an image, and an audio request. Report measured latencies before proceeding.
