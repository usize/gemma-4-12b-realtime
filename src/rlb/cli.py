"""`rlb` command-line entrypoint.

Phase 0 commands:
  rlb doctor          environment + config + inference-endpoint health report
  rlb smoke [--sim]   connect to the robot (real or MuJoCo) and do a safe wiggle
  rlb bench [...]     latency benchmark against the inference endpoint

`rlb up` (full supervisor) arrives as services land in later phases.
"""

from __future__ import annotations

import asyncio
import importlib.util

import typer
from rich.console import Console
from rich.table import Table

from rlb.config import load_config

app = typer.Typer(add_completion=False, help="Reachy Local Brain control CLI.")
console = Console()

# Modules we expect, grouped by phase so `doctor` can show what's ready.
_DEP_GROUPS = {
    "core": ["pydantic", "msgpack", "zmq", "httpx", "yaml", "numpy"],
    "audio/tts": ["sounddevice", "kokoro_onnx", "onnxruntime"],
    "robot": ["reachy_mini", "cv2"],
    "perception (Phase 1)": ["mediapipe"],
    "sim": ["mujoco"],
}


def _present(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ModuleNotFoundError, ValueError):
        return False


@app.command()
def doctor() -> None:
    """Report dependency, config, and inference-endpoint health."""
    cfg = load_config()

    deps = Table(title="Dependencies", show_header=True, header_style="bold")
    deps.add_column("group")
    deps.add_column("module")
    deps.add_column("status")
    for group, mods in _DEP_GROUPS.items():
        for mod in mods:
            ok = _present(mod)
            deps.add_row(group, mod, "[green]OK[/]" if ok else "[red]missing[/]")
    console.print(deps)

    cfgt = Table(title="Config", show_header=False)
    cfgt.add_row("inference.base_url", cfg.inference.base_url)
    cfgt.add_row("inference.model", cfg.inference.model)
    cfgt.add_row("robot.backend", cfg.robot.backend)
    cfgt.add_row("tts.backend", f"{cfg.tts.backend} ({cfg.tts.voice})")
    cfgt.add_row("cognition.wake_mode", cfg.cognition.wake_mode)
    console.print(cfgt)

    # Endpoint liveness (async health probe).
    healthy = asyncio.run(_check_endpoint(cfg))
    if healthy:
        console.print(f"[green]✓[/] inference endpoint reachable at {cfg.inference.base_url}")
    else:
        console.print(
            f"[yellow]![/] inference endpoint NOT reachable at {cfg.inference.base_url} "
            "— start a server (llama.cpp / LM Studio / MLX) or set RLB_INFERENCE__BASE_URL"
        )


async def _check_endpoint(cfg) -> bool:
    from rlb.inference import InferenceClient

    async with InferenceClient(cfg.inference) as ic:
        return await ic.health()


@app.command()
def smoke(
    sim: bool = typer.Option(False, "--sim", help="Use the MuJoCo sim backend instead of hardware."),
) -> None:
    """Connect to the robot and perform a small, safe motion (Phase-0 gate)."""
    from rlb.robot import smoke as run_smoke

    cfg = load_config()
    mode = "MuJoCo sim" if sim else "real hardware"
    console.print(f"[bold]Connecting to robot ({mode})…[/]")
    obs = run_smoke(cfg, use_sim=sim)
    t = Table(show_header=False, title="Smoke test results")
    for k, v in obs.items():
        t.add_row(k, str(v))
    console.print(t)
    console.print("[green]✓ smoke test complete[/]")


def _resolve_services(names: list[str] | None):
    from rlb.services import build_services

    registry = build_services(load_config())
    if not names:
        return list(registry.values())
    chosen = []
    for n in names:
        if n not in registry:
            raise typer.BadParameter(f"unknown service {n!r}; known: {', '.join(registry)}")
        chosen.append(registry[n])
    return chosen


@app.command()
def up(services: list[str] = typer.Argument(None, help="Service names; default all.")) -> None:
    """Start service(s): inference (Gemma), daemon (robot). Idempotent."""
    for svc in _resolve_services(services):
        console.print(f"[bold]{svc.name}[/]: {svc.start()}")


@app.command()
def down(services: list[str] = typer.Argument(None, help="Service names; default all.")) -> None:
    """Stop service(s). No pkill needed."""
    for svc in _resolve_services(services):
        console.print(f"[bold]{svc.name}[/]: {svc.stop()}")


@app.command()
def status() -> None:
    """Show running state of all services."""
    t = Table(title="Services", header_style="bold")
    t.add_column("service"); t.add_column("running"); t.add_column("pid")
    t.add_column("port"); t.add_column("port open")
    for svc in _resolve_services(None):
        s = svc.status()
        run = "[green]yes[/]" if s["running"] else "[dim]no[/]"
        po = "" if s["port_open"] is None else ("[green]✓[/]" if s["port_open"] else "[red]✗[/]")
        t.add_row(s["name"], run, str(s["pid"] or "—"), str(s["port"] or "—"), po)
    console.print(t)


@app.command()
def logs(
    service: str = typer.Argument(..., help="Service name (inference|daemon)."),
    follow: bool = typer.Option(False, "-f", "--follow", help="Tail and follow."),
    lines: int = typer.Option(40, "-n", help="Lines to show."),
) -> None:
    """Show (or follow) a service's log."""
    import subprocess as sp

    (svc,) = _resolve_services([service])
    if not svc.log_file.exists():
        console.print(f"[yellow]no log yet at {svc.log_file}[/]")
        return
    args = ["tail", f"-n{lines}"] + (["-f"] if follow else []) + [str(svc.log_file)]
    sp.run(args)


@app.command()
def chat(
    greeting: str = typer.Option(
        "",
        help="Spoken on startup; empty to skip.",
    ),
) -> None:
    """Hands-free conversation: talk to Reachy. Ctrl-C to stop."""
    import threading

    from rlb.cognition.orchestrator import Orchestrator
    from rlb.robot import connect

    cfg = load_config()
    # Pre-flight: the ML services must be up (they run in .venv-ml).
    import httpx

    checks = {
        "inference": f"{cfg.inference.base_url}/models",
        "asr": f"{cfg.audio.asr.base_url}/health",
        "tts": f"{cfg.tts.base_url}/health",
    }
    down = []
    for name, url in checks.items():
        try:
            console.print(f"checking health: {name}, {url}")
            if httpx.get(url, timeout=2.0).status_code != 200:
                down.append(name)
        except httpx.HTTPError:
            down.append(name)
    if down:
        console.print(f"[red]services down:[/] {', '.join(down)} — run [bold]rlb up "
                      f"{' '.join(down)}[/] first (they load models; give them a moment).")
        raise typer.Exit(1)

    console.print("[bold]Connecting… then just talk. Ctrl-C to stop.[/]")
    with connect(cfg, use_sim=False) as sess:
        orch = Orchestrator(sess, cfg)
        stop = threading.Event()
        try:
            orch.run(stop, greeting=greeting or None)
        except KeyboardInterrupt:
            console.print("\n[dim]stopping…[/]")
            stop.set()
    console.print("[green]✓ conversation ended[/]")


@app.command()
def say(
    text: str = typer.Argument(..., help="What Reachy should say."),
    mac: bool = typer.Option(False, help="Play through Mac speakers instead of the robot."),
    instruct: str = typer.Option("", help="Override the style/emotion instruction."),
    speed: float = typer.Option(0.0, help="Playback speed multiplier (>1 faster); 0 = config default."),
) -> None:
    """Speak text via the TTS service (robot speaker by default)."""
    from rlb.audio import TtsClient
    from rlb.robot import connect

    cfg = load_config()
    if speed:
        cfg.tts.speed = speed
    if mac:
        cfg.tts.sink = "mac"
        tts = TtsClient(cfg.tts)
        tts.speak(text, instruct=instruct or None)
        tts.close()
        console.print("[green]✓ spoke (mac)[/]")
        return
    with connect(cfg, use_sim=False) as sess:
        tts = TtsClient(cfg.tts, robot_session=sess)
        tts.speak(text, instruct=instruct or None)
        tts.close()
    console.print("[green]✓ spoke (robot speaker)[/]")

@app.command()
def embody(
    request: str = typer.Argument(..., help="A natural-language instruction for the robot."),
    sim: bool = typer.Option(False, "--sim"),
    speak: bool = typer.Option(False, help="Speak the reply via macOS `say` (placeholder TTS)."),
) -> None:
    """One embodied turn: inject live pose, let Gemma pick skills, execute them."""
    import asyncio
    import subprocess as sp

    from rlb.cognition import assemble_messages
    from rlb.embodiment import EmbodimentSkills, read_state, skills_schema
    from rlb.inference import InferenceClient, text_part
    from rlb.robot import connect

    cfg = load_config()
    with connect(cfg, use_sim=sim) as sess:
        sess.mini.wake_up()
        skills = EmbodimentSkills(sess, cfg)
        before = read_state(sess.mini)
        console.print(f"[dim]state:[/] {before.line()}")

        async def turn() -> tuple[str, list]:
            async with InferenceClient(cfg.inference) as ic:
                msgs = assemble_messages([], [text_part(request)], state_line=before.line())
                out = await ic.complete(msgs, tools=skills_schema(), max_tokens=200, temperature=0.4)
                return out.text, out.tool_calls

        reply, calls = asyncio.run(turn())
        for tc in calls:
            result = skills.execute(tc.name, tc.arguments)
            console.print(f"[cyan]skill[/] {tc.name}({tc.arguments}) -> {result}")
        import time as _t
        _t.sleep(1.0)
        console.print(f"[dim]state:[/] {read_state(sess.mini).line()}")
        if reply:
            console.print(f"[bold green]Reachy:[/] {reply.strip()}")
            if speak:
                sp.run(["say", reply.strip()])
    console.print("[green]✓ done[/]")


@app.command()
def motion(
    sim: bool = typer.Option(False, "--sim", help="Use MuJoCo sim instead of hardware."),
    seconds: float = typer.Option(15.0, "-s", help="How long to run the loop."),
    demo: bool = typer.Option(False, help="Drive synthetic gaze sweeps + speech pulses."),
) -> None:
    """Run the layered 100 Hz head controller (Phase 1). Ctrl-C to stop early."""
    import math
    import threading
    import time

    from rlb.motion import MotionController
    from rlb.motion.pose import HeadOffset
    from rlb.robot import connect

    cfg = load_config()
    console.print(f"[bold]Motion loop ({'sim' if sim else 'real'}), {seconds:.0f}s"
                  f"{' + demo drivers' if demo else ''}…[/]")
    with connect(cfg, use_sim=sim) as sess:
        sess.mini.wake_up()
        ctrl = MotionController(sess, cfg)
        stop = threading.Event()
        loop = threading.Thread(target=ctrl.run, args=(stop,), daemon=True)
        loop.start()
        try:
            t0 = time.perf_counter()
            while (t := time.perf_counter() - t0) < seconds and loop.is_alive():
                if demo:
                    # Sweep gaze left/right, with a periodic "speaking" pulse.
                    yaw = 18.0 * math.sin(2 * math.pi * t / 6.0)
                    pitch = 8.0 * math.sin(2 * math.pi * t / 4.0)
                    ctrl.set_gaze_target(HeadOffset(yaw=yaw, pitch=pitch) if (t % 12) < 8 else None)
                    ctrl.set_speech_level(max(0.0, math.sin(2 * math.pi * t / 3.0)))
                time.sleep(0.05)
        except KeyboardInterrupt:
            console.print("\n[dim]interrupted[/]")
        finally:
            stop.set()
            loop.join(timeout=2.0)
    console.print("[green]✓ motion loop ended cleanly[/]")


@app.command()
def bench(
    image: bool = typer.Option(True, help="Include an image part in one of the trials."),
    audio: bool = typer.Option(False, help="Include an audio part (needs a real audio-capable server)."),
) -> None:
    """Benchmark time-to-first-token and tok/s for text / +image / +audio."""
    from rlb.bench import run_bench  # lazy: keeps CLI import light

    asyncio.run(run_bench(load_config(), do_image=image, do_audio=audio, console=console))


if __name__ == "__main__":
    app()
