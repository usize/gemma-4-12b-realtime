#!/usr/bin/env python
"""Standalone runner for the inference latency benchmark.

Usage:
    .venv/bin/python scripts/bench_latency.py [--image] [--audio]

Thin wrapper around rlb.bench.run_bench so the benchmark is runnable both via the
`rlb bench` CLI and directly during endpoint bring-up.
"""

from __future__ import annotations

import argparse
import asyncio

from rlb.bench import run_bench
from rlb.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", action="store_true", help="include a text+image trial")
    ap.add_argument("--audio", action="store_true", help="include a text+audio trial")
    args = ap.parse_args()
    asyncio.run(run_bench(load_config(), do_image=args.image, do_audio=args.audio))


if __name__ == "__main__":
    main()
