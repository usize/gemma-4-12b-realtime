"""Reachy Local Brain (rlb) — a fully-local embodied assistant for Reachy Mini Lite.

See reachy-local-brain-plan.md for the architecture. The package is organized as
independent services (motion, perception, cognition, tts) that communicate over a
typed message bus, plus a thin inference client against an OpenAI-compatible endpoint.
"""

__version__ = "0.1.0"
