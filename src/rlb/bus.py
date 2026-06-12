"""The message bus: typed schemas + msgpack serialization + transports.

Topology (plan §7): high-rate state (perception, tts envelope) and commands flow
over ZeroMQ PUB/SUB. Because PUB/SUB is many-to-many, the supervisor runs a single
XPUB/XSUB *forwarder* (see `run_forwarder`); every service connects to it rather
than binding its own socket, so any service can be restarted independently.

An in-process transport (`InprocBus`) implements the same `publish`/`subscribe`
surface for unit tests and the MuJoCo-only CI path, with no sockets involved.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any, ClassVar

import msgpack
from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #
class Topic:
    PERCEPTION_STATE = "perception.state"      # PerceptionState @ frame rate
    TTS_ENVELOPE = "tts.envelope"              # SpeechEnvelope while speaking
    MOTION_COMMAND = "motion.command"          # MotionCommand (gestures, look_at)
    UTTERANCE = "audio.utterance"              # Utterance (closed VAD segment)
    SPEAK = "tts.speak"                        # SpeakRequest (text to synthesize)
    BARGE_IN = "audio.barge_in"               # BargeIn (cancel current speech/gen)
    LOG = "system.log"                         # structured cross-service log lines


# --------------------------------------------------------------------------- #
# Message schemas
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    """Base for all bus messages. `topic` is the routing key on the wire."""

    topic: ClassVar[str] = ""


class Face(BaseModel):
    # Normalized [0,1] image coords; (cx,cy) center, (w,h) box size, area for scoring.
    cx: float
    cy: float
    w: float
    h: float
    score: float = 1.0
    track_id: int | None = None


class PerceptionState(Message):
    topic: ClassVar[str] = Topic.PERCEPTION_STATE
    ts: float
    frame_w: int
    frame_h: int
    faces: list[Face] = []
    primary_face_idx: int | None = None
    motion_saliency: tuple[float, float] | None = None  # normalized (x,y) hotspot


class SpeechEnvelope(Message):
    topic: ClassVar[str] = Topic.TTS_ENVELOPE
    ts: float
    level: float          # 0..1 RMS of the chunk currently playing
    speaking: bool


class MotionCommand(Message):
    topic: ClassVar[str] = Topic.MOTION_COMMAND
    ts: float
    kind: str             # "look_at" | "nod" | "shake" | "emotion" | "dance" | "track"
    args: dict[str, Any] = {}


class Utterance(Message):
    topic: ClassVar[str] = Topic.UTTERANCE
    ts: float
    sample_rate: int
    # Base64-free: raw PCM int16 bytes carried straight through msgpack.
    pcm: bytes
    duration_s: float


class SpeakRequest(Message):
    topic: ClassVar[str] = Topic.SPEAK
    ts: float
    text: str
    interrupt: bool = False  # if True, cancel current playback before speaking


class BargeIn(Message):
    topic: ClassVar[str] = Topic.BARGE_IN
    ts: float


class LogLine(Message):
    topic: ClassVar[str] = Topic.LOG
    ts: float
    service: str
    level: str
    text: str


# Registry so a subscriber can rehydrate the right model from a topic string.
_REGISTRY: dict[str, type[Message]] = {
    m.topic: m
    for m in (
        PerceptionState, SpeechEnvelope, MotionCommand,
        Utterance, SpeakRequest, BargeIn, LogLine,
    )
}


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def pack(msg: Message) -> bytes:
    """Serialize a message body to msgpack bytes (topic travels as the zmq frame 0)."""
    return msgpack.packb(msg.model_dump(), use_bin_type=True)


def unpack(topic: str, body: bytes) -> Message:
    """Rehydrate a message from its topic + msgpack body."""
    model = _REGISTRY.get(topic)
    data = msgpack.unpackb(body, raw=False)
    if model is None:
        raise KeyError(f"unknown topic {topic!r}")
    return model.model_validate(data)


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class InprocBus:
    """Thread-safe in-process pub/sub for tests and the sim-only path."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Any]] = {}
        self._lock = threading.Lock()

    def publish(self, msg: Message) -> None:
        with self._lock:
            queues = list(self._subs.get(msg.topic, []))
        for q in queues:
            q.put_nowait(msg)

    def subscribe(self, *topics: str) -> Iterator[Message]:
        import queue

        q: queue.Queue[Message] = queue.Queue()
        with self._lock:
            for t in topics:
                self._subs.setdefault(t, []).append(q)
        while True:
            yield q.get()


class ZmqBus:
    """ZeroMQ pub/sub against the supervisor's forwarder (plan §7).

    Each service constructs one ZmqBus, connecting its PUB to the forwarder's XSUB
    and (lazily, on first subscribe) its SUB to the forwarder's XPUB.
    """

    def __init__(self, pub_endpoint: str, sub_endpoint: str) -> None:
        import zmq

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._pub_endpoint = pub_endpoint
        self._sub_endpoint = sub_endpoint
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.connect(pub_endpoint)

    def publish(self, msg: Message) -> None:
        # Frame 0 = topic (utf-8) for prefix matching; frame 1 = msgpack body.
        self._pub.send_multipart([msg.topic.encode(), pack(msg)])

    def subscribe(self, *topics: str) -> Iterator[Message]:
        sub = self._ctx.socket(self._zmq.SUB)
        sub.connect(self._sub_endpoint)
        for t in topics:
            sub.setsockopt(self._zmq.SUBSCRIBE, t.encode())
        try:
            while True:
                topic_b, body = sub.recv_multipart()
                yield unpack(topic_b.decode(), body)
        finally:
            sub.close(linger=0)

    def close(self) -> None:
        self._pub.close(linger=0)


def run_forwarder(xsub_endpoint: str, xpub_endpoint: str) -> None:
    """Blocking XSUB/XPUB proxy that fans publishers out to subscribers.

    `xsub_endpoint` is where publishers connect (cmd/pub side); `xpub_endpoint` is
    where subscribers connect. The supervisor runs this in its own thread/process.
    """
    import zmq

    ctx = zmq.Context.instance()
    xsub = ctx.socket(zmq.XSUB)
    xsub.bind(xsub_endpoint)
    xpub = ctx.socket(zmq.XPUB)
    xpub.bind(xpub_endpoint)
    try:
        zmq.proxy(xsub, xpub)
    finally:
        xsub.close(linger=0)
        xpub.close(linger=0)
