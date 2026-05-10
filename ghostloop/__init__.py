"""ghostloop — the agent loop, embodied.

An agent runtime + safety policy pipeline + sim-first execution harness for
embodied AI. A high-level model (LLM, VLA, scripted policy) emits intents like
``move_to``, ``pick``, ``place``, ``scan``. The runtime maps each intent to a
primitive, runs it through a fail-closed policy pipeline (geofence / force cap
/ rate limit / human-in-the-loop), dispatches to a backend (MuJoCo sim today,
PyBullet next, real hardware via ROS 2 later), traces every step, and replays.

Public surface:

  Intent           the structured high-level command emitted by the policy
  Primitive        a callable backend-agnostic action (e.g. move_to(target))
  PrimitiveRegistry the lookup table of intent name -> Primitive
  PolicyGate       a single safety check; gates compose into a Pipeline
  PolicyPipeline   ordered list of gates, fail-closed, returns a Decision
  Backend          execution adapter (Sim, Mock, future ROS2/serial)
  Runtime          orchestrates policy -> safety -> backend -> trace
  Trace            structured event log of one episode, JSON-serialisable

Designed so a new robot is two files: a ``Backend`` and a registry of
``Primitive`` instances bound to that backend's actuators.
"""

from .async_runtime import AsyncPolicyGate, AsyncPolicyPipeline, AsyncRuntime
from .core import (
    Backend,
    Decision,
    Intent,
    MockBackend,
    PolicyGate,
    PolicyPipeline,
    Primitive,
    PrimitiveRegistry,
    Result,
    Runtime,
    Trace,
    TraceEvent,
)
from .observations import ObservationBuffer, ObservationRecord
from .store import EpisodeRow, GhostloopStore, RunRow

__version__ = "0.7.0"

__all__ = [
    "AsyncPolicyGate",
    "AsyncPolicyPipeline",
    "AsyncRuntime",
    "Backend",
    "Decision",
    "EpisodeRow",
    "GhostloopStore",
    "Intent",
    "MockBackend",
    "ObservationBuffer",
    "ObservationRecord",
    "PolicyGate",
    "PolicyPipeline",
    "Primitive",
    "PrimitiveRegistry",
    "Result",
    "RunRow",
    "Runtime",
    "Trace",
    "TraceEvent",
    "__version__",
]
