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

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "Decision",
    "Intent",
    "MockBackend",
    "PolicyGate",
    "PolicyPipeline",
    "Primitive",
    "PrimitiveRegistry",
    "Result",
    "Runtime",
    "Trace",
    "TraceEvent",
    "__version__",
]
