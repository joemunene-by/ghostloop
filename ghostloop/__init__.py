"""ghostloop — the agent loop, embodied.

An agent runtime + safety pipeline + sim-first execution harness +
statistically-rigorous bench harness + post-hoc analysis layer for
embodied AI. A high-level model (LLM, VLA, scripted policy) emits
intents like ``move_to``, ``pick``, ``place``, ``scan``. The runtime
maps each intent to a primitive, runs it through a fail-closed policy
pipeline (geofence / force cap / rate limit / cooldown / time window
/ smoothing / human-in-the-loop), dispatches to a backend (Mock,
MuJoCo, PyBullet, Gymnasium, ROS 2, or RandomizedBackend wrapping any
of the above), and traces every step.

Then on top of that:

  - ``ghostloop.bench`` — Wilson CI / McNemar / Cohen's h paired
    comparisons + Sim2RealBench transfer-gap harness + adversarial
    fuzzing (random / grid / CMA-ES).
  - ``ghostloop.properties`` — declarative invariants with STL
    (Always / Eventually / Until) + auto-mining from a trace corpus.
  - ``ghostloop.judges`` — LLM-as-judge + heuristic rule-based trace
    scoring against a configurable rubric.
  - ``ghostloop.counterfactual`` — replay a trace through a different
    policy, diff their decisions step by step.
  - ``ghostloop.causal`` — leave-one-out attribution: which events
    were causally necessary for a property violation?
  - ``ghostloop.training`` — Constrained-MDP rollout collector +
    Lagrangian multiplier + HER relabeling for goal-conditioned
    policies.
  - ``ghostloop.fleet`` + ``ghostloop.dashboard`` — multi-robot
    abstractions + FastAPI HTTP surface + WebSocket trace streaming.

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

__version__ = "1.0.1"

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
