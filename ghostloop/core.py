"""Core abstractions for the ghostloop runtime.

The data flow per step:

  Policy emits Intent  ->  Runtime resolves Primitive from registry  ->
  PolicyPipeline gates the Primitive call (fail-closed)  ->  Backend executes
  the Primitive  ->  Runtime appends a TraceEvent.

Everything is JSON-serialisable so episodes can be persisted, replayed, or
shipped to a fleet dashboard. There is no global state — pass a Runtime around.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Protocol


# ---------------------------------------------------------------------------
# Intent — what a policy says it wants the robot to do.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intent:
    """A high-level structured command emitted by a policy.

    The ``name`` is looked up in a PrimitiveRegistry. ``args`` is whatever
    keyword arguments the resolved Primitive accepts. ``rationale`` is the
    free-text explanation the policy emitted (useful for traces, audits,
    and replay) — empty string is fine if the policy doesn't produce one.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args, "rationale": self.rationale}


# ---------------------------------------------------------------------------
# Primitive — a backend-bound action.
# ---------------------------------------------------------------------------


@dataclass
class Primitive:
    """A named action a Backend can execute.

    The ``call`` callable receives ``(backend, **args)`` and returns a
    ``Result``. ``description`` is human-readable for tool-card listings (the
    same shape an LLM tool-call schema would consume).
    """

    name: str
    call: Callable[..., "Result"]
    description: str = ""
    arg_schema: dict[str, str] = field(default_factory=dict)


class PrimitiveRegistry:
    """Name -> Primitive lookup the Runtime uses to resolve Intents."""

    def __init__(self, primitives: Iterable[Primitive] = ()) -> None:
        self._by_name: dict[str, Primitive] = {}
        for p in primitives:
            self.register(p)

    def register(self, primitive: Primitive) -> None:
        if primitive.name in self._by_name:
            raise ValueError(f"primitive already registered: {primitive.name}")
        self._by_name[primitive.name] = primitive

    def get(self, name: str) -> Primitive | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return sorted(self._by_name.keys())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name


# ---------------------------------------------------------------------------
# Result — what executing a Primitive returns.
# ---------------------------------------------------------------------------


class ResultStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"  # rejected by the policy pipeline before execution
    TIMEOUT = "timeout"


@dataclass
class Result:
    """Outcome of one Primitive execution (or one blocked attempt)."""

    status: ResultStatus
    observation: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    duration_ms: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "observation": self.observation,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 3),
        }

    @property
    def ok(self) -> bool:
        return self.status is ResultStatus.OK


# ---------------------------------------------------------------------------
# PolicyGate / PolicyPipeline — fail-closed safety pipeline.
# ---------------------------------------------------------------------------


class DecisionAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class Decision:
    """A single PolicyGate's verdict on one Intent.

    ``action`` is allow/deny; pipeline semantics: if ANY gate denies, the
    pipeline as a whole denies (fail-closed). ``reason`` explains why,
    surfaces in the trace, and is what an HITL reviewer or auditor reads.
    """

    action: DecisionAction
    reason: str = ""
    gate_name: str = ""

    @classmethod
    def allow(cls, gate_name: str = "", reason: str = "") -> "Decision":
        return cls(DecisionAction.ALLOW, reason, gate_name)

    @classmethod
    def deny(cls, gate_name: str, reason: str) -> "Decision":
        return cls(DecisionAction.DENY, reason, gate_name)

    def to_json(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "gate_name": self.gate_name,
        }


class PolicyGate(Protocol):
    """A single safety check.

    Implementations receive the Intent + Primitive and return a Decision.
    Gates should be cheap and side-effect-free where possible — they run
    on every step. Examples that ship in v0.1: rate limit, per-primitive
    deny-list, geofence (when 3D arg detected).
    """

    name: str

    def check(self, intent: Intent, primitive: Primitive) -> Decision: ...


@dataclass
class PolicyPipeline:
    """Ordered list of gates with fail-closed semantics.

    On the first DENY, the pipeline stops and returns that decision. If every
    gate ALLOWs, the pipeline returns a synthesised ALLOW with the names of
    all gates that passed (for trace transparency).
    """

    gates: list[PolicyGate] = field(default_factory=list)

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        passed: list[str] = []
        for gate in self.gates:
            decision = gate.check(intent, primitive)
            if decision.action is DecisionAction.DENY:
                return decision
            passed.append(gate.name)
        return Decision.allow(
            gate_name="pipeline",
            reason=f"all {len(self.gates)} gate(s) passed: {', '.join(passed)}"
            if passed
            else "no gates configured",
        )


# ---------------------------------------------------------------------------
# Backend — execution adapter.
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """A robot or simulator the Runtime can target.

    The minimum surface is just (a) a name, and (b) a state snapshot for
    traces. Primitives close over a specific Backend instance via their
    ``call`` callable, so this Protocol stays deliberately small — backends
    don't have to know what primitives exist. v0.1 ships MockBackend; the
    next backends to land are MuJoCoBackend (sim) and PyBulletBackend.
    """

    name: str

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of current backend state."""
        ...


class MockBackend:
    """An in-memory backend used by tests, examples, and the bench harness.

    Tracks a 3D position (the only state worth modelling at this layer);
    everything is deterministic so traces are reproducible across runs.
    """

    def __init__(self, name: str = "mock", initial: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.name = name
        self.position: tuple[float, float, float] = initial
        self.held_object: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "position": list(self.position),
            "held_object": self.held_object,
        }


# ---------------------------------------------------------------------------
# Trace — JSON-serialisable per-episode event log.
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    """One step of one episode: intent, decision, result, before/after state."""

    step: int
    intent: Intent
    decision: Decision
    result: Result
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    timestamp: float

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "intent": self.intent.to_json(),
            "decision": self.decision.to_json(),
            "result": self.result.to_json(),
            "state_before": self.state_before,
            "state_after": self.state_after,
        }


@dataclass
class Trace:
    """Full episode trace. Append-only; serialise via ``to_json``."""

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    backend_name: str = ""
    started_at: float = field(default_factory=time.time)
    events: list[TraceEvent] = field(default_factory=list)

    def append(self, event: TraceEvent) -> None:
        self.events.append(event)

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "backend": self.backend_name,
            "started_at": self.started_at,
            "n_steps": len(self.events),
            "events": [e.to_json() for e in self.events],
        }

    def write_jsonl(self, path: str) -> None:
        """One JSON object per event, plus a header object first."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "episode_id": self.episode_id,
                "backend": self.backend_name,
                "started_at": self.started_at,
                "n_steps": len(self.events),
            }) + "\n")
            for ev in self.events:
                f.write(json.dumps(ev.to_json()) + "\n")


# ---------------------------------------------------------------------------
# Runtime — orchestrator.
# ---------------------------------------------------------------------------


class Runtime:
    """Glue between the policy that emits intents and the backend that runs them.

    Usage::

        runtime = Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([move_to_primitive(backend)]),
            policy_pipeline=PolicyPipeline(gates=[RateLimitGate(per_minute=120)]),
        )
        result = runtime.step(Intent("move_to", {"x": 1.0, "y": 0.0, "z": 0.0}))

    ``step`` runs one intent end-to-end: resolve -> gate -> execute -> trace.
    Returns the Result; the trace is also appended to ``runtime.trace`` so
    callers can inspect, persist, or stream it.
    """

    def __init__(
        self,
        backend: Backend,
        registry: PrimitiveRegistry,
        policy_pipeline: PolicyPipeline | None = None,
        trace: Trace | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.policy_pipeline = policy_pipeline or PolicyPipeline()
        self.trace = trace or Trace(backend_name=backend.name)
        self._step = 0

    def step(self, intent: Intent) -> Result:
        self._step += 1
        state_before = self.backend.snapshot()
        primitive = self.registry.get(intent.name)
        if primitive is None:
            decision = Decision.deny(
                gate_name="resolver",
                reason=f"unknown primitive: {intent.name!r}",
            )
            result = Result(
                status=ResultStatus.BLOCKED,
                message=decision.reason,
            )
            self._record(intent, decision, result, state_before)
            return result

        decision = self.policy_pipeline.check(intent, primitive)
        if decision.action is DecisionAction.DENY:
            result = Result(
                status=ResultStatus.BLOCKED,
                message=f"{decision.gate_name}: {decision.reason}",
            )
            self._record(intent, decision, result, state_before)
            return result

        started = time.monotonic()
        try:
            result = primitive.call(self.backend, **intent.args)
        except Exception as exc:  # noqa: BLE001
            result = Result(
                status=ResultStatus.ERROR,
                message=f"{type(exc).__name__}: {exc}",
            )
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if not result.duration_ms:
            result.duration_ms = elapsed_ms
        self._record(intent, decision, result, state_before)
        return result

    def run(self, intents: Iterable[Intent]) -> list[Result]:
        return [self.step(i) for i in intents]

    def _record(
        self,
        intent: Intent,
        decision: Decision,
        result: Result,
        state_before: dict[str, Any],
    ) -> None:
        self.trace.append(
            TraceEvent(
                step=self._step,
                intent=intent,
                decision=decision,
                result=result,
                state_before=state_before,
                state_after=self.backend.snapshot(),
                timestamp=time.time(),
            )
        )
