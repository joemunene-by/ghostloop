"""AsyncRuntime — coroutine-friendly runtime for control-loop and network workloads.

The synchronous ``Runtime`` is fine for sim demos, scripted tests, and any
single-shot orchestration. Real deployments hit two cases that need async:

  1. Network gates / approvers: HITL approvers polling Slack, dashboard
     queues, or webhook services. Blocking the robot loop on stdin works
     for dev; production needs awaitable approvers.
  2. Network policies / backends: LLMPolicy hits an HTTP endpoint, and
     real-hardware backends speak via ROS 2 / gRPC / MQTT.

AsyncRuntime mirrors Runtime's surface — same ``Intent`` / ``Primitive`` /
``PolicyPipeline`` / ``Backend`` / ``Trace`` types — but ``step`` is
``async def`` and supports ``AsyncPolicyGate`` protocol gates whose
``check`` method is awaitable. Synchronous gates are wrapped transparently
so an existing PolicyPipeline mixes async and sync gates without changes.

Async primitives are dispatched via ``inspect.iscoroutinefunction`` so a
backend can mix sync (``move_to``) and async (``stream_camera``) primitives
in the same registry.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Iterable, Protocol, Union

from .core import (
    Backend,
    Decision,
    DecisionAction,
    Intent,
    PolicyGate,
    PolicyPipeline,
    Primitive,
    PrimitiveRegistry,
    Result,
    ResultStatus,
    Trace,
    TraceEvent,
)


class AsyncPolicyGate(Protocol):
    """An awaitable safety check.

    Same shape as ``PolicyGate`` but ``check`` returns a coroutine. Pipeline
    semantics are identical: fail-closed on first deny, transparent
    reasoning surfaced to the trace.
    """

    name: str

    async def check(self, intent: Intent, primitive: Primitive) -> Decision: ...


GateLike = Union[PolicyGate, AsyncPolicyGate]


@dataclass
class AsyncPolicyPipeline:
    """Awaitable pipeline mixing sync and async gates.

    A sync gate's ``check()`` is invoked directly (instantaneous); an async
    gate's ``check()`` is awaited. Either kind can short-circuit on the
    first deny.
    """

    gates: list[GateLike] = field(default_factory=list)

    async def check(self, intent: Intent, primitive: Primitive) -> Decision:
        passed: list[str] = []
        for gate in self.gates:
            decision_or_coro = gate.check(intent, primitive)
            if inspect.isawaitable(decision_or_coro):
                decision = await decision_or_coro
            else:
                decision = decision_or_coro
            if decision.action is DecisionAction.DENY:
                return decision
            passed.append(gate.name)
        return Decision.allow(
            gate_name="pipeline",
            reason=f"all {len(self.gates)} gate(s) passed: {', '.join(passed)}"
            if passed else "no gates configured",
        )


class AsyncRuntime:
    """The async sibling of Runtime.

    Same constructor shape, but ``step`` is awaitable and supports both
    sync and async gates / primitives. Wrap an existing ``PolicyPipeline``
    by passing it directly — the AsyncPolicyPipeline adapter handles the
    sync gates transparently.
    """

    def __init__(
        self,
        backend: Backend,
        registry: PrimitiveRegistry,
        policy_pipeline: PolicyPipeline | AsyncPolicyPipeline | None = None,
        trace: Trace | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        if isinstance(policy_pipeline, AsyncPolicyPipeline):
            self.policy_pipeline = policy_pipeline
        else:
            sync_pipe = policy_pipeline or PolicyPipeline()
            self.policy_pipeline = AsyncPolicyPipeline(gates=list(sync_pipe.gates))
        self.trace = trace or Trace(backend_name=backend.name)
        self._step = 0

    async def step(self, intent: Intent) -> Result:
        self._step += 1
        state_before = self.backend.snapshot()
        primitive = self.registry.get(intent.name)
        if primitive is None:
            decision = Decision.deny(
                gate_name="resolver",
                reason=f"unknown primitive: {intent.name!r}",
            )
            result = Result(status=ResultStatus.BLOCKED, message=decision.reason)
            self._record(intent, decision, result, state_before)
            return result

        decision = await self.policy_pipeline.check(intent, primitive)
        if decision.action is DecisionAction.DENY:
            result = Result(
                status=ResultStatus.BLOCKED,
                message=f"{decision.gate_name}: {decision.reason}",
            )
            self._record(intent, decision, result, state_before)
            return result

        started = time.monotonic()
        try:
            output = primitive.call(self.backend, **intent.args)
            if inspect.isawaitable(output):
                result = await output
            else:
                result = output
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

    async def run(self, intents: Iterable[Intent]) -> list[Result]:
        return [await self.step(i) for i in intents]

    async def control_loop(
        self,
        next_intent: "callable[[AsyncRuntime, Result | None], Awaitable[Intent | None]]",
        max_steps: int = 1000,
        rate_hz: float | None = None,
    ) -> int:
        """Run a closed control loop driven by ``next_intent``.

        ``next_intent`` is awaitable: it receives the runtime + the previous
        Result (or None on the first call) and returns the next Intent (or
        None to terminate the loop). Optional ``rate_hz`` paces the loop;
        the runtime sleeps to maintain target frequency, accounting for the
        time spent in step + decision.
        """
        period = (1.0 / rate_hz) if rate_hz else 0.0
        last_result: Result | None = None
        steps = 0
        for _ in range(max_steps):
            tick_start = time.monotonic()
            next_obj = next_intent(self, last_result)
            intent = await next_obj if inspect.isawaitable(next_obj) else next_obj
            if intent is None:
                return steps
            last_result = await self.step(intent)
            steps += 1
            if period:
                elapsed = time.monotonic() - tick_start
                sleep_for = period - elapsed
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
        return steps

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
