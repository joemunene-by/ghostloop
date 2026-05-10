"""Live policy intervention — pause, swap, resume the runtime without restart.

When something goes wrong in production — a property fires, an alarm
trips, a human notices the robot is doing the wrong thing — the
fastest recovery isn't a process restart. It's an in-place policy
swap: replace the current policy with a known-safe fallback (or with
a manually-controlled teleop policy), then resume.

This module provides the runtime-side hook: a wrapper around any
policy callable that supports atomic pause / swap / resume from a
control thread, with the dashboard's alarm bus + the property engine
as natural drivers.

Three primitives:

  - ``InterventionState``: PAUSED / RUNNING / SWAPPING enum.
  - ``LivePolicyController``: wraps a policy callable, exposes
    ``act()`` for the runtime to call AND control methods (``pause()``,
    ``resume()``, ``swap_to(new_policy)``) for ops.
  - ``intervention_gate(controller)``: PolicyGate that denies
    primitive dispatch while the controller is PAUSED. Drop it in the
    safety pipeline; ops triggering pause now stops every command in
    flight.

Thread-safe (``threading.Lock``); not async-runtime-aware. For an async
controller, mirror the same shape with ``asyncio.Lock``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .core import Decision, Intent, Primitive


class InterventionState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    SWAPPING = "swapping"
    EMERGENCY_STOP = "emergency_stop"


PolicyFn = Callable[[Any], Intent]   # state -> Intent


@dataclass
class InterventionEvent:
    """One control-plane event for audit trails."""

    timestamp: float
    state_before: InterventionState
    state_after: InterventionState
    operator: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LivePolicyController:
    """Wrap a policy callable with pause / resume / swap controls.

    The runtime's loop calls ``controller.act(state) -> Intent``. Out
    of band, an operator (or an alarm-driven worker) calls
    ``controller.pause(...)`` or ``controller.swap_to(new_policy)``;
    those changes take effect on the next ``act()`` call.

    A short-circuit fallback policy can be configured for PAUSED state
    so the runtime doesn't block — useful when the policy must keep
    issuing safe-fallback intents (e.g. ``stop`` for a mobile base) to
    keep the robot's controller alive.
    """

    policy: PolicyFn
    fallback_policy: PolicyFn | None = None
    name: str = "live_policy"

    _state: InterventionState = field(default=InterventionState.RUNNING, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _history: list[InterventionEvent] = field(default_factory=list, init=False)
    _stuck_intent: Intent | None = field(default=None, init=False)

    @property
    def state(self) -> InterventionState:
        with self._lock:
            return self._state

    def act(self, state: Any) -> Intent:
        """Get the next Intent. Honours pause / fallback / swap state."""
        with self._lock:
            cur = self._state
            current_policy = self.policy
            stuck = self._stuck_intent
            fallback = self.fallback_policy
        if cur is InterventionState.EMERGENCY_STOP:
            # Single sticky intent — usually 'stop' or 'land' or similar.
            if stuck is not None:
                return stuck
            return Intent("stop", {}, rationale="emergency_stop")
        if cur is InterventionState.PAUSED:
            if fallback is not None:
                return fallback(state)
            # No fallback configured; emit an emit_event so the runtime
            # has something to dispatch (gates may deny; that's fine).
            return Intent(
                "emit_event",
                {"kind": "paused", "message": "policy paused; waiting for resume"},
                rationale="intervention.paused",
            )
        # SWAPPING and RUNNING both call the active policy. SWAPPING is a
        # transient state during the atomic swap; act() blocks behind the
        # lock briefly so callers see the new policy on the next call.
        return current_policy(state)

    def pause(self, *, operator: str = "operator", reason: str = "") -> None:
        self._transition(InterventionState.PAUSED, operator=operator, reason=reason)

    def resume(self, *, operator: str = "operator", reason: str = "") -> None:
        self._transition(InterventionState.RUNNING, operator=operator, reason=reason)

    def emergency_stop(
        self,
        stop_intent: Intent | None = None,
        *,
        operator: str = "operator",
        reason: str = "emergency_stop",
    ) -> None:
        """Latch the controller into a sticky-intent state.

        While in EMERGENCY_STOP, ``act()`` returns ``stop_intent``
        regardless of state. Operator must call ``resume()`` explicitly
        to leave this state.
        """
        with self._lock:
            self._stuck_intent = stop_intent or Intent("stop", {}, rationale=reason)
        self._transition(
            InterventionState.EMERGENCY_STOP, operator=operator, reason=reason,
        )

    def swap_to(
        self, new_policy: PolicyFn,
        *,
        operator: str = "operator",
        reason: str = "swap",
    ) -> None:
        """Replace the active policy atomically. Resumes if previously paused."""
        with self._lock:
            self._state = InterventionState.SWAPPING
            self.policy = new_policy
            self._stuck_intent = None
        self._transition(
            InterventionState.RUNNING, operator=operator, reason=f"swap: {reason}",
        )

    def _transition(
        self,
        new_state: InterventionState,
        *,
        operator: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
        self._history.append(InterventionEvent(
            timestamp=time.time(),
            state_before=old,
            state_after=new_state,
            operator=operator,
            reason=reason,
            metadata=metadata or {},
        ))

    def history(self, limit: int = 50) -> list[InterventionEvent]:
        return list(self._history)[-limit:]


@dataclass
class InterventionGate:
    """PolicyGate that denies dispatch while the controller is PAUSED.

    Drop in the pipeline AFTER cheap gates so the deny reason mentions
    the operator + reason from the most recent intervention. EMERGENCY_STOP
    state passes through (the controller's ``act`` already returns the
    sticky stop intent, and we want it actually dispatched).
    """

    controller: LivePolicyController
    name: str = "intervention"

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        state = self.controller.state
        if state in (InterventionState.RUNNING, InterventionState.SWAPPING):
            return Decision.allow(self.name, f"controller {state.value}")
        if state is InterventionState.EMERGENCY_STOP:
            # The stop intent IS the safe action; let it through.
            if intent.name in ("stop", "land", "lie_down", "emit_event"):
                return Decision.allow(
                    self.name,
                    f"emergency_stop in effect; {intent.name} permitted as safe action",
                )
            return Decision.deny(
                self.name,
                f"controller in EMERGENCY_STOP; only stop/land/lie_down/emit_event permitted",
            )
        # PAUSED.
        last = self.controller.history(limit=1)
        reason_suffix = ""
        if last:
            reason_suffix = f" (paused by {last[0].operator}: {last[0].reason})"
        return Decision.deny(self.name, f"controller PAUSED{reason_suffix}")
