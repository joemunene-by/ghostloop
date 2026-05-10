"""Temporal property combinators (Signal Temporal Logic).

The properties shipped in ``builtins.py`` evaluate every event in
isolation — point-in-time invariants like "position stays inside the
workspace box". Real safety claims need TIME:

  - "force never exceeds N for more than 100ms" (windowed Always)
  - "within 5 seconds of detecting an obstacle, robot decelerates"
    (Eventually conditioned on a precursor)
  - "after a HITL approval, the next motion intent runs within 10s"
    (Until — pending state until trigger arrives)

This module adds three temporal operators that operate on a sliding
window of events, defined by ``window_s`` seconds. Each takes either
an ``EventPredicate`` (callable ``(TraceEvent) -> bool``) or another
Property and combines via standard STL semantics:

    Always(P, window_s)        — every event in the window satisfies P
    Eventually(P, window_s)    — at least one event in the window satisfies P
    Until(P, Q, window_s)      — P holds until Q within the window

These return ``Property`` objects so they integrate with the existing
``PropertyEngine`` and combine with ``And`` / ``Or`` / ``Not`` from
``combinators.py`` to express arbitrarily nuanced safety claims.

Built on stdlib only — no numpy / scipy. Each evaluator slides a deque
of pending events through the trace and decides held / violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core import Trace, TraceEvent
from .core import Property, PropertyResult, Severity


EventPredicate = Callable[[TraceEvent], bool]


def _eval_predicate(p: EventPredicate | Property, ev: TraceEvent) -> bool:
    """Check a single event against either a callable predicate or a Property.

    Properties usually evaluate the whole trace at once, so for windowed
    use we wrap the single-event case as a one-event Trace and let the
    Property check it.
    """
    if callable(p) and not hasattr(p, "check"):
        return bool(p(ev))
    # It's a Property — wrap the event.
    sub_trace = Trace(events=[ev])
    return p.check(sub_trace).held


# ---------------------------------------------------------------------------
# Always(phi, window_s) — phi must hold at every event in the last window.
# ---------------------------------------------------------------------------


@dataclass
class Always:
    """STL G[0,W] phi — always within a window of W seconds, phi holds.

    Slides a window of events whose timestamps lie within the last
    ``window_s``. If any event in the window fails ``phi``, log a
    violation and the property is NOT held overall. With ``window_s=0``
    this is a "for every event" property — equivalent to evaluating
    phi on each event independently.
    """

    phi: EventPredicate | Property
    window_s: float = 0.0
    name: str = "always"
    description: str = ""
    severity: Severity = Severity.ERROR

    def check(self, trace: Trace) -> PropertyResult:
        violations: list[dict[str, Any]] = []
        # Sliding window — for each event, check that phi held for every
        # event whose timestamp is within window_s of the current one.
        for i, ev in enumerate(trace.events):
            cutoff = ev.timestamp - self.window_s
            for past in trace.events[: i + 1]:
                if past.timestamp < cutoff:
                    continue
                if not _eval_predicate(self.phi, past):
                    violations.append({
                        "step": past.step,
                        "timestamp": past.timestamp,
                        "reason": (
                            f"phi failed at step {past.step} "
                            f"(within last {self.window_s:g}s of step {ev.step})"
                        ),
                    })
                    break  # one violation per anchor event is enough
        return PropertyResult(
            name=self.name,
            severity=self.severity,
            held=len(violations) == 0,
            description=self.description or f"Always (window={self.window_s:g}s)",
            violations=violations,
        )


# ---------------------------------------------------------------------------
# Eventually(phi, window_s) — phi must hold at least once in the last window.
# ---------------------------------------------------------------------------


@dataclass
class Eventually:
    """STL F[0,W] phi — eventually within a window of W seconds, phi holds.

    For each event, checks that AT LEAST ONE event in the past
    ``window_s`` (including the current event) satisfies ``phi``.
    Useful for "responsiveness" claims: 'after a brake intent the
    actual velocity must drop within 1s'.

    With ``window_s == 0`` this is "phi holds at the current event" —
    same as Always with window_s=0.
    """

    phi: EventPredicate | Property
    window_s: float = float("inf")
    name: str = "eventually"
    description: str = ""
    severity: Severity = Severity.WARN

    def check(self, trace: Trace) -> PropertyResult:
        violations: list[dict[str, Any]] = []
        for i, ev in enumerate(trace.events):
            cutoff = ev.timestamp - self.window_s
            window = [
                past for past in trace.events[: i + 1]
                if past.timestamp >= cutoff
            ]
            if not any(_eval_predicate(self.phi, past) for past in window):
                violations.append({
                    "step": ev.step,
                    "timestamp": ev.timestamp,
                    "reason": (
                        f"phi did not hold within last "
                        f"{self.window_s:g}s ending at step {ev.step}"
                    ),
                })
        return PropertyResult(
            name=self.name,
            severity=self.severity,
            held=len(violations) == 0,
            description=self.description or f"Eventually (window={self.window_s:g}s)",
            violations=violations,
        )


# ---------------------------------------------------------------------------
# Until(phi, psi, window_s) — phi holds until psi happens, all within window.
# ---------------------------------------------------------------------------


@dataclass
class Until:
    """STL phi U[0,W] psi — phi holds at every step until psi triggers.

    For each "anchor" event ev_i, looks at events ev_i, ev_{i+1}, ...
    within ``window_s`` seconds. The property holds for that anchor iff
    phi is true at every step BEFORE the first step where psi is true,
    AND psi becomes true within the window. Violations record the
    earliest anchor where the chain breaks.

    Use case: "after motion authorisation, every step must be a
    dispatched move (phi) until a 'goal_reached' event (psi), within
    30 seconds".
    """

    phi: EventPredicate | Property
    psi: EventPredicate | Property
    window_s: float = float("inf")
    name: str = "until"
    description: str = ""
    severity: Severity = Severity.WARN

    def check(self, trace: Trace) -> PropertyResult:
        violations: list[dict[str, Any]] = []
        for i, anchor in enumerate(trace.events):
            cutoff = anchor.timestamp + self.window_s
            psi_seen = False
            phi_failed_step: int | None = None
            for past in trace.events[i:]:
                if past.timestamp > cutoff:
                    break
                if _eval_predicate(self.psi, past):
                    psi_seen = True
                    break
                if not _eval_predicate(self.phi, past):
                    phi_failed_step = past.step
                    break
            if phi_failed_step is not None:
                violations.append({
                    "step": phi_failed_step,
                    "anchor_step": anchor.step,
                    "reason": (
                        f"phi failed at step {phi_failed_step} before psi "
                        f"(anchor step {anchor.step})"
                    ),
                })
            elif not psi_seen:
                violations.append({
                    "step": anchor.step,
                    "reason": (
                        f"psi never observed within {self.window_s:g}s of "
                        f"anchor step {anchor.step}"
                    ),
                })
        return PropertyResult(
            name=self.name,
            severity=self.severity,
            held=len(violations) == 0,
            description=self.description or f"Until (window={self.window_s:g}s)",
            violations=violations,
        )


# ---------------------------------------------------------------------------
# Helper predicate constructors — convenient for inline use without writing
# named lambdas. Pre-baked ones cover the common shapes; assemble custom
# predicates by composing these or just writing a callable.
# ---------------------------------------------------------------------------


def intent_named(name: str) -> EventPredicate:
    """Predicate: the event's intent has the given name."""
    def _p(ev: TraceEvent) -> bool:
        return ev.intent.name == name
    _p.__name__ = f"intent_named({name!r})"
    return _p


def decision_action(action: str) -> EventPredicate:
    """Predicate: the event's safety decision has the given action.

    Action is one of ``"allow"`` / ``"deny"`` / ``"escalate"``.
    """
    def _p(ev: TraceEvent) -> bool:
        return ev.decision.action.value == action
    _p.__name__ = f"decision_action({action!r})"
    return _p


def result_status(status: str) -> EventPredicate:
    """Predicate: the event's result has the given status (``ok`` / ``error`` / etc.)."""
    def _p(ev: TraceEvent) -> bool:
        return ev.result.status.value == status
    _p.__name__ = f"result_status({status!r})"
    return _p


def state_field_below(field_path: str, threshold: float) -> EventPredicate:
    """Predicate: ``state_after[<dotted-path>] < threshold``.

    Misses (key absent) are treated as not satisfying the predicate —
    i.e. the property does NOT hold. That's the safe default for "force
    must stay below N": absence of a force reading is suspicious.
    """
    parts = field_path.split(".")
    def _p(ev: TraceEvent) -> bool:
        cur: Any = ev.state_after
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return False
        try:
            return float(cur) < threshold
        except (TypeError, ValueError):
            return False
    _p.__name__ = f"state_field_below({field_path!r}, {threshold!r})"
    return _p
