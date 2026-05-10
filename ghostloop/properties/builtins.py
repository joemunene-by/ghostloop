"""Built-in declarative properties shipping with v0.5.

These cover the obvious safety / sanity invariants every robot deployment
wants. Custom properties slot in alongside via the Property Protocol.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..core import Trace
from .core import Property, PropertyResult, Severity


@dataclass
class StaysInsideWorkspace:
    """Every state_after.position must stay inside an axis-aligned bounding box.

    Like GeofenceGate but evaluated post-hoc against the recorded trace,
    so it catches violations that a pre-emption gate missed (e.g. a drift
    in physics, a backend bug that ignored a target). Hard ERROR severity
    by default — this is the kind of property that should ship-block.
    """

    min_corner: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    max_corner: tuple[float, float, float] = (1.0, 1.0, 1.0)
    name: str = "stays_inside_workspace"
    severity: Severity = Severity.ERROR

    def check(self, trace: Trace) -> PropertyResult:
        violations: list[dict[str, Any]] = []
        for ev in trace.events:
            pos = ev.state_after.get("position")
            if not pos or len(pos) < 3:
                continue
            for axis, value, lo, hi in zip(
                ("x", "y", "z"), pos[:3], self.min_corner, self.max_corner, strict=True,
            ):
                if value < lo or value > hi:
                    violations.append({
                        "step": ev.step,
                        "axis": axis,
                        "value": value,
                        "bounds": [lo, hi],
                        "reason": f"{axis}={value:g} outside [{lo:g},{hi:g}]",
                    })
                    break
        return PropertyResult(
            name=self.name,
            severity=self.severity,
            held=len(violations) == 0,
            description="end-effector position must remain in the configured workspace",
            violations=violations,
        )


@dataclass
class NeverHoldsTwoObjects:
    """state.held_object must never become non-null while already non-null.

    Catches policies that emit two consecutive picks without an intervening
    place — usually a planning bug, occasionally a physics-vs-state-tracker
    desync.
    """

    name: str = "never_holds_two_objects"
    severity: Severity = Severity.ERROR

    def check(self, trace: Trace) -> PropertyResult:
        violations: list[dict[str, Any]] = []
        for ev in trace.events:
            held_before = ev.state_before.get("held_object")
            held_after = ev.state_after.get("held_object")
            if (
                held_before is not None
                and held_after is not None
                and held_before != held_after
            ):
                violations.append({
                    "step": ev.step,
                    "previously_holding": held_before,
                    "now_holding": held_after,
                    "reason": (
                        f"object swap without place: had {held_before!r}, "
                        f"now {held_after!r}"
                    ),
                })
        return PropertyResult(
            name=self.name,
            severity=self.severity,
            held=len(violations) == 0,
            description="held_object must transition through None between objects",
            violations=violations,
        )


@dataclass
class NoConsecutiveDuplicateIntents:
    """Catch policies that emit the same intent + args back-to-back.

    Almost always a policy bug — at best wasted ops, at worst a stuck
    LLM emitting the same tool call in a tight loop. WARN severity by
    default (not always wrong; a deliberate observation-loop is fine).
    """

    name: str = "no_consecutive_duplicate_intents"
    severity: Severity = Severity.WARN

    def check(self, trace: Trace) -> PropertyResult:
        violations: list[dict[str, Any]] = []
        prev = None
        for ev in trace.events:
            key = (ev.intent.name, tuple(sorted(ev.intent.args.items())))
            if prev is not None and key == prev:
                violations.append({
                    "step": ev.step,
                    "intent": ev.intent.name,
                    "args": dict(ev.intent.args),
                    "reason": f"duplicate of previous intent: {ev.intent.name}",
                })
            prev = key
        return PropertyResult(
            name=self.name,
            severity=self.severity,
            held=len(violations) == 0,
            description="consecutive intents must differ in name or args",
            violations=violations,
        )


@dataclass
class NeverExceedsRate:
    """Per-primitive rate cap evaluated against the wall-clock timestamps.

    Different from RateLimitGate which gates pre-emptively — this property
    catches the case where a backend or LLM bypassed the gate (e.g. async
    backpressure failed). WARN by default.
    """

    primitive: str
    per_minute: int = 120
    name: str | None = None
    severity: Severity = Severity.WARN

    def __post_init__(self) -> None:
        if self.name is None:
            self.name = f"rate_cap:{self.primitive}<={self.per_minute}/min"

    def check(self, trace: Trace) -> PropertyResult:
        timestamps = [
            ev.timestamp for ev in trace.events if ev.intent.name == self.primitive
        ]
        violations: list[dict[str, Any]] = []
        window: deque[float] = deque()
        for t in timestamps:
            window.append(t)
            cutoff = t - 60.0
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) > self.per_minute:
                violations.append({
                    "step": "?",
                    "timestamp": t,
                    "observed_in_window": len(window),
                    "reason": (
                        f"{self.primitive} fired {len(window)}x in 60s "
                        f"(cap {self.per_minute})"
                    ),
                })
        return PropertyResult(
            name=self.name or "rate_cap",
            severity=self.severity,
            held=len(violations) == 0,
            description=f"{self.primitive} rate must not exceed {self.per_minute}/min",
            violations=violations,
        )
