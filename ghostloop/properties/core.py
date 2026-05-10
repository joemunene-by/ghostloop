"""Property protocol + PropertyEngine that scores a Trace against a list of properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Protocol

from ..core import Trace


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"  # ship-blocking


@dataclass
class PropertyResult:
    """Outcome of evaluating one property against one trace."""

    name: str
    held: bool
    severity: Severity
    description: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "held": self.held,
            "severity": self.severity.value,
            "description": self.description,
            "violations": self.violations,
        }


class Property(Protocol):
    """A declarative invariant over a Trace.

    Implementations inspect every event in the trace and decide whether
    the invariant held. Pure function — no side effects, no state shared
    between evaluations.
    """

    name: str
    severity: Severity

    def check(self, trace: Trace) -> PropertyResult: ...


@dataclass
class PropertyEngine:
    """Evaluate a list of Properties against a Trace.

    Returns one PropertyResult per Property plus a summary tuple
    (total, held, violated, error_violations).
    """

    properties: list[Property] = field(default_factory=list)

    def evaluate(self, trace: Trace) -> list[PropertyResult]:
        return [p.check(trace) for p in self.properties]

    def evaluate_summary(self, trace: Trace) -> dict[str, Any]:
        results = self.evaluate(trace)
        held = sum(1 for r in results if r.held)
        violated = sum(1 for r in results if not r.held)
        error_violations = sum(
            1 for r in results if not r.held and r.severity is Severity.ERROR
        )
        return {
            "n_properties": len(results),
            "held": held,
            "violated": violated,
            "error_violations": error_violations,
            "ship_blocked": error_violations > 0,
            "results": [r.to_json() for r in results],
        }

    def render_md(self, trace: Trace) -> str:
        results = self.evaluate(trace)
        lines = [
            "# Property evaluation",
            "",
            f"Trace: `{trace.episode_id}` ({len(trace.events)} events)",
            "",
            "| Property | Severity | Held | Violations |",
            "|---|---|---|---:|",
        ]
        for r in results:
            check = "✓" if r.held else "✗"
            lines.append(
                f"| {r.name} | {r.severity.value} | {check} | {len(r.violations)} |"
            )
        violations_seen = [r for r in results if r.violations]
        if violations_seen:
            lines.append("")
            lines.append("## Violation detail")
            for r in violations_seen:
                lines.append(f"\n### {r.name} ({r.severity.value})")
                for v in r.violations:
                    lines.append(f"  - step {v.get('step')}: {v.get('reason', v)}")
        return "\n".join(lines) + "\n"
