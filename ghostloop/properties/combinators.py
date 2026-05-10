"""Property combinators: AND / OR / NOT over Properties.

Compose simple properties into complex assertions without forcing each
property to know about the others. The combinator's severity defaults
to the maximum severity of its children but can be overridden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import Trace
from .core import Property, PropertyResult, Severity


_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.WARN: 1,
    Severity.ERROR: 2,
}


def _max_severity(children: list[Property]) -> Severity:
    if not children:
        return Severity.INFO
    return max(children, key=lambda c: _SEVERITY_RANK[c.severity]).severity


@dataclass
class AndProperty:
    """Holds iff EVERY child holds. Violations from all children are concatenated."""

    children: list[Property]
    name: str = "and"
    severity: Severity | None = None  # None = max of children

    def check(self, trace: Trace) -> PropertyResult:
        sev = self.severity or _max_severity(self.children)
        results = [c.check(trace) for c in self.children]
        held = all(r.held for r in results)
        violations: list[dict[str, Any]] = []
        for r in results:
            for v in r.violations:
                violations.append({**v, "from": r.name})
        return PropertyResult(
            name=self.name,
            severity=sev,
            held=held,
            description=f"AND of {len(self.children)} children",
            violations=violations,
        )


@dataclass
class OrProperty:
    """Holds iff ANY child holds. On violation, surfaces every child's violations."""

    children: list[Property]
    name: str = "or"
    severity: Severity | None = None

    def check(self, trace: Trace) -> PropertyResult:
        sev = self.severity or _max_severity(self.children)
        results = [c.check(trace) for c in self.children]
        held = any(r.held for r in results)
        violations: list[dict[str, Any]] = []
        if not held:
            for r in results:
                for v in r.violations:
                    violations.append({**v, "from": r.name})
        return PropertyResult(
            name=self.name,
            severity=sev,
            held=held,
            description=f"OR of {len(self.children)} children",
            violations=violations,
        )


@dataclass
class NotProperty:
    """Inverts a single child. Useful for 'must violate this' regression tests."""

    child: Property
    name: str = "not"
    severity: Severity | None = None

    def check(self, trace: Trace) -> PropertyResult:
        sev = self.severity or self.child.severity
        inner = self.child.check(trace)
        held = not inner.held
        return PropertyResult(
            name=self.name,
            severity=sev,
            held=held,
            description=f"NOT of {inner.name}",
            violations=[] if held else [{
                "step": "?",
                "reason": f"child {inner.name!r} held when it shouldn't have",
            }],
        )
