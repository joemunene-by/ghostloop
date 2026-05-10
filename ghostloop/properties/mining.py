"""Property mining — auto-discover invariants from a corpus of traces.

Hand-authoring properties is a discipline; not every team has the
patience. Property mining looks at a corpus of *successful* traces
and surfaces statistical regularities that are good candidates for
formal invariants:

  - "primitive X always followed by Y within W seconds"
    (high-confidence transitions)
  - "decision DENY only ever appears for primitive Z"
    (action-gate restrictions)
  - "observation field F never exceeds threshold T"
    (numeric bounds)

Mined patterns come back as ``MinedProperty`` records with support
(fraction of traces where the pattern holds), confidence (when the
antecedent holds, how often the consequent does), and a ``promote()``
method that returns a real ``Property`` ready to drop into a
``PropertyEngine``.

The discovery is deliberately small-and-explainable. Heavyweight
sequence-mining (Apriori / FP-Growth) is out of scope; we cover the
three highest-value patterns.

Pure stdlib.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..core import Trace
from .builtins import StaysInsideWorkspace
from .core import Property, PropertyResult, Severity
from .temporal import Always, intent_named, state_field_below


@dataclass
class MinedProperty:
    """One discovered candidate property."""

    pattern: str
    description: str
    support: float                    # fraction of traces where pattern holds
    confidence: float                 # P(consequent | antecedent)
    n_supporting_traces: int
    n_total_traces: int
    promote: Any = None               # Callable -> Property, set per-pattern type

    def to_json(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "description": self.description,
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "n_supporting_traces": self.n_supporting_traces,
            "n_total_traces": self.n_total_traces,
        }


# ---------------------------------------------------------------------------
# Pattern: primitive X always followed by primitive Y within window W
# ---------------------------------------------------------------------------


def _mine_followups(
    traces: list[Trace],
    *,
    window_s: float = 5.0,
    min_support: float = 0.9,
) -> list[MinedProperty]:
    """Find pairs (X, Y) such that whenever X fires, Y fires within W seconds."""
    if not traces:
        return []
    # Count pair occurrences per trace and total X occurrences per trace.
    pair_traces: dict[tuple[str, str], int] = defaultdict(int)
    x_traces: dict[str, int] = defaultdict(int)
    for trace in traces:
        x_seen_in_trace: set[str] = set()
        pairs_in_trace: set[tuple[str, str]] = set()
        for i, ev in enumerate(trace.events):
            x_seen_in_trace.add(ev.intent.name)
            for j in range(i + 1, len(trace.events)):
                ev_j = trace.events[j]
                if ev_j.timestamp - ev.timestamp > window_s:
                    break
                pairs_in_trace.add((ev.intent.name, ev_j.intent.name))
        for x in x_seen_in_trace:
            x_traces[x] += 1
        for pair in pairs_in_trace:
            pair_traces[pair] += 1
    n = len(traces)
    out: list[MinedProperty] = []
    for (x, y), pair_count in pair_traces.items():
        if x == y:
            continue
        x_count = x_traces[x]
        if x_count == 0:
            continue
        confidence = pair_count / x_count
        support = pair_count / n
        if support >= min_support and confidence >= min_support:
            mp = MinedProperty(
                pattern="followup",
                description=f"{x!r} always followed by {y!r} within {window_s:g}s",
                support=support,
                confidence=confidence,
                n_supporting_traces=pair_count,
                n_total_traces=n,
            )
            mp.promote = _make_followup_property(x, y, window_s)
            out.append(mp)
    return out


def _make_followup_property(x: str, y: str, window_s: float):
    def _factory() -> Property:
        # "Always (when intent.name == x, intent_named(y) holds within window)"
        # is non-trivial in our STL; approximate as "Always intent_named(x or y)
        # over a 0-window" — really we just want: "every time x fires, y was
        # in a recent window". Use a custom Property.
        @dataclass
        class _Followup:
            name: str = f"followup({x}->{y}<{window_s:g}s)"
            severity: Severity = Severity.WARN
            def check(self, trace: Trace) -> PropertyResult:
                violations = []
                for i, ev in enumerate(trace.events):
                    if ev.intent.name != x:
                        continue
                    cutoff = ev.timestamp + window_s
                    found = any(
                        e2.intent.name == y and e2.timestamp <= cutoff
                        for e2 in trace.events[i + 1:]
                    )
                    if not found:
                        violations.append({
                            "step": ev.step, "reason":
                            f"intent {x!r} not followed by {y!r} within {window_s:g}s"
                        })
                return PropertyResult(
                    name=self.name,
                    severity=self.severity,
                    held=not violations,
                    description=self.name,
                    violations=violations,
                )
        return _Followup()
    return _factory


# ---------------------------------------------------------------------------
# Pattern: numeric observation field never exceeds threshold
# ---------------------------------------------------------------------------


def _mine_numeric_bounds(
    traces: list[Trace],
    *,
    fields: list[str],
    margin_factor: float = 1.1,
) -> list[MinedProperty]:
    """For each named numeric field, the maximum value across all traces is
    a candidate "never exceeds" property. Set the bound at the empirical
    max times ``margin_factor`` so genuinely successful traces still pass.
    """
    if not traces:
        return []
    results: list[MinedProperty] = []
    for field_name in fields:
        max_seen: float | None = None
        n_with_field = 0
        n_total = 0
        for trace in traces:
            seen_in_trace = False
            for ev in trace.events:
                state = ev.state_after if isinstance(ev.state_after, dict) else {}
                if field_name not in state:
                    continue
                try:
                    v = float(state[field_name])
                except (TypeError, ValueError):
                    continue
                seen_in_trace = True
                if max_seen is None or v > max_seen:
                    max_seen = v
            if seen_in_trace:
                n_with_field += 1
            n_total += 1
        if max_seen is None or n_with_field == 0:
            continue
        threshold = max_seen * margin_factor
        mp = MinedProperty(
            pattern="numeric_bound",
            description=f"observation {field_name!r} never exceeds {threshold:.4g}",
            support=n_with_field / n_total,
            confidence=1.0,
            n_supporting_traces=n_with_field,
            n_total_traces=n_total,
        )
        mp.promote = _make_numeric_bound_property(field_name, threshold)
        results.append(mp)
    return results


def _make_numeric_bound_property(field_name: str, threshold: float):
    def _factory() -> Property:
        return Always(
            phi=state_field_below(field_name, threshold),
            window_s=0.0,
            name=f"bound({field_name}<{threshold:.4g})",
            severity=Severity.WARN,
        )
    return _factory


# ---------------------------------------------------------------------------
# Pattern: workspace confinement (every state.position inside an AABB)
# ---------------------------------------------------------------------------


def _mine_workspace_bounds(
    traces: list[Trace], *, margin: float = 0.05,
) -> list[MinedProperty]:
    """Compute the empirical bounding box of every recorded position; emit
    a StaysInsideWorkspace candidate."""
    if not traces:
        return []
    x_lo = y_lo = z_lo = float("inf")
    x_hi = y_hi = z_hi = float("-inf")
    n_with_position = 0
    for trace in traces:
        seen = False
        for ev in trace.events:
            pos = (
                ev.state_after.get("position")
                if isinstance(ev.state_after, dict) else None
            )
            if not pos or len(pos) < 3:
                continue
            seen = True
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            x_lo = min(x_lo, x); x_hi = max(x_hi, x)
            y_lo = min(y_lo, y); y_hi = max(y_hi, y)
            z_lo = min(z_lo, z); z_hi = max(z_hi, z)
        if seen:
            n_with_position += 1
    if x_lo == float("inf"):
        return []
    bounds_min = (x_lo - margin, y_lo - margin, z_lo - margin)
    bounds_max = (x_hi + margin, y_hi + margin, z_hi + margin)
    mp = MinedProperty(
        pattern="workspace_bounds",
        description=(
            f"position remains in AABB [{bounds_min}..{bounds_max}] "
            f"with margin {margin:g}"
        ),
        support=n_with_position / max(1, len(traces)),
        confidence=1.0,
        n_supporting_traces=n_with_position,
        n_total_traces=len(traces),
    )
    mp.promote = lambda: StaysInsideWorkspace(
        min_corner=bounds_min, max_corner=bounds_max,
        name=f"mined_workspace",
        severity=Severity.WARN,
    )
    return [mp]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mine_properties(
    traces: list[Trace],
    *,
    window_s: float = 5.0,
    min_support: float = 0.9,
    numeric_fields: list[str] | None = None,
    margin_factor: float = 1.1,
    workspace_margin: float = 0.05,
) -> list[MinedProperty]:
    """Run all built-in pattern miners over a corpus of (successful) traces.

    Args:
        traces: corpus of recorded traces. Mining works best on
            "known-good" traces — passing in failed runs pollutes the
            statistics.
        window_s: temporal window for follow-up pattern mining.
        min_support: minimum fraction of traces where a pattern must
            hold to be returned. 0.9 = "holds in 90% of traces".
        numeric_fields: which observation fields to bound-mine. None
            for default ("force", "force_norm", "velocity").
        margin_factor: bound mining widens the empirical max by this
            factor so genuine traces don't false-positive.
        workspace_margin: padding around the empirical position-bounding
            box for workspace mining.

    Returns:
        Sorted (descending support) list of MinedProperty candidates.
    """
    fields = numeric_fields or ["force", "force_norm", "velocity"]
    out: list[MinedProperty] = []
    out.extend(_mine_followups(
        traces, window_s=window_s, min_support=min_support,
    ))
    out.extend(_mine_numeric_bounds(
        traces, fields=fields, margin_factor=margin_factor,
    ))
    out.extend(_mine_workspace_bounds(traces, margin=workspace_margin))
    return sorted(out, key=lambda mp: -mp.support)
