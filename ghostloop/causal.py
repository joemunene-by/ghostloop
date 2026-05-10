"""Causal failure attribution — find the events responsible for a violation.

When a property fails on a recorded trace, the obvious question is
"which events caused it?". The simple answer is "the one whose
state_after first violated the predicate", but that's the SYMPTOM
event, not necessarily the CAUSE. The actual cause is whichever
upstream events SET UP the conditions for the symptom — they may be
many steps back.

This module ships an ablation-based attribution algorithm:

  for each event E_i in the failing trace:
      replay = trace minus E_i
      if property holds without E_i:
          E_i is causally NECESSARY for the failure
      else:
          E_i is INDEPENDENT of the failure

Each event gets a ``necessity`` score in [0, 1]: the fraction of the
post-E_i prefix where the property holds when E_i is masked out. By
sorting events by descending necessity, the user gets a ranked list
of "blame" — the highest-scored event is the most-likely root cause.

This is leave-one-out — O(N) property evaluations per attribution.
For shorter traces (under 200 events) it's fast enough to run
synchronously; for longer ones, sample uniformly via ``max_events``.

The algorithm is shamelessly inspired by causal-inference do-calculus
masking but kept stdlib-simple so it runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import Trace, TraceEvent
from .properties.core import Property


@dataclass
class CauseAttribution:
    """One event's contribution to a property violation."""

    event: TraceEvent
    necessity: float
    became_held: bool                # property held without this event

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.event.step,
            "timestamp": self.event.timestamp,
            "intent_name": self.event.intent.name,
            "decision_action": self.event.decision.action.value,
            "necessity": round(self.necessity, 4),
            "became_held": self.became_held,
        }


@dataclass
class FailureAnalysis:
    """The full causal attribution: which events caused the property to fail."""

    property_name: str
    held: bool
    n_events: int
    n_necessary: int
    attributions: list[CauseAttribution] = field(default_factory=list)

    def top_k(self, k: int = 5) -> list[CauseAttribution]:
        return sorted(self.attributions, key=lambda c: -c.necessity)[:k]

    def to_json(self) -> dict[str, Any]:
        return {
            "property_name": self.property_name,
            "held": self.held,
            "n_events": self.n_events,
            "n_necessary": self.n_necessary,
            "top_5_causes": [c.to_json() for c in self.top_k(5)],
        }

    def render_md(self) -> str:
        lines = [
            f"# Causal failure analysis — `{self.property_name}`",
            "",
            f"- Property held: {self.held}",
            f"- Events: {self.n_events}",
            f"- Causally necessary: {self.n_necessary}",
            "",
            "## Top 5 most-likely causes",
            "",
            "| Step | Intent | Decision | Necessity | Became-held? |",
            "|---:|---|---|---:|:---:|",
        ]
        for c in self.top_k(5):
            held_mark = "✓" if c.became_held else "✗"
            lines.append(
                f"| {c.event.step} | {c.event.intent.name} | "
                f"{c.event.decision.action.value} | {c.necessity:.3f} | {held_mark} |"
            )
        return "\n".join(lines) + "\n"


def attribute_failure(
    trace: Trace,
    prop: Property,
    *,
    max_events: int | None = None,
) -> FailureAnalysis:
    """Leave-one-out causal attribution for a property violation.

    For each event E_i in the trace, evaluates the property on the
    trace WITHOUT E_i. Events whose removal makes the property hold
    are flagged ``became_held=True`` and assigned a high necessity
    score; events whose removal leaves the property still failing
    are assigned 0.0 (independent of the failure).

    If the property already holds on the full trace, returns an empty
    attribution — there's no failure to attribute.

    Args:
        trace: the failing trace.
        prop: the property whose violation is being analysed.
        max_events: optional cap on how many events to ablate. Useful
            for very long traces where O(N) is expensive. Sampled
            uniformly across the trace timeline.

    Returns:
        A ``FailureAnalysis`` with per-event necessity scores. Use
        ``analysis.top_k(k)`` to surface the most-likely root causes.
    """
    full_check = prop.check(trace)
    if full_check.held:
        return FailureAnalysis(
            property_name=prop.name,
            held=True,
            n_events=len(trace.events),
            n_necessary=0,
        )
    indices = list(range(len(trace.events)))
    if max_events is not None and len(indices) > max_events:
        # Sample uniformly across the trace.
        step = max(1, len(indices) // max_events)
        indices = indices[::step]
    attributions: list[CauseAttribution] = []
    for i in indices:
        ablated = Trace(
            episode_id=trace.episode_id,
            backend_name=trace.backend_name,
            started_at=trace.started_at,
            events=trace.events[:i] + trace.events[i + 1:],
        )
        sub = prop.check(ablated)
        # Necessity: did property turn from FAIL -> HOLD when this event removed?
        if sub.held:
            necessity = 1.0
        else:
            # Partial credit: violations decreased meaningfully?
            n_violations_full = len(full_check.violations) or 1
            n_violations_sub = len(sub.violations)
            decrease = max(0.0, n_violations_full - n_violations_sub) / n_violations_full
            necessity = decrease * 0.5  # cap at 0.5 since failure still occurs
        attributions.append(CauseAttribution(
            event=trace.events[i],
            necessity=necessity,
            became_held=sub.held,
        ))
    return FailureAnalysis(
        property_name=prop.name,
        held=False,
        n_events=len(trace.events),
        n_necessary=sum(1 for a in attributions if a.became_held),
        attributions=attributions,
    )


def minimal_cause_set(
    trace: Trace,
    prop: Property,
    *,
    max_set_size: int = 4,
) -> list[TraceEvent] | None:
    """Greedy minimal cause set for a property violation.

    Iteratively builds a set S of events such that removing all of S
    from the trace makes the property hold. At each step picks the
    event whose removal most reduces violation count.

    Returns ``None`` if no cause set up to ``max_set_size`` is found.
    Useful when a single-event ablation isn't enough — multi-event
    causal patterns (e.g. "two specific moves in sequence are required
    to violate the property") need this.
    """
    if prop.check(trace).held:
        return []
    selected: list[TraceEvent] = []
    remaining = list(trace.events)
    for _ in range(max_set_size):
        best_idx = None
        best_remaining_violations = float("inf")
        for i, ev in enumerate(remaining):
            sub_events = [e for j, e in enumerate(remaining) if j != i]
            ablated = Trace(events=selected + sub_events, started_at=trace.started_at)
            # Build a clean trace with just the kept events; selected
            # events are NOT included since they're "ablated out" by the
            # cause set construction.
            kept = Trace(events=sub_events, started_at=trace.started_at)
            res = prop.check(kept)
            if res.held:
                # Found a sufficient cause.
                return selected + [ev]
            n_viol = len(res.violations)
            if n_viol < best_remaining_violations:
                best_remaining_violations = n_viol
                best_idx = i
        if best_idx is None:
            return None
        selected.append(remaining.pop(best_idx))
    return None
