"""HeuristicJudge — rule-based trace scoring without an LLM in the loop.

For environments where calling an LLM is undesirable (cost, latency,
hermetic CI, air-gapped deployments) the rule-based judge gives you
the same scoring shape backed by typed predicates. Each ``JudgeRule``
contributes a 0..1 sub-score; the final score is the weighted mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core import Trace


PredicateFn = Callable[[Trace], float]   # returns score in [0, 1]


@dataclass
class JudgeRule:
    """One scoring rule. ``predicate`` returns a 0..1 sub-score."""

    name: str
    predicate: PredicateFn
    weight: float = 1.0
    description: str = ""


@dataclass
class RubricScore:
    """The judge's output for one trace under one rubric."""

    score: float
    label: str
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "label": self.label,
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
        }


@dataclass
class HeuristicJudge:
    """Score traces by summing weighted predicate sub-scores."""

    rules: list[JudgeRule] = field(default_factory=list)
    pass_threshold: float = 0.7
    fail_threshold: float = 0.4

    def score(self, trace: Trace) -> RubricScore:
        breakdown: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0
        for rule in self.rules:
            v = float(rule.predicate(trace))
            v = max(0.0, min(1.0, v))
            breakdown[rule.name] = v
            weighted_sum += v * rule.weight
            total_weight += rule.weight
        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        if score >= self.pass_threshold:
            label = "pass"
        elif score < self.fail_threshold:
            label = "fail"
        else:
            label = "marginal"
        return RubricScore(score=score, label=label, breakdown=breakdown)


# ---------------------------------------------------------------------------
# A handful of pre-baked rule predicates the user can compose.
# ---------------------------------------------------------------------------


def no_violations() -> PredicateFn:
    """1.0 iff no event has a DENY decision."""
    def _p(trace: Trace) -> float:
        if not trace.events:
            return 1.0
        n = sum(1 for e in trace.events if e.decision.action.value == "deny")
        return 1.0 if n == 0 else 0.0
    return _p


def fraction_allowed() -> PredicateFn:
    """Fraction of events whose decision is ALLOW (not DENY/ESCALATE)."""
    def _p(trace: Trace) -> float:
        if not trace.events:
            return 1.0
        allow = sum(1 for e in trace.events if e.decision.action.value == "allow")
        return allow / len(trace.events)
    return _p


def fraction_ok() -> PredicateFn:
    """Fraction of events whose result.status is OK."""
    def _p(trace: Trace) -> float:
        if not trace.events:
            return 0.0
        ok = sum(1 for e in trace.events if e.result.status.value == "ok")
        return ok / len(trace.events)
    return _p


def step_count_below(threshold: int) -> PredicateFn:
    """1.0 if trace length <= threshold, otherwise linear decay to 0 at 2x."""
    def _p(trace: Trace) -> float:
        n = len(trace.events)
        if n <= threshold:
            return 1.0
        if n >= 2 * threshold:
            return 0.0
        return 1.0 - (n - threshold) / float(threshold)
    return _p
