"""Counterfactual trace replay — what would policy B have done on policy A's trace?

Standard practice in LLM safety / red-teaming is to replay a recorded
conversation through a different model and diff its outputs. Robotics
tooling has no equivalent: when an incident happens, you can't ask
"if we'd shipped policy B instead of A, would this have been
prevented?" without re-running the entire scenario in sim.

This module gives you that for free, against any recorded ``Trace``:

  cf = replay_with_policy(original_trace, new_policy)
  print(cf.divergence_rate, cf.first_divergence_step)
  print(cf.render_md())

The new policy is asked, at each event, "given this state_before, what
intent would you emit?" — its answer is captured as the counterfactual
intent. We don't actually re-execute the world (we'd need to roll the
simulator back), so this is shadow-mode reasoning: the comparison is
"would the policy have CHOSEN differently". For a full re-execution
plug the counterfactual intents into a fresh Runtime + your sim
Backend.

Pure stdlib; works against any callable ``new_policy(state_before) ->
Intent | None``. Returning None is allowed and means "policy declined
to act" (which is itself a meaningful divergence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import Intent, Trace, TraceEvent


CounterfactualPolicy = Callable[[dict[str, Any]], "Intent | None"]


def _args_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Cheap ad-hoc distance between two intent.args dicts.

    Numeric fields contribute |a-b|; mismatched types or string fields
    contribute 1.0 for a name mismatch, 0.0 if equal. Returns a
    normalized 0..inf score; 0 means identical args.
    """
    total = 0.0
    for k in set(a.keys()) | set(b.keys()):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            total += abs(float(va) - float(vb))
        elif isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)) and len(va) == len(vb):
            total += sum(abs(float(x) - float(y)) for x, y in zip(va, vb, strict=False))
        elif va != vb:
            total += 1.0
    return total


@dataclass
class CounterfactualEvent:
    """One step's comparison between original + counterfactual policy."""

    step: int
    timestamp: float
    state_before: dict[str, Any]
    original_intent: Intent
    counterfactual_intent: Intent | None
    intent_match: bool
    name_match: bool
    args_distance: float

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "original_intent": self.original_intent.to_json(),
            "counterfactual_intent": (
                self.counterfactual_intent.to_json()
                if self.counterfactual_intent is not None else None
            ),
            "intent_match": self.intent_match,
            "name_match": self.name_match,
            "args_distance": round(self.args_distance, 6),
        }


@dataclass
class CounterfactualTrace:
    """The full original-vs-counterfactual comparison."""

    original_episode_id: str
    new_policy_name: str
    events: list[CounterfactualEvent] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.events)

    @property
    def n_divergent(self) -> int:
        return sum(1 for e in self.events if not e.intent_match)

    @property
    def divergence_rate(self) -> float:
        return self.n_divergent / self.n if self.n else 0.0

    @property
    def first_divergence_step(self) -> int | None:
        for e in self.events:
            if not e.intent_match:
                return e.step
        return None

    @property
    def mean_args_distance(self) -> float:
        return (
            sum(e.args_distance for e in self.events) / self.n if self.n else 0.0
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "original_episode_id": self.original_episode_id,
            "new_policy_name": self.new_policy_name,
            "n": self.n,
            "n_divergent": self.n_divergent,
            "divergence_rate": round(self.divergence_rate, 4),
            "first_divergence_step": self.first_divergence_step,
            "mean_args_distance": round(self.mean_args_distance, 4),
            "events": [e.to_json() for e in self.events],
        }

    def render_md(self) -> str:
        lines = [
            f"# Counterfactual replay — {self.new_policy_name} vs original",
            "",
            f"- Episode: `{self.original_episode_id}`",
            f"- Steps: {self.n}",
            f"- Divergence rate: {self.divergence_rate:.1%} ({self.n_divergent}/{self.n})",
            f"- First divergence: "
            f"{'step ' + str(self.first_divergence_step) if self.first_divergence_step is not None else '(none)'}",
            f"- Mean args distance: {self.mean_args_distance:.4g}",
            "",
            "| Step | Original | Counterfactual | Match | Args ∆ |",
            "|---:|---|---|:---:|---:|",
        ]
        for e in self.events:
            cf_name = e.counterfactual_intent.name if e.counterfactual_intent else "(declined)"
            check = "✓" if e.intent_match else "✗"
            lines.append(
                f"| {e.step} | {e.original_intent.name} | {cf_name} "
                f"| {check} | {e.args_distance:.3g} |"
            )
        return "\n".join(lines) + "\n"


def replay_with_policy(
    original: Trace,
    new_policy: CounterfactualPolicy,
    *,
    new_policy_name: str = "counterfactual",
) -> CounterfactualTrace:
    """Walk the original trace, asking ``new_policy`` what IT would emit at each step.

    The new policy receives ``state_before`` (the world state BEFORE the
    original event executed) and must return either an Intent (its
    chosen action) or None (declined). Returns a CounterfactualTrace
    aligning every step with both choices.
    """
    cf = CounterfactualTrace(
        original_episode_id=original.episode_id,
        new_policy_name=new_policy_name,
    )
    for ev in original.events:
        try:
            cf_intent = new_policy(ev.state_before)
        except Exception as exc:  # noqa: BLE001
            cf_intent = None
            cf.events.append(CounterfactualEvent(
                step=ev.step,
                timestamp=ev.timestamp,
                state_before=ev.state_before,
                original_intent=ev.intent,
                counterfactual_intent=None,
                intent_match=False,
                name_match=False,
                args_distance=math.inf,
            ))
            continue
        name_match = cf_intent is not None and cf_intent.name == ev.intent.name
        args_dist = (
            _args_distance(cf_intent.args, ev.intent.args)
            if cf_intent is not None else math.inf
        )
        cf.events.append(CounterfactualEvent(
            step=ev.step,
            timestamp=ev.timestamp,
            state_before=ev.state_before,
            original_intent=ev.intent,
            counterfactual_intent=cf_intent,
            intent_match=name_match and args_dist < 1e-9,
            name_match=name_match,
            args_distance=args_dist,
        ))
    return cf
