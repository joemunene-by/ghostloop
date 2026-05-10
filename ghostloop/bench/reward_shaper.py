"""Declarative reward shaping over Intents + Decisions + Results.

Hand-authored reward functions inside primitive ``call`` bodies are
hard to compose, hard to reason about, and bury the actual reward
shape in glue code. ``RewardShaper`` lets you declare reward as a list
of typed components — each fires when its predicate matches, adding
its own contribution. The total reward for a step is the sum of every
matching component's value.

Example:

    shaper = RewardShaper([
        OnPrimitive("pick", reward=+1.0, when_status="ok"),
        OnPrimitive("pick", reward=-2.0, when_status="error"),
        OnDecision("deny", reward=-5.0),       # any deny in any gate
        StepCost(-0.01),                        # mild time penalty
        OnObservation("force_norm", below=10.0, reward=+0.1),
    ])

    reward = shaper.score(intent, decision, result)

Composes with ``train_safe`` from the v0.8 training harness — the
collector's ``reward_fn`` argument can be ``shaper.score_event``.
Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..core import Decision, Intent, Result


@dataclass
class RewardComponent:
    """Base class for a single reward term. Subclasses implement ``score``."""

    reward: float = 0.0
    label: str = ""

    def score(self, intent: Intent, decision: Decision, result: Result) -> float:
        return 0.0  # overridden by subclasses


@dataclass
class OnPrimitive(RewardComponent):
    """Reward when intent.name matches AND (optionally) result.status matches."""

    primitive_name: str = ""
    when_status: str | None = None

    def score(self, intent: Intent, decision: Decision, result: Result) -> float:
        if intent.name != self.primitive_name:
            return 0.0
        if self.when_status is not None and result.status.value != self.when_status:
            return 0.0
        return self.reward


@dataclass
class OnDecision(RewardComponent):
    """Reward when decision.action matches (e.g. 'deny' for safety penalties)."""

    action: str = "deny"
    gate_name: str | None = None  # optional gate-name filter

    def score(self, intent: Intent, decision: Decision, result: Result) -> float:
        if decision.action.value != self.action:
            return 0.0
        if self.gate_name is not None and decision.gate_name != self.gate_name:
            return 0.0
        return self.reward


@dataclass
class OnObservation(RewardComponent):
    """Reward when ``result.observation[field]`` falls in the configured range.

    At least one of ``below`` / ``above`` must be set. Both can be
    given for bounded windows. Missing keys yield zero contribution.
    """

    field_name: str = ""
    below: float | None = None
    above: float | None = None

    def score(self, intent: Intent, decision: Decision, result: Result) -> float:
        if not isinstance(result.observation, dict):
            return 0.0
        v = result.observation.get(self.field_name)
        try:
            x = float(v)
        except (TypeError, ValueError):
            return 0.0
        if self.below is not None and x >= self.below:
            return 0.0
        if self.above is not None and x <= self.above:
            return 0.0
        return self.reward


@dataclass
class StepCost(RewardComponent):
    """Constant per-step reward (negative for time-penalty)."""

    def score(self, intent: Intent, decision: Decision, result: Result) -> float:
        return self.reward


@dataclass
class CustomReward(RewardComponent):
    """Escape hatch: arbitrary callable. Use sparingly; prefer typed components."""

    fn: Callable[[Intent, Decision, Result], float] | None = None

    def score(self, intent: Intent, decision: Decision, result: Result) -> float:
        if self.fn is None:
            return 0.0
        return float(self.fn(intent, decision, result))


@dataclass
class RewardShaper:
    """Compose a list of reward components. Total = sum of all matching components.

    Components are evaluated independently — multiple can fire on the
    same step (e.g. StepCost + OnPrimitive + OnObservation all
    contributing). The order doesn't affect the total but is preserved
    for the per-component breakdown returned by ``score_with_breakdown``.
    """

    components: list[RewardComponent] = field(default_factory=list)

    def score(
        self, intent: Intent, decision: Decision, result: Result,
    ) -> float:
        return sum(c.score(intent, decision, result) for c in self.components)

    def score_with_breakdown(
        self, intent: Intent, decision: Decision, result: Result,
    ) -> dict[str, Any]:
        """Same as ``score`` but also returns the per-component breakdown.

        Useful for trace annotations / debug logs / TensorBoard scalar plots.
        """
        per_component: list[dict[str, Any]] = []
        total = 0.0
        for c in self.components:
            v = c.score(intent, decision, result)
            if v != 0.0:
                per_component.append({
                    "component": c.__class__.__name__,
                    "label": c.label,
                    "value": v,
                })
                total += v
        return {"total": total, "components": per_component}

    # Adapter for the v0.8 training harness's reward_fn signature
    # (which takes a single dict observation). Looks up the most-recent
    # event in the runtime trace and uses its decision/result.
    def score_event(self, observation: dict[str, Any]) -> float:
        """Convenience: score from an observation dict only.

        For full reward shaping wire ``shaper.score`` into your custom
        reward_fn that has access to the Intent + Decision + Result.
        This adapter handles the simpler "reward solely from observation
        fields" case.
        """
        intent = Intent("(unknown)", {})
        decision = Decision.allow("(none)", "")
        result_obs = observation if isinstance(observation, dict) else {}
        from ..core import Result, ResultStatus  # avoid top-level cycle
        synthetic_result = Result(status=ResultStatus.OK, observation=result_obs)
        return self.score(intent, decision, synthetic_result)


def from_dict(spec: list[dict[str, Any]]) -> RewardShaper:
    """Build a ``RewardShaper`` from a JSON-friendly spec list.

    Each entry must include a ``"type"`` key. Example:

        from_dict([
            {"type": "OnPrimitive", "primitive_name": "pick", "reward": 1.0},
            {"type": "StepCost", "reward": -0.01},
        ])
    """
    type_map: dict[str, type[RewardComponent]] = {
        "OnPrimitive": OnPrimitive,
        "OnDecision": OnDecision,
        "OnObservation": OnObservation,
        "StepCost": StepCost,
        "CustomReward": CustomReward,
    }
    components: list[RewardComponent] = []
    for entry in spec:
        kind = entry.get("type")
        if kind not in type_map:
            raise ValueError(f"unknown reward component {kind!r}")
        cls = type_map[kind]
        kwargs = {k: v for k, v in entry.items() if k != "type"}
        components.append(cls(**kwargs))
    return RewardShaper(components=components)
