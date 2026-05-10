"""Planner Protocol + PlanResult dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from ..core import Intent


@dataclass
class PlanResult:
    """Output of one Planner.plan() call."""

    name: str
    intents: list[Intent] = field(default_factory=list)
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.intents)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_steps": self.n_steps,
            "rationale": self.rationale,
            "intents": [i.to_json() for i in self.intents],
            "metadata": self.metadata,
        }


class Planner(Protocol):
    """A high-level goal -> Intent sequence decomposer.

    Pure function — no runtime state, no side effects. The runtime
    handles execution and the safety pipeline gates each emitted intent
    independently. Planners can take ``goal`` as a structured dict, a
    natural-language string, or any custom shape; ghostloop's bench
    harness accepts any of them via the ``policy(runtime)`` callable.
    """

    name: str

    def plan(self, goal: Any) -> PlanResult: ...
