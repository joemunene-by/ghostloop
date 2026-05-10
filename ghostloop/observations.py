"""ObservationBuffer: short-term memory exposed to policies.

Policies often need more than the current backend snapshot — they need
recency: "what was the last thing I observed", "did this primitive
succeed last time", "have I tried this target before". The runtime
records every step in the trace already, but trace iteration on every
policy call is wasteful and the policy shouldn't depend on the trace
shape.

ObservationBuffer is a fixed-size deque the runtime can populate after
each step, and policies can read from. Keeps the policy code simple and
backend-agnostic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .core import Intent, Result


@dataclass
class ObservationRecord:
    """One step's snapshot from the policy's point of view."""

    intent_name: str
    intent_args: dict[str, Any]
    status: str  # ok / error / blocked / timeout
    observation: dict[str, Any]
    state_after: dict[str, Any]
    duration_ms: float

    def to_json(self) -> dict[str, Any]:
        return {
            "intent_name": self.intent_name,
            "intent_args": self.intent_args,
            "status": self.status,
            "observation": self.observation,
            "state_after": self.state_after,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ObservationBuffer:
    """Fixed-size deque of recent ObservationRecords.

    Capacity controls how far back the policy can see. Default 32 covers
    most use cases (LLM context windows fill quickly, VLA models look
    at the present mostly).
    """

    capacity: int = 32
    _records: deque[ObservationRecord] = field(default_factory=deque)

    def append(self, intent: Intent, result: Result, state_after: dict[str, Any]) -> None:
        rec = ObservationRecord(
            intent_name=intent.name,
            intent_args=dict(intent.args),
            status=result.status.value,
            observation=dict(result.observation),
            state_after=dict(state_after),
            duration_ms=result.duration_ms,
        )
        self._records.append(rec)
        while len(self._records) > self.capacity:
            self._records.popleft()

    def latest(self) -> ObservationRecord | None:
        return self._records[-1] if self._records else None

    def n_recent(self, n: int = 5) -> list[ObservationRecord]:
        if n <= 0:
            return []
        return list(self._records)[-n:]

    def filter_by_intent(self, name: str) -> list[ObservationRecord]:
        return [r for r in self._records if r.intent_name == name]

    def n_blocked(self) -> int:
        return sum(1 for r in self._records if r.status == "blocked")

    def n_errored(self) -> int:
        return sum(1 for r in self._records if r.status == "error")

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0

    def to_json(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "n_records": len(self._records),
            "records": [r.to_json() for r in self._records],
        }
