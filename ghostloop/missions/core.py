"""Mission DAG runner."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from ..core import Intent, Result, Runtime


class StepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class MissionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"      # every required step succeeded
    FAILED = "failed"             # at least one required step failed
    PARTIAL = "partial"           # mix of succeeded + skipped


# A step's emit callable returns either a single Intent OR a list of Intents.
# The runner dispatches each one through the runtime and considers the step
# successful iff EVERY emitted intent resulted in an OK status.
StepEmit = Callable[["Mission", "Step"], Intent | list[Intent]]


@dataclass
class Step:
    """One node in the Mission DAG.

    Args:
        name: unique identifier within the mission.
        emit: callable that produces the Intent(s) to dispatch.
        depends_on: names of upstream steps that must succeed first.
        max_attempts: hard cap on retries when the step fails.
        required: if False the mission tolerates this step failing
            (useful for optional verification / cleanup steps).
        labels: free-form tags for observability / dashboards.
    """

    name: str
    emit: StepEmit
    depends_on: list[str] = field(default_factory=list)
    max_attempts: int = 1
    required: bool = True
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class StepResult:
    name: str
    status: StepStatus
    attempts: int
    intents: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "attempts": self.attempts,
            "intents": self.intents,
            "failure_reason": self.failure_reason,
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class Mission:
    """A named DAG of Steps."""

    name: str
    steps: list[Step]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate step names in mission {self.name!r}: {names}")
        all_names = set(names)
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in all_names:
                    raise ValueError(
                        f"step {s.name!r} depends on unknown step {dep!r}"
                    )
        # Cycle detection (Kahn's).
        indeg = {s.name: 0 for s in self.steps}
        out_edges = {s.name: [] for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                indeg[s.name] += 1
                out_edges[dep].append(s.name)
        ready = deque([n for n, d in indeg.items() if d == 0])
        seen = 0
        while ready:
            n = ready.popleft()
            seen += 1
            for nxt in out_edges[n]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        if seen != len(self.steps):
            raise ValueError(f"mission {self.name!r} has a dependency cycle")


@dataclass
class MissionResult:
    """Aggregated output of one mission run."""

    mission_name: str
    status: MissionStatus
    steps: list[StepResult]
    started_at: float
    finished_at: float

    @property
    def n_succeeded(self) -> int:
        return sum(1 for s in self.steps if s.status is StepStatus.SUCCEEDED)

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.steps if s.status is StepStatus.FAILED)

    @property
    def n_skipped(self) -> int:
        return sum(1 for s in self.steps if s.status is StepStatus.SKIPPED)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def to_json(self) -> dict[str, Any]:
        return {
            "mission_name": self.mission_name,
            "status": self.status.value,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
            "duration_s": round(self.duration_s, 4),
            "steps": [s.to_json() for s in self.steps],
        }


class MissionRunner:
    """Execute a Mission DAG against a Runtime under the safety pipeline."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def run(self, mission: Mission) -> MissionResult:
        steps_by_name = {s.name: s for s in mission.steps}
        statuses: dict[str, StepStatus] = {s.name: StepStatus.PENDING for s in mission.steps}
        results: dict[str, StepResult] = {}

        def _can_run(step: Step) -> bool:
            return all(
                statuses.get(dep) is StepStatus.SUCCEEDED
                or (
                    statuses.get(dep) in (StepStatus.SKIPPED, StepStatus.FAILED)
                    and not steps_by_name[dep].required
                )
                for dep in step.depends_on
            )

        started_at = time.time()
        progressed = True
        while progressed:
            progressed = False
            for step in mission.steps:
                if statuses[step.name] is not StepStatus.PENDING:
                    continue
                # Mark FAILED if any required dependency failed.
                blocked_by_failure = False
                for dep in step.depends_on:
                    dep_status = statuses[dep]
                    if dep_status is StepStatus.FAILED and steps_by_name[dep].required:
                        blocked_by_failure = True
                        break
                if blocked_by_failure:
                    statuses[step.name] = StepStatus.SKIPPED
                    results[step.name] = StepResult(
                        name=step.name,
                        status=StepStatus.SKIPPED,
                        attempts=0,
                        failure_reason="upstream step failed",
                        started_at=time.time(),
                        finished_at=time.time(),
                    )
                    progressed = True
                    continue
                if not _can_run(step):
                    continue
                # Run it.
                progressed = True
                result = self._run_step(mission, step)
                results[step.name] = result
                statuses[step.name] = result.status

        finished_at = time.time()
        any_failed_required = any(
            s.name in results and results[s.name].status is StepStatus.FAILED and s.required
            for s in mission.steps
        )
        any_skipped = any(
            s.name in results and results[s.name].status is StepStatus.SKIPPED
            for s in mission.steps
        )
        any_failed_optional = any(
            s.name in results and results[s.name].status is StepStatus.FAILED and not s.required
            for s in mission.steps
        )
        if any_failed_required:
            status = MissionStatus.FAILED
        elif any_skipped or any_failed_optional:
            status = MissionStatus.PARTIAL
        else:
            status = MissionStatus.SUCCEEDED

        return MissionResult(
            mission_name=mission.name,
            status=status,
            steps=[results[s.name] for s in mission.steps],
            started_at=started_at,
            finished_at=finished_at,
        )

    def _run_step(self, mission: Mission, step: Step) -> StepResult:
        attempt = 0
        last_failure = ""
        intents_out: list[dict[str, Any]] = []
        started_at = time.time()
        while attempt < step.max_attempts:
            attempt += 1
            try:
                emitted = step.emit(mission, step)
            except Exception as exc:  # noqa: BLE001
                last_failure = f"emit raised {type(exc).__name__}: {exc}"
                continue
            intents = emitted if isinstance(emitted, list) else [emitted]
            attempt_results: list[Result] = []
            attempt_success = True
            for intent in intents:
                result = self.runtime.step(intent)
                attempt_results.append(result)
                if not result.ok:
                    attempt_success = False
                    last_failure = (
                        f"step {step.name!r} intent {intent.name!r}: "
                        f"{result.status.value} — {result.message}"
                    )
                    break
            intents_out = [
                {**intent.to_json(), "result": result.to_json()}
                for intent, result in zip(intents, attempt_results, strict=False)
            ]
            if attempt_success:
                return StepResult(
                    name=step.name,
                    status=StepStatus.SUCCEEDED,
                    attempts=attempt,
                    intents=intents_out,
                    started_at=started_at,
                    finished_at=time.time(),
                )
        return StepResult(
            name=step.name,
            status=StepStatus.FAILED,
            attempts=attempt,
            intents=intents_out,
            failure_reason=last_failure,
            started_at=started_at,
            finished_at=time.time(),
        )
