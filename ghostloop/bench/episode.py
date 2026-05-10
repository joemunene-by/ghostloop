"""Episode: the smallest unit of a benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..core import Backend, MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime, Trace


@dataclass
class Episode:
    """One benchmark trial.

    ``policy`` is anything callable that yields Intents given a Runtime.
    Two shapes are supported:

      def scripted(runtime) -> Iterable[Intent]: ...

    or a closure that calls runtime.step itself and returns the trace
    (for adapters like LLMPolicy that own their own loop). The harness
    accepts either via ``EpisodeRunner.from_policy``.

    ``success_predicate`` receives the final Trace + Backend snapshot and
    returns True iff the goal is satisfied. Declarative, side-effect free.
    """

    name: str
    goal: str
    setup: Callable[[], Backend]
    policy: Callable[[Runtime], Any]
    success_predicate: Callable[[Trace, dict[str, Any]], bool]
    primitives: Callable[[], list]  # () -> list of Primitive instances
    pipeline: PolicyPipeline | None = None
    max_steps: int = 32
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Outcome of one Episode under one Runtime."""

    episode_name: str
    passed: bool
    steps: int
    final_state: dict[str, Any]
    trace: Trace
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_name": self.episode_name,
            "passed": self.passed,
            "steps": self.steps,
            "final_state": self.final_state,
            "n_events": len(self.trace.events),
            "notes": self.notes,
        }


@dataclass
class EpisodeRunner:
    """Drives one Episode end-to-end and returns an EpisodeResult."""

    def run(self, episode: Episode) -> EpisodeResult:
        backend = episode.setup()
        registry = PrimitiveRegistry(episode.primitives())
        pipeline = episode.pipeline or PolicyPipeline()
        runtime = Runtime(backend=backend, registry=registry, policy_pipeline=pipeline)

        # Two policy shapes:
        # 1. Iterable of Intents -> drive the loop here.
        # 2. Callable that runs its own loop and returns a summary -> already done.
        produced = episode.policy(runtime)
        steps = 0
        if produced is None:
            steps = len(runtime.trace.events)
        elif isinstance(produced, dict) and "steps" in produced:
            steps = int(produced["steps"])
        else:
            for intent in produced:
                runtime.step(intent)
                steps += 1
                if steps >= episode.max_steps:
                    break

        final_state = backend.snapshot()
        passed = bool(episode.success_predicate(runtime.trace, final_state))
        return EpisodeResult(
            episode_name=episode.name,
            passed=passed,
            steps=steps,
            final_state=final_state,
            trace=runtime.trace,
        )

    def run_all(self, episodes: Iterable[Episode]) -> list[EpisodeResult]:
        return [self.run(ep) for ep in episodes]
