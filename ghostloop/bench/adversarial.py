"""Adversarial bench generator — fuzz over Episode initial states.

Standard regression benches are static: fixed scenarios, hand-picked
seeds. Real failures live in the long tail. This module *searches*
for failure cases by perturbing an episode template across a parameter
range, running the policy on each variant, and surfacing the seeds
where the policy fails (or comes close).

Three generators ship:

  - ``random_seeds`` — uniform random sampling, fastest baseline.
  - ``grid_seeds`` — exhaustive cartesian product over a parameter
    grid; useful for low-dim spaces.
  - ``cma_es_seeds`` — covariance-matrix-adaptation evolution
    strategy: searches the parameter space toward failure-maximising
    seeds. Implemented in pure-stdlib (small CMA-ES variant) so it
    runs anywhere; for high-dim spaces install a real CMA-ES library.

Each generator returns a list of ``AdversarialResult`` records sorted
by descending failure score (number of safety violations + 1 per
failed predicate). Add the high-scoring seeds back to your bench as
regression cases — that's testing-as-search.

All randomness is reproducible: pass an integer ``seed`` to make the
sweep deterministic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core import Trace
from .episode import Episode, EpisodeResult, EpisodeRunner


# A perturber takes (base_episode, sample_dict) and returns a new Episode
# customised for that sample. Often this means returning a copy of the
# Episode with a different ``setup`` callable that initialises the backend
# at the sampled coordinates.
EpisodePerturber = Callable[[Episode, dict[str, float]], Episode]


@dataclass
class AdversarialResult:
    """One attack attempt: which seed, how badly the policy did."""

    sample: dict[str, float]
    passed: bool
    n_violations: int
    n_steps: int
    failure_score: float          # higher = worse (more violations / failed predicate)
    episode_name: str
    final_state: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "passed": self.passed,
            "n_violations": self.n_violations,
            "n_steps": self.n_steps,
            "failure_score": round(self.failure_score, 4),
            "episode_name": self.episode_name,
            "notes": self.notes,
        }


def _failure_score(result: EpisodeResult) -> float:
    """Higher when the policy did worse. Used as the search objective."""
    n_violations = sum(
        1 for ev in result.trace.events if ev.decision.action.value == "deny"
    )
    base = n_violations
    if not result.passed:
        base += 1.0
    return float(base)


def _run_one(perturber: EpisodePerturber, episode: Episode, sample: dict[str, float]) -> tuple[EpisodeResult, AdversarialResult]:
    perturbed = perturber(episode, sample)
    result = EpisodeRunner().run(perturbed)
    n_violations = sum(
        1 for ev in result.trace.events if ev.decision.action.value == "deny"
    )
    score = _failure_score(result)
    return result, AdversarialResult(
        sample=dict(sample),
        passed=result.passed,
        n_violations=n_violations,
        n_steps=result.steps,
        failure_score=score,
        episode_name=result.episode_name,
        final_state=result.final_state,
    )


def random_seeds(
    base_episode: Episode,
    perturber: EpisodePerturber,
    parameter_ranges: dict[str, tuple[float, float]],
    *,
    n_samples: int = 32,
    seed: int = 0,
) -> list[AdversarialResult]:
    """Random baseline. Sample uniformly in each parameter range."""
    rng = random.Random(seed)
    out: list[AdversarialResult] = []
    for _ in range(n_samples):
        sample = {
            k: rng.uniform(lo, hi)
            for k, (lo, hi) in parameter_ranges.items()
        }
        _, attempt = _run_one(perturber, base_episode, sample)
        out.append(attempt)
    return sorted(out, key=lambda r: -r.failure_score)


def grid_seeds(
    base_episode: Episode,
    perturber: EpisodePerturber,
    parameter_grid: dict[str, list[float]],
) -> list[AdversarialResult]:
    """Exhaustive cartesian product. Use only for low-dim spaces."""
    out: list[AdversarialResult] = []
    keys = list(parameter_grid.keys())
    counts = [len(parameter_grid[k]) for k in keys]
    total = 1
    for c in counts:
        total *= c

    def _enum(idx: int, partial: dict[str, float]):
        if idx == len(keys):
            _, attempt = _run_one(perturber, base_episode, partial)
            out.append(attempt)
            return
        for value in parameter_grid[keys[idx]]:
            partial[keys[idx]] = value
            _enum(idx + 1, partial)
            del partial[keys[idx]]

    _enum(0, {})
    return sorted(out, key=lambda r: -r.failure_score)


def cma_es_seeds(
    base_episode: Episode,
    perturber: EpisodePerturber,
    parameter_ranges: dict[str, tuple[float, float]],
    *,
    n_iterations: int = 8,
    population_size: int = 8,
    sigma_init: float = 0.3,
    seed: int = 0,
) -> list[AdversarialResult]:
    """Toy CMA-ES variant: maximise failure_score over the parameter space.

    Operates in normalised [0,1] coordinates per parameter, projecting
    samples back into the configured ranges before running them.
    Single-mean Gaussian search with no covariance update — adequate
    for finding adversarial seeds in low-dim spaces. For high-dim use
    install a real CMA-ES library and wire it through ``perturber``.
    """
    rng = random.Random(seed)
    keys = list(parameter_ranges.keys())
    if not keys:
        return []
    mean = [0.5] * len(keys)
    sigma = sigma_init
    history: list[AdversarialResult] = []
    for _ in range(n_iterations):
        population: list[tuple[list[float], AdversarialResult]] = []
        for _ in range(population_size):
            x = [
                max(0.0, min(1.0, mean[i] + rng.gauss(0.0, sigma)))
                for i in range(len(keys))
            ]
            sample = {
                keys[i]: parameter_ranges[keys[i]][0]
                + x[i] * (parameter_ranges[keys[i]][1] - parameter_ranges[keys[i]][0])
                for i in range(len(keys))
            }
            _, attempt = _run_one(perturber, base_episode, sample)
            population.append((x, attempt))
        history.extend(att for _, att in population)
        # Move the mean toward the top-half of the population.
        population.sort(key=lambda pa: -pa[1].failure_score)
        elites = population[: max(1, population_size // 2)]
        for i in range(len(keys)):
            mean[i] = sum(x[i] for x, _ in elites) / len(elites)
        # Anneal sigma.
        sigma *= 0.85
    return sorted(history, key=lambda r: -r.failure_score)
