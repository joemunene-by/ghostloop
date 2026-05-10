"""Run reports and Wilson 95% CIs for binomial proportions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .episode import EpisodeResult


@dataclass
class RunReport:
    """Summary of one policy's results across one bench."""

    run_name: str
    bench_name: str
    results: list[EpisodeResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_ci(self.passed, self.n)

    def to_json(self) -> dict[str, Any]:
        lo, hi = self.ci
        return {
            "run_name": self.run_name,
            "bench_name": self.bench_name,
            "n": self.n,
            "passed": self.passed,
            "rate": round(self.rate, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "results": [r.to_json() for r in self.results],
        }

    def render_md(self) -> str:
        lo, hi = self.ci
        return (
            f"# {self.bench_name} — {self.run_name}\n\n"
            f"**Pass rate**: {self.passed}/{self.n} = {self.rate:.1%} "
            f"(Wilson 95% CI [{lo:.1%}, {hi:.1%}])\n\n"
            "| Episode | Passed | Steps | Notes |\n"
            "|---|---:|---:|---|\n"
            + "\n".join(
                f"| {r.episode_name} | {'✓' if r.passed else '✗'} | "
                f"{r.steps} | {r.notes or ''} |"
                for r in self.results
            )
            + "\n"
        )


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well-behaved for small n and edge proportions.

    Standard Normal-approximation breaks at p≈0 / p≈1 / small n; Wilson is the
    accepted fix and it's what GhostBench uses for the cybersec bets too.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = p + z2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, (center - half) / denom)
    hi = min(1.0, (center + half) / denom)
    return (lo, hi)


def summarize(results: list[EpisodeResult], run_name: str, bench_name: str) -> RunReport:
    return RunReport(run_name=run_name, bench_name=bench_name, results=results)
