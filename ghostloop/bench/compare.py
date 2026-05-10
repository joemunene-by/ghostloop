"""Paired-comparison statistics: McNemar's exact test + Cohen's h effect size."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .episode import EpisodeResult
from .report import RunReport, wilson_ci


@dataclass
class PairedComparison:
    """Comparison of two RunReports on the same bench (same episodes).

    The key statistic is McNemar's: counts the number of episodes where A
    passed and B failed (``a_only``) versus the reverse (``b_only``). The
    null hypothesis is that the two policies are interchangeable on this
    bench. Pairs where both pass or both fail are uninformative and dropped
    from the test (standard McNemar — only discordant pairs matter).
    """

    bench_name: str
    a_name: str
    b_name: str
    n: int
    a_passed: int
    b_passed: int
    both_passed: int
    only_a: int  # A passed, B failed
    only_b: int  # B passed, A failed
    neither: int
    p_value: float
    effect_h: float

    @property
    def a_rate(self) -> float:
        return self.a_passed / self.n if self.n else 0.0

    @property
    def b_rate(self) -> float:
        return self.b_passed / self.n if self.n else 0.0

    @property
    def diff(self) -> float:
        return self.b_rate - self.a_rate

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def render_md(self) -> str:
        a_lo, a_hi = wilson_ci(self.a_passed, self.n)
        b_lo, b_hi = wilson_ci(self.b_passed, self.n)
        return (
            f"# Paired comparison: {self.b_name} vs {self.a_name}\n\n"
            f"Bench: **{self.bench_name}** · n={self.n}\n\n"
            f"| Run | Passed | Rate | Wilson 95% CI |\n"
            f"|---|---:|---:|---|\n"
            f"| {self.a_name} | {self.a_passed} | {self.a_rate:.1%} | "
            f"[{a_lo:.1%}, {a_hi:.1%}] |\n"
            f"| {self.b_name} | {self.b_passed} | {self.b_rate:.1%} | "
            f"[{b_lo:.1%}, {b_hi:.1%}] |\n\n"
            f"**Discordant pairs**: only-A={self.only_a}, only-B={self.only_b}\n\n"
            f"**McNemar exact p**: {self.p_value:.4f}"
            f" {'(significant at α=0.05)' if self.significant else '(not significant)'}\n\n"
            f"**Cohen's h**: {self.effect_h:+.3f} ({effect_label(self.effect_h)})\n"
        )


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value via the binomial distribution.

    With b discordant pairs out of (a + b), the test asks whether b is
    extreme enough to reject the null p=0.5. Uses log-arithmetic to stay
    numerically stable for large pair counts.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    # Probability of ≤k or ≥(n−k) under Binom(n, 0.5).
    log_half_n = -n * math.log(2.0)
    log_terms = []
    for i in range(k + 1):
        log_binom = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        log_terms.append(log_binom + log_half_n)
    one_tail = sum(math.exp(t) for t in log_terms)
    p = min(1.0, 2 * one_tail)
    return p


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions: arcsine-transformed difference.

    Standard small/medium/large thresholds: |h| in [0.2, 0.5, 0.8].
    """
    def phi(p: float) -> float:
        return 2 * math.asin(math.sqrt(max(0.0, min(1.0, p))))
    return phi(p2) - phi(p1)


def effect_label(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def paired_compare(a: RunReport, b: RunReport) -> PairedComparison:
    """Pair a vs b episode-by-episode (matched on ``episode_name``)."""
    if a.bench_name != b.bench_name:
        raise ValueError(
            f"benches differ: {a.bench_name!r} vs {b.bench_name!r}"
        )
    a_by_name = {r.episode_name: r for r in a.results}
    b_by_name = {r.episode_name: r for r in b.results}
    common = sorted(a_by_name.keys() & b_by_name.keys())
    if not common:
        raise ValueError("no shared episodes between the two reports")

    both = only_a = only_b = neither = 0
    for name in common:
        ap = a_by_name[name].passed
        bp = b_by_name[name].passed
        if ap and bp:
            both += 1
        elif ap and not bp:
            only_a += 1
        elif not ap and bp:
            only_b += 1
        else:
            neither += 1

    n = len(common)
    a_passed = both + only_a
    b_passed = both + only_b
    p = mcnemar_p(only_a, only_b)
    h = cohens_h(a_passed / n, b_passed / n)
    return PairedComparison(
        bench_name=a.bench_name,
        a_name=a.run_name,
        b_name=b.run_name,
        n=n,
        a_passed=a_passed,
        b_passed=b_passed,
        both_passed=both,
        only_a=only_a,
        only_b=only_b,
        neither=neither,
        p_value=p,
        effect_h=h,
    )
