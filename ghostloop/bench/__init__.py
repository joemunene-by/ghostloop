"""ghostloop.bench — statistically-rigorous episode benchmarking.

Mirrors the GhostBench pattern from GhostLM:
  - Wilson 95% confidence intervals for binomial proportions
  - McNemar's exact test for paired comparisons (same episodes, two policies)
  - Cohen's h effect-size labelling (small / medium / large)

A bench is a list of ``Episode``s. Each episode declares an initial backend
state, a goal description, and a ``success_predicate`` that scores the final
trace. The harness runs every episode against a policy callable and produces
a ``RunReport`` with per-episode pass/fail + summary statistics. Two run
reports on the same bench can be paired-compared for significance.
"""

from .episode import Episode, EpisodeRunner, EpisodeResult
from .report import RunReport, summarize, wilson_ci
from .compare import paired_compare, mcnemar_p, cohens_h, effect_label

__all__ = [
    "Episode",
    "EpisodeRunner",
    "EpisodeResult",
    "RunReport",
    "summarize",
    "wilson_ci",
    "paired_compare",
    "mcnemar_p",
    "cohens_h",
    "effect_label",
]
