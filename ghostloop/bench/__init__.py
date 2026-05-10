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
from .catalogue import (
    geofence_violations,
    pick_and_place_pairs,
    preset_geofence_smoke,
    preset_pick_and_place_4,
    preset_reach_8,
    reach_targets,
    scan_at_targets,
)
from .reward_shaper import (
    CustomReward,
    OnDecision,
    OnObservation,
    OnPrimitive,
    RewardComponent,
    RewardShaper,
    StepCost,
    from_dict as reward_shaper_from_dict,
)
from .sim2real import Sim2RealBench, Sim2RealReport

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
    # catalogue
    "reach_targets",
    "pick_and_place_pairs",
    "scan_at_targets",
    "geofence_violations",
    "preset_reach_8",
    "preset_pick_and_place_4",
    "preset_geofence_smoke",
    # reward shaper
    "CustomReward",
    "OnDecision",
    "OnObservation",
    "OnPrimitive",
    "RewardComponent",
    "RewardShaper",
    "StepCost",
    "reward_shaper_from_dict",
    # sim2real
    "Sim2RealBench",
    "Sim2RealReport",
]
