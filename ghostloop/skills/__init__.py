"""Skill graph — DAG of robot skills with prerequisite + refinement edges.

A skill is a primitive (or composite primitive) tagged with two kinds
of edges:

  - prerequisites: skills that must be available before this one is
    callable ("you can't pick before you can move_to")
  - refines: optional parent — this skill is a more-specific variant
    of another ("approach_grasp_top_down refines approach_grasp")

The graph lets you reason about coverage ("what skills do I need to
support to enable mission X?") and reuse skills across robots
(morphology adapters can register their concrete primitive under
the appropriate generic skill name).

Stdlib only — Kahn's algorithm for cycle detection + topological
ordering.
"""

from .graph import (
    Skill,
    SkillGraph,
    SkillError,
    skill_from_primitive,
)

__all__ = [
    "Skill",
    "SkillGraph",
    "SkillError",
    "skill_from_primitive",
]
