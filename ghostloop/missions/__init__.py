"""Mission orchestrator: multi-step missions with dependencies + retry semantics.

A Mission is a directed-acyclic graph of Steps, where each Step produces
an Intent (or a sequence of Intents from a Planner) and may depend on
earlier Steps having completed successfully. The MissionRunner executes
the DAG: ready steps go into a queue, dependent steps unlock as their
parents finish, failed steps optionally retry per the configured policy.

Use cases:
  - Multi-stage warehouse picks ("scan -> approach -> pick -> transit -> drop -> verify").
  - Inspection routes with conditional branches ("scan, if hot-spot then take_video else continue").
  - Nightly maintenance schedules with cross-robot dependencies.
"""

from .core import (
    Mission,
    MissionResult,
    MissionRunner,
    MissionStatus,
    Step,
    StepResult,
    StepStatus,
)

__all__ = [
    "Mission",
    "MissionResult",
    "MissionRunner",
    "MissionStatus",
    "Step",
    "StepResult",
    "StepStatus",
]
