"""Safe-RL training harness — train policies under the ghostloop safety pipeline.

Constrained-MDP training is the right shape for embodied agents: the
agent maximises reward, but the safety pipeline imposes hard
constraints (geofence, force cap, time window, cooldown, ...). Naive RL
either ignores constraints (unsafe) or compiles them into a single
scalar reward (loses interpretability and gets gamed).

This module ships a small framework that any policy library (PyTorch,
JAX, NumPyRO, scripted) can plug into:

  - SafeRolloutCollector drives a Runtime over an episode horizon and
    captures (obs, action, reward, info, was_denied) tuples.
  - LagrangianMultiplier tracks a Lagrange multiplier for the constraint
    violation rate (target rate set as a hyperparameter).
  - PolicyAdapter Protocol is the contract every policy must satisfy
    (callable obs -> action, callable update(rollout)).

The harness is pure-stdlib + optional numpy. No PyTorch / JAX dep —
your policy provides those.
"""

from .core import (
    LagrangianMultiplier,
    PolicyAdapter,
    Rollout,
    RolloutBatch,
    SafeRolloutCollector,
    Transition,
    train_safe,
)

__all__ = [
    "LagrangianMultiplier",
    "PolicyAdapter",
    "Rollout",
    "RolloutBatch",
    "SafeRolloutCollector",
    "Transition",
    "train_safe",
]
