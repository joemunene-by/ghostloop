"""Hindsight Experience Replay (HER) for goal-conditioned policies.

Andrychowicz et al. 2017's classic insight: if a robot tries to reach
goal A but ends up at B, the trajectory IS a successful demonstration
of "reach B". Relabel the trajectory's goals retroactively, recompute
rewards, and feed the relabeled experience back into training. Sparse
goal-conditioned tasks become dense.

ghostloop's training harness ships ``Rollout`` records with
(obs, action, reward, next_obs, done, ...) per transition. This
module relabels those rollouts:

  - ``hindsight_relabel(rollout, goal_extractor, reward_fn)``:
    re-tag each transition's goal with the trajectory's achieved
    end-state and recompute reward against the new goal.
  - ``HindsightStrategy``: future / final / random / episode strategies
    for picking which achieved state becomes the new goal.

Pure-stdlib + the v0.8 Rollout types.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .core import Rollout, Transition


GoalExtractor = Callable[[Any], Any]
RewardFn = Callable[[Any, Any], float]


class HindsightStrategy(str, Enum):
    """Which achieved state becomes the new goal."""
    FINAL = "final"          # always use the final state (simplest)
    FUTURE = "future"        # for each transition i, sample a state from i..end
    EPISODE = "episode"      # sample one state per relabeled rollout
    RANDOM = "random"        # uniform random achieved state


def hindsight_relabel(
    rollout: Rollout,
    *,
    goal_extractor: GoalExtractor,
    reward_fn: RewardFn,
    strategy: HindsightStrategy = HindsightStrategy.FINAL,
    n_per_transition: int = 1,
    seed: int = 0,
) -> Rollout:
    """Return a new Rollout with relabeled goals + recomputed rewards.

    Args:
        rollout: source trajectory.
        goal_extractor: ``(obs) -> goal`` — pulls the goal-relevant
            field out of an observation. Often this is just an
            attribute select like ``lambda o: o["position"]``.
        reward_fn: ``(achieved, goal) -> float`` — sparse +1 / 0 in
            the original HER paper, but supports any dense fn.
        strategy: how to pick the achieved state for each transition.
        n_per_transition: how many extra HER transitions to spawn per
            original transition (FUTURE / RANDOM strategies). 1 = one
            relabeled transition per source.
        seed: reproducible RNG.

    Returns:
        New Rollout containing the relabeled transitions. Original
        rollout is unchanged.
    """
    rng = random.Random(seed)
    if not rollout.transitions:
        return Rollout(
            transitions=[], started_at=rollout.started_at,
            finished_at=rollout.finished_at,
        )
    n = len(rollout.transitions)
    new_transitions: list[Transition] = []
    final_goal = goal_extractor(rollout.transitions[-1].next_obs)
    episode_idx = rng.randint(0, n - 1)
    episode_goal = goal_extractor(rollout.transitions[episode_idx].next_obs)
    for i, tr in enumerate(rollout.transitions):
        for _ in range(n_per_transition):
            if strategy is HindsightStrategy.FINAL:
                new_goal = final_goal
            elif strategy is HindsightStrategy.FUTURE:
                if i >= n:
                    continue
                j = rng.randint(i, n - 1)
                new_goal = goal_extractor(rollout.transitions[j].next_obs)
            elif strategy is HindsightStrategy.EPISODE:
                new_goal = episode_goal
            elif strategy is HindsightStrategy.RANDOM:
                j = rng.randint(0, n - 1)
                new_goal = goal_extractor(rollout.transitions[j].next_obs)
            else:
                new_goal = final_goal
            achieved = goal_extractor(tr.next_obs)
            new_reward = float(reward_fn(achieved, new_goal))
            relabeled_obs = _attach_goal(tr.obs, new_goal)
            relabeled_next = _attach_goal(tr.next_obs, new_goal)
            new_transitions.append(Transition(
                obs=relabeled_obs,
                action=tr.action,
                reward=new_reward,
                next_obs=relabeled_next,
                done=tr.done,
                violated=tr.violated,
                decision=tr.decision,
                info={**(tr.info or {}), "her_goal": _to_jsonable(new_goal)},
                intent_name=tr.intent_name,
            ))
    return Rollout(
        transitions=new_transitions,
        started_at=rollout.started_at,
        finished_at=rollout.finished_at,
    )


def sparse_indicator_reward(threshold: float = 0.05) -> RewardFn:
    """Standard HER reward: +1 if achieved within threshold of goal, else 0.

    Goal and achieved are assumed to be 1D / 3D iterables (most common:
    (x, y, z) end-effector positions).
    """
    def _reward(achieved: Any, goal: Any) -> float:
        try:
            ach = list(achieved) if not isinstance(achieved, (int, float)) else [achieved]
            g = list(goal) if not isinstance(goal, (int, float)) else [goal]
        except TypeError:
            return 0.0
        if len(ach) != len(g):
            return 0.0
        sq = sum((float(a) - float(b)) ** 2 for a, b in zip(ach, g, strict=False))
        return 1.0 if sq ** 0.5 <= threshold else 0.0
    return _reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attach_goal(obs: Any, goal: Any) -> Any:
    """Add a ``goal`` field to an observation; preserve dict shape, wrap others."""
    if isinstance(obs, dict):
        out = dict(obs)
        out["goal"] = _to_jsonable(goal)
        return out
    return {"obs": obs, "goal": _to_jsonable(goal)}


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, bool, str)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if hasattr(v, "tolist"):
        try:
            return v.tolist()
        except Exception:  # noqa: BLE001
            pass
    return repr(v)
