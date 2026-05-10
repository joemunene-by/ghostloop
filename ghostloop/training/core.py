"""Safe-RL training core: rollout collector, Lagrangian, policy adapter Protocol.

The Protocol-style design means any policy framework can plug in.
A PyTorch policy implements ``act`` (calls ``net(obs)``) and ``update``
(its own optim step); the harness drives the Runtime, captures
trajectories under the ghostloop safety pipeline, and feeds them back
to the policy's update.

Constraint accounting is first-class. Every transition records a
``violated`` flag (set by the safety pipeline DENY decisions). The
LagrangianMultiplier tracks observed-vs-target violation rate and
provides a multiplier the policy update can use:

    actor_loss = -E[ logπ(a|s) * advantage(s,a) ] + λ * E[ violation(s,a) ]

When violations are above target, λ grows -> the policy learns to avoid
denied actions; below target, λ shrinks -> back to maximising reward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from ..core import Decision, DecisionAction, Intent, Runtime, Trace


@dataclass
class Transition:
    """One environment step under the safety pipeline.

    Carries everything an RL update typically needs: pre-state,
    intended action, post-state, reward, done flag, plus the safety
    pipeline's decision so the policy can learn from BOTH "what
    happened" AND "what was forbidden".
    """

    obs: Any
    action: Any
    reward: float
    next_obs: Any
    done: bool
    violated: bool                  # True iff the safety pipeline denied
    decision: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    intent_name: str = ""


@dataclass
class Rollout:
    """One episode's worth of transitions.

    The ``return_`` field is the un-discounted episode return; ``length``
    is the number of transitions; ``violation_rate`` is what
    LagrangianMultiplier consumes during update.
    """

    transitions: list[Transition]
    started_at: float
    finished_at: float

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def return_(self) -> float:
        return sum(t.reward for t in self.transitions)

    @property
    def violation_rate(self) -> float:
        if not self.transitions:
            return 0.0
        return sum(1 for t in self.transitions if t.violated) / len(self.transitions)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


@dataclass
class RolloutBatch:
    """A batch of rollouts grouped together for a single policy update."""

    rollouts: list[Rollout]

    @property
    def n_episodes(self) -> int:
        return len(self.rollouts)

    @property
    def n_steps(self) -> int:
        return sum(r.length for r in self.rollouts)

    @property
    def mean_return(self) -> float:
        if not self.rollouts:
            return 0.0
        return sum(r.return_ for r in self.rollouts) / len(self.rollouts)

    @property
    def mean_violation_rate(self) -> float:
        total = sum(r.length for r in self.rollouts)
        if total == 0:
            return 0.0
        violations = sum(
            sum(1 for t in r.transitions if t.violated) for r in self.rollouts
        )
        return violations / total

    def all_transitions(self) -> list[Transition]:
        out: list[Transition] = []
        for r in self.rollouts:
            out.extend(r.transitions)
        return out


class PolicyAdapter(Protocol):
    """Contract every policy must implement to plug into ``train_safe``.

    ``act`` produces the next action given an observation. The Intent
    that wraps the action is constructed by ``intent_factory`` in
    ``SafeRolloutCollector``, so the policy can stay observation -> action.

    ``update`` is called once per ``RolloutBatch``; the adapter performs
    its own optim step and returns a metrics dict for logging.
    """

    def act(self, obs: Any) -> Any: ...

    def update(self, batch: RolloutBatch, lagrangian: float) -> dict[str, float]: ...


# ---------------------------------------------------------------------------
# Rollout collector
# ---------------------------------------------------------------------------


@dataclass
class SafeRolloutCollector:
    """Drive a Runtime through episodes under a Policy + the safety pipeline.

    Each ``act`` call yields an action; the collector wraps it into an
    Intent via ``intent_factory(obs, action) -> Intent``, dispatches it
    through the runtime, and records the resulting Transition. Reward
    comes from ``reward_fn(transition_partial) -> float``; when the
    Backend is a GymnasiumBackend the reward is naturally extracted from
    the result observation, but ``reward_fn`` lets you customise.
    """

    runtime: Runtime
    intent_factory: Callable[[Any, Any], Intent]
    reset_fn: Callable[[], Any] | None = None
    reward_fn: Callable[[dict[str, Any]], float] | None = None
    done_fn: Callable[[dict[str, Any]], bool] | None = None
    max_steps_per_episode: int = 200

    def collect_episode(self, policy: PolicyAdapter) -> Rollout:
        started_at = time.time()
        if self.reset_fn is not None:
            obs = self.reset_fn()
        else:
            obs = self.runtime.backend.snapshot()
        transitions: list[Transition] = []
        for _ in range(self.max_steps_per_episode):
            action = policy.act(obs)
            intent = self.intent_factory(obs, action)
            result = self.runtime.step(intent)
            decision = (
                self.runtime.trace.events[-1].decision
                if self.runtime.trace.events
                else None
            )
            violated = decision is not None and decision.action == DecisionAction.DENY
            next_obs = self.runtime.backend.snapshot()
            obs_dict = result.observation if isinstance(result.observation, dict) else {}
            reward = self.reward_fn(obs_dict) if self.reward_fn else float(obs_dict.get("reward", 0.0))
            done = self.done_fn(obs_dict) if self.done_fn else bool(
                obs_dict.get("terminated", False) or obs_dict.get("truncated", False)
            )
            transitions.append(Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                violated=violated,
                decision=decision.to_json() if decision else {},
                info=obs_dict.get("info", {}) if isinstance(obs_dict, dict) else {},
                intent_name=intent.name,
            ))
            if done:
                break
            obs = next_obs
        return Rollout(
            transitions=transitions, started_at=started_at, finished_at=time.time(),
        )

    def collect_batch(self, policy: PolicyAdapter, n_episodes: int) -> RolloutBatch:
        return RolloutBatch(
            rollouts=[self.collect_episode(policy) for _ in range(n_episodes)]
        )


# ---------------------------------------------------------------------------
# Lagrangian multiplier
# ---------------------------------------------------------------------------


@dataclass
class LagrangianMultiplier:
    """Adaptive Lagrangian for constraint violation rate.

    ``target_rate`` is the maximum acceptable violation rate (e.g.
    0.05 = 5%). The multiplier ``value`` rises when observed > target
    (penalty grows -> policy learns to avoid violations) and falls
    when observed < target (back to maximising reward).

    ``learning_rate`` controls how quickly λ adapts; ``min_value`` /
    ``max_value`` clamp the range so λ doesn't run away in degenerate
    early training.
    """

    target_rate: float = 0.05
    learning_rate: float = 0.1
    value: float = 0.0
    min_value: float = 0.0
    max_value: float = 100.0

    def update(self, observed_rate: float) -> float:
        gradient = observed_rate - self.target_rate
        new_value = self.value + self.learning_rate * gradient
        self.value = max(self.min_value, min(self.max_value, new_value))
        return self.value


# ---------------------------------------------------------------------------
# Outer training loop
# ---------------------------------------------------------------------------


def train_safe(
    collector: SafeRolloutCollector,
    policy: PolicyAdapter,
    lagrangian: LagrangianMultiplier,
    *,
    n_iterations: int = 100,
    episodes_per_iteration: int = 8,
    log_every: int = 1,
    on_iteration: Callable[[int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Top-level constrained-MDP training loop.

    Each iteration: collect ``episodes_per_iteration`` rollouts, update
    the Lagrangian on the observed violation rate, call the policy's
    update with the rollouts and the current Lagrangian. Returns the
    list of per-iteration logs.

    Notes:
      - The harness DOES NOT enforce policy update semantics; the
        adapter is free to use PPO / SAC / off-policy / scripted.
      - Violations come from the runtime's safety pipeline, not from
        the environment — that's the whole point: training under the
        safety pipeline is what makes the trained policy deployable
        without removing it.
    """
    history: list[dict[str, Any]] = []
    for it in range(n_iterations):
        batch = collector.collect_batch(policy, n_episodes=episodes_per_iteration)
        lagrangian.update(batch.mean_violation_rate)
        update_metrics = policy.update(batch, lagrangian.value)
        record = {
            "iteration": it,
            "mean_return": batch.mean_return,
            "mean_violation_rate": batch.mean_violation_rate,
            "lagrangian": lagrangian.value,
            "n_steps": batch.n_steps,
            **{f"policy/{k}": v for k, v in update_metrics.items()},
        }
        history.append(record)
        if on_iteration is not None:
            on_iteration(it, record)
        elif log_every > 0 and it % log_every == 0:
            # Cheap default progress line.
            print(
                f"[train_safe] iter={it} return={record['mean_return']:.3f} "
                f"violation_rate={record['mean_violation_rate']:.3f} "
                f"lambda={record['lagrangian']:.3f}"
            )
    return history
