"""Domain-randomized Backend wrapper for sim-to-real transfer.

How sim-trained agents survive in the real world: train under noise.
Position observations gain Gaussian noise; step calls gain timing
jitter; some fraction of action commands silently fail. The policy
converges to a robust solution because it cannot rely on any single
sensor or actuator being clean.

`RandomizedBackend(base, ...)` wraps any Backend and applies a
configurable set of perturbations to its `snapshot()` and to whatever
`apply_*` calls the inner primitives make. The wrapper is itself a
Backend so the policy / runtime / safety pipeline is unchanged.

Three perturbations available:

  pos_noise_std       Gaussian noise on (x, y, z) entries in snapshots
  timing_jitter_s     uniform jitter added to each apply_action sleep
  action_drop_prob    probability that an action is silently dropped

All deterministic given a `seed`; reproducibility for bench harnesses.

Stdlib only (`random`, `time`, `dataclasses`) — no numpy required.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class Backend(Protocol):
    name: str
    def snapshot(self) -> dict[str, Any]: ...


@dataclass
class RandomizationConfig:
    """One stop for every randomization knob."""

    pos_noise_std: float = 0.0
    """Gaussian std-dev added to (x, y, z) snapshot fields. 0 disables."""

    timing_jitter_s: float = 0.0
    """Uniform jitter in [-J, +J] applied as a synthetic sleep before each step."""

    action_drop_prob: float = 0.0
    """Probability that ``apply_action`` is silently dropped (no-op + warning in info)."""

    snapshot_dropout: list[str] = field(default_factory=list)
    """Field names randomly removed from snapshots to simulate sensor dropout."""

    snapshot_dropout_prob: float = 0.0
    """Probability that any of ``snapshot_dropout`` fields are dropped per snapshot."""

    sticky_action_prob: float = 0.0
    """Probability that the previous action is repeated instead of the requested one
    (Atari-style sticky actions; tests robustness to delayed control)."""


@dataclass
class RandomizedBackend:
    """Wraps any Backend with reproducible perturbations.

    The wrapped backend is treated opaquely — we only mutate snapshots
    and intercept commonly-named action methods (``apply_action``,
    ``apply_target``, ``apply_torque``). Custom backend methods pass
    through untouched.
    """

    base: Any
    config: RandomizationConfig = field(default_factory=RandomizationConfig)
    seed: int | None = None
    name: str = "randomized"

    _rng: random.Random = field(init=False, default=None)
    _last_action: Any = field(default=None, init=False, repr=False)
    _drop_count: int = field(default=0, init=False, repr=False)
    _sticky_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        # Inherit the wrapped backend's name if not overridden so traces
        # remain meaningful (e.g. "randomized<mujoco:franka>").
        if self.name == "randomized" and hasattr(self.base, "name"):
            self.name = f"randomized<{self.base.name}>"

    # ------------------------------------------------------------------
    # Backend Protocol — snapshot is the only required method.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        snap = dict(self.base.snapshot())
        if self.config.pos_noise_std > 0:
            for axis in ("x", "y", "z"):
                if axis in snap:
                    try:
                        snap[axis] = float(snap[axis]) + self._rng.gauss(
                            0.0, self.config.pos_noise_std
                        )
                    except (TypeError, ValueError):
                        pass
            if "position" in snap and isinstance(snap["position"], (list, tuple)):
                pos = list(snap["position"])
                for i in range(min(3, len(pos))):
                    try:
                        pos[i] = float(pos[i]) + self._rng.gauss(
                            0.0, self.config.pos_noise_std
                        )
                    except (TypeError, ValueError):
                        pass
                snap["position"] = pos
        # Random sensor dropout.
        if self.config.snapshot_dropout and self.config.snapshot_dropout_prob > 0:
            for k in self.config.snapshot_dropout:
                if k in snap and self._rng.random() < self.config.snapshot_dropout_prob:
                    del snap[k]
        snap["_randomized"] = True
        return snap

    # ------------------------------------------------------------------
    # Action interception — proxy any method that looks like an actuator
    # call, applying jitter / drop / sticky-repeat as configured.
    # ------------------------------------------------------------------

    def apply_action(self, action: Any) -> dict[str, Any]:
        return self._maybe_apply("apply_action", action)

    def apply_target(self, target: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._maybe_apply("apply_target", target, *args, **kwargs)

    def apply_torque(self, torques: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._maybe_apply("apply_torque", torques, *args, **kwargs)

    def reset_env(self) -> dict[str, Any]:
        if hasattr(self.base, "reset_env"):
            return self.base.reset_env()
        return {}

    def _maybe_apply(self, method: str, action: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # Timing jitter — synthetic sleep before any call so policies
        # learn to tolerate variable control-loop period.
        if self.config.timing_jitter_s > 0:
            jitter = self._rng.uniform(
                -self.config.timing_jitter_s, self.config.timing_jitter_s
            )
            if jitter > 0:
                time.sleep(jitter)
        # Action drop — silent no-op.
        if self._rng.random() < self.config.action_drop_prob:
            self._drop_count += 1
            return {
                "_randomized": True,
                "dropped": True,
                "drop_count": self._drop_count,
            }
        # Sticky action — repeat the last successful action.
        if (
            self._last_action is not None
            and self._rng.random() < self.config.sticky_action_prob
        ):
            self._sticky_count += 1
            action = self._last_action
        self._last_action = action
        target = getattr(self.base, method, None)
        if target is None:
            raise AttributeError(
                f"wrapped backend {self.base.__class__.__name__} has no {method}"
            )
        result = target(action, *args, **kwargs)
        if isinstance(result, dict):
            result = {**result, "_randomized": True}
        return result

    # ------------------------------------------------------------------
    # Diagnostics — counters useful for sim2real progress reports.
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "actions_dropped": self._drop_count,
            "actions_sticky": self._sticky_count,
        }
