"""Action smoothing — velocity / acceleration limits between consecutive moves.

VLA models and freshly-trained RL policies emit jerky action sequences:
the controller commands a 30cm hop in one step then a 5mm tap the next.
On real hardware that translates to torque spikes and overshoot. This
module provides two complementary tools:

- ``ActionSmoothingGate`` (PolicyGate) — denies a motion intent whose
  implied velocity / acceleration exceeds configured bounds. Pairs
  with the existing ForceCapGate to give a complete motion-limiting
  story (force AND speed AND smoothness all gated independently).

- ``smooth_target(prev, target, max_step)`` — pure helper for
  primitives that prefer to *clip* rather than *deny*. Useful inside
  a custom Primitive's ``call`` when you want safe motion regardless
  of policy behaviour.

The gate inspects ``intent.args`` for either ``target=(x,y,z)`` or
explicit ``x/y/z``, and (optionally) ``duration`` to compute velocity.
Without duration it falls back to position deltas only, treating each
intent as a discrete step.

Pure stdlib math.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

from ..core import Decision, Intent, Primitive


Point3 = tuple[float, float, float]


def _extract_target(args: dict) -> Point3 | None:
    if "target" in args:
        t: Sequence[float] = args["target"]
        if len(t) == 3:
            return float(t[0]), float(t[1]), float(t[2])
    if all(k in args for k in ("x", "y", "z")):
        return float(args["x"]), float(args["y"]), float(args["z"])
    return None


def _distance(a: Point3, b: Point3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


@dataclass
class ActionSmoothingGate:
    """Deny motion intents whose implied velocity / acceleration exceeds bounds.

    Tracks the most-recent target per primitive name and the timestamp
    when it was set. On each call:

      step_distance = ||new_target - prev_target||
      dt = max(now - prev_time, 1e-3)        # avoid divide-by-zero
      velocity = step_distance / dt
      acceleration = (velocity - prev_velocity) / dt

    Denies if velocity > ``max_velocity`` or acceleration >
    ``max_acceleration``. First call for any primitive is always
    allowed (no prev to compare).

    Primitives without a 3-axis target pass through unchanged.
    """

    max_velocity: float = 1.0          # m/s
    max_acceleration: float = 5.0       # m/s^2
    per_primitive_velocity: dict[str, float] = field(default_factory=dict)
    per_primitive_acceleration: dict[str, float] = field(default_factory=dict)
    name: str = "action_smoothing"

    _last_target: dict[str, Point3] = field(default_factory=dict, init=False, repr=False)
    _last_time: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _last_velocity: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def reset(self, primitive_name: str | None = None) -> None:
        """Forget the last-seen state for one or all primitives.

        Useful between episodes so a long pause doesn't get treated as
        a low-velocity move and let the next intent slip through with
        unrealistic computed acceleration.
        """
        if primitive_name is None:
            self._last_target.clear()
            self._last_time.clear()
            self._last_velocity.clear()
        else:
            self._last_target.pop(primitive_name, None)
            self._last_time.pop(primitive_name, None)
            self._last_velocity.pop(primitive_name, None)

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        target = _extract_target(intent.args)
        if target is None:
            return Decision.allow(self.name, "no positional target in intent")
        v_max = self.per_primitive_velocity.get(primitive.name, self.max_velocity)
        a_max = self.per_primitive_acceleration.get(primitive.name, self.max_acceleration)
        prev = self._last_target.get(primitive.name)
        if prev is None:
            self._last_target[primitive.name] = target
            self._last_time[primitive.name] = time.monotonic()
            self._last_velocity[primitive.name] = 0.0
            return Decision.allow(self.name, "first call — no smoothing baseline")
        now = time.monotonic()
        dt = max(now - self._last_time[primitive.name], 1e-3)
        step = _distance(target, prev)
        velocity = step / dt
        if velocity > v_max:
            return Decision.deny(
                self.name,
                f"velocity {velocity:.3g} m/s > max {v_max:g} (step={step:.3g} dt={dt:.3g})",
            )
        prev_v = self._last_velocity.get(primitive.name, 0.0)
        accel = (velocity - prev_v) / dt
        if abs(accel) > a_max:
            return Decision.deny(
                self.name,
                f"acceleration {accel:.3g} m/s^2 > max {a_max:g}",
            )
        # Update state on ALLOW so the next intent compares against the freshest values.
        self._last_target[primitive.name] = target
        self._last_time[primitive.name] = now
        self._last_velocity[primitive.name] = velocity
        return Decision.allow(
            self.name,
            f"v={velocity:.3g} a={accel:.3g} within bounds",
        )


def smooth_target(prev: Point3, target: Point3, max_step: float) -> Point3:
    """Clip ``target`` so it's at most ``max_step`` away from ``prev``.

    If ``target`` is already close enough, returns it unchanged.
    Otherwise interpolates to the boundary along the direction prev->target.
    Useful inside a custom Primitive ``call`` when the policy
    occasionally emits unsafe deltas and you'd rather correct than deny.
    """
    step = _distance(target, prev)
    if step <= max_step:
        return target
    scale = max_step / step
    return (
        prev[0] + (target[0] - prev[0]) * scale,
        prev[1] + (target[1] - prev[1]) * scale,
        prev[2] + (target[2] - prev[2]) * scale,
    )
