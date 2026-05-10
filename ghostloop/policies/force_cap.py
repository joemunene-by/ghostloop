"""ForceCapGate: deny intents whose declared force / torque / velocity exceeds a cap.

Inspects ``intent.args`` for keys named ``force``, ``torque``, ``velocity``,
``acceleration``, or ``effort`` (case-insensitive). If any value (in the
units the intent uses) exceeds the corresponding cap, the gate denies.
Pass-through for intents without any of those keys."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core import Decision, Intent, Primitive


_CAP_KEYS = {"force", "torque", "velocity", "acceleration", "effort", "speed"}


@dataclass
class ForceCapGate:
    """Reject intents that request mechanical effort above the configured caps.

    Caps default to None (= unlimited). Set the keys you care about:

        ForceCapGate(force_max=50.0, velocity_max=1.0)

    The gate only checks keys the user configured; an intent declaring a
    velocity passes the velocity check if no velocity_max was set.
    """

    force_max: float | None = None
    torque_max: float | None = None
    velocity_max: float | None = None
    acceleration_max: float | None = None
    effort_max: float | None = None
    name: str = "force_cap"

    def _cap_for(self, key: str) -> float | None:
        return {
            "force": self.force_max,
            "torque": self.torque_max,
            "velocity": self.velocity_max,
            "speed": self.velocity_max,
            "acceleration": self.acceleration_max,
            "effort": self.effort_max,
        }.get(key)

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        for raw_key, value in intent.args.items():
            key = raw_key.lower()
            if key not in _CAP_KEYS:
                continue
            cap = self._cap_for(key)
            if cap is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if abs(v) > cap:
                return Decision.deny(
                    self.name,
                    f"{key}={v:g} exceeds cap {cap:g}",
                )
        return Decision.allow(self.name, "all force/effort values within caps")
