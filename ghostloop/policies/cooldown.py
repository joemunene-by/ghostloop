"""CooldownGate: minimum interval between calls to the same primitive.

RateLimitGate caps total calls per minute. CooldownGate enforces a
hard minimum delay between consecutive invocations of the same
primitive — useful for "don't call the model again within 5 seconds"
or "don't open the gripper twice in under 200ms" patterns where
absolute spacing matters more than aggregate rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core import Decision, Intent, Primitive


@dataclass
class CooldownGate:
    """Reject ``primitive`` calls that arrive within ``cooldown_s`` of the previous one.

    The gate is per-primitive: scan and pick have independent timers.
    The cooldown defaults can be overridden per primitive name via
    the ``per_primitive`` map.
    """

    default_s: float = 0.0
    per_primitive: dict[str, float] = field(default_factory=dict)
    name: str = "cooldown"
    _last_call: dict[str, float] = field(default_factory=dict)

    def _required(self, name: str) -> float:
        if name in self.per_primitive:
            return self.per_primitive[name]
        return self.default_s

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        cooldown = self._required(primitive.name)
        if cooldown <= 0:
            return Decision.allow(self.name, "no cooldown configured")
        now = time.monotonic()
        last = self._last_call.get(primitive.name)
        if last is not None:
            elapsed = now - last
            if elapsed < cooldown:
                wait = cooldown - elapsed
                return Decision.deny(
                    self.name,
                    f"{primitive.name} on cooldown ({elapsed:.3g}s < {cooldown:g}s, "
                    f"wait {wait:.3g}s)",
                )
        self._last_call[primitive.name] = now
        return Decision.allow(self.name, f"{primitive.name} cooldown clear")
