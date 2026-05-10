"""TimeWindowGate: only allow primitives during configured time windows.

Common ops constraint: the warehouse robot can drive 06:00-22:00 local
but not overnight; the agricultural robot only operates between sunrise
and sunset; the prototype lab arm is only allowed to move during
business hours so a human is around. The gate uses the system's local
clock by default — pass an explicit ``now`` callable for testing or for
robots running in different timezones from the operator.

Windows are open-ended on either side; pass-through for primitives not
in the configured map.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Callable

from ..core import Decision, Intent, Primitive


TimeOfDay = _dt.time


@dataclass
class Window:
    """A daily time window. ``end_before_start=True`` means the window
    crosses midnight (e.g. 22:00 -> 06:00 night ops).
    """

    start: TimeOfDay
    end: TimeOfDay
    end_before_start: bool = False

    def contains(self, t: TimeOfDay) -> bool:
        if self.end_before_start:
            return t >= self.start or t < self.end
        return self.start <= t < self.end


@dataclass
class TimeWindowGate:
    """Reject calls to primitives whose configured windows don't include now.

    ``per_primitive`` maps primitive name -> list of Windows. Primitives
    NOT in the map pass through (gate doesn't apply).
    """

    per_primitive: dict[str, list[Window]] = field(default_factory=dict)
    now: Callable[[], _dt.datetime] = field(default=lambda: _dt.datetime.now())
    name: str = "time_window"

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        windows = self.per_primitive.get(primitive.name)
        if not windows:
            return Decision.allow(self.name, "no time window configured")
        current = self.now().time()
        for w in windows:
            if w.contains(current):
                return Decision.allow(
                    self.name,
                    f"current {current.strftime('%H:%M')} inside window "
                    f"{w.start.strftime('%H:%M')}-{w.end.strftime('%H:%M')}",
                )
        descriptions = ", ".join(
            f"{w.start.strftime('%H:%M')}-{w.end.strftime('%H:%M')}" for w in windows
        )
        return Decision.deny(
            self.name,
            f"current {current.strftime('%H:%M')} outside any allowed "
            f"window for {primitive.name}: {descriptions}",
        )
