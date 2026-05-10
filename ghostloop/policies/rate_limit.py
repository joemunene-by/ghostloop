"""Per-primitive rate limiting using a sliding window."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..core import Decision, Intent, Primitive


@dataclass
class RateLimitGate:
    """Allow at most ``per_minute`` calls per primitive name in any 60s window.

    Sliding window (deque of timestamps), not fixed-bucket, so a bursty caller
    can't beat it by aligning to the boundary. Per-primitive so a chatty
    ``observe`` doesn't starve a critical ``stop``.
    """

    per_minute: int = 120
    name: str = "rate_limit"
    _windows: dict[str, deque[float]] = field(default_factory=dict)

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        now = time.monotonic()
        window = self._windows.setdefault(primitive.name, deque())
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.per_minute:
            return Decision.deny(
                self.name,
                f"{primitive.name} exceeded {self.per_minute}/min "
                f"(observed {len(window)} in last 60s)",
            )
        window.append(now)
        return Decision.allow(
            self.name,
            f"{len(window)}/{self.per_minute} per minute",
        )
