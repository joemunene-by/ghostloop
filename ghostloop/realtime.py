"""Real-time deadline scheduler — detect missed deadlines in the agent loop.

Robotics control loops have hard timing requirements: a 100 Hz arm
controller can't pause for 50ms while an LLM thinks. The v0.3
``AsyncRuntime`` provides rate-Hz pacing but no detection of missed
deadlines — a loop that lags doesn't know it's lagging.

This module adds:

  - ``DeadlineMonitor`` — observes step timestamps, computes the period
    between them, and emits a violation when the period exceeds a
    configurable threshold. Tracks rolling jitter (std of period
    around the target) so the operator can dashboard timing health.
  - ``DeadlineMissed`` — typed event the monitor produces. Hooks into
    the existing trace + alarm infrastructure.
  - ``deadline_gate(monitor)`` — a PolicyGate factory that DENIES the
    next intent if the loop is currently in a deadline-violated state.
    Use this to escalate to a safe-fallback policy when timing tanks.

Pure-stdlib + the v0.6 alarm registry surface for easy production
integration.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import Decision, Intent, Primitive


@dataclass
class DeadlineMissed:
    """One missed-deadline event."""

    timestamp: float
    expected_period_s: float
    actual_period_s: float
    overrun_ratio: float                     # actual / expected (>1.0 means missed)
    rolling_jitter_s: float                  # std of recent periods
    consecutive_misses: int

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "expected_period_s": round(self.expected_period_s, 6),
            "actual_period_s": round(self.actual_period_s, 6),
            "overrun_ratio": round(self.overrun_ratio, 4),
            "rolling_jitter_s": round(self.rolling_jitter_s, 6),
            "consecutive_misses": self.consecutive_misses,
        }


@dataclass
class DeadlineMonitor:
    """Tracks step-to-step periods against a target.

    Construct with the target rate (Hz) and a tolerance multiplier
    (overrun_threshold). Every call to ``observe()`` records one step
    timestamp; periods exceeding ``target_period_s * overrun_threshold``
    fire an entry into ``misses`` and increment ``consecutive_misses``.
    A period within tolerance resets the consecutive counter.

    Args:
        target_hz: desired control rate.
        overrun_threshold: ratio (actual / expected) above which the
            period counts as a missed deadline. Default 1.2 = 20%
            slack.
        history_size: rolling-window size for jitter computation.
        on_miss: optional callback fired with each ``DeadlineMissed``.
            Wire to AlarmRegistry for production setups.
    """

    target_hz: float
    overrun_threshold: float = 1.2
    history_size: int = 64
    on_miss: Callable[[DeadlineMissed], None] | None = None

    _periods: deque = field(default_factory=lambda: deque(maxlen=64))
    _last_ts: float | None = field(default=None, init=False)
    _consecutive_misses: int = field(default=0, init=False)
    misses: list[DeadlineMissed] = field(default_factory=list)

    @property
    def target_period_s(self) -> float:
        return 1.0 / max(self.target_hz, 1e-9)

    @property
    def threshold_s(self) -> float:
        return self.target_period_s * self.overrun_threshold

    def observe(self, timestamp: float | None = None) -> DeadlineMissed | None:
        """Record one step timestamp. Returns a DeadlineMissed if this
        step exceeded the threshold, else None."""
        ts = timestamp if timestamp is not None else time.monotonic()
        if self._last_ts is None:
            self._last_ts = ts
            return None
        period = ts - self._last_ts
        self._last_ts = ts
        # History queue uses the same maxlen as configured; overwrite the
        # default deque if the user changed history_size after init.
        if self._periods.maxlen != self.history_size:
            self._periods = deque(self._periods, maxlen=self.history_size)
        self._periods.append(period)
        if period <= self.threshold_s:
            self._consecutive_misses = 0
            return None
        self._consecutive_misses += 1
        jitter = statistics.pstdev(self._periods) if len(self._periods) > 1 else 0.0
        miss = DeadlineMissed(
            timestamp=ts,
            expected_period_s=self.target_period_s,
            actual_period_s=period,
            overrun_ratio=period / self.target_period_s,
            rolling_jitter_s=jitter,
            consecutive_misses=self._consecutive_misses,
        )
        self.misses.append(miss)
        if self.on_miss is not None:
            try:
                self.on_miss(miss)
            except Exception:  # noqa: BLE001
                pass
        return miss

    def reset(self) -> None:
        """Forget timing history. Call between episodes."""
        self._last_ts = None
        self._periods.clear()
        self._consecutive_misses = 0
        self.misses.clear()

    @property
    def in_violation(self) -> bool:
        """True iff the most recent observation missed the deadline."""
        return self._consecutive_misses > 0

    def stats(self) -> dict[str, Any]:
        if not self._periods:
            return {
                "target_hz": self.target_hz,
                "n_observations": 0,
                "n_misses": len(self.misses),
            }
        periods = list(self._periods)
        return {
            "target_hz": self.target_hz,
            "n_observations": len(periods),
            "n_misses": len(self.misses),
            "min_period_s": round(min(periods), 6),
            "max_period_s": round(max(periods), 6),
            "mean_period_s": round(statistics.mean(periods), 6),
            "median_period_s": round(statistics.median(periods), 6),
            "p99_period_s": round(
                statistics.quantiles(periods, n=100)[-1] if len(periods) >= 100 else max(periods),
                6,
            ),
            "jitter_s": round(statistics.pstdev(periods) if len(periods) > 1 else 0.0, 6),
            "in_violation": self.in_violation,
        }


@dataclass
class DeadlineGate:
    """PolicyGate that denies new intents while in a deadline-violated state.

    Drop into a PolicyPipeline AFTER the runtime's per-step
    ``monitor.observe()`` call. Useful to trigger a safe-fallback when
    timing degrades (e.g. force the runtime into a hold pattern + raise
    an alarm).

    Args:
        monitor: a shared ``DeadlineMonitor`` that the outer control
            loop also calls ``observe()`` on.
        max_consecutive: gate denies once this many consecutive misses
            have happened. Default 3 — single transient slow steps
            don't trigger.
        denied_message: included in the deny reason for trace clarity.
    """

    monitor: DeadlineMonitor
    max_consecutive: int = 3
    denied_message: str = "loop is missing deadlines; transition to safe-fallback"
    name: str = "deadline"

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        if self.monitor._consecutive_misses < self.max_consecutive:
            return Decision.allow(
                self.name,
                f"timing OK ({self.monitor._consecutive_misses}/{self.max_consecutive} consecutive misses)",
            )
        last = self.monitor.misses[-1] if self.monitor.misses else None
        detail = ""
        if last is not None:
            detail = (
                f" — last actual_period={last.actual_period_s:.3g}s "
                f"expected={last.expected_period_s:.3g}s "
                f"overrun={last.overrun_ratio:.2f}x"
            )
        return Decision.deny(
            self.name,
            f"{self.denied_message}{detail}",
        )
