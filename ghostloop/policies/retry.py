"""RetryPolicy: re-dispatch transient ERROR results from the runtime.

Wraps a Runtime so that ``step(intent)`` automatically retries on errors
matching a configurable predicate. Backoff is exponential with optional
jitter; max_attempts caps the loop. BLOCKED results are NOT retried —
those came from the safety pipeline and the policy should respect them.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from ..core import Intent, Result, ResultStatus, Runtime


def is_transient_error(result: Result) -> bool:
    """Default error classifier — catches network / timeout / busy patterns."""
    if result.status is not ResultStatus.ERROR:
        return False
    msg = (result.message or "").lower()
    transient_markers = (
        "timeout", "temporarily", "try again", "rate limit", "busy",
        "connection reset", "broken pipe", "503", "502", "429",
    )
    return any(m in msg for m in transient_markers)


def is_any_error(result: Result) -> bool:
    """Predicate retrying ALL errors — use only when you trust the underlying API."""
    return result.status is ResultStatus.ERROR


@dataclass
class RetryPolicy:
    """Wrap a Runtime and re-dispatch ERROR results matching ``classify``.

    Args:
        runtime: the Runtime to wrap.
        max_attempts: hard cap including the first try (so 3 = 1 + 2 retries).
        backoff_initial_s: first sleep before retry.
        backoff_factor: multiplier per attempt (2.0 = exponential).
        backoff_max_s: ceiling for sleep duration.
        jitter_frac: random jitter as a fraction of computed backoff
            (0.1 = +/-10% on every sleep). Avoids thundering herd.
        classify: predicate(result) -> bool. Default ``is_transient_error``.
    """

    runtime: Runtime
    max_attempts: int = 3
    backoff_initial_s: float = 0.1
    backoff_factor: float = 2.0
    backoff_max_s: float = 5.0
    jitter_frac: float = 0.1
    classify: Callable[[Result], bool] = field(default=is_transient_error)

    def step(self, intent: Intent) -> Result:
        backoff = self.backoff_initial_s
        last_result: Result | None = None
        for attempt in range(1, self.max_attempts + 1):
            result = self.runtime.step(intent)
            last_result = result
            if not self.classify(result):
                return result
            if attempt >= self.max_attempts:
                return result
            sleep_for = min(backoff, self.backoff_max_s)
            if self.jitter_frac > 0:
                jitter = sleep_for * self.jitter_frac * (2 * random.random() - 1)
                sleep_for = max(0.0, sleep_for + jitter)
            time.sleep(sleep_for)
            backoff *= self.backoff_factor
        assert last_result is not None
        return last_result

    def run(self, intents) -> list[Result]:
        return [self.step(i) for i in intents]
