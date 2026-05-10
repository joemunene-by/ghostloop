"""FleetRegistry + RobotHandle + FleetDispatcher core types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import cycle
from typing import Any, Iterator

from ..core import Intent, Result, Runtime


class RobotStatus(str, Enum):
    """Lifecycle of a robot inside the fleet registry."""

    IDLE = "idle"          # online, ready for work
    BUSY = "busy"          # currently executing
    OFFLINE = "offline"    # explicitly removed / heartbeat lapsed
    ERROR = "error"        # last operation errored, needs operator review


class FleetError(RuntimeError):
    """Raised when a fleet operation can't find a target robot."""


@dataclass
class RobotHandle:
    """One robot in the fleet: a Runtime + lifecycle state + heartbeat metadata."""

    name: str
    runtime: Runtime
    status: RobotStatus = RobotStatus.IDLE
    labels: dict[str, str] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)

    def heartbeat(self) -> None:
        self.last_seen = time.time()

    def step(self, intent: Intent) -> Result:
        """Execute one intent, updating status."""
        self.status = RobotStatus.BUSY
        try:
            result = self.runtime.step(intent)
        except Exception:
            self.status = RobotStatus.ERROR
            raise
        finally:
            self.heartbeat()
        if result.status.value == "error":
            self.status = RobotStatus.ERROR
        elif result.status.value in ("ok", "blocked"):
            self.status = RobotStatus.IDLE
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "labels": dict(self.labels),
            "last_seen": self.last_seen,
            "n_events": len(self.runtime.trace.events),
            "backend": self.runtime.backend.name,
            "snapshot": self.runtime.backend.snapshot(),
        }


@dataclass
class FleetRegistry:
    """Name -> RobotHandle index plus selection helpers."""

    _robots: dict[str, RobotHandle] = field(default_factory=dict)

    def register(self, robot: RobotHandle) -> None:
        if robot.name in self._robots:
            raise FleetError(f"robot already registered: {robot.name!r}")
        self._robots[robot.name] = robot

    def deregister(self, name: str) -> None:
        if name not in self._robots:
            raise FleetError(f"unknown robot: {name!r}")
        self._robots[name].status = RobotStatus.OFFLINE
        del self._robots[name]

    def get(self, name: str) -> RobotHandle | None:
        return self._robots.get(name)

    def all(self) -> list[RobotHandle]:
        return list(self._robots.values())

    def names(self) -> list[str]:
        return sorted(self._robots.keys())

    def filter_by_status(self, status: RobotStatus) -> list[RobotHandle]:
        return [r for r in self._robots.values() if r.status is status]

    def filter_by_label(self, key: str, value: str) -> list[RobotHandle]:
        return [r for r in self._robots.values() if r.labels.get(key) == value]

    def stale(self, timeout_s: float) -> list[RobotHandle]:
        """Robots whose last_seen is older than ``timeout_s`` ago."""
        cutoff = time.time() - timeout_s
        return [r for r in self._robots.values() if r.last_seen < cutoff]

    def snapshot(self) -> "FleetSnapshot":
        return FleetSnapshot(robots=[r.to_json() for r in self._robots.values()])

    def __len__(self) -> int:
        return len(self._robots)

    def __contains__(self, name: object) -> bool:
        return name in self._robots


class DispatchStrategy(str, Enum):
    FIRST_IDLE = "first_idle"      # pick the first IDLE robot
    ROUND_ROBIN = "round_robin"     # cycle through every robot in name order
    LEAST_BUSY = "least_busy"       # robot with the fewest trace events


@dataclass
class Dispatch:
    """Result of one dispatcher call."""

    robot_name: str
    intent: Intent
    result: Result


@dataclass
class FleetDispatcher:
    """Submit intents to whichever robot the strategy picks.

    Idempotent if the strategy is idempotent (FIRST_IDLE, LEAST_BUSY).
    ROUND_ROBIN cycles state lives inside the dispatcher.
    """

    registry: FleetRegistry
    strategy: DispatchStrategy = DispatchStrategy.FIRST_IDLE
    _round_robin_iter: Iterator[RobotHandle] | None = field(default=None, init=False)

    def _select(self) -> RobotHandle:
        if not self.registry:
            raise FleetError("no robots registered in fleet")
        if self.strategy is DispatchStrategy.FIRST_IDLE:
            for r in self.registry.all():
                if r.status is RobotStatus.IDLE:
                    return r
            # Fallback: return first robot regardless of status (let caller handle).
            return self.registry.all()[0]
        if self.strategy is DispatchStrategy.ROUND_ROBIN:
            if self._round_robin_iter is None:
                self._round_robin_iter = cycle(self.registry.all())
            return next(self._round_robin_iter)
        if self.strategy is DispatchStrategy.LEAST_BUSY:
            return min(self.registry.all(), key=lambda r: len(r.runtime.trace.events))
        raise FleetError(f"unknown strategy: {self.strategy}")

    def dispatch(self, intent: Intent, *, robot: str | None = None) -> Dispatch:
        """Run one intent. ``robot`` pins to a specific name; default uses strategy."""
        if robot is not None:
            handle = self.registry.get(robot)
            if handle is None:
                raise FleetError(f"unknown robot: {robot!r}")
        else:
            handle = self._select()
        result = handle.step(intent)
        return Dispatch(robot_name=handle.name, intent=intent, result=result)

    def dispatch_many(self, intents) -> list[Dispatch]:
        return [self.dispatch(i) for i in intents]


@dataclass
class FleetSnapshot:
    """JSON-safe view of every robot's state — for dashboards / fleet APIs."""

    robots: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        n_idle = sum(1 for r in self.robots if r["status"] == "idle")
        n_busy = sum(1 for r in self.robots if r["status"] == "busy")
        n_error = sum(1 for r in self.robots if r["status"] == "error")
        n_offline = sum(1 for r in self.robots if r["status"] == "offline")
        return {
            "n_robots": len(self.robots),
            "n_idle": n_idle,
            "n_busy": n_busy,
            "n_error": n_error,
            "n_offline": n_offline,
            "robots": self.robots,
        }
