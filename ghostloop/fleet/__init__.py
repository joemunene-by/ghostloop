"""Fleet primitives — multi-robot abstractions over single-robot Runtimes.

Single-robot ghostloop is a great agent loop. Real deployments are
fleets of robots, each with its own backend, registry, and trace, all
coordinated by a fleet operator. This module adds:

  RobotHandle      one robot — runtime + status + last_seen
  FleetRegistry    name -> RobotHandle, the index every fleet op uses
  FleetDispatcher  load-balanced submission across robots
  FleetSnapshot    JSON-safe view of every robot's state for dashboards

The dispatcher is intentionally simple — round-robin or first-available
strategies. Sophisticated routing (skill matching, geographic affinity,
priority queues) is a future-release follow-on.
"""

from .core import (
    Dispatch,
    DispatchStrategy,
    FleetDispatcher,
    FleetError,
    FleetRegistry,
    FleetSnapshot,
    RobotHandle,
    RobotStatus,
)

__all__ = [
    "Dispatch",
    "DispatchStrategy",
    "FleetDispatcher",
    "FleetError",
    "FleetRegistry",
    "FleetSnapshot",
    "RobotHandle",
    "RobotStatus",
]
