"""Trajectory primitives: waypoint sequences with optional timing.

A `follow_trajectory` primitive accepts a list of (x, y, z) waypoints
plus an optional per-waypoint dwell time. Internally it dispatches the
underlying single-shot move_to per waypoint against MockBackend, so each
waypoint hop is recorded as one trace event from the runtime's view.
For sim/hardware backends the equivalent primitives can integrate the
trajectory through their controllers.

Useful for cases where the policy already knows the path (computed plan,
recorded teleop, hand-tuned approach maneuver) and the safety pipeline
should still gate every waypoint.
"""

from __future__ import annotations

import time
from typing import Any

from ..core import MockBackend, Primitive, Result, ResultStatus


def _follow_call(
    backend: MockBackend,
    waypoints: list[list[float]],
    dwell_s: float = 0.0,
) -> Result:
    if not isinstance(waypoints, list) or not waypoints:
        return Result(
            status=ResultStatus.ERROR,
            message="waypoints must be a non-empty list of [x, y, z] points",
        )
    visited: list[list[float]] = []
    started = time.monotonic()
    for i, wp in enumerate(waypoints):
        if not isinstance(wp, (list, tuple)) or len(wp) != 3:
            return Result(
                status=ResultStatus.ERROR,
                message=f"waypoint {i} not a 3-element point: {wp!r}",
            )
        backend.position = (float(wp[0]), float(wp[1]), float(wp[2]))
        visited.append([float(wp[0]), float(wp[1]), float(wp[2])])
        if dwell_s > 0:
            time.sleep(dwell_s)
    elapsed = time.monotonic() - started
    return Result(
        status=ResultStatus.OK,
        observation={
            "waypoints_visited": visited,
            "n_waypoints": len(waypoints),
            "elapsed_s": round(elapsed, 4),
        },
        message=f"followed {len(waypoints)}-point trajectory in {elapsed:.3g}s",
    )


def follow_trajectory() -> Primitive:
    """Follow a list of [x, y, z] waypoints with optional per-waypoint dwell.

    On MockBackend each waypoint is a teleport. Real backends implement
    interpolated motion and return per-waypoint timing, joint paths,
    and end-effector wrench data in the observation.
    """
    return Primitive(
        name="follow_trajectory",
        call=_follow_call,
        description="Visit a list of Cartesian waypoints in sequence.",
        arg_schema={
            "waypoints": "list[[x, y, z]]",
            "dwell_s": "float (optional, default 0.0)",
        },
    )


def linear_interpolate(start: list[float], end: list[float], n: int = 10) -> list[list[float]]:
    """Generate ``n`` evenly-spaced waypoints from ``start`` to ``end`` (inclusive).

    Helper for callers who know "approach this target along a straight
    line in N steps" without wanting to roll their own arithmetic.
    """
    if n < 2:
        return [list(end)]
    out: list[list[float]] = []
    for i in range(n):
        t = i / (n - 1)
        out.append([
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t,
        ])
    return out
