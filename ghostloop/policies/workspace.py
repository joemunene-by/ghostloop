"""WorkspaceModel + ObstacleAvoidanceGate — geometry richer than the axis-box geofence.

GeofenceGate handles the simplest case: rectangular workspace bounds.
Real deployments also need:
  - Forbidden regions (table-mounted obstacles, no-fly zones).
  - Convex hulls (more flexible than axis-aligned boxes).
  - Inflation radii (don't approach within X cm of an obstacle).

WorkspaceModel composes those pieces. ObstacleAvoidanceGate plugs the
model into the policy pipeline so any motion intent that would land
inside a forbidden region is denied with a structured reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..core import Decision, Intent, Primitive


Point3 = tuple[float, float, float]


@dataclass
class Sphere:
    """A spherical obstacle: deny moves whose target is within ``radius`` of ``center``.

    ``inflation`` adds an extra safety margin beyond the physical sphere
    radius. A 5cm-radius cup with a 2cm inflation rejects targets within
    7cm of the centre.
    """

    center: Point3
    radius: float
    inflation: float = 0.0
    label: str = ""

    def contains(self, p: Point3) -> bool:
        d = math.sqrt(sum((p[i] - self.center[i]) ** 2 for i in range(3)))
        return d <= self.radius + self.inflation


@dataclass
class AxisAlignedBox:
    """An axis-aligned box obstacle: deny targets inside the box."""

    min_corner: Point3
    max_corner: Point3
    inflation: float = 0.0
    label: str = ""

    def contains(self, p: Point3) -> bool:
        for i in range(3):
            lo = self.min_corner[i] - self.inflation
            hi = self.max_corner[i] + self.inflation
            if p[i] < lo or p[i] > hi:
                return False
        return True


Obstacle = Sphere | AxisAlignedBox


@dataclass
class WorkspaceModel:
    """Composable workspace: outer-bound box PLUS a list of forbidden obstacles.

    Targets are valid iff they're inside the outer bounds AND outside
    every obstacle. A motion intent that lands at a valid target passes
    the workspace check; anything else gets a Decision.deny with the
    specific reason (out-of-bounds OR obstacle name).
    """

    bounds_min: Point3 = (-1.0, -1.0, 0.0)
    bounds_max: Point3 = (1.0, 1.0, 1.0)
    obstacles: list[Obstacle] = field(default_factory=list)

    def add_sphere(self, center: Point3, radius: float, *,
                   inflation: float = 0.0, label: str = "") -> None:
        self.obstacles.append(
            Sphere(center=center, radius=radius, inflation=inflation, label=label)
        )

    def add_box(self, min_corner: Point3, max_corner: Point3, *,
                inflation: float = 0.0, label: str = "") -> None:
        self.obstacles.append(
            AxisAlignedBox(
                min_corner=min_corner, max_corner=max_corner,
                inflation=inflation, label=label,
            )
        )

    def violates(self, p: Point3) -> str | None:
        """Return None iff p is valid, else a human-readable violation reason."""
        for axis_idx, axis in enumerate("xyz"):
            lo, hi = self.bounds_min[axis_idx], self.bounds_max[axis_idx]
            if p[axis_idx] < lo or p[axis_idx] > hi:
                return f"{axis}={p[axis_idx]:g} outside workspace [{lo:g},{hi:g}]"
        for obs in self.obstacles:
            if obs.contains(p):
                kind = "sphere" if isinstance(obs, Sphere) else "box"
                tag = f" {obs.label!r}" if obs.label else ""
                return f"target inside {kind} obstacle{tag}"
        return None


@dataclass
class ObstacleAvoidanceGate:
    """Deny motion intents whose target lies outside the workspace OR inside an obstacle.

    Inspects ``intent.args`` for ``x``/``y``/``z`` (or ``target`` as a
    3-tuple). Pass-through for intents without a positional target.
    """

    workspace: WorkspaceModel
    name: str = "workspace"

    def _extract_target(self, args: dict) -> Point3 | None:
        if "target" in args:
            t: Sequence[float] = args["target"]
            if len(t) == 3:
                return float(t[0]), float(t[1]), float(t[2])
        if all(k in args for k in ("x", "y", "z")):
            return float(args["x"]), float(args["y"]), float(args["z"])
        return None

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        target = self._extract_target(intent.args)
        if target is None:
            return Decision.allow(self.name, "no target coords in intent")
        violation = self.workspace.violates(target)
        if violation is not None:
            return Decision.deny(self.name, violation)
        return Decision.allow(self.name, f"target {target} valid")
