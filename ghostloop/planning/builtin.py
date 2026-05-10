"""Built-in planners: PickAndPlacePlanner and TraversePlanner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..core import Intent
from ..primitives.trajectory import linear_interpolate
from .core import PlanResult


Point3 = tuple[float, float, float] | list[float]


@dataclass
class PickAndPlacePlanner:
    """Goal: pick ``object_id`` at ``pickup`` -> drop at ``drop``.

    Emits 5 intents: optional initial scan, approach, pick, transfer,
    place. Configurable scan_first / approach_offset / use_trajectory
    knobs let callers swap in interpolated approaches when the safety
    pipeline includes obstacle avoidance.
    """

    name: str = "pick_and_place"
    scan_first: bool = True
    scan_radius: float = 0.5
    approach_offset_z: float = 0.0
    use_trajectory: bool = False
    trajectory_steps: int = 5

    def plan(self, goal: dict) -> PlanResult:
        """Goal dict shape::

            {"object_id": "widget-7",
             "pickup": (x, y, z),
             "drop": (x, y, z)}
        """
        obj = str(goal["object_id"])
        pickup: Point3 = tuple(goal["pickup"])
        drop: Point3 = tuple(goal["drop"])

        intents: list[Intent] = []
        if self.scan_first:
            intents.append(Intent(
                name="scan",
                args={"radius": self.scan_radius},
                rationale="initial workspace scan",
            ))

        intents.extend(self._move_chain(
            from_=(0.0, 0.0, 0.0),  # planner doesn't know start; backend handles delta
            to=pickup,
            why=f"approach pickup site for {obj!r}",
        ))
        intents.append(Intent(
            name="pick",
            args={"object_id": obj},
            rationale=f"acquire {obj!r}",
        ))
        intents.extend(self._move_chain(
            from_=pickup, to=drop,
            why=f"transfer {obj!r} to drop site",
        ))
        intents.append(Intent(
            name="place",
            args={},
            rationale=f"release {obj!r} at drop site",
        ))
        return PlanResult(
            name=self.name,
            intents=intents,
            rationale=f"5-stage pick-and-place for {obj!r}: {pickup} -> {drop}",
            metadata={"object_id": obj, "pickup": list(pickup), "drop": list(drop)},
        )

    def _move_chain(self, from_: Point3, to: Point3, why: str) -> list[Intent]:
        """Either a single move_to or an interpolated trajectory, per config."""
        if self.use_trajectory:
            waypoints = linear_interpolate(list(from_), list(to), n=self.trajectory_steps)
            return [Intent(
                name="follow_trajectory",
                args={"waypoints": waypoints},
                rationale=why,
            )]
        return [Intent(
            name="move_to",
            args={"x": float(to[0]), "y": float(to[1]), "z": float(to[2])},
            rationale=why,
        )]


@dataclass
class TraversePlanner:
    """Goal: visit a list of waypoints in order, optionally scanning at each."""

    name: str = "traverse"
    scan_at_each: bool = False
    scan_radius: float = 0.5

    def plan(self, goal: Sequence[Point3]) -> PlanResult:
        intents: list[Intent] = []
        for i, wp in enumerate(goal):
            intents.append(Intent(
                name="move_to",
                args={"x": float(wp[0]), "y": float(wp[1]), "z": float(wp[2])},
                rationale=f"traverse waypoint {i + 1}/{len(goal)}",
            ))
            if self.scan_at_each:
                intents.append(Intent(
                    name="scan",
                    args={"radius": self.scan_radius},
                    rationale=f"observe at waypoint {i + 1}",
                ))
        return PlanResult(
            name=self.name,
            intents=intents,
            rationale=f"traverse {len(goal)} waypoints"
            + (" with scans" if self.scan_at_each else ""),
            metadata={"n_waypoints": len(goal)},
        )
