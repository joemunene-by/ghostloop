"""Pre-built Episode catalogue for the v0.3 bench harness.

Each builder returns a list of Episodes that share a goal shape but differ
in initial state / target / object. This is the equivalent of GhostBench's
12 differentiation bets in GhostLM, transposed to robot tasks.

Suites included in v0.3:
  reach_targets        — move to N points, pass iff position == target
  pick_and_place_pairs — pick object A, drop at location B
  scan_at_targets      — visit waypoints and call scan, score for coverage
  geofence_violations  — half-inside, half-outside (regression suite for
                         the safety pipeline)
"""

from __future__ import annotations

from typing import Callable, Iterable

from ..core import Intent, MockBackend
from ..primitives import move_to, pick, place, scan
from .episode import Episode


def _mock_setup() -> MockBackend:
    return MockBackend()


def _registry_factory():
    return [move_to(), scan(), pick(), place()]


def reach_targets(
    targets: Iterable[tuple[str, tuple[float, float, float]]],
) -> list[Episode]:
    """Suite: scripted move_to to each target, pass iff position == target."""
    out = []
    for name, t in targets:
        def _policy(_runtime, _t=t):
            return [Intent("move_to", {"x": _t[0], "y": _t[1], "z": _t[2]})]

        def _pred(_trace, state, _t=t):
            return tuple(state["position"]) == _t

        out.append(Episode(
            name=name,
            goal=f"reach {t}",
            setup=_mock_setup,
            policy=_policy,
            success_predicate=_pred,
            primitives=_registry_factory,
        ))
    return out


def pick_and_place_pairs(
    pairs: Iterable[tuple[str, str, tuple[float, float, float], tuple[float, float, float]]],
) -> list[Episode]:
    """Suite: pick object_id at A, place at B. Pass iff backend.held_object is None
    AND backend.position == B at the end (the object got placed at B)."""
    out = []
    for name, obj_id, pickup, drop in pairs:
        def _policy(_runtime, _obj=obj_id, _p=pickup, _d=drop):
            return [
                Intent("move_to", {"x": _p[0], "y": _p[1], "z": _p[2]}),
                Intent("pick", {"object_id": _obj}),
                Intent("move_to", {"x": _d[0], "y": _d[1], "z": _d[2]}),
                Intent("place", {}),
            ]

        def _pred(_trace, state, _d=drop):
            return state["held_object"] is None and tuple(state["position"]) == _d

        out.append(Episode(
            name=name,
            goal=f"pick {obj_id} at {pickup} -> drop at {drop}",
            setup=_mock_setup,
            policy=_policy,
            success_predicate=_pred,
            primitives=_registry_factory,
        ))
    return out


def scan_at_targets(
    waypoints: Iterable[tuple[str, list[tuple[float, float, float]]]],
) -> list[Episode]:
    """Suite: visit a list of waypoints, call scan at each. Pass iff
    every waypoint produced an OK scan event in the trace."""
    out = []
    for name, points in waypoints:
        def _policy(_runtime, _pts=points):
            intents = []
            for p in _pts:
                intents.append(Intent("move_to", {"x": p[0], "y": p[1], "z": p[2]}))
                intents.append(Intent("scan", {"radius": 0.5}))
            return intents

        def _pred(trace, _state, _pts=points):
            scans_ok = sum(
                1 for ev in trace.events
                if ev.intent.name == "scan" and ev.result.status.value == "ok"
            )
            return scans_ok == len(_pts)

        out.append(Episode(
            name=name,
            goal=f"scan at each of {len(points)} waypoints",
            setup=_mock_setup,
            policy=_policy,
            success_predicate=_pred,
            primitives=_registry_factory,
        ))
    return out


def geofence_violations(
    inside_targets: list[tuple[str, tuple[float, float, float]]],
    outside_targets: list[tuple[str, tuple[float, float, float]]],
) -> list[Episode]:
    """Regression suite for the safety pipeline.

    Half the episodes target points inside the workspace (should pass with
    or without geofence). Half target points outside (should pass without
    geofence, fail with geofence). Use this with paired_compare to confirm
    the gate is doing its job.
    """
    return reach_targets(inside_targets) + reach_targets(outside_targets)


# ---------------------------------------------------------------------------
# Convenience presets — handy default fixtures for quick demos / smoke tests.
# ---------------------------------------------------------------------------


def preset_reach_8() -> list[Episode]:
    """8 reach targets spanning the workspace corners + axes."""
    return reach_targets([
        (f"reach-{name}", t) for name, t in [
            ("origin", (0.0, 0.0, 0.0)),
            ("px", (0.5, 0.0, 0.0)),
            ("nx", (-0.5, 0.0, 0.0)),
            ("py", (0.0, 0.5, 0.0)),
            ("ny", (0.0, -0.5, 0.0)),
            ("pz", (0.0, 0.0, 0.5)),
            ("corner-near", (0.4, 0.4, 0.4)),
            ("corner-far", (-0.4, -0.4, 0.4)),
        ]
    ])


def preset_pick_and_place_4() -> list[Episode]:
    return pick_and_place_pairs([
        ("widget-east-to-west", "widget-1", (0.4, 0.0, 0.1), (-0.4, 0.0, 0.1)),
        ("cup-up-to-down", "cup-1", (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)),
        ("brick-corner-shuffle", "brick-1", (0.3, 0.3, 0.0), (-0.3, -0.3, 0.0)),
        ("ball-axis-flip", "ball-1", (0.5, 0.5, 0.5), (-0.5, -0.5, -0.5)),
    ])


def preset_geofence_smoke() -> list[Episode]:
    """The 8-episode geofence regression suite used by examples/."""
    return geofence_violations(
        inside_targets=[
            ("inside-1", (0.3, 0.0, 0.0)),
            ("inside-2", (-0.5, 0.2, 0.1)),
            ("inside-3", (0.0, 0.7, 0.5)),
            ("inside-4", (0.4, -0.4, 0.2)),
        ],
        outside_targets=[
            ("outside-1", (5.0, 0.0, 0.0)),
            ("outside-2", (-3.0, 0.0, 0.0)),
            ("outside-3", (0.0, 9.0, 0.0)),
            ("outside-4", (0.0, 0.0, 12.0)),
        ],
    )
