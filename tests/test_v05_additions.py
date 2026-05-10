"""Tests for v0.5: VLAPolicy, properties engine, workspace+obstacle gate,
trajectory primitives, and the planner module."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import (
    Intent,
    MockBackend,
    PolicyPipeline,
    PrimitiveRegistry,
    Runtime,
    Trace,
)
from ghostloop.planning import PickAndPlacePlanner, TraversePlanner
from ghostloop.policies import (
    DeltaXYZDecoder,
    GeofenceGate,
    ObstacleAvoidanceGate,
    Sphere,
    VLAPolicy,
    WorkspaceModel,
    vla_policy_loop,
)
from ghostloop.primitives import (
    follow_trajectory,
    linear_interpolate,
    move_to,
    pick,
    place,
    scan,
)
from ghostloop.properties import (
    NeverHoldsTwoObjects,
    NoConsecutiveDuplicateIntents,
    PropertyEngine,
    Severity,
    StaysInsideWorkspace,
)


def _registry(extra=()):
    return PrimitiveRegistry([
        move_to(), scan(), pick(), place(), follow_trajectory(), *extra,
    ])


# ---------------------------------------------------------------------------
# DeltaXYZDecoder + VLAPolicy
# ---------------------------------------------------------------------------


class TestDeltaXYZDecoder:
    def test_position_delta_emits_move_to(self):
        d = DeltaXYZDecoder(delta_scale=1.0, deadband=0.0)
        intent = d.decode([0.1, -0.05, 0.2, 0, 0, 0, 0],
                          {"position": [1.0, 2.0, 3.0]})
        assert intent is not None
        assert intent.name == "move_to"
        assert intent.args["x"] == pytest.approx(1.1)
        assert intent.args["y"] == pytest.approx(1.95)
        assert intent.args["z"] == pytest.approx(3.2)

    def test_below_deadband_emits_none(self):
        d = DeltaXYZDecoder(delta_scale=1.0, deadband=0.01)
        intent = d.decode([0.001, 0.001, 0.001, 0, 0, 0, 0],
                          {"position": [0, 0, 0]})
        assert intent is None

    def test_gripper_close_transition_emits_pick(self):
        d = DeltaXYZDecoder()
        d.last_gripper = False
        intent = d.decode([0, 0, 0, 0, 0, 0, 0.9], {"position": [0, 0, 0]})
        assert intent is not None
        assert intent.name == "pick"
        assert d.last_gripper is True

    def test_gripper_open_transition_emits_place(self):
        d = DeltaXYZDecoder()
        d.last_gripper = True
        intent = d.decode([0, 0, 0, 0, 0, 0, 0.0], {"position": [0, 0, 0]})
        assert intent is not None
        assert intent.name == "place"
        assert d.last_gripper is False

    def test_no_repeat_picks_without_intervening_place(self):
        d = DeltaXYZDecoder()
        d.last_gripper = False
        first = d.decode([0, 0, 0, 0, 0, 0, 1.0], {"position": [0, 0, 0]})
        second = d.decode([0, 0, 0, 0, 0, 0, 1.0], {"position": [0, 0, 0]})
        assert first is not None and first.name == "pick"
        assert second is None  # gripper still closed → no transition → no-op

    def test_short_action_vector_returns_none(self):
        d = DeltaXYZDecoder()
        assert d.decode([0.1], {"position": [0, 0, 0]}) is None


class TestVLAPolicy:
    def test_loop_runs_until_idle_termination(self):
        # Mock model: emits one delta then returns zeros forever.
        actions = [
            [0.5, 0.0, 0.0, 0, 0, 0, 0],  # move_to delta
            [0.0, 0.0, 0.0, 0, 0, 0, 0],  # idle
            [0.0, 0.0, 0.0, 0, 0, 0, 0],  # idle
            [0.0, 0.0, 0.0, 0, 0, 0, 0],  # idle
            [0.0, 0.0, 0.0, 0, 0, 0, 0],  # idle (4th idle = terminate)
        ]
        idx = {"i": 0}
        def model(_obs):
            i = idx["i"]
            idx["i"] = i + 1
            return actions[min(i, len(actions) - 1)]
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        summary = vla_policy_loop(model, rt, max_steps=10)
        assert summary["steps"] == 1  # only 1 non-idle action
        assert summary["terminated"] in ("idle", "no_action")
        assert rt.backend.position == (0.5, 0.0, 0.0)

    def test_safety_pipeline_blocks_oversized_vla_step(self):
        # Pipeline has a tight geofence; VLA emits a giant delta.
        rt = Runtime(
            backend=MockBackend(),
            registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-0.5, -0.5, -0.5),
                             max_corner=(0.5, 0.5, 0.5)),
            ]),
        )
        def model(_obs):
            return [10.0, 0.0, 0.0, 0, 0, 0, 0]
        summary = vla_policy_loop(model, rt, max_steps=2)
        # Step took place (intent dispatched), but result was BLOCKED.
        events = rt.trace.events
        assert any(ev.result.status.value == "blocked" for ev in events)
        # Backend must not have moved.
        assert rt.backend.position == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# WorkspaceModel + ObstacleAvoidanceGate
# ---------------------------------------------------------------------------


class TestWorkspaceModel:
    def test_inside_bounds_no_obstacles_passes(self):
        ws = WorkspaceModel(bounds_min=(-1, -1, -1), bounds_max=(1, 1, 1))
        assert ws.violates((0.0, 0.0, 0.0)) is None

    def test_outside_bounds_violates(self):
        ws = WorkspaceModel(bounds_min=(-1, -1, -1), bounds_max=(1, 1, 1))
        v = ws.violates((5.0, 0.0, 0.0))
        assert v is not None and "outside workspace" in v

    def test_sphere_obstacle_violates(self):
        ws = WorkspaceModel()
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.2, label="cup")
        v = ws.violates((0.1, 0.0, 0.0))  # within sphere
        assert v is not None and "sphere" in v
        assert "cup" in v

    def test_sphere_with_inflation(self):
        ws = WorkspaceModel()
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.1, inflation=0.2, label="hot-zone")
        # 0.25 from center: outside physical sphere (0.1) but inside inflation (0.3).
        v = ws.violates((0.25, 0.0, 0.0))
        assert v is not None
        # Outside everything (>0.3): passes.
        assert ws.violates((0.4, 0.0, 0.0)) is None

    def test_box_obstacle_violates(self):
        ws = WorkspaceModel()
        ws.add_box((0.1, 0.1, 0.0), (0.3, 0.3, 0.3), label="table")
        assert ws.violates((0.2, 0.2, 0.2)) is not None
        assert ws.violates((0.5, 0.5, 0.5)) is None


class TestObstacleAvoidanceGate:
    def test_gate_blocks_targets_inside_obstacle(self):
        ws = WorkspaceModel()
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.1)
        rt = Runtime(
            backend=MockBackend(),
            registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[ObstacleAvoidanceGate(workspace=ws)]),
        )
        result = rt.step(Intent("move_to", {"x": 0.05, "y": 0.0, "z": 0.0}))
        assert result.status.value == "blocked"
        assert "obstacle" in result.message

    def test_gate_passes_clear_targets(self):
        ws = WorkspaceModel()
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.1)
        rt = Runtime(
            backend=MockBackend(),
            registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[ObstacleAvoidanceGate(workspace=ws)]),
        )
        result = rt.step(Intent("move_to", {"x": 0.5, "y": 0.5, "z": 0.5}))
        assert result.ok

    def test_gate_passes_argless_intents(self):
        ws = WorkspaceModel()
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.5)
        rt = Runtime(
            backend=MockBackend(),
            registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[ObstacleAvoidanceGate(workspace=ws)]),
        )
        # scan has no x/y/z — gate passes through.
        result = rt.step(Intent("scan", {"radius": 0.5}))
        assert result.ok


# ---------------------------------------------------------------------------
# Trajectory primitive
# ---------------------------------------------------------------------------


class TestTrajectory:
    def test_follow_trajectory_visits_each_waypoint(self):
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        result = rt.step(Intent("follow_trajectory", {
            "waypoints": [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
        }))
        assert result.ok
        assert len(result.observation["waypoints_visited"]) == 3
        assert rt.backend.position == (0.3, 0.0, 0.0)

    def test_follow_trajectory_empty_waypoints_errors(self):
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        result = rt.step(Intent("follow_trajectory", {"waypoints": []}))
        assert result.status.value == "error"
        assert "non-empty" in result.message

    def test_linear_interpolate_endpoints(self):
        wps = linear_interpolate([0, 0, 0], [1, 2, 3], n=5)
        assert wps[0] == [0.0, 0.0, 0.0]
        assert wps[-1] == [1.0, 2.0, 3.0]
        assert len(wps) == 5

    def test_linear_interpolate_minimum(self):
        wps = linear_interpolate([0, 0, 0], [1, 1, 1], n=1)
        assert wps == [[1, 1, 1]]


# ---------------------------------------------------------------------------
# Properties engine
# ---------------------------------------------------------------------------


def _trace_with(intents):
    rt = Runtime(backend=MockBackend(), registry=_registry(),
                 policy_pipeline=PolicyPipeline())
    rt.run(intents)
    return rt.trace


class TestPropertyEngine:
    def test_stays_inside_workspace_holds(self):
        trace = _trace_with([
            Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0}),
            Intent("move_to", {"x": 0.2, "y": 0.0, "z": 0.0}),
        ])
        prop = StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        result = prop.check(trace)
        assert result.held
        assert result.violations == []

    def test_stays_inside_workspace_caught_violation(self):
        trace = _trace_with([
            Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0}),  # actually moved (no gate)
        ])
        prop = StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        result = prop.check(trace)
        assert not result.held
        assert len(result.violations) == 1
        assert result.violations[0]["axis"] == "x"

    def test_never_holds_two_objects_violation(self):
        # Force a state inconsistency by manually appending a second pick on top of held.
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        rt.step(Intent("pick", {"object_id": "a"}))
        # Force the backend to a different object (simulating a backend bug).
        rt.backend.held_object = "a"
        # Now manually call pick — it'll error normally; but if we set held_object=None
        # first then pick, we get a clean state. To trigger the violation we need a
        # state_before non-null + state_after non-null with different value.
        # Easier: hand-craft a Trace.
        from ghostloop import TraceEvent, Decision
        from ghostloop.core import Result, ResultStatus
        ev1 = rt.trace.events[0]
        # Append a fabricated event simulating a backend bug.
        ev_bug = TraceEvent(
            step=2,
            intent=Intent("pick", {"object_id": "b"}),
            decision=Decision.allow("test", "fake"),
            result=Result(status=ResultStatus.OK, message="ok"),
            state_before={"position": [0, 0, 0], "held_object": "a"},
            state_after={"position": [0, 0, 0], "held_object": "b"},
            timestamp=time.time(),
        )
        rt.trace.append(ev_bug)
        prop = NeverHoldsTwoObjects()
        result = prop.check(rt.trace)
        assert not result.held
        assert any("a" in v["reason"] and "b" in v["reason"] for v in result.violations)

    def test_no_consecutive_duplicate_intents_warning(self):
        trace = _trace_with([
            Intent("scan", {"radius": 0.5}),
            Intent("scan", {"radius": 0.5}),
        ])
        prop = NoConsecutiveDuplicateIntents()
        result = prop.check(trace)
        assert not result.held
        assert result.severity is Severity.WARN

    def test_engine_summary_counts(self):
        trace = _trace_with([
            Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0}),
            Intent("scan", {"radius": 0.5}),
            Intent("scan", {"radius": 0.5}),  # duplicate -> warn
        ])
        engine = PropertyEngine(properties=[
            StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            NoConsecutiveDuplicateIntents(),
        ])
        summary = engine.evaluate_summary(trace)
        assert summary["n_properties"] == 2
        assert summary["held"] == 1
        assert summary["violated"] == 1
        # No ERROR-severity violations -> not ship-blocked.
        assert summary["ship_blocked"] is False

    def test_engine_renders_md(self):
        trace = _trace_with([Intent("scan", {"radius": 0.5})])
        engine = PropertyEngine(properties=[
            StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
        ])
        md = engine.render_md(trace)
        assert "Property evaluation" in md
        assert "stays_inside_workspace" in md


# ---------------------------------------------------------------------------
# Planning module
# ---------------------------------------------------------------------------


class TestPlanners:
    def test_pick_and_place_plan_shape(self):
        planner = PickAndPlacePlanner()
        plan = planner.plan({
            "object_id": "widget-7",
            "pickup": (0.4, 0.0, 0.1),
            "drop": (-0.4, 0.0, 0.1),
        })
        names = [i.name for i in plan.intents]
        # scan, move, pick, move, place
        assert names == ["scan", "move_to", "pick", "move_to", "place"]
        assert plan.metadata["object_id"] == "widget-7"

    def test_pick_and_place_use_trajectory(self):
        planner = PickAndPlacePlanner(use_trajectory=True, trajectory_steps=4)
        plan = planner.plan({
            "object_id": "widget-7",
            "pickup": (0.4, 0.0, 0.1),
            "drop": (-0.4, 0.0, 0.1),
        })
        names = [i.name for i in plan.intents]
        # scan, follow_trajectory, pick, follow_trajectory, place
        assert names == ["scan", "follow_trajectory", "pick", "follow_trajectory", "place"]

    def test_pick_and_place_no_scan(self):
        planner = PickAndPlacePlanner(scan_first=False)
        plan = planner.plan({
            "object_id": "x", "pickup": (0, 0, 0), "drop": (0.1, 0, 0),
        })
        assert plan.intents[0].name == "move_to"

    def test_traverse_planner(self):
        planner = TraversePlanner()
        plan = planner.plan([(0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0)])
        assert all(i.name == "move_to" for i in plan.intents)
        assert plan.n_steps == 3

    def test_traverse_with_scan(self):
        planner = TraversePlanner(scan_at_each=True)
        plan = planner.plan([(0.1, 0, 0), (0.2, 0, 0)])
        names = [i.name for i in plan.intents]
        assert names == ["move_to", "scan", "move_to", "scan"]

    def test_plan_executes_through_runtime(self):
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        plan = PickAndPlacePlanner().plan({
            "object_id": "widget-7",
            "pickup": (0.4, 0.0, 0.1),
            "drop": (-0.4, 0.0, 0.1),
        })
        results = rt.run(plan.intents)
        assert all(r.ok for r in results)
        # Final state: at drop site, holding nothing.
        assert rt.backend.position == (-0.4, 0.0, 0.1)
        assert rt.backend.held_object is None
