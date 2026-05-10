"""Tests for v0.9: ROS2Backend (live-gated), action smoothing, safe-action
projection, reward shaper, sim2real bench harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import (
    Decision,
    Intent,
    MockBackend,
    PolicyPipeline,
    PrimitiveRegistry,
    Result,
    Runtime,
)
from ghostloop.backends import ros2_available
from ghostloop.bench import (
    Episode,
    OnDecision,
    OnObservation,
    OnPrimitive,
    RewardShaper,
    Sim2RealBench,
    StepCost,
    reward_shaper_from_dict,
)
from ghostloop.core import DecisionAction, Primitive, ResultStatus
from ghostloop.policies import (
    ActionSmoothingGate,
    GeofenceGate,
    project_to_sdf,
    project_to_workspace,
    smooth_target,
)
from ghostloop.policies.workspace import WorkspaceModel
from ghostloop.primitives import move_to, scan


def _registry() -> PrimitiveRegistry:
    return PrimitiveRegistry([move_to(), scan()])


# ---------------------------------------------------------------------------
# ROS2Backend (live-gated — only runs when rclpy is installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ros2_available(), reason="rclpy not installed")
class TestROS2BackendLive:
    def test_construction(self):
        from ghostloop.backends import ROS2Backend
        backend = ROS2Backend(node_name="ghostloop_test")
        try:
            snap = backend.snapshot()
            assert snap["backend"] == "ros2"
            assert "node" in snap
        finally:
            backend.shutdown()


class TestROS2BackendUnavailable:
    def test_import_raises_with_install_hint(self):
        if ros2_available():
            pytest.skip("rclpy is installed; can't test the unavailable path")
        from ghostloop.backends import ROS2Backend
        with pytest.raises(ImportError) as excinfo:
            ROS2Backend(node_name="dummy")
        assert "rclpy" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Action smoothing
# ---------------------------------------------------------------------------


class TestActionSmoothingGate:
    def test_first_call_always_allowed(self):
        gate = ActionSmoothingGate(max_velocity=0.001, max_acceleration=0.001)
        intent = Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0})
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        d = gate.check(intent, prim)
        assert d.action is DecisionAction.ALLOW

    def test_high_velocity_denied(self):
        gate = ActionSmoothingGate(max_velocity=0.01, max_acceleration=1e9)
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        # First call sets baseline.
        gate.check(Intent("move_to", {"x": 0.0, "y": 0.0, "z": 0.0}), prim)
        # Big jump = high velocity.
        d = gate.check(Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0}), prim)
        assert d.action is DecisionAction.DENY
        assert "velocity" in d.reason

    def test_no_target_passes_through(self):
        gate = ActionSmoothingGate()
        intent = Intent("scan", {"radius": 0.3})
        prim = Primitive(name="scan", call=lambda *a, **k: None)
        d = gate.check(intent, prim)
        assert d.action is DecisionAction.ALLOW

    def test_per_primitive_overrides(self):
        gate = ActionSmoothingGate(
            max_velocity=10.0,
            per_primitive_velocity={"move_to": 0.001},
        )
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        gate.check(Intent("move_to", {"x": 0.0, "y": 0.0, "z": 0.0}), prim)
        d = gate.check(Intent("move_to", {"x": 1.0, "y": 0.0, "z": 0.0}), prim)
        assert d.action is DecisionAction.DENY

    def test_reset_clears_state(self):
        gate = ActionSmoothingGate(max_velocity=0.01)
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        gate.check(Intent("move_to", {"x": 0.0, "y": 0.0, "z": 0.0}), prim)
        gate.reset("move_to")
        # After reset, second call should be treated as first.
        d = gate.check(Intent("move_to", {"x": 9.0, "y": 0.0, "z": 0.0}), prim)
        assert d.action is DecisionAction.ALLOW


class TestSmoothTarget:
    def test_within_step_unchanged(self):
        result = smooth_target((0, 0, 0), (0.05, 0, 0), max_step=0.1)
        assert result == (0.05, 0, 0)

    def test_clip_to_max_step(self):
        result = smooth_target((0, 0, 0), (1.0, 0, 0), max_step=0.1)
        assert result[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Safe-action projection
# ---------------------------------------------------------------------------


class TestProjectToWorkspace:
    def test_inside_unchanged(self):
        ws = WorkspaceModel(bounds_min=(-1, -1, -1), bounds_max=(1, 1, 1))
        intent = Intent("move_to", {"x": 0.5, "y": 0.0, "z": 0.0})
        result = project_to_workspace(intent, ws)
        assert result is intent

    def test_outside_clamped_to_bounds(self):
        ws = WorkspaceModel(bounds_min=(-1, -1, -1), bounds_max=(1, 1, 1))
        intent = Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0})
        result = project_to_workspace(intent, ws)
        assert result.args["x"] == 1.0
        assert result.args["projected_from"] == [5.0, 0.0, 0.0]

    def test_inside_sphere_pushed_out(self):
        ws = WorkspaceModel(bounds_min=(-2, -2, -2), bounds_max=(2, 2, 2))
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.3)
        intent = Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0})
        result = project_to_workspace(intent, ws)
        # Now sits at radius 0.3 from origin.
        x, y, z = result.args["x"], result.args["y"], result.args["z"]
        from math import sqrt
        assert sqrt(x*x + y*y + z*z) == pytest.approx(0.3, abs=1e-6)

    def test_no_target_passes_through(self):
        ws = WorkspaceModel(bounds_min=(-1, -1, -1), bounds_max=(1, 1, 1))
        intent = Intent("scan", {"radius": 0.5})
        result = project_to_workspace(intent, ws)
        assert result is intent


class TestProjectToSDF:
    def test_inside_obstacle_pushed_out(self):
        ws = WorkspaceModel(bounds_min=(-2, -2, -2), bounds_max=(2, 2, 2))
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.3)
        intent = Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0})
        result = project_to_sdf(intent, ws, n_steps=200, learning_rate=0.05)
        x, y, z = result.args["x"], result.args["y"], result.args["z"]
        from math import sqrt
        assert sqrt(x*x + y*y + z*z) >= 0.3 - 0.05  # close to surface


# ---------------------------------------------------------------------------
# Reward shaper DSL
# ---------------------------------------------------------------------------


class TestRewardShaper:
    def _ev(
        self, intent_name: str, status: str = "ok", obs: dict | None = None,
        decision_action: str = "allow", gate_name: str = "test",
    ):
        intent = Intent(intent_name, {})
        decision = (
            Decision.allow(gate_name, "ok") if decision_action == "allow"
            else Decision.deny(gate_name, "no")
        )
        result_status = ResultStatus.OK if status == "ok" else ResultStatus.ERROR
        result = Result(status=result_status, observation=obs or {})
        return intent, decision, result

    def test_on_primitive_matches_status(self):
        shaper = RewardShaper([
            OnPrimitive(primitive_name="pick", reward=1.0, when_status="ok"),
            OnPrimitive(primitive_name="pick", reward=-2.0, when_status="error"),
        ])
        ev_ok = self._ev("pick", status="ok")
        ev_err = self._ev("pick", status="error")
        assert shaper.score(*ev_ok) == 1.0
        assert shaper.score(*ev_err) == -2.0

    def test_step_cost_always_fires(self):
        shaper = RewardShaper([StepCost(reward=-0.01)])
        assert shaper.score(*self._ev("move_to")) == -0.01

    def test_on_decision_deny_penalty(self):
        shaper = RewardShaper([OnDecision(action="deny", reward=-5.0)])
        assert shaper.score(*self._ev("move_to", decision_action="deny")) == -5.0
        assert shaper.score(*self._ev("move_to", decision_action="allow")) == 0.0

    def test_on_observation_below_threshold(self):
        shaper = RewardShaper([
            OnObservation(field_name="force", below=10.0, reward=0.5),
        ])
        assert shaper.score(*self._ev("move_to", obs={"force": 5.0})) == 0.5
        assert shaper.score(*self._ev("move_to", obs={"force": 20.0})) == 0.0

    def test_compositional_total(self):
        shaper = RewardShaper([
            OnPrimitive(primitive_name="pick", reward=1.0, when_status="ok"),
            StepCost(reward=-0.01),
        ])
        # Both fire on a successful pick.
        assert shaper.score(*self._ev("pick", status="ok")) == pytest.approx(0.99)

    def test_score_with_breakdown(self):
        shaper = RewardShaper([
            StepCost(reward=-0.01),
            OnPrimitive(primitive_name="pick", reward=1.0, when_status="ok"),
        ])
        result = shaper.score_with_breakdown(*self._ev("pick", status="ok"))
        assert result["total"] == pytest.approx(0.99)
        assert len(result["components"]) == 2

    def test_from_dict(self):
        shaper = reward_shaper_from_dict([
            {"type": "OnPrimitive", "primitive_name": "pick", "reward": 1.0},
            {"type": "StepCost", "reward": -0.01},
        ])
        assert len(shaper.components) == 2

    def test_unknown_component_type_raises(self):
        with pytest.raises(ValueError):
            reward_shaper_from_dict([{"type": "NotAThing", "reward": 1.0}])


# ---------------------------------------------------------------------------
# Sim2Real bench harness
# ---------------------------------------------------------------------------


class TestSim2RealBench:
    def _episode(self, name: str, target: float, expect_pass: bool) -> Episode:
        def setup():
            return MockBackend()

        def policy(runtime):
            runtime.step(Intent("move_to", {"x": target, "y": 0.0, "z": 0.0}))
            return None

        def predicate(trace, snap):
            # Pass iff the trace's first event was allowed (target inside geofence).
            if not trace.events:
                return False
            return trace.events[0].decision.action is DecisionAction.ALLOW

        return Episode(
            name=name, goal=f"reach x={target}",
            setup=setup,
            policy=policy,
            success_predicate=predicate,
            primitives=lambda: [move_to(), scan()],
            pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]) if expect_pass else PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-0.1, -0.1, -0.1), max_corner=(0.1, 0.1, 0.1)),
            ]),
        )

    def test_paired_run_produces_report(self):
        sim_episodes = [
            self._episode("ep1", 0.5, expect_pass=True),
            self._episode("ep2", 0.0, expect_pass=True),
        ]
        # "Real" has tighter pipeline so big targets fail.
        real_episodes = [
            self._episode("ep1", 0.5, expect_pass=False),
            self._episode("ep2", 0.0, expect_pass=False),  # 0,0,0 is inside even tight box
        ]
        bench = Sim2RealBench(
            sim_episodes=sim_episodes, real_episodes=real_episodes,
            sim_label="mujoco", real_label="randomized_mujoco",
        )
        report = bench.run()
        # ep1 sim passes (target=0.5 in [-1,1]) but real fails (out of [-0.1,0.1]).
        # ep2 sim passes; real also passes (0 inside both boxes).
        assert report.sim_report.passed == 2
        assert report.real_report.passed == 1
        assert report.transfer_gap == pytest.approx(0.5)
        # Report has the expected fields.
        rendered = report.render_md()
        assert "mujoco" in rendered
        assert "randomized_mujoco" in rendered

    def test_mismatched_lengths_raise(self):
        ep = self._episode("a", 0.0, expect_pass=True)
        with pytest.raises(ValueError):
            Sim2RealBench(sim_episodes=[ep, ep], real_episodes=[ep])

    def test_action_kl_zero_when_distributions_match(self):
        ep = self._episode("ep1", 0.5, expect_pass=True)
        bench = Sim2RealBench(sim_episodes=[ep], real_episodes=[ep])
        report = bench.run()
        # Same episode, same primitive distribution -> KL ~ 0.
        for v in report.action_kl.values():
            assert v < 1e-3
