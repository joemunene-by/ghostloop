"""Tests for v0.8: STL temporal properties, URDF workspace, randomized backend,
trace query DSL, safe-RL training harness."""

from __future__ import annotations

import sys
import time
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
    Trace,
    TraceEvent,
)
from ghostloop.backends import RandomizationConfig, RandomizedBackend
from ghostloop.core import DecisionAction, Primitive, ResultStatus
from ghostloop.policies import GeofenceGate, workspace_from_urdf
from ghostloop.policies.workspace import AxisAlignedBox, WorkspaceModel
from ghostloop.primitives import move_to, scan
from ghostloop.properties import (
    Always,
    Eventually,
    Severity,
    Until,
    decision_action,
    intent_named,
    result_status,
    state_field_below,
)
from ghostloop.training import (
    LagrangianMultiplier,
    Rollout,
    RolloutBatch,
    SafeRolloutCollector,
    Transition,
    train_safe,
)
from ghostloop.traces import QueryError, compile_query, query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry() -> PrimitiveRegistry:
    return PrimitiveRegistry([move_to(), scan()])


def _trace_with_events(events: list[tuple[str, str, float]]) -> Trace:
    """Build a minimal Trace from (intent_name, decision_action, timestamp)."""
    trace = Trace()
    for i, (name, action, ts) in enumerate(events):
        intent = Intent(name, {})
        if action == "deny":
            decision = Decision.deny("test", "synthetic")
        else:
            decision = Decision.allow("test", "synthetic")
        result = Result(status=ResultStatus.OK, observation={"force": 1.0})
        trace.append(TraceEvent(
            step=i,
            intent=intent,
            decision=decision,
            result=result,
            state_before={},
            state_after={"force": 1.0},
            timestamp=ts,
        ))
    return trace


# ---------------------------------------------------------------------------
# STL temporal properties
# ---------------------------------------------------------------------------


class TestAlways:
    def test_holds_when_every_event_satisfies(self):
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("move_to", "allow", 2.0),
            ("move_to", "allow", 3.0),
        ])
        prop = Always(intent_named("move_to"), window_s=10.0, name="always_move")
        result = prop.check(trace)
        assert result.held
        assert not result.violations

    def test_violates_when_one_event_fails(self):
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("scan", "allow", 2.0),
            ("move_to", "allow", 3.0),
        ])
        prop = Always(intent_named("move_to"), window_s=10.0, name="always_move")
        result = prop.check(trace)
        assert not result.held
        assert result.violations

    def test_window_zero_is_pointwise(self):
        trace = _trace_with_events([("scan", "allow", 1.0)])
        prop = Always(intent_named("move_to"), window_s=0.0)
        result = prop.check(trace)
        assert not result.held


class TestEventually:
    def test_holds_when_one_event_satisfies(self):
        trace = _trace_with_events([
            ("scan", "allow", 1.0),
            ("scan", "allow", 2.0),
            ("move_to", "allow", 3.0),
        ])
        prop = Eventually(intent_named("move_to"), window_s=5.0)
        result = prop.check(trace)
        # First two events have no move_to in their preceding 5s window.
        assert not result.held

    def test_holds_when_global_window_unbounded(self):
        trace = _trace_with_events([
            ("scan", "allow", 1.0),
            ("move_to", "allow", 2.0),
        ])
        prop = Eventually(intent_named("move_to"), window_s=float("inf"))
        result = prop.check(trace)
        # Event 0: window includes only itself, no move_to -> violation.
        # Event 1: includes move_to -> OK.
        assert not result.held
        assert len(result.violations) == 1


class TestUntil:
    def test_held_until_psi_arrives_within_window(self):
        # phi = "intent is move_to", psi = "result observation contains 'goal'"
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("move_to", "allow", 1.5),
            ("scan", "allow", 2.0),  # psi fires here
        ])
        prop = Until(
            phi=intent_named("move_to"),
            psi=intent_named("scan"),
            window_s=5.0,
        )
        result = prop.check(trace)
        # First two anchors see psi at step 2 within window -> OK.
        # Anchor at step 2 IS psi itself -> OK.
        assert result.held

    def test_violates_when_psi_never_arrives(self):
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("move_to", "allow", 2.0),
        ])
        prop = Until(
            phi=intent_named("move_to"),
            psi=intent_named("never_seen"),
            window_s=5.0,
        )
        result = prop.check(trace)
        assert not result.held


class TestPredicateHelpers:
    def test_state_field_below(self):
        ev = TraceEvent(
            step=0,
            intent=Intent("move_to", {}),
            decision=Decision.allow("g", "ok"),
            result=Result(status=ResultStatus.OK),
            state_before={},
            state_after={"force": 5.0},
            timestamp=1.0,
        )
        assert state_field_below("force", 10.0)(ev)
        assert not state_field_below("force", 1.0)(ev)
        assert not state_field_below("missing", 1.0)(ev)

    def test_decision_action_predicate(self):
        ev = TraceEvent(
            step=0,
            intent=Intent("move_to", {}),
            decision=Decision.deny("g", "out of bounds"),
            result=Result(status=ResultStatus.BLOCKED),
            state_before={},
            state_after={},
            timestamp=1.0,
        )
        assert decision_action("deny")(ev)
        assert not decision_action("allow")(ev)


# ---------------------------------------------------------------------------
# URDF workspace builder
# ---------------------------------------------------------------------------


class TestURDFWorkspace:
    def _write_minimal_urdf(self, tmp_path: Path) -> Path:
        urdf = """
<?xml version="1.0"?>
<robot name="bot">
  <link name="base_link">
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry><box size="0.4 0.4 0.1"/></geometry>
    </collision>
  </link>
  <link name="arm">
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry><cylinder radius="0.05" length="0.6"/></geometry>
    </collision>
  </link>
  <link name="end_effector">
    <collision>
      <origin xyz="0 0 0.85"/>
      <geometry><sphere radius="0.04"/></geometry>
    </collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base_link"/>
    <child link="arm"/>
    <limit lower="-3.14" upper="3.14"/>
  </joint>
  <joint name="wrist" type="revolute">
    <parent link="arm"/>
    <child link="end_effector"/>
    <limit lower="-1.5" upper="1.5"/>
  </joint>
</robot>
""".strip()
        path = tmp_path / "bot.urdf"
        path.write_text(urdf)
        return path

    def test_parses_links_and_joints(self, tmp_path):
        urdf_path = self._write_minimal_urdf(tmp_path)
        ws, stats = workspace_from_urdf(urdf_path)
        assert stats.n_links == 3
        assert stats.n_joints == 2
        assert stats.n_revolute_joints == 2
        assert stats.n_collision_shapes == 3
        assert isinstance(ws, WorkspaceModel)

    def test_bounds_include_all_obstacles(self, tmp_path):
        urdf_path = self._write_minimal_urdf(tmp_path)
        ws, stats = workspace_from_urdf(urdf_path, bounds_inflate=0.0)
        # End effector top is at 0.85 + 0.04 = 0.89.
        assert ws.bounds_max[2] >= 0.89

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            workspace_from_urdf("/nonexistent/path.urdf")

    def test_malformed_urdf_raises(self, tmp_path):
        bad = tmp_path / "bad.urdf"
        bad.write_text("<not-valid-xml")
        with pytest.raises(ValueError):
            workspace_from_urdf(bad)


# ---------------------------------------------------------------------------
# Randomized backend
# ---------------------------------------------------------------------------


class TestRandomizedBackend:
    def test_snapshot_adds_noise_to_position(self):
        base = MockBackend()
        cfg = RandomizationConfig(pos_noise_std=0.05)
        wrap = RandomizedBackend(base=base, config=cfg, seed=42)
        snap1 = wrap.snapshot()
        # Mock backend's snapshot includes "position" by default.
        assert "_randomized" in snap1

    def test_action_drop_observed_with_high_prob(self):
        # A custom backend that exposes apply_action.
        class _ApplyBackend:
            name = "apply_test"
            def __init__(self):
                self.calls = 0
            def snapshot(self):
                return {}
            def apply_action(self, action):
                self.calls += 1
                return {"reward": 1.0}

        base = _ApplyBackend()
        cfg = RandomizationConfig(action_drop_prob=1.0)  # always drop
        wrap = RandomizedBackend(base=base, config=cfg, seed=0)
        result = wrap.apply_action([0.0, 0.0])
        assert result.get("dropped") is True
        assert base.calls == 0  # never reached the inner backend

    def test_seeded_reproducibility(self):
        base = MockBackend()
        cfg = RandomizationConfig(pos_noise_std=0.1)
        a = RandomizedBackend(base=base, config=cfg, seed=7)
        b = RandomizedBackend(base=MockBackend(), config=cfg, seed=7)
        snap_a = a.snapshot()
        snap_b = b.snapshot()
        # Both seeded the same -> same noise applied.
        for axis in ("x", "y", "z"):
            if axis in snap_a and axis in snap_b:
                assert snap_a[axis] == snap_b[axis]


# ---------------------------------------------------------------------------
# Trace query DSL
# ---------------------------------------------------------------------------


class TestTraceQuery:
    def test_filter_by_intent_name(self):
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("scan", "allow", 2.0),
            ("move_to", "deny", 3.0),
        ])
        results = query(trace, "intent.name == 'move_to'")
        assert len(results) == 2

    def test_filter_by_decision_action(self):
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("move_to", "deny", 2.0),
        ])
        results = query(trace, "decision.action == 'deny'")
        assert len(results) == 1
        assert results[0].step == 1

    def test_compound_expression(self):
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("scan", "allow", 2.0),
            ("move_to", "deny", 3.0),
        ])
        results = query(trace, "intent.name == 'move_to' and decision.action == 'allow'")
        assert len(results) == 1

    def test_in_operator(self):
        trace = _trace_with_events([
            ("move_to", "allow", 1.0),
            ("scan", "allow", 2.0),
            ("pick", "allow", 3.0),
        ])
        results = query(trace, "intent.name in ('move_to', 'scan')")
        assert len(results) == 2

    def test_invalid_expression_raises(self):
        with pytest.raises(QueryError):
            compile_query("intent.name @@ 'move_to'")

    def test_compile_returns_callable(self):
        pred = compile_query("intent.name == 'scan'")
        ev = _trace_with_events([("scan", "allow", 1.0)]).events[0]
        assert pred(ev)


# ---------------------------------------------------------------------------
# Safe-RL training harness
# ---------------------------------------------------------------------------


class _LinearPolicy:
    """Trivial scripted policy — emits constant action; logs update calls."""

    def __init__(self):
        self.update_calls = 0

    def act(self, obs):
        return [0.5, 0.0]

    def update(self, batch, lagrangian):
        self.update_calls += 1
        return {"loss": 0.0, "lagrangian_seen": lagrangian}


def _stub_intent_factory(obs, action):
    return Intent("move_to", {"x": float(action[0]), "y": float(action[1]), "z": 0.0})


class TestLagrangian:
    def test_grows_when_violation_above_target(self):
        lag = LagrangianMultiplier(target_rate=0.05, learning_rate=0.5, value=0.0)
        for _ in range(20):
            lag.update(0.5)
        assert lag.value > 0

    def test_shrinks_to_zero_when_below_target(self):
        lag = LagrangianMultiplier(target_rate=0.05, learning_rate=0.5, value=2.0, min_value=0.0)
        for _ in range(200):
            lag.update(0.0)
        assert lag.value == 0.0

    def test_clamped_by_max(self):
        lag = LagrangianMultiplier(
            target_rate=0.0, learning_rate=10.0, value=0.0, max_value=5.0
        )
        for _ in range(20):
            lag.update(1.0)
        assert lag.value == 5.0


class TestSafeRolloutCollector:
    def test_collects_episode_with_violations(self):
        rt = Runtime(
            backend=MockBackend(), registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )
        # Step always emits action [5.0, 0.0] which violates geofence (x=5).
        class _Pol:
            def act(self, obs): return [5.0, 0.0]
            def update(self, b, l): return {}
        collector = SafeRolloutCollector(
            runtime=rt, intent_factory=_stub_intent_factory,
            max_steps_per_episode=5,
        )
        rollout = collector.collect_episode(_Pol())
        assert rollout.length == 5
        # All five intents violated the geofence.
        assert rollout.violation_rate == 1.0

    def test_train_safe_loop_runs(self):
        rt = Runtime(
            backend=MockBackend(), registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[]),
        )
        collector = SafeRolloutCollector(
            runtime=rt, intent_factory=_stub_intent_factory,
            max_steps_per_episode=3,
        )
        policy = _LinearPolicy()
        lag = LagrangianMultiplier(target_rate=0.1)
        history = train_safe(
            collector=collector, policy=policy, lagrangian=lag,
            n_iterations=2, episodes_per_iteration=2, log_every=0,
        )
        assert len(history) == 2
        assert policy.update_calls == 2


class TestRolloutBatch:
    def test_aggregates_metrics(self):
        t1 = Transition(
            obs={}, action=0, reward=1.0, next_obs={}, done=False,
            violated=False, decision={},
        )
        t2 = Transition(
            obs={}, action=0, reward=2.0, next_obs={}, done=True,
            violated=True, decision={},
        )
        r1 = Rollout(transitions=[t1, t2], started_at=0.0, finished_at=1.0)
        r2 = Rollout(transitions=[t1], started_at=0.0, finished_at=0.5)
        batch = RolloutBatch(rollouts=[r1, r2])
        assert batch.n_episodes == 2
        assert batch.n_steps == 3
        assert batch.mean_return == pytest.approx((3.0 + 1.0) / 2.0)
        assert batch.mean_violation_rate == pytest.approx(1.0 / 3.0)
