"""Tests for v0.7: GymnasiumBackend, CooldownGate, TimeWindowGate, SDF/convex,
composite primitives, mission orchestrator, WebSocket streaming."""

from __future__ import annotations

import asyncio
import datetime as _dt
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
    Result,
    Runtime,
)
from ghostloop.backends import gymnasium_available
from ghostloop.core import Primitive, ResultStatus
from ghostloop.missions import (
    Mission,
    MissionRunner,
    MissionStatus,
    Step,
    StepStatus,
)
from ghostloop.policies import (
    CooldownGate,
    GeofenceGate,
    TimeWindowGate,
    Window,
)
from ghostloop.policies.sdf import (
    ConvexPolytope,
    HalfSpace,
    signed_distance,
)
from ghostloop.policies.workspace import WorkspaceModel
from ghostloop.primitives import (
    composite_primitive,
    move_to,
    pick,
    place,
    scan,
)


def _registry(extra=()):
    return PrimitiveRegistry([move_to(), scan(), pick(), place(), *extra])


# ---------------------------------------------------------------------------
# Gymnasium backend (conditional)
# ---------------------------------------------------------------------------


class TestGymnasiumConditional:
    def test_helper_returns_bool(self):
        assert isinstance(gymnasium_available(), bool)

    @pytest.mark.skipif(gymnasium_available(), reason="gymnasium IS installed")
    def test_construction_without_gym_raises(self):
        from ghostloop.backends import GymnasiumBackend
        with pytest.raises(ImportError) as exc:
            GymnasiumBackend(env_id="CartPole-v1")
        msg = str(exc.value)
        assert "pip install gymnasium" in msg
        assert "ghostloop[gym]" in msg


# ---------------------------------------------------------------------------
# CooldownGate
# ---------------------------------------------------------------------------


class TestCooldownGate:
    def test_no_cooldown_passes_everything(self):
        gate = CooldownGate(default_s=0)
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        for _ in range(5):
            assert rt.step(Intent("scan", {"radius": 0.3})).ok

    def test_default_cooldown_blocks_back_to_back(self):
        gate = CooldownGate(default_s=10.0)
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        first = rt.step(Intent("scan", {"radius": 0.3}))
        second = rt.step(Intent("scan", {"radius": 0.3}))
        assert first.ok
        assert second.status.value == "blocked"
        assert "cooldown" in second.message

    def test_per_primitive_override(self):
        gate = CooldownGate(default_s=10.0, per_primitive={"scan": 0.0})
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        # scan has 0s cooldown -> back-to-back works.
        assert rt.step(Intent("scan", {"radius": 0.1})).ok
        assert rt.step(Intent("scan", {"radius": 0.1})).ok
        # move_to inherits default 10s -> second blocks.
        assert rt.step(Intent("move_to", {"x": 0, "y": 0, "z": 0})).ok
        assert rt.step(Intent("move_to", {"x": 0.1, "y": 0, "z": 0})).status.value == "blocked"

    def test_separate_timers_per_primitive(self):
        gate = CooldownGate(default_s=10.0)
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        # scan cools down independently from move_to.
        rt.step(Intent("scan", {"radius": 0.1}))
        # Different primitive: passes immediately.
        assert rt.step(Intent("move_to", {"x": 0.1, "y": 0, "z": 0})).ok


# ---------------------------------------------------------------------------
# TimeWindowGate
# ---------------------------------------------------------------------------


class TestTimeWindowGate:
    def test_no_windows_pass_through(self):
        gate = TimeWindowGate()  # empty per_primitive
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        assert rt.step(Intent("scan", {"radius": 0.3})).ok

    def test_inside_window_allows(self):
        gate = TimeWindowGate(
            per_primitive={"move_to": [Window(_dt.time(0, 0), _dt.time(23, 59))]},
            now=lambda: _dt.datetime(2026, 5, 10, 14, 0, 0),  # 14:00
        )
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        assert rt.step(Intent("move_to", {"x": 0.1, "y": 0, "z": 0})).ok

    def test_outside_window_blocks(self):
        gate = TimeWindowGate(
            per_primitive={"move_to": [Window(_dt.time(9, 0), _dt.time(17, 0))]},
            now=lambda: _dt.datetime(2026, 5, 10, 22, 0, 0),  # 22:00
        )
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        result = rt.step(Intent("move_to", {"x": 0.1, "y": 0, "z": 0}))
        assert result.status.value == "blocked"
        assert "outside any allowed window" in result.message

    def test_overnight_window(self):
        gate = TimeWindowGate(
            per_primitive={
                "move_to": [Window(_dt.time(22, 0), _dt.time(6, 0),
                                    end_before_start=True)]
            },
            now=lambda: _dt.datetime(2026, 5, 10, 23, 30),  # 23:30 (inside overnight)
        )
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        assert rt.step(Intent("move_to", {"x": 0.1, "y": 0, "z": 0})).ok

    def test_unconfigured_primitive_passes(self):
        gate = TimeWindowGate(
            per_primitive={"move_to": [Window(_dt.time(9, 0), _dt.time(17, 0))]},
            now=lambda: _dt.datetime(2026, 5, 10, 22, 0, 0),
        )
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline(gates=[gate]))
        # scan isn't in per_primitive -> always passes.
        assert rt.step(Intent("scan", {"radius": 0.3})).ok


# ---------------------------------------------------------------------------
# SDF / convex hull workspace geometry
# ---------------------------------------------------------------------------


class TestSDF:
    def test_signed_distance_outside_clear(self):
        ws = WorkspaceModel(bounds_min=(-1, -1, -1), bounds_max=(1, 1, 1))
        d = signed_distance((0.0, 0.0, 0.0), ws)
        assert d == pytest.approx(1.0)  # 1 unit clearance to nearest wall

    def test_signed_distance_inside_sphere_negative(self):
        ws = WorkspaceModel(bounds_min=(-1, -1, -1), bounds_max=(1, 1, 1))
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.2)
        d = signed_distance((0.05, 0.0, 0.0), ws)
        # Inside the 0.2-radius sphere by 0.15 units -> distance is negative.
        assert d < 0
        assert d == pytest.approx(-0.15, abs=1e-6)

    def test_signed_distance_outside_sphere_positive(self):
        ws = WorkspaceModel(bounds_min=(-2, -2, -2), bounds_max=(2, 2, 2))
        ws.add_sphere((0.0, 0.0, 0.0), radius=0.2)
        d = signed_distance((0.5, 0.0, 0.0), ws)
        assert d > 0
        assert d == pytest.approx(0.3, abs=1e-6)  # 0.5 - 0.2 from sphere

    def test_halfspace_obstacle(self):
        ws = WorkspaceModel(bounds_min=(-10, -10, -10), bounds_max=(10, 10, 10))
        # Half-space: z < 0 is forbidden (floor).
        floor = HalfSpace(normal=(0, 0, -1), offset=0.0, label="floor")
        # Above floor -> positive.
        d_above = signed_distance((0, 0, 0.5), ws, extras=[floor])
        assert d_above > 0
        # Below floor -> negative.
        d_below = signed_distance((0, 0, -0.5), ws, extras=[floor])
        assert d_below < 0

    def test_convex_polytope(self):
        # Cube [-0.5, 0.5]^3 as 6 half-spaces (outward normals).
        # ConvexPolytope convention: interior is where n.p + offset <= 0
        # for every face, i.e. each plane n.p + offset = 0 is a face and
        # the outward normal points OUT of the polytope.
        cube = ConvexPolytope(faces=[
            HalfSpace(normal=(1, 0, 0), offset=-0.5),    # x = 0.5
            HalfSpace(normal=(-1, 0, 0), offset=-0.5),   # x = -0.5
            HalfSpace(normal=(0, 1, 0), offset=-0.5),    # y = 0.5
            HalfSpace(normal=(0, -1, 0), offset=-0.5),   # y = -0.5
            HalfSpace(normal=(0, 0, 1), offset=-0.5),    # z = 0.5
            HalfSpace(normal=(0, 0, -1), offset=-0.5),   # z = -0.5
        ], label="unit_cube")
        # Point inside the cube -> contains() returns True.
        assert cube.contains((0.0, 0.0, 0.0))
        # Point outside the cube -> contains() returns False.
        assert not cube.contains((1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Composite primitives
# ---------------------------------------------------------------------------


class TestCompositePrimitive:
    def test_sequential_dispatch_succeeds(self):
        # approach_grasp = move_to + pick.
        def map_args(kwargs, idx, prim):
            if prim.name == "move_to":
                return {"x": kwargs.get("x", 0), "y": kwargs.get("y", 0), "z": kwargs.get("z", 0)}
            if prim.name == "pick":
                return {"object_id": kwargs.get("object_id", "unknown")}
            return kwargs

        approach = composite_primitive(
            "approach_grasp",
            steps=[move_to(), pick()],
            description="Move to pose then pick.",
            arg_schema={"x": "float", "y": "float", "z": "float", "object_id": "str"},
            map_args=map_args,
        )
        rt = Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([approach, move_to(), scan(), pick(), place()]),
            policy_pipeline=PolicyPipeline(),
        )
        result = rt.step(Intent("approach_grasp", {
            "x": 0.5, "y": 0.0, "z": 0.1, "object_id": "widget-7",
        }))
        assert result.ok
        assert "substep_0_move_to" in result.observation
        assert "substep_1_pick" in result.observation
        assert rt.backend.held_object == "widget-7"

    def test_failure_short_circuits(self):
        def map_args(kwargs, idx, prim):
            if prim.name == "pick":
                return {"object_id": kwargs.get("object_id", "x")}
            return kwargs

        # First pick succeeds, second pick fails (already holding).
        bad = composite_primitive(
            "double_pick",
            steps=[pick(), pick()],
            map_args=map_args,
        )
        rt = Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([bad, move_to(), scan(), pick(), place()]),
            policy_pipeline=PolicyPipeline(),
        )
        result = rt.step(Intent("double_pick", {"object_id": "a"}))
        assert not result.ok
        # Second pick should be where it failed.
        assert result.observation.get("step_index") == 1
        assert "already holding" in result.message


# ---------------------------------------------------------------------------
# Mission orchestrator
# ---------------------------------------------------------------------------


class TestMission:
    def _runtime(self):
        return Runtime(backend=MockBackend(), registry=_registry(),
                       policy_pipeline=PolicyPipeline())

    def test_simple_mission_succeeds(self):
        rt = self._runtime()

        def s1_emit(_m, _s):
            return Intent("scan", {"radius": 0.3})

        def s2_emit(_m, _s):
            return Intent("move_to", {"x": 0.4, "y": 0, "z": 0})

        mission = Mission(
            name="basic",
            steps=[
                Step(name="scan", emit=s1_emit),
                Step(name="approach", emit=s2_emit, depends_on=["scan"]),
            ],
        )
        result = MissionRunner(rt).run(mission)
        assert result.status is MissionStatus.SUCCEEDED
        assert result.n_succeeded == 2
        # Order: scan ran before approach.
        assert [s.name for s in result.steps] == ["scan", "approach"]

    def test_mission_with_list_emit(self):
        rt = self._runtime()

        def emit_list(_m, _s):
            return [
                Intent("scan", {"radius": 0.3}),
                Intent("move_to", {"x": 0.1, "y": 0, "z": 0}),
            ]

        mission = Mission(
            name="batch",
            steps=[Step(name="batch", emit=emit_list)],
        )
        result = MissionRunner(rt).run(mission)
        assert result.status is MissionStatus.SUCCEEDED
        # The single step ran two intents.
        assert len(result.steps[0].intents) == 2

    def test_required_failure_skips_dependents(self):
        rt = Runtime(
            backend=MockBackend(), registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )

        def fail_emit(_m, _s):
            return Intent("move_to", {"x": 5.0, "y": 0, "z": 0})  # blocked

        def downstream_emit(_m, _s):
            return Intent("scan", {"radius": 0.3})

        mission = Mission(
            name="will_fail",
            steps=[
                Step(name="bad", emit=fail_emit),
                Step(name="downstream", emit=downstream_emit, depends_on=["bad"]),
            ],
        )
        result = MissionRunner(rt).run(mission)
        assert result.status is MissionStatus.FAILED
        assert result.steps[0].status is StepStatus.FAILED
        assert result.steps[1].status is StepStatus.SKIPPED

    def test_optional_failure_does_not_block_dependents(self):
        rt = Runtime(
            backend=MockBackend(), registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )

        def opt_fail_emit(_m, _s):
            return Intent("move_to", {"x": 5.0, "y": 0, "z": 0})

        def downstream_emit(_m, _s):
            return Intent("scan", {"radius": 0.3})

        mission = Mission(
            name="optional",
            steps=[
                Step(name="cleanup", emit=opt_fail_emit, required=False),
                Step(name="real_work", emit=downstream_emit, depends_on=["cleanup"]),
            ],
        )
        result = MissionRunner(rt).run(mission)
        # Mission status PARTIAL because cleanup was skipped after failure;
        # downstream still ran because cleanup was optional.
        # NOTE: in current impl, FAILED optional steps mark dependents as
        # PENDING -> they run when ready. Verify real_work ran.
        downstream = next(s for s in result.steps if s.name == "real_work")
        assert downstream.status is StepStatus.SUCCEEDED

    def test_retry_on_transient_failure(self):
        # Custom flaky primitive that succeeds on attempt 2.
        attempts = {"n": 0}

        def _flaky_call(_backend, **_kw):
            attempts["n"] += 1
            if attempts["n"] < 2:
                return Result(status=ResultStatus.ERROR, message="transient")
            return Result(status=ResultStatus.OK, observation={"attempts": attempts["n"]})

        flaky = Primitive(name="flaky", call=_flaky_call, description="test")
        rt = Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([flaky]),
            policy_pipeline=PolicyPipeline(),
        )

        def emit(_m, _s):
            return Intent("flaky", {})

        mission = Mission(
            name="retry",
            steps=[Step(name="risky", emit=emit, max_attempts=3)],
        )
        result = MissionRunner(rt).run(mission)
        assert result.status is MissionStatus.SUCCEEDED
        assert result.steps[0].attempts == 2

    def test_mission_rejects_cycles(self):
        with pytest.raises(ValueError, match="cycle"):
            Mission(
                name="cyclic",
                steps=[
                    Step(name="a", emit=lambda *a: Intent("scan", {}), depends_on=["b"]),
                    Step(name="b", emit=lambda *a: Intent("scan", {}), depends_on=["a"]),
                ],
            )

    def test_mission_rejects_unknown_dep(self):
        with pytest.raises(ValueError, match="unknown step"):
            Mission(
                name="bad",
                steps=[Step(name="a", emit=lambda *a: Intent("scan", {}), depends_on=["ghost"])],
            )

    def test_mission_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="duplicate"):
            Mission(
                name="dup",
                steps=[
                    Step(name="a", emit=lambda *a: Intent("scan", {})),
                    Step(name="a", emit=lambda *a: Intent("scan", {})),
                ],
            )


# ---------------------------------------------------------------------------
# WebSocket streaming (StreamManager logic; ws endpoint conditional on fastapi)
# ---------------------------------------------------------------------------


class TestStreamManager:
    def test_publish_appends_to_history(self):
        from ghostloop.dashboard import StreamManager
        mgr = StreamManager(max_history=4)
        for i in range(6):
            mgr.publish("alpha", {"step": i, "status": "ok"})
        history = mgr.history("alpha")
        # Bounded ring buffer keeps last 4.
        assert len(history) == 4
        assert history[0]["step"] == 2
        assert history[-1]["step"] == 5
        assert all(e["robot"] == "alpha" for e in history)

    def test_subscriber_receives_published_events(self):
        from ghostloop.dashboard import StreamManager

        async def _drive():
            mgr = StreamManager()
            q = await mgr.subscribe()
            mgr.publish("alpha", {"step": 1})
            mgr.publish("beta", {"step": 2})
            envelopes = []
            for _ in range(2):
                envelopes.append(await asyncio.wait_for(q.get(), timeout=1.0))
            await mgr.unsubscribe(q)
            return envelopes

        envelopes = asyncio.run(_drive())
        assert envelopes[0]["robot"] == "alpha"
        assert envelopes[1]["robot"] == "beta"

    def test_history_per_robot(self):
        from ghostloop.dashboard import StreamManager
        mgr = StreamManager()
        mgr.publish("alpha", {"step": 1})
        mgr.publish("beta", {"step": 99})
        assert mgr.history("alpha")[0]["step"] == 1
        assert mgr.history("beta")[0]["step"] == 99
        assert mgr.history("nonexistent") == []
