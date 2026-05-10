"""Tests for v0.6: ObservationBuffer, RetryPolicy, property combinators,
TraceDiff, LLMPlanner, FleetRegistry/Dispatcher, dashboard, eval/diff CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import (
    Intent,
    MockBackend,
    ObservationBuffer,
    PolicyPipeline,
    PrimitiveRegistry,
    Result,
    Runtime,
    Trace,
)
from ghostloop.core import ResultStatus
from ghostloop.fleet import (
    Dispatch,
    DispatchStrategy,
    FleetDispatcher,
    FleetError,
    FleetRegistry,
    RobotHandle,
    RobotStatus,
)
from ghostloop.planning import LLMPlanner
from ghostloop.policies import (
    GeofenceGate,
    LLMPolicyConfig,
    RetryPolicy,
    is_any_error,
    is_transient_error,
)
from ghostloop.primitives import move_to, pick, place, scan
from ghostloop.properties import (
    AndProperty,
    NoConsecutiveDuplicateIntents,
    NotProperty,
    OrProperty,
    Severity,
    StaysInsideWorkspace,
)
from ghostloop.traces import diff_events, diff_traces, load_trace


def _registry():
    return PrimitiveRegistry([move_to(), scan(), pick(), place()])


# ---------------------------------------------------------------------------
# ObservationBuffer
# ---------------------------------------------------------------------------


class TestObservationBuffer:
    def test_append_and_latest(self):
        buf = ObservationBuffer(capacity=3)
        intent = Intent("scan", {"radius": 0.5})
        result = Result(status=ResultStatus.OK, observation={"detections": []})
        buf.append(intent, result, {"position": [0, 0, 0]})
        latest = buf.latest()
        assert latest is not None
        assert latest.intent_name == "scan"
        assert latest.status == "ok"

    def test_capacity_evicts_oldest(self):
        buf = ObservationBuffer(capacity=2)
        for i in range(5):
            buf.append(
                Intent("move_to", {"x": i, "y": 0, "z": 0}),
                Result(status=ResultStatus.OK),
                {"position": [i, 0, 0]},
            )
        assert len(buf) == 2
        recent = buf.n_recent(2)
        assert recent[0].intent_args["x"] == 3
        assert recent[1].intent_args["x"] == 4

    def test_filter_by_intent(self):
        buf = ObservationBuffer()
        buf.append(Intent("scan", {}), Result(status=ResultStatus.OK), {})
        buf.append(Intent("move_to", {"x": 0.1, "y": 0, "z": 0}),
                   Result(status=ResultStatus.OK), {})
        buf.append(Intent("scan", {}), Result(status=ResultStatus.OK), {})
        scans = buf.filter_by_intent("scan")
        assert len(scans) == 2

    def test_n_blocked_and_errored_counters(self):
        buf = ObservationBuffer()
        buf.append(Intent("a", {}), Result(status=ResultStatus.OK), {})
        buf.append(Intent("b", {}), Result(status=ResultStatus.BLOCKED), {})
        buf.append(Intent("c", {}), Result(status=ResultStatus.ERROR), {})
        buf.append(Intent("d", {}), Result(status=ResultStatus.BLOCKED), {})
        assert buf.n_blocked() == 2
        assert buf.n_errored() == 1

    def test_to_json_roundtrip(self):
        buf = ObservationBuffer()
        buf.append(Intent("scan", {"radius": 0.3}),
                   Result(status=ResultStatus.OK, observation={"x": 1}),
                   {"position": [0, 0, 0]})
        payload = buf.to_json()
        json.dumps(payload)  # must not raise
        assert payload["n_records"] == 1


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_transient_error_classifier(self):
        ok = Result(status=ResultStatus.OK)
        err_transient = Result(status=ResultStatus.ERROR, message="HTTP 503 service busy")
        err_permanent = Result(status=ResultStatus.ERROR, message="syntax error in arg")
        blocked = Result(status=ResultStatus.BLOCKED, message="timeout reason")
        assert is_transient_error(ok) is False
        assert is_transient_error(err_transient) is True
        assert is_transient_error(err_permanent) is False
        # Even with the word "timeout", BLOCKED is not retried.
        assert is_transient_error(blocked) is False

    def test_any_error_classifier(self):
        assert is_any_error(Result(status=ResultStatus.ERROR)) is True
        assert is_any_error(Result(status=ResultStatus.OK)) is False

    def test_succeeds_on_first_try(self):
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        retry = RetryPolicy(runtime=rt, max_attempts=3, backoff_initial_s=0.0,
                             jitter_frac=0.0)
        result = retry.step(Intent("move_to", {"x": 0.5, "y": 0, "z": 0}))
        assert result.ok
        assert len(rt.trace.events) == 1  # only one attempt needed

    def test_does_not_retry_blocked(self):
        # Pipeline blocks oversized moves; retry should NOT loop on BLOCKED.
        rt = Runtime(
            backend=MockBackend(), registry=_registry(),
            policy_pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )
        retry = RetryPolicy(runtime=rt, max_attempts=5, backoff_initial_s=0.0,
                             jitter_frac=0.0)
        result = retry.step(Intent("move_to", {"x": 9.0, "y": 0, "z": 0}))
        assert result.status.value == "blocked"
        assert len(rt.trace.events) == 1  # exactly one attempt, no retry

    def test_retries_transient_error(self):
        # Use a custom primitive that fails twice then succeeds.
        from ghostloop.core import Primitive
        attempts = {"n": 0}
        def _flaky(_backend, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return Result(status=ResultStatus.ERROR, message="upstream 503 try again")
            return Result(status=ResultStatus.OK, observation={"attempts": attempts["n"]})
        flaky_prim = Primitive(name="flaky", call=_flaky, description="test")
        reg = PrimitiveRegistry([flaky_prim])
        rt = Runtime(backend=MockBackend(), registry=reg, policy_pipeline=PolicyPipeline())
        retry = RetryPolicy(runtime=rt, max_attempts=5, backoff_initial_s=0.0,
                             jitter_frac=0.0)
        result = retry.step(Intent("flaky", {}))
        assert result.ok
        assert result.observation["attempts"] == 3
        assert attempts["n"] == 3  # 2 failures + 1 success


# ---------------------------------------------------------------------------
# Property combinators
# ---------------------------------------------------------------------------


class TestPropertyCombinators:
    def _trace_with(self, intents):
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        rt.run(intents)
        return rt.trace

    def test_and_holds_when_both_hold(self):
        trace = self._trace_with([
            Intent("scan", {"radius": 0.5}),
            Intent("move_to", {"x": 0.1, "y": 0, "z": 0}),
        ])
        prop = AndProperty(children=[
            StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            NoConsecutiveDuplicateIntents(),
        ])
        result = prop.check(trace)
        assert result.held

    def test_and_fails_when_one_fails(self):
        trace = self._trace_with([
            Intent("scan", {"radius": 0.5}),
            Intent("scan", {"radius": 0.5}),  # duplicate -> NoConsecutive fails
        ])
        prop = AndProperty(children=[
            StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            NoConsecutiveDuplicateIntents(),
        ])
        result = prop.check(trace)
        assert not result.held
        assert any("from" in v for v in result.violations)

    def test_or_holds_if_one_holds(self):
        trace = self._trace_with([
            Intent("scan", {"radius": 0.5}),
            Intent("scan", {"radius": 0.5}),
        ])
        # First child fails, second holds -> OR holds.
        prop = OrProperty(children=[
            NoConsecutiveDuplicateIntents(),
            StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
        ])
        result = prop.check(trace)
        assert result.held

    def test_not_inverts(self):
        trace = self._trace_with([Intent("scan", {"radius": 0.5})])
        # StaysInsideWorkspace holds -> NOT fails.
        prop = NotProperty(child=StaysInsideWorkspace(
            min_corner=(-1, -1, -1), max_corner=(1, 1, 1),
        ))
        result = prop.check(trace)
        assert not result.held
        # Inverse case.
        trace2 = self._trace_with([Intent("scan", {"radius": 0.5}),
                                    Intent("scan", {"radius": 0.5})])
        prop2 = NotProperty(child=NoConsecutiveDuplicateIntents())
        result2 = prop2.check(trace2)
        assert result2.held  # child failed, so NOT holds

    def test_severity_defaults_to_max_of_children(self):
        prop = AndProperty(children=[
            NoConsecutiveDuplicateIntents(),  # WARN
            StaysInsideWorkspace(),            # ERROR
        ])
        trace = self._trace_with([Intent("scan", {})])
        result = prop.check(trace)
        assert result.severity is Severity.ERROR


# ---------------------------------------------------------------------------
# TraceDiff
# ---------------------------------------------------------------------------


class TestTraceDiff:
    def _write_trace(self, tmp_path, intents, name):
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        rt.run(intents)
        path = tmp_path / name
        rt.trace.write_jsonl(str(path))
        return path

    def test_identical_traces(self, tmp_path):
        ints = [Intent("scan", {"radius": 0.3}),
                Intent("move_to", {"x": 0.1, "y": 0, "z": 0})]
        a = self._write_trace(tmp_path, ints, "a.jsonl")
        b = self._write_trace(tmp_path, ints, "b.jsonl")
        diff = diff_traces(a, b)
        assert diff.n_identical == 2
        assert diff.n_diverged == 0
        assert diff.first_divergence() is None

    def test_diverged_traces(self, tmp_path):
        a = self._write_trace(tmp_path, [
            Intent("scan", {"radius": 0.3}),
            Intent("move_to", {"x": 0.1, "y": 0, "z": 0}),
        ], "a.jsonl")
        b = self._write_trace(tmp_path, [
            Intent("scan", {"radius": 0.3}),
            Intent("move_to", {"x": 0.5, "y": 0, "z": 0}),  # different args
        ], "b.jsonl")
        diff = diff_traces(a, b)
        assert diff.n_identical == 1
        assert diff.n_diverged == 1
        first = diff.first_divergence()
        assert first is not None and first.step == 2

    def test_only_a_only_b(self, tmp_path):
        a = self._write_trace(tmp_path, [Intent("scan", {})], "a.jsonl")
        b = self._write_trace(tmp_path, [
            Intent("scan", {}), Intent("scan", {}),
        ], "b.jsonl")
        diff = diff_traces(a, b)
        assert diff.n_only_b == 1

    def test_render_md(self, tmp_path):
        a = self._write_trace(tmp_path, [Intent("scan", {})], "a.jsonl")
        b = self._write_trace(tmp_path, [Intent("scan", {})], "b.jsonl")
        md = diff_traces(a, b).render_md()
        assert "Trace diff" in md
        assert "Identical" in md


# ---------------------------------------------------------------------------
# LLMPlanner
# ---------------------------------------------------------------------------


class TestLLMPlanner:
    def test_plan_parses_submit_plan_tool_call(self):
        planner = LLMPlanner(registry=_registry(), config=LLMPolicyConfig())
        steps = [
            {"name": "scan", "args": {"radius": 0.5}, "rationale": "look around"},
            {"name": "move_to", "args": {"x": 0.5, "y": 0, "z": 0}, "rationale": "approach"},
            {"name": "pick", "args": {"object_id": "widget-7"}, "rationale": "grasp"},
        ]
        upstream = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "submit_plan",
                            "arguments": json.dumps({
                                "steps": steps,
                                "rationale": "5-stage pick approach",
                            }),
                        },
                    }],
                },
            }],
        }
        with patch("ghostloop.planning.llm_planner.urllib.request.urlopen") as mock_url:
            resp = MagicMock()
            resp.read.return_value = json.dumps(upstream).encode()
            mock_url.return_value.__enter__.return_value = resp
            result = planner.plan("pick widget-7 from the table")
        assert result.n_steps == 3
        assert result.intents[0].name == "scan"
        assert result.intents[2].args["object_id"] == "widget-7"

    def test_plan_raises_when_no_tool_call(self):
        from ghostloop.policies.llm import LLMPolicyError
        planner = LLMPlanner(registry=_registry(), config=LLMPolicyConfig())
        upstream = {"choices": [{"message": {"role": "assistant", "content": "no plan"}}]}
        with patch("ghostloop.planning.llm_planner.urllib.request.urlopen") as mock_url:
            resp = MagicMock()
            resp.read.return_value = json.dumps(upstream).encode()
            mock_url.return_value.__enter__.return_value = resp
            with pytest.raises(LLMPolicyError, match="submit_plan"):
                planner.plan("test")


# ---------------------------------------------------------------------------
# Fleet
# ---------------------------------------------------------------------------


def _make_robot(name: str) -> RobotHandle:
    return RobotHandle(
        name=name,
        runtime=Runtime(
            backend=MockBackend(name=f"mock-{name}"),
            registry=_registry(),
            policy_pipeline=PolicyPipeline(),
        ),
    )


class TestFleet:
    def test_register_and_lookup(self):
        fleet = FleetRegistry()
        fleet.register(_make_robot("alpha"))
        fleet.register(_make_robot("beta"))
        assert "alpha" in fleet
        assert len(fleet) == 2
        assert fleet.names() == ["alpha", "beta"]

    def test_duplicate_register_raises(self):
        fleet = FleetRegistry()
        fleet.register(_make_robot("alpha"))
        with pytest.raises(FleetError, match="already registered"):
            fleet.register(_make_robot("alpha"))

    def test_deregister_marks_offline_and_removes(self):
        fleet = FleetRegistry()
        r = _make_robot("alpha")
        fleet.register(r)
        fleet.deregister("alpha")
        assert "alpha" not in fleet
        assert r.status is RobotStatus.OFFLINE

    def test_filter_by_status_and_label(self):
        fleet = FleetRegistry()
        r1 = _make_robot("a"); r1.labels = {"site": "lab"}
        r2 = _make_robot("b"); r2.labels = {"site": "field"}; r2.status = RobotStatus.BUSY
        fleet.register(r1); fleet.register(r2)
        idle = fleet.filter_by_status(RobotStatus.IDLE)
        assert {r.name for r in idle} == {"a"}
        lab = fleet.filter_by_label("site", "lab")
        assert {r.name for r in lab} == {"a"}

    def test_dispatch_first_idle(self):
        fleet = FleetRegistry()
        fleet.register(_make_robot("a"))
        fleet.register(_make_robot("b"))
        disp = FleetDispatcher(registry=fleet, strategy=DispatchStrategy.FIRST_IDLE)
        result = disp.dispatch(Intent("move_to", {"x": 0.1, "y": 0, "z": 0}))
        assert result.robot_name == "a"
        assert result.result.ok

    def test_dispatch_round_robin(self):
        fleet = FleetRegistry()
        fleet.register(_make_robot("a"))
        fleet.register(_make_robot("b"))
        disp = FleetDispatcher(registry=fleet, strategy=DispatchStrategy.ROUND_ROBIN)
        d1 = disp.dispatch(Intent("scan", {"radius": 0.3}))
        d2 = disp.dispatch(Intent("scan", {"radius": 0.3}))
        d3 = disp.dispatch(Intent("scan", {"radius": 0.3}))
        names = [d1.robot_name, d2.robot_name, d3.robot_name]
        assert names[0] != names[1]  # cycles through
        assert names[0] == names[2]  # round-robin wraps

    def test_dispatch_pinned_robot(self):
        fleet = FleetRegistry()
        fleet.register(_make_robot("a"))
        fleet.register(_make_robot("b"))
        disp = FleetDispatcher(registry=fleet)
        result = disp.dispatch(Intent("scan", {"radius": 0.3}), robot="b")
        assert result.robot_name == "b"

    def test_dispatch_unknown_robot_errors(self):
        fleet = FleetRegistry()
        fleet.register(_make_robot("a"))
        disp = FleetDispatcher(registry=fleet)
        with pytest.raises(FleetError, match="unknown robot"):
            disp.dispatch(Intent("scan", {}), robot="ghost")

    def test_dispatch_empty_fleet_errors(self):
        disp = FleetDispatcher(registry=FleetRegistry())
        with pytest.raises(FleetError, match="no robots"):
            disp.dispatch(Intent("scan", {}))

    def test_snapshot_aggregates(self):
        fleet = FleetRegistry()
        fleet.register(_make_robot("a"))
        fleet.register(_make_robot("b"))
        snap = fleet.snapshot().to_json()
        assert snap["n_robots"] == 2
        assert snap["n_idle"] == 2
        assert snap["n_busy"] == 0


# ---------------------------------------------------------------------------
# Dashboard (conditional FastAPI)
# ---------------------------------------------------------------------------


from ghostloop.dashboard import dashboard_available  # noqa: E402


class TestDashboard:
    def test_helper_returns_bool(self):
        assert isinstance(dashboard_available(), bool)

    @pytest.mark.skipif(dashboard_available(), reason="fastapi IS installed")
    def test_create_app_without_fastapi_raises(self, tmp_path):
        from ghostloop import GhostloopStore
        from ghostloop.dashboard import create_dashboard_app
        with pytest.raises(ImportError) as exc:
            create_dashboard_app(GhostloopStore(str(tmp_path / "g.db")))
        assert "fastapi" in str(exc.value).lower()

    @pytest.mark.skipif(not dashboard_available(), reason="fastapi not installed")
    def test_dashboard_serves_stats(self, tmp_path):
        from fastapi.testclient import TestClient  # type: ignore
        from ghostloop import GhostloopStore
        from ghostloop.dashboard import create_dashboard_app
        store = GhostloopStore(str(tmp_path / "g.db"))
        app = create_dashboard_app(store)
        client = TestClient(app)
        r = client.get("/healthz")
        assert r.status_code == 200
        r = client.get("/v1/store/stats")
        assert r.status_code == 200
        assert r.json()["episodes"] == 0


# ---------------------------------------------------------------------------
# CLI: eval + diff subcommands
# ---------------------------------------------------------------------------


class TestNewCLI:
    def test_eval_runs_and_renders(self, capsys):
        from ghostloop.__main__ import main
        rc = main(["eval", "--preset", "reach_8"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Wilson 95% CI" in out

    def test_eval_with_geofence_blocks_some(self, capsys):
        from ghostloop.__main__ import main
        rc = main(["eval", "--preset", "geofence_smoke", "--with-geofence"])
        assert rc == 0
        out = capsys.readouterr().out
        # 4 inside + 4 outside; geofence blocks the outside ones.
        assert "4/8" in out

    def test_eval_writes_to_file(self, tmp_path, capsys):
        from ghostloop.__main__ import main
        out = tmp_path / "eval.md"
        rc = main(["eval", "--preset", "reach_8", "--out", str(out)])
        assert rc == 0
        assert out.read_text().count("Episode") > 0

    def test_diff_subcommand(self, tmp_path, capsys):
        # Build two trace files and diff via CLI.
        rt = Runtime(backend=MockBackend(), registry=_registry(),
                     policy_pipeline=PolicyPipeline())
        rt.step(Intent("scan", {"radius": 0.5}))
        a = tmp_path / "a.jsonl"; rt.trace.write_jsonl(str(a))
        rt2 = Runtime(backend=MockBackend(), registry=_registry(),
                      policy_pipeline=PolicyPipeline())
        rt2.step(Intent("move_to", {"x": 0.1, "y": 0, "z": 0}))
        b = tmp_path / "b.jsonl"; rt2.trace.write_jsonl(str(b))
        from ghostloop.__main__ import main
        rc = main(["diff", str(a), str(b)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Trace diff" in out
        assert "Diverged: 1" in out
