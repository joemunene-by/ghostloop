"""Tests for v1.0.2 additions: distillation, deadline scheduler, live
policy intervention, calibration."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import (
    CalibrationReport,
    DeadlineGate,
    DeadlineMissed,
    DeadlineMonitor,
    Decision,
    Intent,
    InterventionGate,
    InterventionState,
    LivePolicyController,
    MockBackend,
    PolicyPipeline,
    PrimitiveRegistry,
    Result,
    Runtime,
    Trace,
    TraceEvent,
    analyse_capture,
    calibration_episode,
)
from ghostloop.calibration import CalibrationCapture, CalibrationCaptureLog
from ghostloop.core import DecisionAction, Primitive, ResultStatus
from ghostloop.primitives import move_to, scan
from ghostloop.training import (
    DistillationDataset,
    DistillationSample,
    DistillationTrainer,
    collect_dataset,
)


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------


def _trace_with_events(specs: list[tuple[str, dict, str]]) -> Trace:
    trace = Trace()
    for i, (name, args, action) in enumerate(specs):
        intent = Intent(name, args)
        decision = (
            Decision.allow("test", "ok") if action == "allow"
            else Decision.deny("test", "blocked")
        )
        result = Result(
            status=ResultStatus.OK if action == "allow" else ResultStatus.BLOCKED,
            observation={"force": 1.0},
        )
        trace.append(TraceEvent(
            step=i, intent=intent, decision=decision, result=result,
            state_before={"position": [0, 0, 0]},
            state_after={"position": [args.get("x", 0.0), 0, 0]},
            timestamp=float(i),
        ))
    return trace


class TestDistillation:
    def test_dataset_filters_denied_by_default(self):
        trace = _trace_with_events([
            ("move_to", {"x": 0.5}, "allow"),
            ("move_to", {"x": 5.0}, "deny"),
            ("scan", {"radius": 0.3}, "allow"),
        ])
        ds = DistillationDataset()
        added = ds.add_trace(trace)
        assert added == 2
        names = [s.intent_name for s in ds.samples]
        assert "scan" in names
        assert ds.samples[0].intent_name == "move_to"

    def test_dataset_includes_denied_when_flagged(self):
        trace = _trace_with_events([
            ("move_to", {"x": 5.0}, "deny"),
        ])
        ds = DistillationDataset()
        ds.add_trace(trace, include_denied=True)
        assert ds.n == 1
        assert ds.samples[0].metadata["was_denied"] is True

    def test_split_train_val(self):
        trace = _trace_with_events([("move_to", {"x": float(i)}, "allow") for i in range(10)])
        ds = DistillationDataset()
        ds.add_trace(trace)
        train, val = ds.split(train_frac=0.7)
        assert train.n == 7
        assert val.n == 3

    def test_jsonl_round_trip(self, tmp_path):
        trace = _trace_with_events([("move_to", {"x": 0.5}, "allow")])
        ds = DistillationDataset()
        ds.add_trace(trace)
        path = tmp_path / "ds.jsonl"
        ds.write_jsonl(path)
        loaded = DistillationDataset.read_jsonl(path)
        assert loaded.n == ds.n
        assert loaded.samples[0].intent_name == "move_to"

    def test_trainer_calls_student_update(self):
        trace = _trace_with_events([("move_to", {"x": float(i)}, "allow") for i in range(20)])
        ds = DistillationDataset()
        ds.add_trace(trace)

        class _Student:
            def __init__(self):
                self.update_calls = 0
            def act(self, state):
                return Intent("move_to", {})
            def update(self, batch):
                self.update_calls += 1
                return {"loss": 0.1}

        student = _Student()
        trainer = DistillationTrainer(
            student=student, dataset=ds,
            n_epochs=2, batch_size=5, log_every=0,
        )
        history = trainer.train()
        assert len(history) == 2
        assert student.update_calls > 0

    def test_collect_dataset_helper(self):
        trace = _trace_with_events([("move_to", {"x": 0.0}, "allow")])
        ds = collect_dataset(lambda: trace)
        assert ds.n == 1


# ---------------------------------------------------------------------------
# Deadline scheduler
# ---------------------------------------------------------------------------


class TestDeadlineMonitor:
    def test_no_miss_at_target_rate(self):
        m = DeadlineMonitor(target_hz=10.0, overrun_threshold=1.5)
        # First observe: no period to compute.
        assert m.observe(timestamp=0.0) is None
        # 0.1s later = exactly at 10 Hz.
        assert m.observe(timestamp=0.1) is None
        assert not m.in_violation

    def test_miss_when_period_exceeds_threshold(self):
        m = DeadlineMonitor(target_hz=10.0, overrun_threshold=1.2)
        m.observe(timestamp=0.0)
        # 0.5s later — way over the 0.12s threshold.
        miss = m.observe(timestamp=0.5)
        assert miss is not None
        assert miss.actual_period_s == pytest.approx(0.5)
        assert miss.consecutive_misses == 1
        assert m.in_violation

    def test_miss_callback_fires(self):
        events: list[DeadlineMissed] = []
        m = DeadlineMonitor(
            target_hz=10.0, overrun_threshold=1.2,
            on_miss=lambda e: events.append(e),
        )
        m.observe(timestamp=0.0)
        m.observe(timestamp=1.0)
        assert len(events) == 1

    def test_stats(self):
        m = DeadlineMonitor(target_hz=10.0)
        for i in range(5):
            m.observe(timestamp=i * 0.1)
        s = m.stats()
        assert s["n_observations"] == 4
        assert s["target_hz"] == 10.0


class TestDeadlineGate:
    def test_allows_when_no_misses(self):
        m = DeadlineMonitor(target_hz=10.0)
        m.observe(timestamp=0.0)
        m.observe(timestamp=0.1)
        gate = DeadlineGate(monitor=m, max_consecutive=3)
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        d = gate.check(Intent("move_to", {}), prim)
        assert d.action is DecisionAction.ALLOW

    def test_denies_after_max_consecutive_misses(self):
        m = DeadlineMonitor(target_hz=10.0, overrun_threshold=1.2)
        m.observe(timestamp=0.0)
        for i in range(5):
            m.observe(timestamp=(i + 1) * 1.0)  # all misses
        gate = DeadlineGate(monitor=m, max_consecutive=3)
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        d = gate.check(Intent("move_to", {}), prim)
        assert d.action is DecisionAction.DENY


# ---------------------------------------------------------------------------
# Live policy intervention
# ---------------------------------------------------------------------------


class TestLivePolicyController:
    def test_act_runs_policy(self):
        policy = lambda state: Intent("move_to", {"x": 1.0})
        c = LivePolicyController(policy=policy)
        intent = c.act({})
        assert intent.name == "move_to"

    def test_pause_uses_fallback(self):
        c = LivePolicyController(
            policy=lambda s: Intent("move_to", {}),
            fallback_policy=lambda s: Intent("stop", {}),
        )
        c.pause(operator="alice", reason="test")
        intent = c.act({})
        assert intent.name == "stop"
        assert c.state is InterventionState.PAUSED

    def test_pause_no_fallback_emits_event(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        c.pause()
        intent = c.act({})
        assert intent.name == "emit_event"

    def test_swap_replaces_policy(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        c.swap_to(lambda s: Intent("scan", {}))
        intent = c.act({})
        assert intent.name == "scan"
        assert c.state is InterventionState.RUNNING

    def test_emergency_stop_latches(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        c.emergency_stop(stop_intent=Intent("land", {}))
        assert c.act({}).name == "land"
        assert c.act({}).name == "land"
        c.resume()
        assert c.act({}).name == "move_to"

    def test_history_records_transitions(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        c.pause(operator="alice", reason="check")
        c.resume(operator="alice", reason="ok")
        events = c.history()
        assert len(events) == 2
        assert events[0].operator == "alice"


class TestInterventionGate:
    def test_allow_when_running(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        gate = InterventionGate(controller=c)
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        d = gate.check(Intent("move_to", {}), prim)
        assert d.action is DecisionAction.ALLOW

    def test_deny_when_paused(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        c.pause(operator="alice", reason="halting")
        gate = InterventionGate(controller=c)
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        d = gate.check(Intent("move_to", {}), prim)
        assert d.action is DecisionAction.DENY
        assert "alice" in d.reason

    def test_emergency_stop_allows_safe_actions(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        c.emergency_stop()
        gate = InterventionGate(controller=c)
        prim = Primitive(name="stop", call=lambda *a, **k: None)
        d = gate.check(Intent("stop", {}), prim)
        assert d.action is DecisionAction.ALLOW

    def test_emergency_stop_denies_other_actions(self):
        c = LivePolicyController(policy=lambda s: Intent("move_to", {}))
        c.emergency_stop()
        gate = InterventionGate(controller=c)
        prim = Primitive(name="move_to", call=lambda *a, **k: None)
        d = gate.check(Intent("move_to", {}), prim)
        assert d.action is DecisionAction.DENY


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def _runtime(self):
        return Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([move_to(), scan()]),
            policy_pipeline=PolicyPipeline(),
        )

    def test_capture_records_events(self):
        rt = self._runtime()
        log = calibration_episode(rt, n_repeats=1)
        assert len(log.captures) > 0

    def test_report_extracts_workspace(self):
        rt = self._runtime()
        log = calibration_episode(rt, n_repeats=1)
        report = analyse_capture(log)
        assert isinstance(report, CalibrationReport)
        assert report.n_captures == len(log.captures)
        # MockBackend tracks position; should produce bounds.
        assert report.workspace_min is not None
        assert report.workspace_max is not None

    def test_report_handles_empty_log(self):
        report = analyse_capture(CalibrationCaptureLog())
        assert report.n_captures == 0
        assert "no captures recorded" in report.warnings

    def test_recommended_profile(self):
        rt = self._runtime()
        log = calibration_episode(rt, n_repeats=1)
        report = analyse_capture(log)
        profile = report.recommended_profile(name="my_robot")
        assert profile.name == "my_robot"
        if report.workspace_min is not None:
            assert profile.workspace_bounds is not None
