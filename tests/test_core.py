"""Tests for ghostloop.core: registry, runtime, trace, decision flow."""

import json
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
    Runtime,
)
from ghostloop.policies import DenyListGate, GeofenceGate, RateLimitGate
from ghostloop.primitives import move_to, pick, place, scan


def _runtime(*, gates=None, registry=None):
    backend = MockBackend()
    reg = registry or PrimitiveRegistry([move_to(), scan(), pick(), place()])
    pipe = PolicyPipeline(gates=gates or [])
    return Runtime(backend=backend, registry=reg, policy_pipeline=pipe)


class TestRegistry:
    def test_register_and_lookup(self):
        reg = PrimitiveRegistry([move_to()])
        assert "move_to" in reg
        assert reg.get("move_to") is not None
        assert reg.get("nonexistent") is None
        assert reg.names() == ["move_to"]

    def test_duplicate_raises(self):
        reg = PrimitiveRegistry([move_to()])
        with pytest.raises(ValueError, match="already registered"):
            reg.register(move_to())


class TestRuntimeStep:
    def test_unknown_primitive_blocks(self):
        rt = _runtime()
        result = rt.step(Intent("does_not_exist", {}))
        assert result.status.value == "blocked"
        assert "unknown primitive" in result.message
        assert len(rt.trace.events) == 1

    def test_move_to_updates_backend_position(self):
        rt = _runtime()
        result = rt.step(Intent("move_to", {"x": 0.5, "y": 0.0, "z": 0.2}))
        assert result.ok
        assert rt.backend.position == (0.5, 0.0, 0.2)
        assert result.observation["distance"] == pytest.approx(0.5385, abs=1e-3)

    def test_pick_then_place_round_trip(self):
        rt = _runtime()
        rt.step(Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0}))
        r1 = rt.step(Intent("pick", {"object_id": "cup-1"}))
        assert r1.ok
        assert rt.backend.held_object == "cup-1"
        r2 = rt.step(Intent("place", {}))
        assert r2.ok
        assert rt.backend.held_object is None

    def test_pick_when_already_holding_errors(self):
        rt = _runtime()
        rt.step(Intent("pick", {"object_id": "a"}))
        result = rt.step(Intent("pick", {"object_id": "b"}))
        assert result.status.value == "error"
        assert "already holding" in result.message

    def test_place_when_empty_errors(self):
        rt = _runtime()
        result = rt.step(Intent("place", {}))
        assert result.status.value == "error"
        assert "nothing to place" in result.message

    def test_primitive_exception_becomes_error_result(self):
        rt = _runtime()
        # move_to without args should raise TypeError, runtime catches it.
        result = rt.step(Intent("move_to", {}))
        assert result.status.value == "error"
        assert "TypeError" in result.message


class TestPolicyPipeline:
    def test_deny_list_blocks(self):
        rt = _runtime(gates=[DenyListGate(denied={"move_to"})])
        result = rt.step(Intent("move_to", {"x": 0.0, "y": 0.0, "z": 0.0}))
        assert result.status.value == "blocked"
        assert "deny_list" in result.message

    def test_geofence_blocks_outside(self):
        rt = _runtime(gates=[
            GeofenceGate(min_corner=(-1, -1, 0), max_corner=(1, 1, 1)),
        ])
        result = rt.step(Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0}))
        assert result.status.value == "blocked"
        assert "outside workspace" in result.message
        # Backend must not have moved.
        assert rt.backend.position == (0.0, 0.0, 0.0)

    def test_geofence_allows_inside(self):
        rt = _runtime(gates=[
            GeofenceGate(min_corner=(-1, -1, 0), max_corner=(1, 1, 1)),
        ])
        result = rt.step(Intent("move_to", {"x": 0.5, "y": 0.5, "z": 0.5}))
        assert result.ok

    def test_geofence_passes_through_intents_without_target(self):
        rt = _runtime(gates=[
            GeofenceGate(min_corner=(0, 0, 0), max_corner=(0.1, 0.1, 0.1)),
        ])
        # scan has no x/y/z args so the geofence should not apply.
        result = rt.step(Intent("scan", {"radius": 0.5}))
        assert result.ok

    def test_rate_limit_blocks_after_threshold(self):
        rt = _runtime(gates=[RateLimitGate(per_minute=2)])
        r1 = rt.step(Intent("scan", {}))
        r2 = rt.step(Intent("scan", {}))
        r3 = rt.step(Intent("scan", {}))
        assert r1.ok and r2.ok
        assert r3.status.value == "blocked"
        assert "exceeded" in r3.message

    def test_rate_limit_is_per_primitive(self):
        rt = _runtime(gates=[RateLimitGate(per_minute=1)])
        # Each primitive has its own bucket: the second scan would block,
        # but a different primitive should still go through.
        rt.step(Intent("scan", {}))
        result = rt.step(Intent("move_to", {"x": 0, "y": 0, "z": 0}))
        assert result.ok

    def test_pipeline_short_circuits_on_first_deny(self):
        rt = _runtime(gates=[
            DenyListGate(denied={"move_to"}),
            GeofenceGate(min_corner=(0, 0, 0), max_corner=(0.1, 0.1, 0.1)),
        ])
        result = rt.step(Intent("move_to", {"x": 5, "y": 5, "z": 5}))
        assert result.status.value == "blocked"
        # Should be denied by deny_list, not by geofence.
        event = rt.trace.events[-1]
        assert event.decision.gate_name == "deny_list"


class TestTrace:
    def test_events_cover_every_step(self):
        rt = _runtime()
        rt.run([
            Intent("scan", {}),
            Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0}),
            Intent("pick", {"object_id": "x"}),
        ])
        assert len(rt.trace.events) == 3
        steps = [e.step for e in rt.trace.events]
        assert steps == [1, 2, 3]

    def test_trace_is_json_serialisable(self):
        rt = _runtime()
        rt.step(Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0}))
        payload = rt.trace.to_json()
        json.dumps(payload)  # must not raise
        assert payload["n_steps"] == 1
        assert payload["events"][0]["intent"]["name"] == "move_to"

    def test_state_before_and_after_differ_for_movement(self):
        rt = _runtime()
        rt.step(Intent("move_to", {"x": 0.3, "y": 0.4, "z": 0.0}))
        ev = rt.trace.events[0]
        assert ev.state_before["position"] == [0.0, 0.0, 0.0]
        assert ev.state_after["position"] == [0.3, 0.4, 0.0]

    def test_write_jsonl(self, tmp_path):
        rt = _runtime()
        rt.step(Intent("scan", {"radius": 0.5}))
        out = tmp_path / "trace.jsonl"
        rt.trace.write_jsonl(str(out))
        lines = out.read_text().strip().splitlines()
        # 1 header + 1 event = 2 lines.
        assert len(lines) == 2
        header = json.loads(lines[0])
        assert header["n_steps"] == 1
        ev = json.loads(lines[1])
        assert ev["step"] == 1


class TestDecision:
    def test_allow_factory(self):
        d = Decision.allow("test_gate", "ok")
        assert d.action.value == "allow"
        assert d.gate_name == "test_gate"

    def test_deny_factory(self):
        d = Decision.deny("test_gate", "bad")
        assert d.action.value == "deny"
        assert d.reason == "bad"

    def test_to_json(self):
        d = Decision.deny("g", "r")
        payload = d.to_json()
        assert payload == {"action": "deny", "reason": "r", "gate_name": "g"}


class TestEmptyPipeline:
    def test_no_gates_allows_everything_in_workspace(self):
        rt = _runtime(gates=[])
        result = rt.step(Intent("move_to", {"x": 100, "y": 100, "z": 100}))
        # No geofence gate -> goes through.
        assert result.ok
