"""Tests for the v0.3 additions: PyBullet conditional, ForceCap + HITL gates,
episode catalogue, trace replay, menagerie loader, CLI subcommands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

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
from ghostloop.backends import (
    KNOWN_MODELS,
    PyBulletBackend,
    pybullet_available,
    resolve_model,
)
from ghostloop.backends.menagerie import MenagerieError
from ghostloop.bench import (
    EpisodeRunner,
    paired_compare,
    preset_geofence_smoke,
    preset_pick_and_place_4,
    preset_reach_8,
    summarize,
)
from ghostloop.policies import (
    ForceCapGate,
    GeofenceGate,
    HumanInTheLoopGate,
    always_approve,
    always_deny,
)
from ghostloop.primitives import move_to, pick, place, scan
from ghostloop.traces import iter_events, load_trace, summarize_trace


# ----------------------------------------------------------------------
# PyBullet conditional import.
# ----------------------------------------------------------------------


class TestPyBulletConditional:
    def test_helper_returns_bool(self):
        assert isinstance(pybullet_available(), bool)

    @pytest.mark.skipif(pybullet_available(), reason="pybullet IS installed")
    def test_construction_without_pybullet_raises_with_install_hint(self):
        with pytest.raises(ImportError) as exc:
            PyBulletBackend(model_path="/nonexistent.urdf")
        msg = str(exc.value)
        assert "pip install pybullet" in msg
        assert "ghostloop[pybullet]" in msg


# ----------------------------------------------------------------------
# ForceCapGate.
# ----------------------------------------------------------------------


class TestForceCapGate:
    """Unit-test the gate.check() method directly so primitive arg signatures
    don't get in the way of the gate's behaviour."""

    def _check(self, gate, args):
        intent = Intent("test", args)
        # The gate doesn't actually look at the primitive object, so a stub
        # with .name='test' is enough.
        from ghostloop.core import Primitive
        prim = Primitive(name="test", call=lambda b, **kw: None, description="")
        return gate.check(intent, prim)

    def test_no_caps_set_passes_everything(self):
        d = self._check(ForceCapGate(), {"force": 9999, "torque": 9999})
        assert d.action.value == "allow"

    def test_force_above_cap_denies(self):
        d = self._check(ForceCapGate(force_max=50.0), {"force": 75.0})
        assert d.action.value == "deny"
        assert "force=75" in d.reason

    def test_velocity_cap_via_speed_alias(self):
        d = self._check(ForceCapGate(velocity_max=1.0), {"speed": 5.0})
        assert d.action.value == "deny"

    def test_intent_without_force_keys_passes(self):
        d = self._check(ForceCapGate(force_max=10.0), {"x": 0.1, "y": 0.0, "z": 0.0})
        assert d.action.value == "allow"

    def test_negative_value_uses_abs(self):
        d = self._check(ForceCapGate(torque_max=2.0), {"torque": -5.0})
        assert d.action.value == "deny"
        assert "torque=-5" in d.reason

    def test_unparseable_value_skipped(self):
        # Non-numeric force value silently passes (don't crash on weird input).
        d = self._check(ForceCapGate(force_max=10.0), {"force": "high"})
        assert d.action.value == "allow"


# ----------------------------------------------------------------------
# HumanInTheLoopGate.
# ----------------------------------------------------------------------


class TestHumanInTheLoopGate:
    def _runtime(self, gate):
        return Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([move_to(), scan(), pick(), place()]),
            policy_pipeline=PolicyPipeline(gates=[gate]),
        )

    def test_pass_through_for_unprotected_primitive(self):
        gate = HumanInTheLoopGate(requires_approval={"pick"}, approver=always_deny)
        rt = self._runtime(gate)
        # scan isn't on the list -> passes.
        result = rt.step(Intent("scan", {"radius": 0.5}))
        assert result.ok

    def test_protected_primitive_with_always_deny(self):
        gate = HumanInTheLoopGate(requires_approval={"pick"}, approver=always_deny)
        rt = self._runtime(gate)
        result = rt.step(Intent("pick", {"object_id": "x"}))
        assert result.status.value == "blocked"
        assert "denied" in result.message.lower()

    def test_protected_primitive_with_always_approve(self):
        gate = HumanInTheLoopGate(requires_approval={"pick"}, approver=always_approve)
        rt = self._runtime(gate)
        result = rt.step(Intent("pick", {"object_id": "x"}))
        assert result.ok
        assert rt.backend.held_object == "x"

    def test_approver_exception_blocks(self):
        def boom(_intent, _primitive):
            raise RuntimeError("approval system down")
        gate = HumanInTheLoopGate(requires_approval={"pick"}, approver=boom)
        rt = self._runtime(gate)
        result = rt.step(Intent("pick", {"object_id": "x"}))
        assert result.status.value == "blocked"
        assert "raised" in result.message.lower()

    def test_custom_predicate(self):
        # Approve picks of object_id starting with 'safe-'.
        def picky(intent, _primitive):
            return str(intent.args.get("object_id", "")).startswith("safe-")
        gate = HumanInTheLoopGate(requires_approval={"pick"}, approver=picky)
        rt = self._runtime(gate)
        assert rt.step(Intent("pick", {"object_id": "danger-1"})).status.value == "blocked"
        assert rt.step(Intent("pick", {"object_id": "safe-1"})).ok


# ----------------------------------------------------------------------
# Episode catalogue presets.
# ----------------------------------------------------------------------


class TestCataloguePresets:
    def test_reach_8_returns_8_episodes(self):
        eps = preset_reach_8()
        assert len(eps) == 8
        for ep in eps:
            assert ep.name.startswith("reach-")

    def test_reach_8_all_pass_with_default_runtime(self):
        eps = preset_reach_8()
        results = EpisodeRunner().run_all(eps)
        assert sum(1 for r in results if r.passed) == 8

    def test_pick_and_place_4(self):
        eps = preset_pick_and_place_4()
        assert len(eps) == 4
        results = EpisodeRunner().run_all(eps)
        # All 4 should pass: scripted pick + move + place.
        assert sum(1 for r in results if r.passed) == 4

    def test_geofence_smoke_8_episodes(self):
        eps = preset_geofence_smoke()
        assert len(eps) == 8

    def test_geofence_smoke_paired_compare_shows_only_b_zero(self):
        # Without geofence everyone passes; with geofence outsiders fail.
        eps_a = preset_geofence_smoke()
        res_a = EpisodeRunner().run_all(eps_a)
        rep_a = summarize(res_a, "no-fence", "smoke")

        eps_b = preset_geofence_smoke()
        for ep in eps_b:
            ep.pipeline = PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ])
        res_b = EpisodeRunner().run_all(eps_b)
        rep_b = summarize(res_b, "with-fence", "smoke")

        comp = paired_compare(rep_a, rep_b)
        assert comp.a_passed == 8
        assert comp.b_passed == 4
        assert comp.only_a == 4  # no-fence passes 4 cases the fenced run fails
        assert comp.only_b == 0


# ----------------------------------------------------------------------
# Trace replay.
# ----------------------------------------------------------------------


class TestTraceReplay:
    def _write_trace(self, tmp_path) -> Path:
        rt = Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([move_to(), scan(), pick(), place()]),
            policy_pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )
        rt.run([
            Intent("scan", {"radius": 0.5}),
            Intent("move_to", {"x": 0.5, "y": 0.0, "z": 0.0}),
            Intent("move_to", {"x": 9.0, "y": 0.0, "z": 0.0}),  # blocked
            Intent("pick", {"object_id": "x"}),
        ])
        path = tmp_path / "trace.jsonl"
        rt.trace.write_jsonl(str(path))
        return path

    def test_load_trace_round_trip(self, tmp_path):
        path = self._write_trace(tmp_path)
        header, events = load_trace(path)
        assert header.n_steps == 4
        assert len(events) == 4
        assert events[0].intent_name == "scan"
        assert events[2].decision_action == "deny"
        assert events[2].result_status == "blocked"

    def test_iter_events_skips_header(self, tmp_path):
        path = self._write_trace(tmp_path)
        events = list(iter_events(path))
        assert len(events) == 4

    def test_summarize_trace(self, tmp_path):
        path = self._write_trace(tmp_path)
        s = summarize_trace(path)
        assert s["n_events"] == 4
        assert s["by_status"]["ok"] == 3
        assert s["by_status"]["blocked"] == 1
        assert s["denied"] == 1
        assert "geofence" in s["deny_reasons"][0]


# ----------------------------------------------------------------------
# Menagerie loader (offline-only — git clone gated behind allow_clone).
# ----------------------------------------------------------------------


class TestMenagerieLoader:
    def test_known_models_includes_franka(self):
        assert "franka" in KNOWN_MODELS
        assert "ur5e" in KNOWN_MODELS
        assert "stretch" in KNOWN_MODELS

    def test_resolve_unknown_without_clone_raises(self, tmp_path):
        # Point at an empty dir, disable clone — should error cleanly.
        with patch.dict(__import__("os").environ, {"MENAGERIE_PATH": str(tmp_path)}):
            with pytest.raises(MenagerieError):
                # Bypass clone by using internal helper.
                from ghostloop.backends.menagerie import ensure_menagerie
                ensure_menagerie(allow_clone=False)

    def test_absolute_path_passes_through(self, tmp_path):
        # Create a fake xml; resolve_model should return its path unchanged.
        xml = tmp_path / "fake.xml"
        xml.write_text("<mujoco/>")
        assert resolve_model(str(xml)) == str(xml)


# ----------------------------------------------------------------------
# CLI subcommands.
# ----------------------------------------------------------------------


class TestCLI:
    def test_info_runs(self, capsys):
        from ghostloop.__main__ import main
        rc = main(["info"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ghostloop" in out
        assert "MockBackend" in out
        assert "Primitives" in out

    def test_demo_runs(self, capsys):
        from ghostloop.__main__ import main
        rc = main(["demo"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[OK ]" in out
        # Sixth intent must be blocked by the geofence.
        assert "[BLK]" in out

    def test_bench_runs(self, capsys):
        from ghostloop.__main__ import main
        rc = main(["bench"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Wilson 95% CI" in out
        assert "McNemar" in out

    def test_replay_summary(self, tmp_path, capsys):
        # Build a tiny trace and replay-summarise it.
        rt = Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([move_to(), scan(), pick(), place()]),
            policy_pipeline=PolicyPipeline(),
        )
        rt.step(Intent("scan", {"radius": 0.3}))
        rt.step(Intent("move_to", {"x": 0.1, "y": 0, "z": 0}))
        path = tmp_path / "t.jsonl"
        rt.trace.write_jsonl(str(path))

        from ghostloop.__main__ import main
        rc = main(["replay", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Episode" in out
        assert "Backend" in out
        assert "Events" in out

    def test_replay_json_mode(self, tmp_path, capsys):
        rt = Runtime(
            backend=MockBackend(),
            registry=PrimitiveRegistry([move_to(), scan(), pick(), place()]),
            policy_pipeline=PolicyPipeline(),
        )
        rt.step(Intent("scan", {"radius": 0.3}))
        path = tmp_path / "t.jsonl"
        rt.trace.write_jsonl(str(path))

        from ghostloop.__main__ import main
        rc = main(["replay", str(path), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["n_events"] == 1
