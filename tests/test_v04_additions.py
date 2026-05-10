"""Tests for v0.4 additions: AsyncRuntime, store, vision sensors, MCP, telemetry."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import (
    AsyncPolicyPipeline,
    AsyncRuntime,
    Decision,
    GhostloopStore,
    Intent,
    MockBackend,
    PolicyPipeline,
    PrimitiveRegistry,
    Runtime,
)
from ghostloop.bench import EpisodeRunner, paired_compare, preset_geofence_smoke, summarize
from ghostloop.policies import GeofenceGate
from ghostloop.primitives import move_to, pick, place, scan
from ghostloop.sensors import Camera, CameraFrame, CameraIntrinsics, MockCamera, capture_camera
from ghostloop.telemetry import otel_available, step_span
from ghostloop.mcp_server import mcp_available


# ----------------------------------------------------------------------
# AsyncRuntime
# ----------------------------------------------------------------------


def _registry():
    return PrimitiveRegistry([move_to(), scan(), pick(), place()])


class TestAsyncRuntime:
    def test_step_runs_under_asyncio(self):
        rt = AsyncRuntime(backend=MockBackend(), registry=_registry())
        result = asyncio.run(rt.step(Intent("move_to", {"x": 0.4, "y": 0.0, "z": 0.0})))
        assert result.ok
        assert rt.backend.position == (0.4, 0.0, 0.0)

    def test_async_pipeline_blocks_via_async_gate(self):
        async def deny_check(intent, primitive):
            await asyncio.sleep(0)  # demonstrates awaitable
            return Decision.deny("async_test", "denied async")

        class _AsyncGate:
            name = "async_test"
            check = staticmethod(deny_check)

        pipe = AsyncPolicyPipeline(gates=[_AsyncGate()])
        rt = AsyncRuntime(backend=MockBackend(), registry=_registry(), policy_pipeline=pipe)
        result = asyncio.run(rt.step(Intent("scan", {"radius": 0.3})))
        assert result.status.value == "blocked"
        assert "denied async" in result.message

    def test_sync_pipeline_wrapped_transparently(self):
        # Pass an existing sync PolicyPipeline; AsyncRuntime adapts it.
        sync_pipe = PolicyPipeline(gates=[
            GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
        ])
        rt = AsyncRuntime(backend=MockBackend(), registry=_registry(), policy_pipeline=sync_pipe)
        result = asyncio.run(rt.step(Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0})))
        assert result.status.value == "blocked"

    def test_async_primitive_supported(self):
        from ghostloop.core import Primitive, Result, ResultStatus

        async def _async_call(backend, value: float = 1.0):
            await asyncio.sleep(0)
            return Result(status=ResultStatus.OK, observation={"value": value})

        async_prim = Primitive(name="async_op", call=_async_call, description="async test")
        reg = PrimitiveRegistry([async_prim])
        rt = AsyncRuntime(backend=MockBackend(), registry=reg)
        result = asyncio.run(rt.step(Intent("async_op", {"value": 42.0})))
        assert result.ok
        assert result.observation["value"] == 42.0

    def test_control_loop_terminates_on_none(self):
        rt = AsyncRuntime(backend=MockBackend(), registry=_registry())
        plan = [
            Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0}),
            Intent("move_to", {"x": 0.2, "y": 0.0, "z": 0.0}),
            None,
        ]
        idx = {"i": 0}
        async def next_intent(_rt, _last):
            i = idx["i"]; idx["i"] = i + 1
            return plan[i] if i < len(plan) else None
        steps = asyncio.run(rt.control_loop(next_intent, max_steps=10))
        assert steps == 2
        assert rt.backend.position == (0.2, 0.0, 0.0)

    def test_run_runs_a_list(self):
        rt = AsyncRuntime(backend=MockBackend(), registry=_registry())
        results = asyncio.run(rt.run([
            Intent("move_to", {"x": 0.1, "y": 0.0, "z": 0.0}),
            Intent("scan", {"radius": 0.5}),
        ]))
        assert all(r.ok for r in results)
        assert len(rt.trace.events) == 2


# ----------------------------------------------------------------------
# GhostloopStore (SQLite)
# ----------------------------------------------------------------------


class TestStore:
    def _trace(self):
        rt = Runtime(backend=MockBackend(), registry=_registry(), policy_pipeline=PolicyPipeline())
        rt.run([
            Intent("scan", {"radius": 0.5}),
            Intent("move_to", {"x": 0.4, "y": 0.0, "z": 0.0}),
        ])
        return rt.trace

    def test_save_and_load_episode(self, tmp_path):
        store = GhostloopStore(str(tmp_path / "g.db"))
        episode_id = store.save_episode(self._trace())
        loaded = store.load_episode(episode_id)
        assert loaded is not None
        assert loaded["n_steps"] == 2
        assert loaded["events"][0]["intent"]["name"] == "scan"

    def test_list_episodes(self, tmp_path):
        store = GhostloopStore(str(tmp_path / "g.db"))
        store.save_episode(self._trace())
        store.save_episode(self._trace())
        eps = store.list_episodes()
        assert len(eps) == 2
        assert all(ep.backend == "mock" for ep in eps)

    def test_save_run_report_round_trip(self, tmp_path):
        store = GhostloopStore(str(tmp_path / "g.db"))
        eps = preset_geofence_smoke()
        results = EpisodeRunner().run_all(eps)
        report = summarize(results, run_name="no-gates", bench_name="smoke")
        run_id = store.save_run_report(report)
        loaded = store.load_run(run_id)
        assert loaded is not None
        assert loaded["passed"] == 8

    def test_save_paired_comparison(self, tmp_path):
        store = GhostloopStore(str(tmp_path / "g.db"))
        eps_a = preset_geofence_smoke()
        rep_a = summarize(EpisodeRunner().run_all(eps_a), "a", "smoke")
        eps_b = preset_geofence_smoke()
        for ep in eps_b:
            ep.pipeline = PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ])
        rep_b = summarize(EpisodeRunner().run_all(eps_b), "b", "smoke")
        comp = paired_compare(rep_a, rep_b)
        comp_id = store.save_comparison(comp)
        assert isinstance(comp_id, str)
        stats = store.stats()
        assert stats["comparisons"] == 1

    def test_stats(self, tmp_path):
        store = GhostloopStore(str(tmp_path / "g.db"))
        store.save_episode(self._trace())
        s = store.stats()
        assert s["episodes"] == 1
        assert s["run_reports"] == 0
        assert s["comparisons"] == 0


# ----------------------------------------------------------------------
# Vision / Camera
# ----------------------------------------------------------------------


class TestCameraSensors:
    def test_mock_camera_capture_shape(self):
        cam = MockCamera(width=8, height=6)
        frame = cam.capture()
        assert frame.rgb_shape == (6, 8, 3)
        assert frame.depth_shape == (6, 8)
        assert isinstance(frame.intrinsics, CameraIntrinsics)
        assert frame.intrinsics.width == 8

    def test_capture_camera_primitive_falls_back_to_mock(self):
        backend = MockBackend()
        # backend has no .cameras — primitive must auto-mock.
        prim = capture_camera()
        result = prim.call(backend, camera="default")
        assert result.ok
        assert result.observation["frame"]["name"] == "default"

    def test_capture_camera_uses_attached_camera(self):
        backend = MockBackend()
        backend.cameras = {"hand_cam": MockCamera(name="hand_cam", width=4, height=4)}
        prim = capture_camera()
        result = prim.call(backend, camera="hand_cam")
        assert result.ok
        assert result.observation["frame"]["rgb_shape"] == [4, 4, 3]

    def test_camera_frame_metadata_is_json_serialisable(self):
        import json
        frame = MockCamera().capture()
        json.dumps(frame.metadata())  # must not raise

    def test_intrinsics_to_json(self):
        intr = CameraIntrinsics(width=64, height=48, fx=64.0, fy=48.0, cx=32.0, cy=24.0)
        d = intr.to_json()
        assert d["width"] == 64
        assert d["fx"] == 64.0


# ----------------------------------------------------------------------
# Telemetry / OpenTelemetry
# ----------------------------------------------------------------------


class TestTelemetry:
    def test_otel_available_returns_bool(self):
        assert isinstance(otel_available(), bool)

    def test_step_span_no_op_without_otel(self):
        # Even if OTel is not configured, step_span must yield cleanly.
        intent = Intent("scan", {"radius": 0.5})
        with step_span(intent) as span:
            pass  # no error means no-op worked
        # span may be None when OTel is absent or unconfigured
        assert span is None or hasattr(span, "set_attribute")


# ----------------------------------------------------------------------
# MCP server (conditional)
# ----------------------------------------------------------------------


class TestMCPServer:
    def test_mcp_available_returns_bool(self):
        assert isinstance(mcp_available(), bool)

    @pytest.mark.skipif(mcp_available(), reason="mcp IS installed; install-hint test n/a")
    def test_build_mcp_server_without_mcp_raises(self):
        from ghostloop.mcp_server import build_mcp_server
        rt = Runtime(backend=MockBackend(), registry=_registry(), policy_pipeline=PolicyPipeline())
        with pytest.raises(ImportError) as exc:
            build_mcp_server(rt)
        assert "pip install mcp" in str(exc.value) or "ghostloop[mcp]" in str(exc.value)

    @pytest.mark.skipif(not mcp_available(), reason="mcp not installed")
    def test_build_mcp_server_returns_fastmcp_instance(self):
        from ghostloop.mcp_server import build_mcp_server
        rt = Runtime(backend=MockBackend(), registry=_registry(), policy_pipeline=PolicyPipeline())
        server = build_mcp_server(rt, server_name="test-ghostloop")
        # FastMCP exposes .name; tools registered eagerly.
        assert server is not None
        assert getattr(server, "name", "") == "test-ghostloop"
