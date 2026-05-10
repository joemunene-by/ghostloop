"""Tests for v1.0: RGB-D fusion + lightweight object detection,
VLA-on-MuJoCo benchmark harness, production fleet dashboard."""

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
)
from ghostloop.bench import (
    BaselineSpec,
    Episode,
    VLABenchmarkSuite,
    catalogue_published,
    suite_with_published_baselines,
)
from ghostloop.core import DecisionAction, Primitive, ResultStatus
from ghostloop.dashboard import (
    AlarmRegistry,
    ProductionConfig,
    RateLimiter,
    StaticTokenAuth,
)
from ghostloop.policies import GeofenceGate
from ghostloop.primitives import move_to, scan
from ghostloop.sensors import (
    BlobDetector,
    CameraFrame,
    CameraIntrinsics,
    CameraProcessorPipeline,
    ColorTarget,
    Detection,
    deproject_depth,
)


# ---------------------------------------------------------------------------
# RGB-D fusion + object detection
# ---------------------------------------------------------------------------


class TestDeprojectDepth:
    def _frame_with_depth(self, depth):
        intr = CameraIntrinsics(width=4, height=4, fx=2.0, fy=2.0, cx=2.0, cy=2.0)
        return CameraFrame(
            name="test",
            timestamp=time.time(),
            intrinsics=intr,
            depth=depth,
        )

    def test_pure_python_path(self):
        # 2x2 depth grid (pure list) -- intrinsics 2x2 to match.
        intr = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
        frame = CameraFrame(
            name="t", timestamp=0.0, intrinsics=intr,
            depth=[[1.0, 2.0], [3.0, 4.0]],
        )
        # Don't import numpy in this test path — call directly.
        from ghostloop.sensors.perception import _deproject_pure
        fused = _deproject_pure(frame, valid_min=1e-3, valid_max=10.0)
        # Pure-python fused.points is a flat list of (x, y, z) tuples.
        assert isinstance(fused.points, list)
        assert len(fused.points) == 4
        # Pixel (0,0) at depth 1.0 -> X = (0-1)/1.0 * 1 = -1, Y = -1, Z = 1.
        first = fused.points[0]
        assert first[2] == 1.0

    def test_rejects_missing_depth(self):
        intr = CameraIntrinsics(width=2, height=2, fx=1.0, fy=1.0, cx=1.0, cy=1.0)
        frame = CameraFrame(
            name="t", timestamp=0.0, intrinsics=intr, depth=None,
        )
        with pytest.raises(ValueError):
            deproject_depth(frame)


class TestBlobDetector:
    def test_finds_red_blob_in_pure_python(self):
        # 4x4 RGB grid where the centre 2x2 is red.
        red, white = (255, 0, 0), (255, 255, 255)
        rgb = [
            [white, white, white, white],
            [white, red,   red,   white],
            [white, red,   red,   white],
            [white, white, white, white],
        ]
        intr = CameraIntrinsics(width=4, height=4, fx=2, fy=2, cx=2, cy=2)
        frame = CameraFrame(
            name="t", timestamp=0.0, intrinsics=intr,
            rgb=rgb, depth=None,
        )
        det = BlobDetector(targets=[ColorTarget(
            label="red", rgb_min=(200, 0, 0), rgb_max=(255, 50, 50),
        )], min_area_px=2)
        # Force pure-python path even if numpy is available.
        results = det._detect_pure(frame)
        assert len(results) == 1
        detection = results[0]
        assert detection.label == "red"
        assert detection.bbox == (1, 1, 2, 2)

    def test_no_blob_returns_empty(self):
        rgb = [
            [(0, 0, 0), (0, 0, 0)],
            [(0, 0, 0), (0, 0, 0)],
        ]
        intr = CameraIntrinsics(width=2, height=2, fx=1, fy=1, cx=1, cy=1)
        frame = CameraFrame(
            name="t", timestamp=0.0, intrinsics=intr, rgb=rgb,
        )
        det = BlobDetector(targets=[ColorTarget(
            label="red", rgb_min=(200, 0, 0), rgb_max=(255, 50, 50),
        )], min_area_px=1)
        results = det._detect_pure(frame)
        assert results == []

    def test_pipeline_aggregates_detectors(self):
        rgb = [
            [(255, 0, 0), (255, 0, 0)],
            [(255, 0, 0), (255, 0, 0)],
        ]
        intr = CameraIntrinsics(width=2, height=2, fx=1, fy=1, cx=1, cy=1)
        frame = CameraFrame(name="t", timestamp=0.0, intrinsics=intr, rgb=rgb)

        class _AlwaysDetect:
            def detect(self, f):
                return [Detection(
                    label="x", score=1.0, bbox=(0, 0, 1, 1),
                    centroid_px=(0.5, 0.5),
                )]

        pipeline = CameraProcessorPipeline(detectors=[_AlwaysDetect(), _AlwaysDetect()])
        out = pipeline.process(frame)
        assert len(out) == 2

    def test_pipeline_swallows_detector_errors(self):
        intr = CameraIntrinsics(width=1, height=1, fx=1, fy=1, cx=0, cy=0)
        frame = CameraFrame(name="t", timestamp=0.0, intrinsics=intr)

        class _Crash:
            def detect(self, f):
                raise RuntimeError("oops")

        pipeline = CameraProcessorPipeline(detectors=[_Crash()])
        # Errors are swallowed; pipeline returns empty.
        assert pipeline.process(frame) == []


# ---------------------------------------------------------------------------
# VLA-on-MuJoCo benchmark
# ---------------------------------------------------------------------------


class TestVLABenchmark:
    def _episode(self, name: str, target_x: float) -> Episode:
        def setup():
            return MockBackend()

        def policy(runtime):
            runtime.step(Intent("move_to", {"x": target_x, "y": 0.0, "z": 0.0}))
            return None

        def predicate(trace, snap):
            return any(
                ev.decision.action is DecisionAction.ALLOW for ev in trace.events
            )

        return Episode(
            name=name, goal=f"reach x={target_x}",
            setup=setup, policy=policy, success_predicate=predicate,
            primitives=lambda: [move_to(), scan()],
            pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )

    def test_run_against_baselines(self):
        eps = [
            self._episode("ep1", 0.0),
            self._episode("ep2", 0.5),
            self._episode("ep3", 5.0),  # out of bounds -> fail
        ]
        suite = VLABenchmarkSuite(
            bench_label="reach_target",
            episodes=eps,
            baselines=[
                BaselineSpec(
                    name="Ref-A", bench_label="reach_target",
                    n=100, pass_rate=0.5,
                    citation="Test 2025",
                ),
            ],
        )
        result = suite.run(policy_label="ghostloop-test")
        assert result.n == 3
        assert result.passed == 2
        # Render markdown for sanity check.
        md = result.render_md()
        assert "Ref-A" in md
        assert "ghostloop-test" in md

    def test_published_catalogue_has_known_benches(self):
        cat = catalogue_published()
        assert "pick_place_widowx" in cat
        assert any(b.name.startswith("OpenVLA") for b in cat["pick_place_widowx"])

    def test_suite_with_published_baselines(self):
        eps = [self._episode("ep1", 0.0)]
        suite = suite_with_published_baselines("manipulation_bridge", eps)
        names = {b.name for b in suite.baselines}
        assert "π0" in names

    def test_baseline_ci_is_sensible(self):
        b = BaselineSpec(
            name="t", bench_label="b", n=100, pass_rate=0.7,
        )
        lo, hi = b.ci
        assert lo < 0.7 < hi
        assert b.passed == 70


# ---------------------------------------------------------------------------
# Production dashboard
# ---------------------------------------------------------------------------


class TestStaticTokenAuth:
    def test_empty_set_allows_all(self):
        auth = StaticTokenAuth(tokens=set())
        assert auth.authorise(None)
        assert auth.authorise("anything")

    def test_valid_token_allowed(self):
        auth = StaticTokenAuth(tokens={"secret"})
        assert auth.authorise("secret")
        assert not auth.authorise("wrong")
        assert not auth.authorise(None)


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        assert rl.check("k")
        assert rl.check("k")
        assert rl.check("k")
        assert not rl.check("k")          # 4th request fails

    def test_separate_keys_separate_buckets(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        assert rl.check("a")
        assert rl.check("b")              # different key, fresh bucket
        assert not rl.check("a")


class TestAlarmRegistry:
    def test_raise_then_ack(self):
        reg = AlarmRegistry()
        a = reg.raise_alarm(kind="test", message="boom", severity="error")
        assert len(reg.active()) == 1
        ack = reg.ack(a.id, by="alice")
        assert ack is not None
        assert ack.acked
        assert ack.acked_by == "alice"
        assert len(reg.active()) == 0

    def test_ack_unknown_returns_none(self):
        reg = AlarmRegistry()
        assert reg.ack("nonexistent") is None

    def test_history_capped(self):
        reg = AlarmRegistry()
        for i in range(10):
            reg.raise_alarm(kind="t", message=f"#{i}")
        history = reg.list_history(limit=5)
        assert len(history) == 5


class TestProductionConfig:
    def test_defaults(self):
        cfg = ProductionConfig()
        assert cfg.title.startswith("ghostloop")
        assert cfg.rate_limit_rps == 60
        assert cfg.metrics_enabled is True
