"""Tests for ghostloop.backends.mujoco.

Most tests run *without* mujoco installed — they verify the conditional
import + helpful error message. Live integration is gated on mujoco being
importable, so the offline test path stays at zero install cost.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop.backends import MuJoCoBackend, mujoco_available


class TestConditionalImport:
    def test_module_imports_without_mujoco(self):
        # The package import in __init__.py succeeded; that's the test.
        from ghostloop.backends.mujoco import MuJoCoBackend, mujoco_available  # noqa
        assert callable(mujoco_available)

    def test_mujoco_available_is_bool(self):
        assert isinstance(mujoco_available(), bool)

    @pytest.mark.skipif(mujoco_available(), reason="mujoco IS installed; install-hint test n/a")
    def test_construction_without_mujoco_raises_with_install_hint(self):
        with pytest.raises(ImportError) as exc:
            MuJoCoBackend(model_path="/nonexistent.xml")
        msg = str(exc.value)
        assert "pip install mujoco" in msg
        assert "ghostloop[mujoco]" in msg


@pytest.mark.skipif(
    not mujoco_available(),
    reason="mujoco not installed (set up via `pip install ghostloop[mujoco]`)",
)
class TestLiveMuJoCo:
    """End-to-end integration tests that only run if mujoco is importable.

    Uses a minimal one-body MJCF written to a tmp file so we don't depend on
    the MuJoCo Menagerie being on disk.
    """

    MJCF = """
    <mujoco>
      <worldbody>
        <body name="end_effector" pos="0 0 0.5">
          <joint name="x" type="slide" axis="1 0 0"/>
          <joint name="y" type="slide" axis="0 1 0"/>
          <joint name="z" type="slide" axis="0 0 1"/>
          <geom type="sphere" size="0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def _backend(self, tmp_path):
        path = tmp_path / "tiny.xml"
        path.write_text(self.MJCF)
        return MuJoCoBackend(model_path=str(path), end_effector="end_effector")

    def test_snapshot_shape(self, tmp_path):
        backend = self._backend(tmp_path)
        snap = backend.snapshot()
        assert snap["backend"] == "mujoco"
        assert len(snap["position"]) == 3
        assert len(snap["qpos"]) == 3

    def test_set_qpos_and_advance(self, tmp_path):
        backend = self._backend(tmp_path)
        backend.set_qpos([0.5, 0.0, 0.0])
        backend.advance(0.01)
        snap = backend.snapshot()
        assert snap["qpos"][0] == pytest.approx(0.5, abs=1e-3)

    def test_set_qpos_wrong_length_raises(self, tmp_path):
        backend = self._backend(tmp_path)
        with pytest.raises(ValueError, match="qpos length"):
            backend.set_qpos([0.0, 0.0])

    def test_move_to_primitive_drives_position(self, tmp_path):
        from ghostloop.backends.mujoco import move_to
        backend = self._backend(tmp_path)
        prim = move_to()
        result = prim.call(backend, x=0.3, y=0.2, z=0.0, duration=0.01)
        assert result.ok
        assert result.observation["target"] == [0.3, 0.2, 0.0]

    def test_scan_returns_detections(self, tmp_path):
        from ghostloop.backends.mujoco import scan
        backend = self._backend(tmp_path)
        prim = scan()
        result = prim.call(backend, radius=2.0)
        assert result.ok
        # World body + end_effector body should both be in range.
        assert len(result.observation["detections"]) >= 1
