"""Tests for the v1.0+ profile system: primitive library, RobotProfile,
preset profiles, YAML loader, custom-primitive resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import Intent, MockBackend
from ghostloop.core import DecisionAction
from ghostloop.primitives import primitives_for_categories
from ghostloop.profiles import (
    RobotProfile,
    SafetyGateSpec,
    build_runtime_from_profile,
    franka_arm,
    humanoid_demo,
    load_profile_yaml,
    spot_quadruped,
    stretch_mobile_arm,
    tello_drone,
    turtlebot_base,
)


# ---------------------------------------------------------------------------
# Cross-morphology primitive library
# ---------------------------------------------------------------------------


class TestPrimitiveLibrary:
    def test_mobile_base_set(self):
        prims = primitives_for_categories(["mobile_base"])
        names = {p.name for p in prims}
        assert {"drive", "stop", "goto", "rotate"}.issubset(names)

    def test_quadruped_set(self):
        prims = primitives_for_categories(["quadruped"])
        names = {p.name for p in prims}
        assert {"sit", "stand", "lie_down", "walk_to"}.issubset(names)

    def test_aerial_set(self):
        prims = primitives_for_categories(["aerial"])
        names = {p.name for p in prims}
        assert {"takeoff", "land", "fly_to", "hover"}.issubset(names)

    def test_unknown_category_silently_skipped(self):
        prims = primitives_for_categories(["nonexistent", "humanoid"])
        names = {p.name for p in prims}
        assert "wave" in names

    def test_dedup_across_categories(self):
        # mobile_base + quadruped both list `stop` / `rotate`.
        prims = primitives_for_categories(["mobile_base", "quadruped"])
        names = [p.name for p in prims]
        assert names.count("stop") == 1
        assert names.count("rotate") == 1


# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------


class TestPresets:
    def test_franka_arm_dispatches(self):
        profile = franka_arm()
        runtime = build_runtime_from_profile(profile)
        assert "set_gripper" in runtime.registry.names()
        # Geofence + ForceCap configured.
        gate_kinds = {g.__class__.__name__ for g in runtime.policy_pipeline.gates}
        assert "GeofenceGate" in gate_kinds
        assert "ForceCapGate" in gate_kinds

    def test_spot_has_quadruped_primitives(self):
        profile = spot_quadruped()
        runtime = build_runtime_from_profile(profile)
        assert {"sit", "stand", "walk_to"}.issubset(set(runtime.registry.names()))
        # walk_to is HITL.
        gates = runtime.policy_pipeline.gates
        hitl = [g for g in gates if g.__class__.__name__ == "HumanInTheLoopGate"]
        assert hitl
        assert "walk_to" in hitl[0].requires_approval

    def test_tello_has_aerial_primitives(self):
        profile = tello_drone()
        runtime = build_runtime_from_profile(profile)
        assert {"takeoff", "land", "fly_to"}.issubset(set(runtime.registry.names()))

    def test_stretch_combines_mobile_and_dexterous(self):
        profile = stretch_mobile_arm()
        runtime = build_runtime_from_profile(profile)
        names = set(runtime.registry.names())
        assert "drive" in names
        assert "set_gripper" in names

    def test_humanoid_demo(self):
        profile = humanoid_demo()
        runtime = build_runtime_from_profile(profile)
        assert {"wave", "look_at", "point_at", "nod"}.issubset(set(runtime.registry.names()))

    def test_turtlebot(self):
        profile = turtlebot_base()
        runtime = build_runtime_from_profile(profile)
        assert "drive" in runtime.registry.names()


# ---------------------------------------------------------------------------
# Profile -> Runtime end-to-end
# ---------------------------------------------------------------------------


class TestProfileBuilder:
    def test_minimal_profile_falls_back_to_arm_primitives(self):
        profile = RobotProfile(name="empty")
        runtime = build_runtime_from_profile(profile)
        # Falls back so something is callable.
        assert "move_to" in runtime.registry.names()

    def test_workspace_bounds_drive_geofence_deny(self):
        profile = RobotProfile(
            name="bounded",
            categories=["aerial"],
            workspace_bounds=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        )
        runtime = build_runtime_from_profile(profile)
        # fly_to has full (x, y, z); GeofenceGate uses 3D and rejects out-of-box.
        runtime.step(Intent("fly_to", {"x": 5.0, "y": 0.0, "z": 0.0}))
        assert runtime.trace.events
        last = runtime.trace.events[-1]
        assert last.decision.action is DecisionAction.DENY

    def test_denied_primitive_blocked(self):
        profile = RobotProfile(
            name="denied_test",
            categories=["aerial"],
            denied_primitives=["takeoff"],
        )
        runtime = build_runtime_from_profile(profile)
        runtime.step(Intent("takeoff", {"altitude": 1.0}))
        last = runtime.trace.events[-1]
        assert last.decision.action is DecisionAction.DENY

    def test_composites_assembled(self):
        profile = RobotProfile(
            name="composite_test",
            categories=["humanoid"],
            composites=[
                {"name": "greet", "steps": ["wave", "nod"], "description": "greeting"},
            ],
        )
        runtime = build_runtime_from_profile(profile)
        assert "greet" in runtime.registry.names()


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


class TestYAMLLoader:
    def _write(self, tmp_path, body):
        p = tmp_path / "profile.yaml"
        p.write_text(body)
        return p

    def test_loads_minimal(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        path = self._write(tmp_path, """
name: minibot
morphology: mobile_base
categories: [mobile_base, generic]
instructions: |
  Test bot.
max_velocity: 0.5
""")
        profile = load_profile_yaml(path)
        assert profile.name == "minibot"
        assert profile.morphology == "mobile_base"
        assert profile.max_velocity == 0.5
        assert "Test bot." in profile.instructions

    def test_workspace_bounds_round_trip(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        path = self._write(tmp_path, """
name: t
workspace_bounds:
  - [-1, -1, 0]
  - [1, 1, 1]
""")
        profile = load_profile_yaml(path)
        assert profile.workspace_bounds == ((-1, -1, 0), (1, 1, 1))

    def test_safety_gate_specs(self, tmp_path):
        try:
            import yaml  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        path = self._write(tmp_path, """
name: t
safety_gates:
  - kind: cooldown
    args:
      default_s: 0.5
""")
        profile = load_profile_yaml(path)
        assert len(profile.safety_gates) == 1
        assert profile.safety_gates[0].kind == "cooldown"
        runtime = build_runtime_from_profile(profile)
        kinds = [g.__class__.__name__ for g in runtime.policy_pipeline.gates]
        assert "CooldownGate" in kinds

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile_yaml("/nonexistent/path.yaml")
