"""Claude Desktop -> ghostloop -> robot arm, end-to-end.

This is the canonical "I want Claude to drive a robot through ghostloop's
safety pipeline" entry point. Claude Desktop (or Cursor / any MCP client)
launches this file as a stdio subprocess, the script constructs a Runtime
with a Backend + safety pipeline, and exposes every Primitive as an MCP
tool the assistant can call.

Run it standalone to verify it boots:

    python3 examples/claude_desktop_mcp_arm.py --selfcheck

Wire it into Claude Desktop by adding this to
``~/Library/Application Support/Claude/claude_desktop_config.json``
(macOS) or ``%APPDATA%\\Claude\\claude_desktop_config.json`` (Windows):

    {
      "mcpServers": {
        "ghostloop": {
          "command": "python3",
          "args": ["/absolute/path/to/ghostloop/examples/claude_desktop_mcp_arm.py"]
        }
      }
    }

Restart Claude Desktop. The new conversation gets ``move_to`` / ``pick`` /
``place`` / ``scan`` / ``state`` / ``recent_trace`` / ``list_primitives``
tools. Try: "Move to (0.4, 0.0, 0.5), then scan with radius 0.3, then
move to (0.6, 0.2, 0.5)." The geofence will block any target outside
[-0.6, 0.6] in xy.

THREE BACKEND CHOICES — pick one by setting GHOSTLOOP_BACKEND below:

  - "mock" (default)   zero install, in-memory. Try it here first.
  - "mujoco"           real physics in MuJoCo. Needs ``pip install mujoco``
                       and a model file. See the BACKEND="mujoco" branch.
  - "ros2"             real hardware via DDS. Needs ROS 2 + rclpy installed
                       AND a robot driver running. See the "ros2" branch.

The safety pipeline is the SAME across all three backends. That's the whole
point: prove the policy + tool surface in sim, then promote to the real arm.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running this script from inside the repo without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import Intent, MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime
from ghostloop.policies import (
    ActionSmoothingGate,
    DenyListGate,
    ForceCapGate,
    GeofenceGate,
    HumanInTheLoopGate,
    RateLimitGate,
    cli_approver,
)
from ghostloop.primitives import move_to, pick, place, scan


# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup.
# ---------------------------------------------------------------------------

BACKEND = os.environ.get("GHOSTLOOP_BACKEND", "mock").lower()

# Workspace bounding box. The geofence rejects any target outside this AABB.
# Override with conservative numbers for your specific arm reach + table.
WORKSPACE_MIN = (-0.6, -0.6, 0.0)
WORKSPACE_MAX = (0.6, 0.6, 1.0)

# Hard force limit. Real arms expose force-torque sensors via a topic /
# state field; the gate consults Result observations for force_norm.
MAX_FORCE_N = 15.0

# Velocity / acceleration caps for ActionSmoothingGate. Tune per arm.
MAX_VELOCITY_MS = 0.5      # m/s
MAX_ACCELERATION_MS2 = 2.0  # m/s^2

# Rate limit: maximum primitive calls per minute, per primitive name.
RATE_LIMIT_PER_MIN = 60

# Operations Claude is NOT allowed to call without explicit human approval.
# Empty list = pure soft pipeline; populate for high-stakes actions.
HITL_OPERATIONS: list[str] = ["pick", "place"]

# Operations to deny outright (e.g. raw torque commands you don't want
# Claude touching). Empty list by default.
DENIED_OPERATIONS: list[str] = []


# ---------------------------------------------------------------------------
# Backend construction.
# ---------------------------------------------------------------------------


def build_backend():
    """Construct the configured Backend.

    The Mock path is zero-install. MuJoCo / ROS 2 require their respective
    extras and the lines below show exactly what to swap in.
    """
    if BACKEND == "mock":
        return MockBackend(name="mock_arm")

    if BACKEND == "mujoco":
        # pip install ghostloop[mujoco]
        from ghostloop.backends import MuJoCoBackend
        # Drop a model path here — Franka, UR5e, Stretch, Spot, Aloha all
        # work via the MuJoCo Menagerie. Pull one with:
        #   from ghostloop.backends import resolve_model
        #   path = resolve_model("franka")          # auto-clones first time
        return MuJoCoBackend(
            model_path=os.environ.get(
                "GHOSTLOOP_MUJOCO_MODEL",
                "franka_panda.xml",
            ),
            end_effector="hand",
        )

    if BACKEND == "ros2":
        # apt install ros-humble-rclpy ...; source /opt/ros/humble/setup.bash
        from ghostloop.backends import ROS2Backend
        return ROS2Backend(
            node_name=os.environ.get("GHOSTLOOP_ROS2_NODE", "ghostloop_arm"),
            cmd_vel_topic=os.environ.get("GHOSTLOOP_CMD_VEL", "/cmd_vel"),
            joint_state_topic=os.environ.get(
                "GHOSTLOOP_JOINT_STATES", "/joint_states",
            ),
            force_torque_topic=os.environ.get(
                "GHOSTLOOP_FORCE_TORQUE", "/wrench",
            ),
        )

    raise ValueError(
        f"unknown GHOSTLOOP_BACKEND={BACKEND!r}. "
        f"Supported: mock / mujoco / ros2."
    )


# ---------------------------------------------------------------------------
# Primitive registry.
# ---------------------------------------------------------------------------


def build_registry():
    """Default primitives bound to MockBackend.

    For MuJoCoBackend, swap to the MuJoCo-bound versions:
        from ghostloop.backends.mujoco import move_to as mujoco_move_to
        from ghostloop.backends.mujoco import scan as mujoco_scan
        return PrimitiveRegistry([mujoco_move_to(), mujoco_scan()])

    For ROS 2, write Franka/UR5e-specific Primitive factories that
    convert (x, y, z) intents into joint goals via your driver's
    MoveIt action server, then list them here.
    """
    return PrimitiveRegistry([move_to(), scan(), pick(), place()])


# ---------------------------------------------------------------------------
# Safety pipeline. Order matters: cheap gates first, expensive ones last.
# ---------------------------------------------------------------------------


def build_safety_pipeline() -> PolicyPipeline:
    gates = []
    if DENIED_OPERATIONS:
        gates.append(DenyListGate(denied=set(DENIED_OPERATIONS)))
    gates.append(RateLimitGate(per_minute=RATE_LIMIT_PER_MIN))
    gates.append(GeofenceGate(
        min_corner=WORKSPACE_MIN, max_corner=WORKSPACE_MAX,
    ))
    gates.append(ForceCapGate(
        force_max=MAX_FORCE_N,
        velocity_max=MAX_VELOCITY_MS,
        acceleration_max=MAX_ACCELERATION_MS2,
    ))
    gates.append(ActionSmoothingGate(
        max_velocity=MAX_VELOCITY_MS,
        max_acceleration=MAX_ACCELERATION_MS2,
    ))
    if HITL_OPERATIONS:
        gates.append(HumanInTheLoopGate(
            requires_approval=set(HITL_OPERATIONS),
            approver=cli_approver,
        ))
    return PolicyPipeline(gates=gates)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def build_runtime() -> Runtime:
    backend = build_backend()
    registry = build_registry()
    pipeline = build_safety_pipeline()
    return Runtime(backend=backend, registry=registry, policy_pipeline=pipeline)


def main() -> None:
    if "--selfcheck" in sys.argv:
        runtime = build_runtime()
        print(f"[ghostloop] backend={runtime.backend.name}")
        print(f"[ghostloop] primitives={runtime.registry.names()}")
        print(f"[ghostloop] gates={[g.__class__.__name__ for g in runtime.policy_pipeline.gates]}")
        # One step to confirm the pipeline runs end-to-end.
        result = runtime.step(Intent("move_to", {"x": 0.0, "y": 0.0, "z": 0.5}))
        print(f"[ghostloop] selfcheck step status={result.status.value} message={result.message}")
        return

    # Default mode: run the MCP server over stdio. Claude Desktop / Cursor
    # spawn this script and speak MCP through stdin/stdout, so DON'T print
    # to stdout from anywhere outside the MCP machinery — it'll corrupt
    # the protocol. Use stderr if you must.
    from ghostloop.mcp_server import run_mcp_server
    runtime = build_runtime()
    print(
        f"[ghostloop] starting MCP server (backend={runtime.backend.name})...",
        file=sys.stderr,
    )
    run_mcp_server(runtime, server_name="ghostloop-arm")


if __name__ == "__main__":
    main()
