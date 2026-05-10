"""MCP-server entry point for ghostloop. Drive a robot arm from any MCP client.

Works with every MCP-aware assistant — Claude Desktop, Cursor, Continue,
Cline, Zed, Gemini CLI, plus any future client that speaks the protocol —
because MCP is the protocol; ghostloop is the server.

Three transports, picked via GHOSTLOOP_TRANSPORT:

  - ``stdio`` (default)        for desktop clients on the same machine.
                                The client spawns this script as a subprocess.
  - ``streamable-http``         the modern HTTP transport. Bind once, then
                                any number of remote clients (including mobile
                                MCP apps + browser-based UIs) connect via URL.
  - ``sse``                     legacy server-sent events HTTP. Same host:port
                                shape. Use only if your client doesn't yet
                                speak streamable-http.

Three backends, picked via GHOSTLOOP_BACKEND:

  - ``mock`` (default)   zero install, in-memory. Try it here first.
  - ``mujoco``           real physics in MuJoCo. ``pip install ghostloop[mujoco]``.
  - ``ros2``             real hardware via DDS. ROS 2 + rclpy + arm driver.

Run it standalone to verify it boots:

    python3 examples/claude_desktop_mcp_arm.py --selfcheck

Or stand up the HTTP server:

    GHOSTLOOP_TRANSPORT=streamable-http \\
    GHOSTLOOP_HOST=0.0.0.0 GHOSTLOOP_PORT=8765 \\
    python3 examples/claude_desktop_mcp_arm.py

Then point any MCP client at ``http://<host>:8765/mcp``.

The safety pipeline is the SAME across all backends + all transports.
That's the whole point: prove the policy + tool surface in sim with one
client, then promote to a real arm and a different client without
changing a line of safety-critical code. See examples/claude_desktop_config.json
for ready-to-paste config snippets per client.
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

    # Default mode: run the MCP server. Transport picked from the env so
    # the same script powers desktop (stdio) AND remote / mobile clients
    # (streamable-http). DON'T print to stdout outside MCP machinery —
    # it'll corrupt the stdio protocol. Use stderr if you must.
    from ghostloop.mcp_server import run_mcp_server
    transport = os.environ.get("GHOSTLOOP_TRANSPORT", "stdio").lower()
    host = os.environ.get("GHOSTLOOP_HOST", "127.0.0.1")
    port = int(os.environ.get("GHOSTLOOP_PORT", "8765"))
    runtime = build_runtime()
    print(
        f"[ghostloop] starting MCP server "
        f"(backend={runtime.backend.name} transport={transport} "
        f"{f'@ {host}:{port}' if transport != 'stdio' else ''})...",
        file=sys.stderr,
    )
    run_mcp_server(
        runtime, server_name="ghostloop-arm",
        transport=transport, host=host, port=port,
    )


if __name__ == "__main__":
    main()
