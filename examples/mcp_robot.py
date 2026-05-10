"""MCP server for any robot — pick a profile or load your own YAML.

This is the general-purpose entry point that supersedes the arm-only
``claude_desktop_mcp_arm.py``. Pick what you control via env vars:

  GHOSTLOOP_PROFILE          name of a built-in preset OR path to a YAML file
                             Built-ins:
                               franka_arm                arm
                               turtlebot                 mobile base
                               spot                      quadruped
                               tello                     drone
                               stretch                   mobile arm
                               humanoid_demo             humanoid

  GHOSTLOOP_BACKEND          mock / mujoco / pybullet / gymnasium / ros2
                             (overrides profile.backend_kind if set)

  GHOSTLOOP_TRANSPORT        stdio (default) / streamable-http / sse
  GHOSTLOOP_HOST             default 127.0.0.1
  GHOSTLOOP_PORT             default 8765

  GHOSTLOOP_INSTRUCTIONS     extra free-form instructions appended to the
                             profile's instructions, sent to the LLM as
                             system prompt. Use for per-deployment context
                             ("the green block is at (0.4, 0.0, 0.05)").

Custom robots: write a YAML profile (see examples/custom_robot.yaml) and
point GHOSTLOOP_PROFILE at the path. Add custom Primitives by creating a
Python module with factory functions and listing them under
``custom_primitives:`` in the YAML.

  python3 examples/mcp_robot.py --selfcheck
  GHOSTLOOP_PROFILE=spot python3 examples/mcp_robot.py --selfcheck
  GHOSTLOOP_PROFILE=examples/custom_robot.yaml python3 examples/mcp_robot.py --selfcheck
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import Intent
from ghostloop.profiles import (
    RobotProfile,
    build_runtime_from_profile,
    franka_arm,
    humanoid_demo,
    load_profile_yaml,
    spot_quadruped,
    stretch_mobile_arm,
    tello_drone,
    turtlebot_base,
)


PRESETS = {
    "franka_arm":     franka_arm,
    "turtlebot":      turtlebot_base,
    "spot":           spot_quadruped,
    "tello":          tello_drone,
    "stretch":        stretch_mobile_arm,
    "humanoid_demo":  humanoid_demo,
}


def resolve_profile() -> RobotProfile:
    """Find the profile from env: built-in preset name OR YAML path."""
    spec = os.environ.get("GHOSTLOOP_PROFILE", "franka_arm")
    if spec in PRESETS:
        return PRESETS[spec]()
    p = Path(spec)
    if p.exists():
        return load_profile_yaml(p)
    raise SystemExit(
        f"GHOSTLOOP_PROFILE={spec!r} is not a built-in preset and not a "
        f"file path that exists. Built-ins: {sorted(PRESETS.keys())}."
    )


def main() -> None:
    profile = resolve_profile()

    # Allow GHOSTLOOP_BACKEND to override the profile's backend.
    backend_override = os.environ.get("GHOSTLOOP_BACKEND")
    if backend_override:
        profile.backend_kind = backend_override

    runtime = build_runtime_from_profile(profile)

    if "--selfcheck" in sys.argv:
        print(f"[ghostloop] profile={profile.name} morphology={profile.morphology}")
        print(f"[ghostloop] backend={runtime.backend.name}")
        print(f"[ghostloop] primitives={runtime.registry.names()}")
        print(
            f"[ghostloop] gates="
            f"{[g.__class__.__name__ for g in runtime.policy_pipeline.gates]}"
        )
        # Pick a primitive that exists in this profile and is harmless.
        sample = "scan" if "scan" in runtime.registry.names() else (
            "stop" if "stop" in runtime.registry.names() else (
                "sense" if "sense" in runtime.registry.names() else None
            )
        )
        if sample:
            args = {"radius": 0.1} if sample == "scan" else {}
            result = runtime.step(Intent(sample, args))
            print(f"[ghostloop] selfcheck step={sample} status={result.status.value}")
        else:
            print(
                f"[ghostloop] selfcheck: no obviously-safe primitive found; "
                f"first call would be {runtime.registry.names()[0]!r}"
            )
        return

    # Run the MCP server. Same transport-by-env scheme as before.
    from ghostloop.mcp_server import run_mcp_server
    transport = os.environ.get("GHOSTLOOP_TRANSPORT", "stdio").lower()
    host = os.environ.get("GHOSTLOOP_HOST", "127.0.0.1")
    port = int(os.environ.get("GHOSTLOOP_PORT", "8765"))

    instructions = profile.instructions
    extra = os.environ.get("GHOSTLOOP_INSTRUCTIONS", "").strip()
    if extra:
        instructions = (instructions + "\n\n" + extra).strip()

    print(
        f"[ghostloop] starting MCP server "
        f"(profile={profile.name} backend={runtime.backend.name} "
        f"transport={transport}"
        f"{f' @ {host}:{port}' if transport != 'stdio' else ''})...",
        file=sys.stderr,
    )
    run_mcp_server(
        runtime,
        server_name=f"ghostloop-{profile.name}",
        transport=transport, host=host, port=port,
        instructions=instructions or None,
    )


if __name__ == "__main__":
    main()
