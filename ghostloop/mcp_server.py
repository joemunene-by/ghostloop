"""MCP server: every Primitive becomes a tool callable from any MCP client.

Claude Desktop, Cursor, any MCP-aware agent can drive a ghostloop runtime
through the safety pipeline. Same FastMCP pattern GhostLM uses.

Conditional import — package itself imports cleanly without ``mcp``;
``run_mcp_server(...)`` raises ImportError with install hint at call time.

  pip install ghostloop[mcp]

Three tools exposed by default:
  list_primitives()                — MCP discovery hook for the agent
  step(name, args, rationale)     — one runtime.step under safety pipeline
  recent_trace(n=10)               — last n events from the active trace

Plus auto-generated tools for every Primitive in the registry, so the
agent can call ``move_to(x, y, z)`` directly without the indirection.
"""

from __future__ import annotations

import json
from typing import Any

from .core import Backend, Intent, PolicyPipeline, PrimitiveRegistry, Runtime


_MCP_INSTALL_HINT = (
    "MCP server requires the mcp package.\n"
    "  pip install mcp\n"
    "or  pip install ghostloop[mcp]\n"
    "Docs: https://github.com/modelcontextprotocol/python-sdk"
)


def mcp_available() -> bool:
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
        return True
    except ImportError:
        return False


def build_mcp_server(
    runtime: Runtime,
    *,
    server_name: str = "ghostloop",
    instructions: str | None = None,
):
    """Construct a FastMCP server bound to a Runtime instance.

    Returns the FastMCP object so the caller can either ``.run()`` it
    (stdio transport, default for Claude Desktop / Cursor) or mount it
    in a custom HTTP / SSE transport. Tools are registered eagerly at
    construction; the registry is consulted at server start, not on
    every call, so adding primitives later requires rebuilding the
    server.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as e:
        raise ImportError(_MCP_INSTALL_HINT) from e

    default_instructions = (
        "ghostloop is the agent loop, embodied. Use list_primitives() to "
        "discover available robot actions, then call them through step() "
        "OR via the per-primitive auto-tools. Every action passes through "
        "the configured safety policy pipeline (geofence / rate-limit / "
        "force-cap / human-in-the-loop). Blocked actions return a "
        "structured BLOCKED result with the gate name and reason — read "
        "those carefully and adjust your plan accordingly."
    )
    server = FastMCP(server_name, instructions=instructions or default_instructions)

    # ------------------------------------------------------------------
    # Discovery + general-purpose step + trace inspection.
    # ------------------------------------------------------------------

    @server.tool()
    def list_primitives() -> dict[str, Any]:
        """List every primitive currently in the runtime registry."""
        out: dict[str, dict[str, Any]] = {}
        for name in runtime.registry.names():
            prim = runtime.registry.get(name)
            assert prim is not None
            out[name] = {
                "description": prim.description,
                "args": prim.arg_schema,
            }
        return {"primitives": out, "backend": runtime.backend.name}

    @server.tool()
    def step(name: str, args: dict[str, Any] | None = None, rationale: str = "") -> dict[str, Any]:
        """Run one step: dispatch the named primitive through the safety pipeline."""
        intent = Intent(name=name, args=args or {}, rationale=rationale)
        result = runtime.step(intent)
        return {
            "status": result.status.value,
            "message": result.message,
            "observation": result.observation,
            "duration_ms": round(result.duration_ms, 3),
            "state": runtime.backend.snapshot(),
        }

    @server.tool()
    def recent_trace(n: int = 10) -> dict[str, Any]:
        """Return the last ``n`` events from the active trace."""
        events = [ev.to_json() for ev in runtime.trace.events[-n:]]
        return {
            "episode_id": runtime.trace.episode_id,
            "n_events_total": len(runtime.trace.events),
            "events": events,
        }

    @server.tool()
    def state() -> dict[str, Any]:
        """Current backend snapshot — pose, joints, held object, etc."""
        return runtime.backend.snapshot()

    # ------------------------------------------------------------------
    # One auto-tool per Primitive so the LLM can call directly.
    # ------------------------------------------------------------------

    for prim_name in runtime.registry.names():
        _register_primitive_tool(server, runtime, prim_name)

    return server


def _register_primitive_tool(server, runtime: Runtime, primitive_name: str) -> None:
    """Bind one Primitive as a top-level FastMCP tool.

    Closes over ``primitive_name`` so the runtime resolves at call time.
    Each tool emits one runtime.step under the safety pipeline.
    """
    prim = runtime.registry.get(primitive_name)
    if prim is None:
        return
    arg_schema = prim.arg_schema
    description = prim.description or f"Run the {primitive_name} primitive."

    def _tool(**kwargs: Any) -> dict[str, Any]:
        intent = Intent(name=primitive_name, args=kwargs)
        result = runtime.step(intent)
        return {
            "status": result.status.value,
            "message": result.message,
            "observation": result.observation,
            "state": runtime.backend.snapshot(),
        }

    _tool.__name__ = primitive_name
    _tool.__doc__ = (
        f"{description}\n\nArgs: " + ", ".join(f"{k}: {v}" for k, v in arg_schema.items())
    )
    server.tool()(_tool)


def run_mcp_server(
    runtime: Runtime,
    *,
    server_name: str = "ghostloop",
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    mount_path: str | None = None,
    instructions: str | None = None,
) -> None:
    """Construct + run the MCP server. Blocks; exits when the client disconnects.

    Three transports supported by FastMCP:

      - ``stdio`` (default) — desktop clients spawn the server as a
        subprocess and speak MCP through stdin/stdout. Right for
        Claude Desktop, Cursor, Continue, Cline, Zed, Gemini CLI on
        the same machine.
      - ``sse`` — server-sent events over HTTP. The server binds to
        ``host:port`` and remote clients connect via URL. Right for
        mobile + cross-machine setups + browser-based MCP clients.
      - ``streamable-http`` — the newer HTTP transport (replaces SSE
        going forward). Same host/port shape; preferred when both
        ends speak it.

    Pick stdio for desktop, streamable-http for mobile / remote. The
    server, the runtime, and every safety gate are unchanged across
    transports — only the wire protocol differs.
    """
    server = build_mcp_server(
        runtime, server_name=server_name, instructions=instructions,
    )
    if transport in ("sse", "streamable-http"):
        # Export host / port via FastMCP's settings before running so
        # the bound port matches the user's intent. FastMCP defaults to
        # 127.0.0.1:8000; we expose explicit host/port so users can
        # bind to 0.0.0.0 for remote-access setups.
        try:
            server.settings.host = host
            server.settings.port = port
        except AttributeError:
            # Older FastMCP — fall back to env vars before .run().
            import os as _os
            _os.environ.setdefault("FASTMCP_HOST", host)
            _os.environ.setdefault("FASTMCP_PORT", str(port))
    if mount_path is not None:
        server.run(transport=transport, mount_path=mount_path)
    else:
        server.run(transport=transport)


def _cli_main() -> None:
    """``python -m ghostloop.mcp_server`` entry point.

    Picks profile / backend / transport from env vars (same scheme as
    examples/mcp_robot.py). Exists so package installers — and Smithery
    in particular — can launch the server with one command line:

        python -m ghostloop.mcp_server

    Without writing the example script.
    """
    import os
    import sys
    from .profiles import (
        build_runtime_from_profile,
        franka_arm,
        humanoid_demo,
        load_profile_yaml,
        spot_quadruped,
        stretch_mobile_arm,
        tello_drone,
        turtlebot_base,
    )
    from pathlib import Path
    presets = {
        "franka_arm":     franka_arm,
        "turtlebot":      turtlebot_base,
        "spot":           spot_quadruped,
        "tello":          tello_drone,
        "stretch":        stretch_mobile_arm,
        "humanoid_demo":  humanoid_demo,
    }
    spec = os.environ.get("GHOSTLOOP_PROFILE", "franka_arm")
    if spec in presets:
        profile = presets[spec]()
    elif Path(spec).exists():
        profile = load_profile_yaml(spec)
    else:
        sys.stderr.write(
            f"GHOSTLOOP_PROFILE={spec!r} is not a built-in preset and not "
            f"a YAML path. Built-ins: {sorted(presets)}\n"
        )
        sys.exit(2)
    if backend := os.environ.get("GHOSTLOOP_BACKEND"):
        profile.backend_kind = backend
    runtime = build_runtime_from_profile(profile)
    transport = os.environ.get("GHOSTLOOP_TRANSPORT", "stdio").lower()
    host = os.environ.get("GHOSTLOOP_HOST", "127.0.0.1")
    port = int(os.environ.get("GHOSTLOOP_PORT", "8765"))
    instructions = profile.instructions
    extra = os.environ.get("GHOSTLOOP_INSTRUCTIONS", "").strip()
    if extra:
        instructions = (instructions + "\n\n" + extra).strip()
    sys.stderr.write(
        f"[ghostloop] starting MCP server "
        f"(profile={profile.name} backend={runtime.backend.name} "
        f"transport={transport})\n"
    )
    run_mcp_server(
        runtime, server_name=f"ghostloop-{profile.name}",
        transport=transport, host=host, port=port,
        instructions=instructions or None,
    )


if __name__ == "__main__":
    _cli_main()
