"""Direct LLM tool-use without MCP. Works with any OpenAI-compatible endpoint.

For chat clients that DO speak MCP (Claude Desktop, Cursor, Continue, Cline,
Zed, Gemini CLI), use ``examples/claude_desktop_mcp_arm.py``. For when you
just want to drive ghostloop from a script with whatever model you can hit
over HTTP — OpenAI, Anthropic via OpenAI-compatible proxy, Google Gemini,
Groq, Ollama, vLLM, llama.cpp server, GhostLM's multi-vendor server — use
THIS file.

The bridge is ``LLMPolicy`` from v0.2: any chat endpoint that accepts an
OpenAI ``tools=[{...}]`` array becomes a ghostloop policy. The runtime
exposes every Primitive as a tool; the model picks one and supplies args;
the safety pipeline gates the resulting Intent; the trace records it.

  ::

      OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... \\
          python3 examples/direct_llm_arm.py

      # Or any OpenAI-compatible endpoint:
      OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_MODEL=qwen2.5:14b \\
          python3 examples/direct_llm_arm.py

      OPENAI_BASE_URL=https://api.groq.com/openai/v1 OPENAI_API_KEY=gsk_... \\
      OPENAI_MODEL=llama-3.3-70b-versatile \\
          python3 examples/direct_llm_arm.py

      # Anthropic via the OpenAI-compatible endpoint shim:
      OPENAI_BASE_URL=https://api.anthropic.com/v1 \\
      OPENAI_API_KEY=sk-ant-... \\
      OPENAI_MODEL=claude-sonnet-4-6 \\
          python3 examples/direct_llm_arm.py

Same safety pipeline as the MCP example — Geofence + ForceCap +
ActionSmoothing + RateLimit + (optional) HumanInTheLoop. Same Backend
choice (Mock / MuJoCo / ROS 2). Only difference: no MCP wire protocol;
the LLM-to-tool plumbing happens in-process via LLMPolicy.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import MockBackend, PolicyPipeline, PrimitiveRegistry, Runtime
from ghostloop.policies import (
    ActionSmoothingGate,
    ForceCapGate,
    GeofenceGate,
    LLMPolicyConfig,
    RateLimitGate,
    llm_policy_loop,
)
from ghostloop.primitives import move_to, pick, place, scan


# ---------------------------------------------------------------------------
# Config — same shape as the MCP example.
# ---------------------------------------------------------------------------

WORKSPACE_MIN = (-0.6, -0.6, 0.0)
WORKSPACE_MAX = (0.6, 0.6, 1.0)
MAX_FORCE_N = 15.0
MAX_VELOCITY_MS = 0.5
MAX_ACCELERATION_MS2 = 2.0


def main() -> None:
    runtime = Runtime(
        backend=MockBackend(name="direct_llm_arm"),
        registry=PrimitiveRegistry([move_to(), scan(), pick(), place()]),
        policy_pipeline=PolicyPipeline(gates=[
            RateLimitGate(per_minute=60),
            GeofenceGate(min_corner=WORKSPACE_MIN, max_corner=WORKSPACE_MAX),
            ForceCapGate(force_max=MAX_FORCE_N),
            ActionSmoothingGate(
                max_velocity=MAX_VELOCITY_MS,
                max_acceleration=MAX_ACCELERATION_MS2,
            ),
        ]),
    )

    config = LLMPolicyConfig(
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        model=os.environ.get("OPENAI_MODEL", "qwen2.5:14b"),
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.0")),
    )

    goal = os.environ.get(
        "GHOSTLOOP_GOAL",
        "Pick widget-7 from (0.4, 0.2, 0.1) and place it at (-0.4, 0.2, 0.1).",
    )

    summary = llm_policy_loop(
        registry=runtime.registry,
        runtime=runtime,
        goal=goal,
        config=config,
        max_steps=int(os.environ.get("GHOSTLOOP_MAX_STEPS", "16")),
    )
    print(f"steps={summary.get('steps')} terminated={summary.get('terminated')}")
    out_path = Path(os.environ.get("GHOSTLOOP_TRACE_OUT", "trace.jsonl"))
    runtime.trace.write_jsonl(str(out_path))
    print(f"trace -> {out_path}")


if __name__ == "__main__":
    main()
