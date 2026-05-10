"""LLMPolicy: any OpenAI-compatible chat endpoint emits Intents through the runtime.

The LLM sees the Primitive registry as a tool schema (the same shape OpenAI /
Anthropic / Gemini / Ollama / vLLM all consume), receives observations from
the previous step, and emits the next Intent as a tool call. ghostloop owns
the loop: dispatch -> safety pipeline -> backend -> trace -> back to the LLM.

This is the pattern from GhostLM's GhostAgent (tool-using runtime over a
GhostLM checkpoint), reshaped so the tools are robot primitives instead of
CVE / MITRE / CWE lookups. Same OpenAI-compatible wire format means a single
adapter works across every model vendor.

No SDK dependency: uses urllib so v0.1's "zero runtime deps" promise holds.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core import Intent, PrimitiveRegistry


_DEFAULT_SYSTEM = (
    "You are a robot policy. On each turn you choose ONE primitive from the "
    "tool registry and call it with the right arguments. After every call "
    "you receive the result and the next state. Stop by calling the special "
    "tool 'done' when the goal is satisfied or unreachable. Be concise; the "
    "rationale field of each tool call is logged for audit."
)


@dataclass
class LLMPolicyConfig:
    """Configuration for an LLMPolicy adapter."""

    base_url: str = "http://localhost:11434/v1"  # Ollama default; override per vendor
    model: str = "qwen2.5:14b"
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 512
    request_timeout: float = 60.0
    system_prompt: str = _DEFAULT_SYSTEM


class LLMPolicyError(RuntimeError):
    """Raised when the upstream endpoint returns an error or an unparseable response."""


def _build_tool_schema(registry: PrimitiveRegistry) -> list[dict[str, Any]]:
    """Convert the PrimitiveRegistry into the OpenAI tools array.

    Each Primitive becomes one tool with name + description + a hand-built
    JSON schema derived from the Primitive's ``arg_schema``. We emit STRINGS
    not types since arg_schema is documentation, not strict typing — the
    LLM tends to handle this well and we re-coerce at the runtime boundary.
    """
    tools: list[dict[str, Any]] = []
    for name in registry.names():
        prim = registry.get(name)
        assert prim is not None
        properties: dict[str, dict[str, str]] = {}
        for arg_name, arg_desc in prim.arg_schema.items():
            properties[arg_name] = {"type": "string", "description": arg_desc}
        tools.append({
            "type": "function",
            "function": {
                "name": prim.name,
                "description": prim.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(prim.arg_schema.keys()),
                },
            },
        })
    # Special "done" pseudo-tool — never registered as a Primitive, but the
    # LLM needs a way to gracefully terminate the episode.
    tools.append({
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the goal is achieved or unreachable. Ends the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why you are stopping."},
                },
                "required": ["reason"],
            },
        },
    })
    return tools


def _coerce_args(prim_arg_schema: dict[str, str], raw: dict[str, Any]) -> dict[str, Any]:
    """Best-effort coerce model-emitted args to the expected types from arg_schema.

    arg_schema entries look like ``"float"`` or ``"str"`` or ``"float (optional)"``.
    LLMs sometimes return numbers as strings (especially with weaker models);
    this catches the common cases without being strict.
    """
    out: dict[str, Any] = {}
    for k, v in raw.items():
        spec = prim_arg_schema.get(k, "str").lower()
        if v is None:
            out[k] = None
            continue
        if "float" in spec or "number" in spec:
            try:
                out[k] = float(v)
                continue
            except (TypeError, ValueError):
                pass
        if "int" in spec:
            try:
                out[k] = int(v)
                continue
            except (TypeError, ValueError):
                pass
        if "bool" in spec:
            if isinstance(v, bool):
                out[k] = v
                continue
            if isinstance(v, str):
                out[k] = v.strip().lower() in ("true", "1", "yes", "y", "on")
                continue
        out[k] = v
    return out


@dataclass
class LLMPolicy:
    """Closes the agent loop: query LLM, parse tool call, return Intent.

    Maintains its own message history (system prompt + alternating user
    observations and assistant tool calls). ``ask(observation)`` posts the
    observation back to the LLM and returns the next Intent (or None if
    the model called the special ``done`` tool).
    """

    registry: PrimitiveRegistry
    config: LLMPolicyConfig = field(default_factory=LLMPolicyConfig)
    messages: list[dict[str, Any]] = field(default_factory=list)
    _tools: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._tools = _build_tool_schema(self.registry)
        if not self.messages:
            self.messages = [
                {"role": "system", "content": self.config.system_prompt},
            ]

    def set_goal(self, goal: str) -> None:
        """Seed the conversation with the user's goal description."""
        self.messages.append({"role": "user", "content": f"Goal: {goal}"})

    def ask(self, observation: str | dict[str, Any] | None = None) -> Intent | None:
        """One round-trip: send observation, return next Intent (or None if done)."""
        if observation is not None:
            content = observation if isinstance(observation, str) else json.dumps(observation)
            self.messages.append({"role": "tool", "content": content, "tool_call_id": "prev"})

        payload = {
            "model": self.config.model,
            "messages": self.messages,
            "tools": self._tools,
            "tool_choice": "auto",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url.rstrip("/") + "/chat/completions"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise LLMPolicyError(
                f"upstream {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise LLMPolicyError(f"upstream unreachable: {e.reason}") from e

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMPolicyError(f"unparseable upstream JSON: {e}") from e

        choices = data.get("choices") or []
        if not choices:
            raise LLMPolicyError(f"no choices in upstream response: {data}")
        message = choices[0].get("message") or {}
        self.messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            # Model emitted a plain text response with no tool call. Treat as done.
            return None

        call = tool_calls[0]
        fn = call.get("function") or {}
        name = fn.get("name")
        if not name:
            raise LLMPolicyError(f"tool_call missing function name: {call}")
        if name == "done":
            return None
        prim = self.registry.get(name)
        if prim is None:
            # Model hallucinated a tool; surface as an Intent so the runtime
            # can BLOCK it via the resolver and the trace records the attempt.
            return Intent(name=name, args={}, rationale=f"hallucinated tool: {name}")

        try:
            raw_args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            raw_args = {}
        args = _coerce_args(prim.arg_schema, raw_args)
        rationale = (message.get("content") or "").strip()
        return Intent(name=name, args=args, rationale=rationale)


def llm_policy_loop(
    registry: PrimitiveRegistry,
    runtime,
    goal: str,
    config: LLMPolicyConfig | None = None,
    max_steps: int = 16,
) -> dict[str, Any]:
    """End-to-end driver: LLMPolicy in front of a Runtime, until done or step cap.

    Returns a summary dict (steps taken, terminated reason, last observation,
    full message log). The runtime's trace remains the canonical event log.
    """
    policy = LLMPolicy(registry=registry, config=config or LLMPolicyConfig())
    policy.set_goal(goal)

    last_observation: dict[str, Any] | None = None
    terminated = "max_steps"
    steps = 0
    for _ in range(max_steps):
        intent = policy.ask(last_observation)
        if intent is None:
            terminated = "done"
            break
        result = runtime.step(intent)
        steps += 1
        last_observation = {
            "status": result.status.value,
            "message": result.message,
            "observation": result.observation,
            "state": runtime.backend.snapshot(),
        }

    return {
        "steps": steps,
        "terminated": terminated,
        "last_observation": last_observation,
        "messages": policy.messages,
    }
