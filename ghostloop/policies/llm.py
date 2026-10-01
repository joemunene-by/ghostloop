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
    price_per_mtok: tuple[float, float] | None = None  # (input, output) USD, for cost accounting

    @classmethod
    def for_provider(cls, provider: str, model: str, api_key: str | None = None, **kwargs: Any) -> "LLMPolicyConfig":
        """Preset base URLs for common OpenAI-compatible endpoints."""
        urls = {
            "ollama": "http://localhost:11434/v1",
            "vllm": "http://localhost:8000/v1",
            "lmstudio": "http://localhost:1234/v1",
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": "https://api.groq.com/openai/v1",
        }
        if provider not in urls:
            raise ValueError(f"unknown provider {provider!r}; known: {sorted(urls)}")
        return cls(base_url=urls[provider], model=model, api_key=api_key, **kwargs)


class LLMPolicyError(RuntimeError):
    """Raised when the upstream endpoint returns an error or an unparseable response."""


DONE_TOOL_DESCRIPTION = "Signal that the goal is achieved or unreachable. Ends the episode."


def _json_type(spec: str) -> dict[str, Any]:
    spec = spec.lower()
    if spec.startswith(("list", "tuple")) or "[" in spec:
        return {"type": "array", "items": {"type": "number"} if "float" in spec else {}}
    if "float" in spec or "number" in spec:
        return {"type": "number"}
    if spec.startswith("int"):
        return {"type": "integer"}
    if spec.startswith("bool"):
        return {"type": "boolean"}
    return {"type": "string"}


def _is_optional(spec: str) -> bool:
    spec = spec.lower()
    return "optional" in spec or "default" in spec


def primitive_input_schema(arg_schema: dict[str, str]) -> dict[str, Any]:
    """JSON Schema for a Primitive's ``arg_schema`` (types inferred from the spec strings)."""
    return {
        "type": "object",
        "properties": {
            name: {**_json_type(spec), "description": spec} for name, spec in arg_schema.items()
        },
        "required": [name for name, spec in arg_schema.items() if not _is_optional(spec)],
    }


def _build_tool_schema(registry: PrimitiveRegistry) -> list[dict[str, Any]]:
    """Convert the PrimitiveRegistry into the OpenAI tools array, plus a ``done`` tool."""
    tools: list[dict[str, Any]] = []
    for name in registry.names():
        prim = registry.get(name)
        assert prim is not None
        tools.append({
            "type": "function",
            "function": {
                "name": prim.name,
                "description": prim.description,
                "parameters": primitive_input_schema(prim.arg_schema),
            },
        })
    # Special "done" pseudo-tool — never registered as a Primitive, but the
    # LLM needs a way to gracefully terminate the episode.
    tools.append({
        "type": "function",
        "function": {
            "name": "done",
            "description": DONE_TOOL_DESCRIPTION,
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
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0})
    stop_reason: str = ""
    _tools: list[dict[str, Any]] = field(default_factory=list, init=False)
    _pending_calls: list[str] = field(default_factory=list, init=False)

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
            if self._pending_calls:
                first, *extra = self._pending_calls
                self.messages.append({"role": "tool", "content": content, "tool_call_id": first})
                for call_id in extra:
                    self.messages.append({
                        "role": "tool", "tool_call_id": call_id,
                        "content": "not executed: ghostloop runs one primitive per step; call it again if still needed",
                    })
            else:
                self.messages.append({"role": "user", "content": f"Observation: {content}"})
            self._pending_calls = []

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
        self.stop_reason = choices[0].get("finish_reason") or ""
        usage = data.get("usage") or {}
        self.usage["input_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.usage["output_tokens"] += int(usage.get("completion_tokens") or 0)
        self.usage["requests"] += 1

        tool_calls = message.get("tool_calls") or []
        self._pending_calls = [c.get("id") or f"call_{i}" for i, c in enumerate(tool_calls)]
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


def estimate_cost(usage: dict[str, int], price_per_mtok: tuple[float, float] | None) -> float | None:
    if price_per_mtok is None:
        return None
    return (usage.get("input_tokens", 0) * price_per_mtok[0]
            + usage.get("output_tokens", 0) * price_per_mtok[1]) / 1_000_000


def llm_policy_loop(
    registry: PrimitiveRegistry,
    runtime,
    goal: str,
    config: LLMPolicyConfig | None = None,
    max_steps: int = 16,
    policy: Any = None,
) -> dict[str, Any]:
    """End-to-end driver: an LLM policy in front of a Runtime, until done or step cap.

    ``policy`` may be any object with ``set_goal``/``ask``/``usage`` (e.g.
    :class:`~ghostloop.policies.claude.ClaudePolicy`); by default an
    OpenAI-compatible :class:`LLMPolicy` is built from ``config``.

    Returns a summary dict (steps, terminated reason, last observation,
    message log, token usage and estimated cost). The runtime's trace
    remains the canonical event log.
    """
    if policy is None:
        policy = LLMPolicy(registry=registry, config=config or LLMPolicyConfig())
    policy.set_goal(goal)

    last_observation: dict[str, Any] | None = None
    terminated = "max_steps"
    steps = 0
    for _ in range(max_steps):
        intent = policy.ask(last_observation)
        if intent is None:
            terminated = "refusal" if getattr(policy, "stop_reason", "") == "refusal" else "done"
            break
        result = runtime.step(intent)
        steps += 1
        last_observation = {
            "status": result.status.value,
            "message": result.message,
            "observation": result.observation,
            "state": runtime.backend.snapshot(),
        }

    usage = dict(getattr(policy, "usage", {}) or {})
    cost = getattr(policy, "estimated_cost", None)
    if cost is None:
        cost = estimate_cost(usage, getattr(getattr(policy, "config", None), "price_per_mtok", None))
    elif callable(cost):
        cost = cost()
    return {
        "steps": steps,
        "terminated": terminated,
        "last_observation": last_observation,
        "messages": policy.messages,
        "usage": usage,
        "cost_usd": cost,
    }


def llm_episode_policy(policy_factory: Any, goal: str | None = None, max_steps: int = 16):
    """Adapt an LLM policy for :class:`~ghostloop.bench.Episode` so benches can score models.

    ``policy_factory(registry)`` must return a fresh policy per episode. The
    returned callable runs the loop and hands usage/cost back to the bench.
    """

    def run(runtime) -> dict[str, Any]:
        policy = policy_factory(runtime.registry)
        summary = llm_policy_loop(runtime.registry, runtime, goal or "", max_steps=max_steps, policy=policy)
        summary.pop("messages", None)
        return summary

    return run
