"""ClaudePolicy: Anthropic's Messages API as a ghostloop policy.

Same ``set_goal`` / ``ask`` surface as :class:`~ghostloop.policies.llm.LLMPolicy`,
so it drops into ``llm_policy_loop`` and the bench. Uses the official SDK
(``pip install ghostloop[anthropic]``); the core package stays dependency-free.

The model gets one primitive per step: ghostloop has to run every call
through the safety pipeline before the next one is chosen, so parallel tool
use is disabled and extra calls are answered with an error result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core import Intent, PrimitiveRegistry
from .llm import (
    DONE_TOOL_DESCRIPTION,
    LLMPolicyError,
    _DEFAULT_SYSTEM,
    _coerce_args,
    primitive_input_schema,
)

# USD per million tokens: (input, output, cache read). Cache writes bill at 1.25x input.
CLAUDE_PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-4-5": (1.0, 5.0, 0.10),
}

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ClaudePolicyConfig:
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000
    system_prompt: str = _DEFAULT_SYSTEM
    fallbacks: bool = True  # server-side refusal fallback; Claude API / Claude Platform on AWS only
    prompt_caching: bool = True
    api_key: str | None = None
    timeout: float = 120.0


def _anthropic():
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ClaudePolicy requires the anthropic SDK:\n  pip install ghostloop[anthropic]"
        ) from e
    return anthropic


def build_claude_tools(registry: PrimitiveRegistry) -> list[dict[str, Any]]:
    tools = []
    for name in registry.names():
        prim = registry.get(name)
        assert prim is not None
        tools.append({
            "name": prim.name,
            "description": prim.description or prim.name,
            "input_schema": primitive_input_schema(prim.arg_schema),
        })
    tools.append({
        "name": "done",
        "description": DONE_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why you are stopping."}},
            "required": ["reason"],
        },
    })
    return tools


@dataclass
class ClaudePolicy:
    """Closes the agent loop over the Claude Messages API.

    ``client`` may be injected (tests, Claude Platform on AWS, custom
    base URLs); otherwise ``anthropic.Anthropic()`` resolves credentials
    from the environment.
    """

    registry: PrimitiveRegistry
    config: ClaudePolicyConfig = field(default_factory=ClaudePolicyConfig)
    client: Any = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0, "requests": 0,
    })
    stop_reason: str = ""
    cost_usd: float = 0.0
    unpriced_models: set[str] = field(default_factory=set)
    _tools: list[dict[str, Any]] = field(default_factory=list, init=False)
    _pending: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._tools = build_claude_tools(self.registry)
        if self.client is None:
            anthropic = _anthropic()
            kwargs: dict[str, Any] = {"timeout": self.config.timeout}
            if self.config.api_key:
                kwargs["api_key"] = self.config.api_key
            self.client = anthropic.Anthropic(**kwargs)

    def set_goal(self, goal: str) -> None:
        self.messages.append({"role": "user", "content": f"Goal: {goal}"})

    def estimated_cost(self) -> float | None:
        """Total spend so far, or None if any response came from a model without a price entry."""
        return None if self.unpriced_models else self.cost_usd

    def _observation_turn(self, observation: str | dict[str, Any]) -> None:
        content = observation if isinstance(observation, str) else json.dumps(observation)
        if not self._pending:
            self.messages.append({"role": "user", "content": f"Observation: {content}"})
            return
        first, *extra = self._pending
        results = [{"type": "tool_result", "tool_use_id": first, "content": content}]
        results += [
            {"type": "tool_result", "tool_use_id": tid, "is_error": True,
             "content": "not executed: ghostloop runs one primitive per step"}
            for tid in extra
        ]
        self.messages.append({"role": "user", "content": results})
        self._pending = []

    def _request(self) -> Any:
        cfg = self.config
        params: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": cfg.system_prompt,
            "tools": self._tools,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "messages": self.messages,
            "output_config": {"effort": cfg.effort},
        }
        if cfg.prompt_caching:
            params["cache_control"] = {"type": "ephemeral"}
        anthropic = _anthropic()
        try:
            if cfg.fallbacks:
                return self.client.beta.messages.create(
                    betas=[_FALLBACK_BETA], fallbacks="default", **params,
                )
            return self.client.messages.create(**params)
        except anthropic.RateLimitError as e:
            raise LLMPolicyError(f"rate limited after SDK retries: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMPolicyError(f"upstream {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMPolicyError(f"upstream unreachable: {e}") from e

    def _account(self, response: Any) -> None:
        u = response.usage
        counts = {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
        for k, v in counts.items():
            self.usage[k] += int(v)
        self.usage["requests"] += 1
        model = getattr(response, "model", None) or self.config.model
        price = CLAUDE_PRICES.get(model)
        if price is None:
            self.unpriced_models.add(model)
            return
        inp, out, cache_read = price
        self.cost_usd += (
            counts["input_tokens"] * inp
            + counts["output_tokens"] * out
            + counts["cache_read_input_tokens"] * cache_read
            + counts["cache_creation_input_tokens"] * inp * 1.25
        ) / 1_000_000

    def ask(self, observation: str | dict[str, Any] | None = None) -> Intent | None:
        if observation is not None:
            self._observation_turn(observation)

        response = self._request()
        self._account(response)
        self.stop_reason = response.stop_reason or ""
        # Append the full content: thinking blocks must be replayed unchanged.
        self.messages.append({"role": "assistant", "content": response.content})

        if self.stop_reason == "refusal":
            return None
        if self.stop_reason == "max_tokens":
            raise LLMPolicyError(f"response hit max_tokens={self.config.max_tokens} before a tool call")

        tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
        self._pending = [b.id for b in tool_uses]
        if not tool_uses:
            return None
        call = tool_uses[0]
        if call.name == "done":
            return None
        rationale = " ".join(
            b.text.strip() for b in response.content
            if getattr(b, "type", None) == "text" and b.text.strip()
        )
        raw = call.input if isinstance(call.input, dict) else {}
        prim = self.registry.get(call.name)
        if prim is None:
            return Intent(name=call.name, args=raw, rationale=f"hallucinated tool: {call.name}")
        return Intent(name=call.name, args=_coerce_args(prim.arg_schema, raw), rationale=rationale)
