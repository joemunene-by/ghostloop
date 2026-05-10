"""LLMPlanner: any OpenAI-compatible chat endpoint emits a structured plan.

Sister to LLMPolicy. Where LLMPolicy emits ONE intent per turn (closed
loop, reactive), LLMPlanner asks the model for the FULL plan up front.
The runtime then executes that plan under the safety pipeline. If the
pipeline blocks an intent mid-plan, the runtime continues with the
remaining steps and surfaces the blocked count.

Use cases:
  - Hand-curated few-shot prompts that ALWAYS produce the same plan
    shape (great for regression tests).
  - Hierarchical control: LLMPlanner does the top-level decomposition,
    a separate VLAPolicy / scripted controller handles each step.
  - Cost optimisation: one LLM call per episode beats N calls.

Same OpenAI-tool-call wire format as LLMPolicy. The prompt asks the
model for a single ``submit_plan(steps=[...])`` tool call rather than
streaming intents.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..core import Intent, PrimitiveRegistry
from ..policies.llm import LLMPolicyConfig, LLMPolicyError
from .core import PlanResult


_PLANNER_SYSTEM = (
    "You are a robot task planner. Given a goal, emit ONE tool call to "
    "submit_plan(steps=[...]) where each step is "
    "{name: <primitive>, args: {...}, rationale: <why>}. The runtime will "
    "execute every step under a fail-closed safety policy pipeline; "
    "blocked steps surface in the trace. Use only the primitives in the "
    "tool registry. Be concise — one well-formed plan beats five sloppy "
    "ones."
)


def _build_planner_tool(registry: PrimitiveRegistry) -> dict[str, Any]:
    """submit_plan tool with a JSON schema describing the steps array."""
    primitive_names = registry.names()
    return {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": (
                "Submit a complete robot plan as a sequence of primitive "
                "calls. Each step uses one of the registered primitive "
                f"names: {', '.join(primitive_names)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "Ordered list of primitive calls.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Primitive name.",
                                    "enum": primitive_names,
                                },
                                "args": {
                                    "type": "object",
                                    "description": "Keyword args for the primitive.",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "Why this step.",
                                },
                            },
                            "required": ["name"],
                        },
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Top-level reasoning for the whole plan.",
                    },
                },
                "required": ["steps"],
            },
        },
    }


@dataclass
class LLMPlanner:
    """LLM-backed Planner emitting a full PlanResult per call.

    Args:
        registry: PrimitiveRegistry the LLM will pick from.
        config: same config shape as LLMPolicy (base_url, model, etc).
        system_prompt: override the default planner system prompt.
    """

    registry: PrimitiveRegistry
    config: LLMPolicyConfig = field(default_factory=LLMPolicyConfig)
    system_prompt: str = _PLANNER_SYSTEM
    name: str = "llm_planner"

    def plan(self, goal: str | dict) -> PlanResult:
        goal_text = goal if isinstance(goal, str) else json.dumps(goal)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Goal: {goal_text}"},
        ]
        tool = _build_planner_tool(self.registry)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": "submit_plan"}},
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
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            raise LLMPolicyError("model did not emit submit_plan tool call")
        call = tool_calls[0]
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            raise LLMPolicyError(f"unparseable plan arguments: {e}") from e

        steps = args.get("steps") or []
        intents: list[Intent] = []
        for s in steps:
            name = s.get("name")
            if not name:
                continue
            intents.append(Intent(
                name=str(name),
                args=s.get("args") or {},
                rationale=str(s.get("rationale") or ""),
            ))
        rationale = str(args.get("rationale") or message.get("content") or "")
        return PlanResult(
            name=self.name,
            intents=intents,
            rationale=rationale,
            metadata={"goal": goal_text, "model": self.config.model},
        )
