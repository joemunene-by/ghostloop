"""Tests for ghostloop.policies.llm — schema build, arg coercion, loop driver."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import (
    Intent,
    MockBackend,
    PolicyPipeline,
    PrimitiveRegistry,
    Runtime,
)
from ghostloop.policies.llm import (
    LLMPolicy,
    LLMPolicyConfig,
    LLMPolicyError,
    _build_tool_schema,
    _coerce_args,
    llm_policy_loop,
)
from ghostloop.primitives import move_to, pick, place, scan


def _registry():
    return PrimitiveRegistry([move_to(), scan(), pick(), place()])


class TestSchemaBuild:
    def test_includes_every_primitive_plus_done(self):
        tools = _build_tool_schema(_registry())
        names = [t["function"]["name"] for t in tools]
        assert "move_to" in names
        assert "scan" in names
        assert "pick" in names
        assert "place" in names
        assert "done" in names  # special termination tool
        assert len(tools) == 5

    def test_move_to_required_args(self):
        tools = _build_tool_schema(_registry())
        move = next(t for t in tools if t["function"]["name"] == "move_to")
        params = move["function"]["parameters"]
        assert set(params["required"]) == {"x", "y", "z"}
        for k in ("x", "y", "z"):
            assert k in params["properties"]

    def test_done_tool_present_with_reason(self):
        tools = _build_tool_schema(_registry())
        done = next(t for t in tools if t["function"]["name"] == "done")
        assert "reason" in done["function"]["parameters"]["required"]


class TestArgCoercion:
    def test_string_floats_become_floats(self):
        out = _coerce_args(
            {"x": "float", "y": "float", "z": "float"},
            {"x": "0.5", "y": "1", "z": -2.0},
        )
        assert out == {"x": 0.5, "y": 1.0, "z": -2.0}

    def test_unknown_keys_pass_through(self):
        out = _coerce_args({}, {"weird_key": "value"})
        assert out == {"weird_key": "value"}

    def test_optional_marker_still_coerces(self):
        out = _coerce_args({"radius": "float (optional, default 1.0)"}, {"radius": "0.5"})
        assert out == {"radius": 0.5}

    def test_int_coercion(self):
        out = _coerce_args({"n": "int"}, {"n": "3"})
        assert out == {"n": 3}

    def test_bool_coercion(self):
        assert _coerce_args({"flag": "bool"}, {"flag": "true"}) == {"flag": True}
        assert _coerce_args({"flag": "bool"}, {"flag": "no"}) == {"flag": False}
        assert _coerce_args({"flag": "bool"}, {"flag": True}) == {"flag": True}


def _mock_chat_response(tool_name: str, args: dict | None = None, content: str = "") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args or {}),
                            },
                        }
                    ],
                }
            }
        ]
    }


class TestLLMPolicy:
    def test_ask_parses_tool_call_into_intent(self):
        policy = LLMPolicy(registry=_registry())
        policy.set_goal("move to (0.1, 0.2, 0.3)")
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                _mock_chat_response("move_to", {"x": "0.1", "y": "0.2", "z": "0.3"},
                                    "moving toward target")
            ).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            intent = policy.ask()
        assert intent is not None
        assert intent.name == "move_to"
        assert intent.args == {"x": 0.1, "y": 0.2, "z": 0.3}
        assert intent.rationale == "moving toward target"

    def test_done_call_returns_none(self):
        policy = LLMPolicy(registry=_registry())
        policy.set_goal("test")
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                _mock_chat_response("done", {"reason": "goal reached"})
            ).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            intent = policy.ask()
        assert intent is None

    def test_no_tool_calls_returns_none(self):
        policy = LLMPolicy(registry=_registry())
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "no tools"}}]
            }).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            intent = policy.ask()
        assert intent is None

    def test_hallucinated_tool_yields_intent_for_runtime_to_block(self):
        policy = LLMPolicy(registry=_registry())
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                _mock_chat_response("teleport", {})
            ).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            intent = policy.ask()
        assert intent is not None
        assert intent.name == "teleport"
        assert "hallucinated" in intent.rationale.lower()

    def test_observation_appears_as_tool_role_message(self):
        policy = LLMPolicy(registry=_registry())
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(
                _mock_chat_response("done", {"reason": "ok"})
            ).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp
            policy.ask({"position": [0, 0, 0]})
        # Last user-role message before the assistant reply should carry the obs.
        roles = [m["role"] for m in policy.messages]
        assert "tool" in roles

    def test_http_error_raises_policy_error(self):
        import urllib.error
        policy = LLMPolicy(registry=_registry())
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            err = urllib.error.HTTPError(
                "url", 500, "server error", {}, None,  # type: ignore
            )
            err.read = lambda: b"{}"  # type: ignore
            mock_urlopen.side_effect = err
            with pytest.raises(LLMPolicyError, match="upstream 500"):
                policy.ask()


class TestLLMPolicyLoop:
    def test_drives_runtime_to_completion(self):
        rt = Runtime(
            backend=MockBackend(),
            registry=_registry(),
            policy_pipeline=PolicyPipeline(),
        )
        # Mock the LLM emitting: move_to -> done.
        responses = [
            _mock_chat_response("move_to", {"x": "0.5", "y": "0.0", "z": "0.0"}),
            _mock_chat_response("done", {"reason": "at target"}),
        ]
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            def _next_response(*a, **k):
                resp = MagicMock()
                resp.read.return_value = json.dumps(responses.pop(0)).encode()
                cm = MagicMock()
                cm.__enter__.return_value = resp
                return cm
            mock_urlopen.side_effect = _next_response
            summary = llm_policy_loop(
                registry=_registry(),
                runtime=rt,
                goal="move to (0.5, 0, 0)",
                config=LLMPolicyConfig(),
                max_steps=4,
            )
        assert summary["terminated"] == "done"
        assert summary["steps"] == 1
        assert rt.backend.position == (0.5, 0.0, 0.0)

    def test_respects_max_steps(self):
        rt = Runtime(
            backend=MockBackend(),
            registry=_registry(),
            policy_pipeline=PolicyPipeline(),
        )
        # LLM never emits 'done' — loop must give up at max_steps.
        with patch("ghostloop.policies.llm.urllib.request.urlopen") as mock_urlopen:
            def _next_response(*a, **k):
                resp = MagicMock()
                resp.read.return_value = json.dumps(
                    _mock_chat_response("scan", {"radius": "0.5"})
                ).encode()
                cm = MagicMock()
                cm.__enter__.return_value = resp
                return cm
            mock_urlopen.side_effect = _next_response
            summary = llm_policy_loop(
                registry=_registry(),
                runtime=rt,
                goal="never finishes",
                max_steps=3,
            )
        assert summary["terminated"] == "max_steps"
        assert summary["steps"] == 3
