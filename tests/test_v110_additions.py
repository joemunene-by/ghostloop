"""v1.1.0: backend registry, canonical frames, cross-backend bench, conformance,
non-finite argument rejection, MuJoCo IK, and ClaudePolicy."""

import json
import math
from types import SimpleNamespace

import pytest

from ghostloop.backends import (
    BackendBinding,
    BackendRegistryError,
    create_backend,
    list_backends,
    mujoco_available,
    register_backend,
    unregister_backend,
)
from ghostloop.bench.catalogue import PRESETS, preset_reach_8
from ghostloop.bench.cross import bind_episodes, cross_backend_compare
from ghostloop.bench.episode import EpisodeRunner
from ghostloop.bench.frames import FramedBackend, Workspace, framed_primitive
from ghostloop.core import (
    Intent,
    MockBackend,
    PolicyPipeline,
    PrimitiveRegistry,
    Runtime,
    non_finite_arg,
)
from ghostloop.policies import ForceCapGate, GeofenceGate, ObstacleAvoidanceGate
from ghostloop.policies.workspace import WorkspaceModel
from ghostloop.primitives import move_to, pick, place, scan
from ghostloop.testing import check_backend_conformance


def _menagerie_ready() -> bool:
    if not mujoco_available():
        return False
    from ghostloop.backends.menagerie import MenagerieError, ensure_menagerie
    try:
        root = ensure_menagerie(allow_clone=False)
    except MenagerieError:
        return False
    return (root / "universal_robots_ur5e").is_dir()


needs_menagerie = pytest.mark.skipif(
    not _menagerie_ready(), reason="mujoco + MuJoCo Menagerie (ur5e) not available",
)


class TestRegistry:
    def test_builtins_listed(self):
        names = {b.name for b in list_backends()}
        assert {"mock", "mujoco", "pybullet", "gym"} <= names

    def test_create_mock(self):
        binding = create_backend("mock")
        assert isinstance(binding, BackendBinding)
        assert {p.name for p in binding.primitives} == {"move_to", "scan", "pick", "place"}
        assert binding.provenance["spec"] == "mock"

    def test_unknown_backend(self):
        with pytest.raises(BackendRegistryError, match="unknown backend"):
            create_backend("nope:thing")

    def test_register_custom_backend_with_variant(self):
        seen = {}

        def factory(variant, **kwargs):
            seen["variant"] = variant
            return BackendBinding(MockBackend(name="custom"), [move_to()], embodiment=variant)

        register_backend("custom-test", factory, description="test")
        try:
            binding = create_backend("custom-test:arm7")
            assert seen["variant"] == "arm7"
            assert binding.embodiment == "arm7"
            with pytest.raises(ValueError, match="already registered"):
                register_backend("custom-test", factory)
        finally:
            unregister_backend("custom-test")

    def test_factory_must_return_binding(self):
        register_backend("bad-test", lambda variant, **kw: MockBackend())
        try:
            with pytest.raises(TypeError, match="BackendBinding"):
                create_backend("bad-test")
        finally:
            unregister_backend("bad-test")

    def test_names_cannot_contain_colon(self):
        with pytest.raises(ValueError):
            register_backend("a:b", lambda v: None)


class TestFrames:
    def test_round_trip(self):
        ws = Workspace(center=(0.45, 0.0, 0.4), scale=0.35)
        p = (0.3, -0.2, 0.1)
        assert ws.to_canonical(ws.to_robot(p)) == pytest.approx(p)

    def test_framed_backend_and_primitive(self):
        ws = Workspace(center=(1.0, 2.0, 3.0), scale=0.5)
        backend = FramedBackend(MockBackend(), ws)
        prim = framed_primitive(move_to(), ws)
        res = prim.call(backend, x=1.0, y=0.0, z=-1.0)
        assert res.ok
        snap = backend.snapshot()
        assert snap["robot_position"] == pytest.approx([1.5, 2.0, 2.5])
        assert snap["position"] == pytest.approx([1.0, 0.0, -1.0])


class TestSeededPresets:
    def test_same_seed_same_targets(self):
        a = [e.metadata["target"] for e in preset_reach_8(seed=7)]
        b = [e.metadata["target"] for e in preset_reach_8(seed=7)]
        c = [e.metadata["target"] for e in preset_reach_8(seed=8)]
        assert a == b
        assert a != c

    def test_unseeded_is_unchanged(self):
        assert preset_reach_8()[1].metadata["target"] == [0.5, 0.0, 0.0]
        assert preset_reach_8()[1].name == "reach-px"

    def test_tolerance_predicate(self):
        ep = preset_reach_8(tolerance=0.1)[1]
        assert ep.success_predicate(None, {"position": [0.55, 0.0, 0.0]})
        assert not ep.success_predicate(None, {"position": [0.65, 0.0, 0.0]})

    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_all_presets_pass_on_mock_without_gates(self, name):
        results = EpisodeRunner().run_all(PRESETS[name](seed=0, tolerance=1e-9))
        assert all(r.passed for r in results)


class TestCrossBackend:
    def test_mock_vs_mock_is_identical(self, tmp_path):
        report = cross_backend_compare(PRESETS["reach_8"], "mock", "mock", seeds=[0, 1])
        s = report.stats
        assert s.sim_report.passed == s.real_report.passed == 16
        assert s.mcnemar_p == 1.0
        assert report.divergence_summary()["max"] == 0.0
        out = report.write(tmp_path)
        data = json.loads((out / "report.json").read_text())
        assert data["provenance"]["seeds"] == [0, 1]
        assert len(list((out / "traces" / "a").glob("*.jsonl"))) == 16

    def test_gates_apply_in_canonical_frame(self):
        fence = lambda: PolicyPipeline(gates=[GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))])
        report = cross_backend_compare(PRESETS["geofence_smoke"], "mock", "mock",
                                       pipeline_factory=fence)
        assert report.stats.sim_report.passed == 4
        assert report.stats.sim_violation_rate == pytest.approx(0.5)

    def test_bound_episode_uses_registry_backend(self):
        register_backend("shifted-test", lambda v, **k: BackendBinding(
            MockBackend(name="shifted"), [move_to(), scan(), pick(), place()], embodiment="shifted"))
        try:
            eps, prov = bind_episodes(preset_reach_8(tolerance=1e-9),
                                      "shifted-test", workspace=Workspace((10.0, 0.0, 0.0), 2.0))
            results = EpisodeRunner().run_all(eps)
            assert all(r.passed for r in results)
            assert results[1].final_state["robot_position"] == pytest.approx([11.0, 0.0, 0.0])
            assert prov["workspace"] == {"center": [10.0, 0.0, 0.0], "scale": 2.0}
        finally:
            unregister_backend("shifted-test")


class TestNonFiniteRejection:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "NaN", " -inf "])
    def test_pipeline_denies_before_gates(self, bad):
        pipe = PolicyPipeline(gates=[GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))])
        d = pipe.check(Intent("move_to", {"x": bad, "y": 0, "z": 0}), move_to())
        assert d.action.value == "deny"
        assert d.gate_name == "finite_args"

    def test_empty_pipeline_still_denies(self):
        rt = Runtime(MockBackend(), PrimitiveRegistry([move_to()]))
        res = rt.step(Intent("move_to", {"x": float("nan"), "y": 0, "z": 0}))
        assert res.status.value == "blocked"
        assert rt.backend.position == (0.0, 0.0, 0.0)

    def test_nested_paths(self):
        assert non_finite_arg({"target": [0, 1, float("nan")]}) == "target[2]"
        assert non_finite_arg({"a": {"b": "inf"}}) == "a.b"
        assert non_finite_arg({"x": 1.0, "label": "infinite loop", "flag": True}) is None

    def test_gates_are_safe_standalone(self):
        nan = float("nan")
        assert GeofenceGate().check(Intent("move_to", {"x": nan, "y": 0, "z": 0.5}), move_to()).action.value == "deny"
        ws = ObstacleAvoidanceGate(workspace=WorkspaceModel())
        assert ws.check(Intent("move_to", {"x": nan, "y": 0, "z": 0.5}), move_to()).action.value == "deny"
        assert ForceCapGate(force_max=5).check(Intent("push", {"force": nan}), move_to()).action.value == "deny"

    def test_async_pipeline(self):
        import asyncio
        from ghostloop.async_runtime import AsyncPolicyPipeline
        d = asyncio.run(AsyncPolicyPipeline().check(Intent("move_to", {"x": "nan", "y": 0, "z": 0}), move_to()))
        assert d.action.value == "deny"


class TestConformance:
    def test_mock_conforms(self):
        report = check_backend_conformance("mock")
        assert report.ok, report.render()

    def test_dishonest_backend_fails(self):
        def _lying_move(backend, x, y, z):
            from ghostloop.core import Result, ResultStatus
            return Result(ResultStatus.OK, message="sure")

        from ghostloop.core import Primitive
        lying = Primitive("move_to", _lying_move, "lies", {"x": "float", "y": "float", "z": "float"})
        register_backend("liar-test", lambda v, **k: BackendBinding(MockBackend(), [lying]))
        try:
            report = check_backend_conformance("liar-test")
            assert not report.ok
            assert any("central workspace target" in name for name, _ in report.failed)
        finally:
            unregister_backend("liar-test")


@needs_menagerie
class TestMuJoCoIK:
    def test_ur5e_reaches_cartesian_target(self):
        binding = create_backend("mujoco:ur5e")
        prim = next(p for p in binding.primitives if p.name == "move_to")
        res = prim.call(binding.backend, x=0.5, y=0.2, z=0.3)
        assert res.ok, res.message
        assert res.observation["tracking_error_m"] < 0.002

    def test_unreachable_target_fails_closed(self):
        binding = create_backend("mujoco:ur5e")
        before = binding.backend.snapshot()["qpos"]
        prim = next(p for p in binding.primitives if p.name == "move_to")
        res = prim.call(binding.backend, x=3.0, y=0.0, z=0.0)
        assert res.status.value == "error"
        assert "unreachable" in res.message
        assert binding.backend.snapshot()["qpos"] == before

    def test_conformance(self):
        for spec in ("mujoco:ur5e", "mujoco:panda"):
            report = check_backend_conformance(spec)
            assert report.ok, report.render()

    def test_provenance_has_no_absolute_paths(self):
        prov = create_backend("mujoco:ur5e").provenance
        assert prov["model"].startswith("menagerie:")
        assert prov["control"] == "ik"

    def test_cross_backend_detects_gravity_sag(self):
        report = cross_backend_compare(
            PRESETS["reach_8"], "mujoco:ur5e", "mujoco:ur5e", seeds=[0],
            tolerance=0.01, b_kwargs={"gravity_compensation": False},
        )
        assert report.stats.sim_report.passed == 8
        assert report.stats.real_report.passed < 8
        assert report.divergence_summary()["mean_m"] > 0.001


class _Block(SimpleNamespace):
    pass


def _claude_response(blocks, stop_reason="tool_use", model="claude-opus-5-5"):
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason, model=model,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                              cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class _FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _fake_client(responses):
    msgs = _FakeMessages(responses)
    return SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)), msgs


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("anthropic") is None, reason="anthropic SDK not installed",
)
class TestClaudePolicy:
    def _registry(self):
        return PrimitiveRegistry([move_to(), scan()])

    def test_tool_call_becomes_intent_and_request_shape(self):
        from ghostloop.policies import ClaudePolicy
        client, msgs = _fake_client([_claude_response([
            _Block(type="text", text="Heading to the target."),
            _Block(type="tool_use", id="toolu_1", name="move_to", input={"x": "0.1", "y": 0.2, "z": 0.3}),
        ])])
        policy = ClaudePolicy(registry=self._registry(), client=client)
        policy.set_goal("go")
        intent = policy.ask()
        assert intent == Intent("move_to", {"x": 0.1, "y": 0.2, "z": 0.3}, "Heading to the target.")
        call = msgs.calls[0]
        assert call["model"] == "claude-opus-5-5"
        assert call["fallbacks"] == "default"
        assert call["betas"] == ["server-side-fallback-2026-07-01"]
        assert call["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert call["output_config"] == {"effort": "medium"}
        assert {t["name"] for t in call["tools"]} == {"move_to", "scan", "done"}

    def test_observation_is_a_tool_result_for_the_right_id(self):
        from ghostloop.policies import ClaudePolicy
        client, msgs = _fake_client([
            _claude_response([_Block(type="tool_use", id="toolu_9", name="scan", input={})]),
            _claude_response([_Block(type="tool_use", id="toolu_10", name="done", input={"reason": "ok"})]),
        ])
        policy = ClaudePolicy(registry=self._registry(), client=client)
        policy.set_goal("look")
        policy.ask()
        assert policy.ask({"status": "ok"}) is None
        last_user = [m for m in policy.messages if m["role"] == "user"][-1]
        assert last_user["content"][0]["tool_use_id"] == "toolu_9"

    def test_refusal_terminates_loop(self):
        from ghostloop.policies import ClaudePolicy, llm_policy_loop
        client, _ = _fake_client([_claude_response([], stop_reason="refusal")])
        reg = self._registry()
        policy = ClaudePolicy(registry=reg, client=client)
        rt = Runtime(MockBackend(), reg)
        summary = llm_policy_loop(reg, rt, "go", policy=policy)
        assert summary["terminated"] == "refusal"
        assert summary["steps"] == 0

    def test_cost_accounting(self):
        from ghostloop.policies import ClaudePolicy
        client, _ = _fake_client([_claude_response([_Block(type="tool_use", id="t", name="done", input={})])])
        policy = ClaudePolicy(registry=self._registry(), client=client)
        policy.set_goal("x")
        policy.ask()
        assert policy.estimated_cost() == pytest.approx((1000 * 4.0 + 200 * 20.0) / 1e6)

    def test_unpriced_model_reports_none(self):
        from ghostloop.policies import ClaudePolicy
        client, _ = _fake_client([_claude_response([], stop_reason="end_turn", model="claude-future-9")])
        policy = ClaudePolicy(registry=self._registry(), client=client)
        policy.set_goal("x")
        policy.ask()
        assert policy.estimated_cost() is None

    def test_drives_episode_and_reports_spend(self):
        from ghostloop.bench import Episode
        from ghostloop.policies import ClaudePolicy, llm_episode_policy
        client, _ = _fake_client([
            _claude_response([_Block(type="tool_use", id="a", name="move_to", input={"x": 0.5, "y": 0, "z": 0})]),
            _claude_response([_Block(type="tool_use", id="b", name="done", input={"reason": "there"})]),
        ])
        ep = Episode(
            name="llm-reach", goal="reach (0.5,0,0)", setup=MockBackend,
            policy=llm_episode_policy(lambda reg: ClaudePolicy(registry=reg, client=client), "reach (0.5,0,0)"),
            success_predicate=lambda tr, st: math.dist(st["position"], (0.5, 0, 0)) < 1e-9,
            primitives=lambda: [move_to(), scan()],
        )
        res = EpisodeRunner().run(ep)
        assert res.passed
        assert res.extras["cost_usd"] > 0
        assert res.extras["usage"]["requests"] == 2
