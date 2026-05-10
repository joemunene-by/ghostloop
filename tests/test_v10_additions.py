"""Tests for v0.10: counterfactual replay, causal attribution, LLM judge,
adversarial bench, property mining, skill graph, hindsight relabeling,
energy ledger, morphology registry."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import (
    Decision,
    Intent,
    MockBackend,
    PolicyPipeline,
    PrimitiveRegistry,
    Result,
    Runtime,
    Trace,
    TraceEvent,
)
from ghostloop.bench import (
    AdversarialResult,
    Episode,
    cma_es_seeds,
    grid_seeds,
    random_seeds,
)
from ghostloop.causal import attribute_failure, minimal_cause_set
from ghostloop.core import DecisionAction, Primitive, ResultStatus
from ghostloop.counterfactual import CounterfactualTrace, replay_with_policy
from ghostloop.judges import (
    HeuristicJudge,
    JudgeRule,
    LLMJudge,
    LLMJudgeConfig,
)
from ghostloop.judges.heuristic import (
    fraction_allowed,
    fraction_ok,
    no_violations,
    step_count_below,
)
from ghostloop.judges.llm_judge import parse_judgement
from ghostloop.policies import GeofenceGate
from ghostloop.primitives import MorphologyError, MorphologyRegistry, move_to, scan
from ghostloop.properties import MinedProperty, mine_properties
from ghostloop.properties.builtins import StaysInsideWorkspace
from ghostloop.skills import Skill, SkillError, SkillGraph, skill_from_primitive
from ghostloop.telemetry import EnergyEstimator, EnergyLedger, PrimitiveEnergyModel, default_estimator
from ghostloop.training import (
    HindsightStrategy,
    Rollout,
    Transition,
    hindsight_relabel,
    sparse_indicator_reward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace(events: list[tuple[str, dict, str, float]]) -> Trace:
    """events: (intent_name, args, decision_action, timestamp)"""
    trace = Trace()
    for i, (name, args, action, ts) in enumerate(events):
        intent = Intent(name, args)
        if action == "deny":
            decision = Decision.deny("test", "synthetic")
        else:
            decision = Decision.allow("test", "synthetic")
        result = Result(status=ResultStatus.OK, observation={"force": 1.0})
        trace.append(TraceEvent(
            step=i, intent=intent, decision=decision, result=result,
            state_before={"position": [0, 0, 0]},
            state_after={"position": [args.get("x", 0.0), 0, 0], "force": 1.0},
            timestamp=ts,
        ))
    return trace


# ---------------------------------------------------------------------------
# Counterfactual replay
# ---------------------------------------------------------------------------


class TestCounterfactual:
    def test_match_when_policy_agrees(self):
        trace = _trace([("move_to", {"x": 0.5}, "allow", 1.0)])
        def same_policy(state):
            return Intent("move_to", {"x": 0.5})
        cf = replay_with_policy(trace, same_policy, new_policy_name="same")
        assert cf.n == 1
        assert cf.events[0].intent_match
        assert cf.divergence_rate == 0.0

    def test_diverge_on_args(self):
        trace = _trace([("move_to", {"x": 0.5}, "allow", 1.0)])
        def shifted(state):
            return Intent("move_to", {"x": 1.0})
        cf = replay_with_policy(trace, shifted, new_policy_name="shifted")
        assert not cf.events[0].intent_match
        assert cf.divergence_rate == 1.0
        assert cf.events[0].args_distance > 0

    def test_decline_treated_as_divergence(self):
        trace = _trace([("move_to", {"x": 0.5}, "allow", 1.0)])
        def decline(state):
            return None
        cf = replay_with_policy(trace, decline)
        assert not cf.events[0].intent_match
        assert cf.events[0].counterfactual_intent is None

    def test_first_divergence_step(self):
        trace = _trace([
            ("move_to", {"x": 0.0}, "allow", 1.0),
            ("move_to", {"x": 1.0}, "allow", 2.0),
            ("scan", {}, "allow", 3.0),
        ])
        def diff_at_2(state):
            x = state.get("position", [0, 0, 0])[0]
            if x > 0.5:
                return Intent("pick", {})
            return Intent("move_to", state.get("position", {"x": 0}) and {"x": 0})
        cf = replay_with_policy(trace, diff_at_2)
        # Some divergence happens; first one >= step 0.
        assert cf.first_divergence_step is not None


# ---------------------------------------------------------------------------
# Causal failure attribution
# ---------------------------------------------------------------------------


class TestCausalAttribution:
    def test_no_attribution_when_property_holds(self):
        trace = _trace([("move_to", {"x": 0.5}, "allow", 1.0)])
        prop = StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        analysis = attribute_failure(trace, prop)
        assert analysis.held
        assert analysis.n_necessary == 0

    def test_identifies_necessary_event(self):
        trace = _trace([
            ("move_to", {"x": 0.0}, "allow", 1.0),
            ("move_to", {"x": 5.0}, "allow", 2.0),  # this puts position outside
        ])
        prop = StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        analysis = attribute_failure(trace, prop)
        assert not analysis.held
        # Removing event 1 should restore the property.
        top = analysis.top_k(1)
        assert top
        assert top[0].event.step == 1


class TestMinimalCauseSet:
    def test_single_event_cause(self):
        trace = _trace([
            ("move_to", {"x": 0.0}, "allow", 1.0),
            ("move_to", {"x": 5.0}, "allow", 2.0),
        ])
        prop = StaysInsideWorkspace(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        cause = minimal_cause_set(trace, prop)
        assert cause is not None
        assert len(cause) == 1


# ---------------------------------------------------------------------------
# Heuristic + LLM judges
# ---------------------------------------------------------------------------


class TestHeuristicJudge:
    def test_no_violations_passes(self):
        trace = _trace([("move_to", {"x": 0.5}, "allow", 1.0)])
        judge = HeuristicJudge(
            rules=[JudgeRule(name="no_violations", predicate=no_violations(), weight=1.0)],
        )
        score = judge.score(trace)
        assert score.label == "pass"

    def test_violations_fail(self):
        trace = _trace([
            ("move_to", {"x": 0.5}, "deny", 1.0),
            ("move_to", {"x": 0.5}, "deny", 2.0),
        ])
        judge = HeuristicJudge(
            rules=[
                JudgeRule(name="no_violations", predicate=no_violations(), weight=1.0),
                JudgeRule(name="frac_allowed", predicate=fraction_allowed(), weight=1.0),
            ],
        )
        score = judge.score(trace)
        assert score.label == "fail"

    def test_step_count_predicate(self):
        trace = _trace([("move_to", {}, "allow", float(i)) for i in range(5)])
        pred = step_count_below(10)
        assert pred(trace) == 1.0

        long_trace = _trace([("move_to", {}, "allow", float(i)) for i in range(20)])
        assert pred(long_trace) == 0.0


class TestLLMJudge:
    def test_parse_well_formed(self):
        raw = '{"task_completion": 1.0, "safety_adherence": 0.8, "efficiency": 0.6, "recoverability": 0.9, "label": "pass", "reasoning": "looks good"}'
        result = parse_judgement(raw)
        assert result.label == "pass"
        assert result.score == pytest.approx((1.0 + 0.8 + 0.6 + 0.9) / 4.0)

    def test_parse_fenced(self):
        raw = '```json\n{"task_completion": 0.5, "safety_adherence": 0.5, "efficiency": 0.5, "recoverability": 0.5, "label": "marginal", "reasoning": ""}\n```'
        result = parse_judgement(raw)
        assert result.label == "marginal"
        assert result.score == pytest.approx(0.5)

    def test_parse_malformed_returns_error(self):
        result = parse_judgement("not json at all")
        assert result.label == "error"

    def test_judge_calls_client_and_parses(self):
        class _StubClient:
            def chat(self, messages, **kw):
                return '{"task_completion": 0.9, "safety_adherence": 0.9, "efficiency": 0.9, "recoverability": 0.9, "label": "pass", "reasoning": "ok"}'

        trace = _trace([("move_to", {"x": 0.0}, "allow", 1.0)])
        judge = LLMJudge(client=_StubClient())
        result = judge.score(trace)
        assert result.label == "pass"


# ---------------------------------------------------------------------------
# Adversarial bench generator
# ---------------------------------------------------------------------------


class TestAdversarial:
    def _episode(self, x: float = 0.0):
        def setup():
            return MockBackend()

        def policy(runtime):
            runtime.step(Intent("move_to", {"x": x, "y": 0.0, "z": 0.0}))
            return None

        def predicate(trace, snap):
            return any(ev.decision.action is DecisionAction.ALLOW for ev in trace.events)

        return Episode(
            name="adv",
            goal=f"reach x={x}",
            setup=setup,
            policy=policy,
            success_predicate=predicate,
            primitives=lambda: [move_to(), scan()],
            pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )

    def test_random_finds_failures(self):
        base = self._episode(x=0.0)

        def _make_policy(x):
            def policy(rt):
                rt.step(Intent("move_to", {"x": x, "y": 0.0, "z": 0.0}))
                return None
            return policy

        def perturber(ep, sample):
            x = sample["x"]
            return Episode(
                name=ep.name + f"_x{x:.2f}", goal=ep.goal,
                setup=ep.setup,
                policy=_make_policy(x),
                success_predicate=ep.success_predicate,
                primitives=ep.primitives, pipeline=ep.pipeline,
            )

        results = random_seeds(
            base, perturber, parameter_ranges={"x": (-5.0, 5.0)},
            n_samples=10, seed=42,
        )
        assert len(results) == 10
        # At least one out-of-bounds seed should fail.
        assert any(not r.passed for r in results)
        # Sorted descending by failure_score.
        for i in range(len(results) - 1):
            assert results[i].failure_score >= results[i + 1].failure_score

    def test_grid_exhaustive(self):
        base = self._episode()

        def _make_policy(x):
            def policy(rt):
                rt.step(Intent("move_to", {"x": x, "y": 0.0, "z": 0.0}))
                return None
            return policy

        def perturber(ep, sample):
            x = sample["x"]
            return Episode(
                name=ep.name, goal=ep.goal, setup=ep.setup,
                policy=_make_policy(x),
                success_predicate=ep.success_predicate,
                primitives=ep.primitives, pipeline=ep.pipeline,
            )

        results = grid_seeds(base, perturber, parameter_grid={"x": [-2.0, 0.0, 2.0]})
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Property mining
# ---------------------------------------------------------------------------


class TestPropertyMining:
    def test_mines_workspace_bounds(self):
        traces = [
            _trace([("move_to", {"x": 0.1}, "allow", 1.0)]),
            _trace([("move_to", {"x": 0.5}, "allow", 1.0)]),
            _trace([("move_to", {"x": 0.3}, "allow", 1.0)]),
        ]
        mined = mine_properties(traces)
        # Should have at least the workspace bounds property.
        ws_mined = [m for m in mined if m.pattern == "workspace_bounds"]
        assert len(ws_mined) == 1
        prop = ws_mined[0].promote()
        # Apply on a trace within bounds — should hold.
        result = prop.check(traces[0])
        assert result.held

    def test_mines_followups(self):
        # Pattern: every move_to is followed by scan within 5s.
        traces = []
        for _ in range(3):
            traces.append(_trace([
                ("move_to", {"x": 0.1}, "allow", 1.0),
                ("scan", {}, "allow", 2.0),
            ]))
        mined = mine_properties(traces, min_support=0.9)
        followups = [m for m in mined if m.pattern == "followup"]
        assert any("scan" in m.description for m in followups)


# ---------------------------------------------------------------------------
# Skill graph
# ---------------------------------------------------------------------------


class TestSkillGraph:
    def test_topological_order(self):
        g = SkillGraph()
        g.add(skill_from_primitive(move_to()))
        g.add(skill_from_primitive(scan(), prerequisites=["move_to"]))
        order = g.topological_order()
        assert order.index("move_to") < order.index("scan")

    def test_cycle_detected(self):
        g = SkillGraph()
        g.add(Skill(name="a", primitive=move_to(), prerequisites=["b"]))
        g.add(Skill(name="b", primitive=scan(), prerequisites=["a"]))
        with pytest.raises(SkillError):
            g.validate()

    def test_unknown_prereq_raises(self):
        g = SkillGraph()
        g.add(Skill(name="a", primitive=move_to(), prerequisites=["nonexistent"]))
        with pytest.raises(SkillError):
            g.validate()

    def test_required_for(self):
        g = SkillGraph()
        g.add(skill_from_primitive(move_to()))
        g.add(Skill(name="approach", primitive=scan(), prerequisites=["move_to"]))
        g.add(Skill(name="grasp", primitive=scan(), prerequisites=["approach"]))
        deps = g.required_for("grasp")
        assert deps == {"move_to", "approach"}

    def test_coverage(self):
        g = SkillGraph()
        g.add(skill_from_primitive(move_to()))
        cov = g.coverage(["move_to", "pick"])
        assert "move_to" in cov["covered"]
        assert "pick" in cov["missing"]


# ---------------------------------------------------------------------------
# Hindsight relabeling
# ---------------------------------------------------------------------------


class TestHindsight:
    def test_final_strategy_relabels_to_end_state(self):
        ts = [
            Transition(
                obs={"position": [0.0, 0.0, 0.0]},
                action=[0.1, 0.0, 0.0],
                reward=0.0,
                next_obs={"position": [0.1, 0.0, 0.0]},
                done=False, violated=False, decision={},
            ),
            Transition(
                obs={"position": [0.1, 0.0, 0.0]},
                action=[0.1, 0.0, 0.0],
                reward=0.0,
                next_obs={"position": [0.5, 0.0, 0.0]},
                done=True, violated=False, decision={},
            ),
        ]
        rollout = Rollout(transitions=ts, started_at=0.0, finished_at=1.0)
        relabeled = hindsight_relabel(
            rollout,
            goal_extractor=lambda obs: obs["position"],
            reward_fn=sparse_indicator_reward(threshold=0.05),
            strategy=HindsightStrategy.FINAL,
        )
        assert relabeled.length == 2
        # Final achieved is [0.5, 0, 0] -> the last transition got reward 1.
        assert relabeled.transitions[-1].reward == 1.0

    def test_sparse_indicator_reward(self):
        rew = sparse_indicator_reward(threshold=0.1)
        assert rew([0, 0, 0], [0, 0, 0]) == 1.0
        assert rew([0, 0, 0], [1, 0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Energy ledger
# ---------------------------------------------------------------------------


class TestEnergyLedger:
    def test_total_skips_denied_events(self):
        trace = _trace([
            ("move_to", {"x": 0.5, "y": 0.0, "z": 0.0}, "allow", 1.0),
            ("move_to", {"x": 5.0, "y": 0.0, "z": 0.0}, "deny", 2.0),
        ])
        ledger = EnergyLedger()
        # Denied event should NOT count toward total.
        total_with_deny = ledger.total(trace)
        # Now make all events allowed.
        for ev in trace.events:
            ev.decision = Decision.allow("g", "ok")
        total_all_allow = ledger.total(trace)
        assert total_all_allow > total_with_deny

    def test_per_primitive_breakdown(self):
        trace = _trace([
            ("move_to", {"x": 1.0, "y": 0.0, "z": 0.0}, "allow", 1.0),
            ("scan", {}, "allow", 2.0),
        ])
        ledger = EnergyLedger()
        breakdown = ledger.by_primitive(trace)
        assert "move_to" in breakdown
        assert "scan" in breakdown
        assert breakdown["scan"] == pytest.approx(0.2)

    def test_constant_model(self):
        model = PrimitiveEnergyModel.constant("pick", 5.0)
        intent = Intent("pick", {})
        result = Result(status=ResultStatus.OK)
        assert model.estimator(intent, result) == 5.0

    def test_linear_in_arg(self):
        model = PrimitiveEnergyModel.linear_in_arg("scan", "radius", slope=2.0, intercept=0.5)
        intent = Intent("scan", {"radius": 3.0})
        result = Result(status=ResultStatus.OK)
        assert model.estimator(intent, result) == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# Morphology registry
# ---------------------------------------------------------------------------


class TestMorphology:
    def test_register_and_build(self):
        reg = MorphologyRegistry()
        reg.register("franka", "pick", lambda: Primitive(name="pick", call=lambda b, **k: None))
        reg.register("ur5e",   "pick", lambda: Primitive(name="pick", call=lambda b, **k: None))
        prims = reg.build("franka", ["pick"])
        assert prims[0].name == "pick"

    def test_unknown_morphology_raises(self):
        reg = MorphologyRegistry()
        with pytest.raises(MorphologyError):
            reg.build_one("franka", "pick")

    def test_coverage_report(self):
        reg = MorphologyRegistry()
        reg.register("spot", "walk", lambda: Primitive(name="walk", call=lambda b, **k: None))
        cov = reg.coverage("spot", ["walk", "pick"])
        assert "walk" in cov["covered"]
        assert "pick" in cov["missing"]

    def test_supported_lists_primitives(self):
        reg = MorphologyRegistry()
        reg.register("a", "x", lambda: Primitive(name="x", call=lambda b, **k: None))
        reg.register("a", "y", lambda: Primitive(name="y", call=lambda b, **k: None))
        reg.register("b", "x", lambda: Primitive(name="x", call=lambda b, **k: None))
        assert reg.supported("a") == ["x", "y"]
        assert reg.morphologies() == ["a", "b"]
