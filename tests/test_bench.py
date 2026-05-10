"""Tests for ghostloop.bench — Wilson CIs, McNemar, episode runner, paired compare."""

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostloop import Intent, MockBackend, PolicyPipeline
from ghostloop.bench import (
    Episode,
    EpisodeResult,
    EpisodeRunner,
    cohens_h,
    effect_label,
    mcnemar_p,
    paired_compare,
    summarize,
    wilson_ci,
)
from ghostloop.policies import GeofenceGate
from ghostloop.primitives import move_to, pick, place, scan


# ------------------------------------------------------------------
# Wilson CI math.
# ------------------------------------------------------------------


class TestWilsonCI:
    def test_ci_zero_n_returns_zero_zero(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_ci_zero_passes(self):
        lo, hi = wilson_ci(0, 20)
        assert lo == 0.0
        assert hi > 0  # upper bound is well above 0
        assert hi < 0.2  # but still tight at n=20

    def test_ci_all_pass(self):
        lo, hi = wilson_ci(20, 20)
        assert hi == pytest.approx(1.0, abs=1e-6)
        assert lo > 0.8

    def test_ci_centered(self):
        lo, hi = wilson_ci(10, 20)
        assert 0.0 < lo < 0.5 < hi < 1.0
        # Roughly symmetric around 0.5 for p=0.5.
        assert abs((lo + hi) / 2 - 0.5) < 0.02

    def test_ci_widens_at_small_n(self):
        # Same proportion, smaller n -> wider CI.
        lo_a, hi_a = wilson_ci(5, 10)
        lo_b, hi_b = wilson_ci(50, 100)
        assert (hi_a - lo_a) > (hi_b - lo_b)


# ------------------------------------------------------------------
# McNemar exact p-value.
# ------------------------------------------------------------------


class TestMcNemar:
    def test_no_discordant_pairs_p_one(self):
        assert mcnemar_p(0, 0) == 1.0

    def test_perfectly_discordant_one_side(self):
        # All discordant pairs go one way -> tiny p.
        p = mcnemar_p(20, 0)
        assert p < 1e-4

    def test_balanced_discordant_p_high(self):
        # Equal split is exactly the null.
        p = mcnemar_p(10, 10)
        assert 0.5 < p <= 1.0

    def test_p_value_in_unit_interval(self):
        for a, b in [(0, 0), (1, 0), (0, 1), (3, 7), (50, 50), (0, 100)]:
            p = mcnemar_p(a, b)
            assert 0.0 <= p <= 1.0


# ------------------------------------------------------------------
# Cohen's h.
# ------------------------------------------------------------------


class TestCohensH:
    def test_zero_when_proportions_equal(self):
        assert cohens_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_p2_greater(self):
        assert cohens_h(0.3, 0.7) > 0

    def test_negative_when_p2_less(self):
        assert cohens_h(0.7, 0.3) < 0

    def test_effect_label_thresholds(self):
        assert effect_label(0.0) == "negligible"
        assert effect_label(0.19) == "negligible"
        assert effect_label(0.3) == "small"
        assert effect_label(0.6) == "medium"
        assert effect_label(0.9) == "large"


# ------------------------------------------------------------------
# Episode runner.
# ------------------------------------------------------------------


def _move_episode(name: str, target: tuple[float, float, float], scripted: bool = True) -> Episode:
    def setup():
        return MockBackend()

    def policy(runtime):
        if scripted:
            return [Intent("move_to", {"x": target[0], "y": target[1], "z": target[2]})]
        return []

    def success(trace, state):
        return tuple(state["position"]) == target

    return Episode(
        name=name,
        goal=f"move to {target}",
        setup=setup,
        policy=policy,
        success_predicate=success,
        primitives=lambda: [move_to(), scan(), pick(), place()],
    )


class TestEpisodeRunner:
    def test_passing_episode(self):
        ep = _move_episode("move-A", (0.5, 0.0, 0.0))
        result = EpisodeRunner().run(ep)
        assert result.passed
        assert result.episode_name == "move-A"
        assert result.steps == 1
        assert result.final_state["position"] == [0.5, 0.0, 0.0]

    def test_failing_episode(self):
        # Geofence the workspace tightly so the move target is outside.
        ep = _move_episode("move-out", (5.0, 0.0, 0.0))
        ep.pipeline = PolicyPipeline(gates=[
            GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
        ])
        result = EpisodeRunner().run(ep)
        assert not result.passed
        # Backend never moved because the move was BLOCKED.
        assert result.final_state["position"] == [0.0, 0.0, 0.0]

    def test_run_all(self):
        eps = [
            _move_episode("a", (0.1, 0.0, 0.0)),
            _move_episode("b", (0.2, 0.0, 0.0)),
            _move_episode("c", (0.3, 0.0, 0.0)),
        ]
        results = EpisodeRunner().run_all(eps)
        assert len(results) == 3
        assert all(r.passed for r in results)


# ------------------------------------------------------------------
# Run report + paired compare.
# ------------------------------------------------------------------


def _scripted_run(eps: list[Episode], pipeline: PolicyPipeline | None = None) -> list[EpisodeResult]:
    out = []
    for ep in eps:
        if pipeline is not None:
            ep.pipeline = pipeline
        out.append(EpisodeRunner().run(ep))
    return out


class TestRunReport:
    def test_report_shape(self):
        eps = [_move_episode(f"e{i}", (0.1, 0.0, 0.0)) for i in range(5)]
        results = _scripted_run(eps)
        report = summarize(results, run_name="baseline", bench_name="move-suite")
        assert report.n == 5
        assert report.passed == 5
        assert report.rate == 1.0

    def test_report_md_renders(self):
        eps = [_move_episode("e", (0.1, 0.0, 0.0))]
        results = _scripted_run(eps)
        report = summarize(results, run_name="x", bench_name="y")
        md = report.render_md()
        assert "Pass rate" in md
        assert "Episode" in md
        assert "Wilson 95% CI" in md


class TestPairedCompare:
    def test_two_perfect_runs_no_significance(self):
        eps = [_move_episode(f"e{i}", (0.1, 0.0, 0.0)) for i in range(5)]
        a_res = _scripted_run(eps)
        # Recreate eps so EpisodeRunner gets fresh state.
        eps_b = [_move_episode(f"e{i}", (0.1, 0.0, 0.0)) for i in range(5)]
        b_res = _scripted_run(eps_b)
        a_rep = summarize(a_res, "a", "bench")
        b_rep = summarize(b_res, "b", "bench")
        comp = paired_compare(a_rep, b_rep)
        assert comp.n == 5
        assert comp.a_passed == 5 and comp.b_passed == 5
        assert comp.p_value == 1.0
        assert not comp.significant

    def test_complete_dominance_yields_significance(self):
        # A passes 0/8, B passes 8/8 — maximally discordant.
        eps_a = [_move_episode(f"e{i}", (5.0, 0.0, 0.0)) for i in range(8)]
        a_res = _scripted_run(
            eps_a,
            pipeline=PolicyPipeline(gates=[
                GeofenceGate(min_corner=(-1, -1, -1), max_corner=(1, 1, 1)),
            ]),
        )
        eps_b = [_move_episode(f"e{i}", (0.5, 0.0, 0.0)) for i in range(8)]
        b_res = _scripted_run(eps_b)
        # Rename to overlap.
        for i, r in enumerate(a_res):
            r.episode_name = f"shared{i}"
        for i, r in enumerate(b_res):
            r.episode_name = f"shared{i}"
        a_rep = summarize(a_res, "no-fence", "shared-bench")
        b_rep = summarize(b_res, "with-fence", "shared-bench")
        comp = paired_compare(a_rep, b_rep)
        assert comp.only_b == 8
        assert comp.only_a == 0
        assert comp.significant
        assert comp.effect_h > 1.5  # massive effect

    def test_compare_rejects_different_benches(self):
        a = summarize([], "a", "bench-x")
        b = summarize([], "b", "bench-y")
        with pytest.raises(ValueError, match="benches differ"):
            paired_compare(a, b)

    def test_compare_rejects_no_overlap(self):
        eps_a = [_move_episode("a-only", (0.1, 0.0, 0.0))]
        eps_b = [_move_episode("b-only", (0.1, 0.0, 0.0))]
        a = summarize(_scripted_run(eps_a), "a", "shared")
        b = summarize(_scripted_run(eps_b), "b", "shared")
        with pytest.raises(ValueError, match="no shared episodes"):
            paired_compare(a, b)
