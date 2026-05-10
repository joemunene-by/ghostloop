"""Sim-to-real benchmark harness — paired comparison across two backends.

Wraps the v0.2 paired-comparison machinery (Wilson CI, McNemar, Cohen's h)
specifically for the question every robotics team eventually asks:

  "We trained in sim. How much does the policy degrade on the real
   robot — and how much of that gap closes when we add domain
   randomization to the sim training?"

Sim2RealBench runs the SAME bench against TWO backends (a "calibration"
backend like the deterministic MuJoCo + a "deployment" backend like the
real-robot adapter or RandomizedBackend), pairs the per-episode results,
and reports:

  - per-backend pass rate + Wilson 95% CI
  - paired McNemar p value for the success-rate gap
  - Cohen's h effect size + qualitative label
  - action-distribution KL divergence (per-primitive)
  - mean violation-rate gap (safety pipeline DENY rate per backend)
  - per-episode-pair breakdown for forensic analysis

Stdlib math; no scipy / numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .compare import cohens_h, effect_label, mcnemar_p
from .episode import Episode, EpisodeResult, EpisodeRunner
from .report import RunReport, summarize, wilson_ci


@dataclass
class Sim2RealReport:
    """Output of one Sim2RealBench run.

    Carries the two RunReports plus the cross-backend statistics so
    the caller can pretty-print and / or persist to the dashboard.
    """

    sim_label: str
    real_label: str
    sim_report: RunReport
    real_report: RunReport
    paired_pairs: list[tuple[bool, bool]]
    transfer_gap: float                      # sim_rate - real_rate
    transfer_gap_ci: tuple[float, float]
    mcnemar_p: float
    cohens_h: float
    cohens_label: str
    action_kl: dict[str, float]              # per-primitive symmetric KL
    sim_violation_rate: float
    real_violation_rate: float

    def to_json(self) -> dict[str, Any]:
        return {
            "sim_label": self.sim_label,
            "real_label": self.real_label,
            "sim_report": self.sim_report.to_json(),
            "real_report": self.real_report.to_json(),
            "transfer_gap": self.transfer_gap,
            "transfer_gap_ci": list(self.transfer_gap_ci),
            "mcnemar_p": self.mcnemar_p,
            "cohens_h": self.cohens_h,
            "cohens_label": self.cohens_label,
            "action_kl_per_primitive": self.action_kl,
            "sim_violation_rate": self.sim_violation_rate,
            "real_violation_rate": self.real_violation_rate,
        }

    def render_md(self) -> str:
        sim = self.sim_report
        real = self.real_report
        sim_lo, sim_hi = sim.ci
        real_lo, real_hi = real.ci
        gap_lo, gap_hi = self.transfer_gap_ci
        lines = [
            f"# Sim-to-Real Bench — {self.sim_label} vs {self.real_label}",
            "",
            f"- **{self.sim_label}**: {sim.passed}/{sim.n}"
            f" = {sim.rate:.1%}  (95% CI [{sim_lo:.1%}, {sim_hi:.1%}])",
            f"- **{self.real_label}**: {real.passed}/{real.n}"
            f" = {real.rate:.1%}  (95% CI [{real_lo:.1%}, {real_hi:.1%}])",
            f"- **Transfer gap**: {self.transfer_gap:+.1%}"
            f" (95% CI [{gap_lo:.1%}, {gap_hi:.1%}])",
            f"- **McNemar p**: {self.mcnemar_p:.4g}",
            f"- **Cohen's h**: {self.cohens_h:.3g} ({self.cohens_label})",
            f"- **Violation rate**: {self.sim_label} {self.sim_violation_rate:.1%},"
            f" {self.real_label} {self.real_violation_rate:.1%}",
            "",
            "## Action distribution KL (per primitive)",
            "",
        ]
        if self.action_kl:
            lines.append("| Primitive | symmetric KL |")
            lines.append("|---|---:|")
            for prim, kl in sorted(self.action_kl.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {prim} | {kl:.4g} |")
        else:
            lines.append("(no shared primitives)")
        return "\n".join(lines) + "\n"


def _compute_action_kl(
    sim_results: list[EpisodeResult], real_results: list[EpisodeResult],
) -> dict[str, float]:
    """Symmetric KL of the per-primitive call-frequency distributions.

    Cheap proxy for "policy did the same kind of work in sim vs real".
    A high KL means primitives were dispatched at very different rates
    on the two backends — a red flag for transfer.

    Add-1 smoothing to avoid log(0) on primitives present in only one
    backend.
    """
    sim_counts: dict[str, int] = {}
    real_counts: dict[str, int] = {}
    for res in sim_results:
        for ev in res.trace.events:
            sim_counts[ev.intent.name] = sim_counts.get(ev.intent.name, 0) + 1
    for res in real_results:
        for ev in res.trace.events:
            real_counts[ev.intent.name] = real_counts.get(ev.intent.name, 0) + 1
    primitives = set(sim_counts.keys()) | set(real_counts.keys())
    if not primitives:
        return {}
    sim_total = sum(sim_counts.values()) + len(primitives)
    real_total = sum(real_counts.values()) + len(primitives)
    out: dict[str, float] = {}
    for prim in primitives:
        p = (sim_counts.get(prim, 0) + 1) / sim_total
        q = (real_counts.get(prim, 0) + 1) / real_total
        # Symmetric KL = 0.5 * (KL(p||q) + KL(q||p)).
        kl = 0.5 * (p * math.log(p / q) + q * math.log(q / p))
        out[prim] = kl
    return out


def _violation_rate(results: list[EpisodeResult]) -> float:
    n_events = 0
    n_denied = 0
    for res in results:
        for ev in res.trace.events:
            n_events += 1
            if ev.decision.action.value == "deny":
                n_denied += 1
    return n_denied / n_events if n_events > 0 else 0.0


def _gap_wilson_ci(
    n: int, paired: list[tuple[bool, bool]],
) -> tuple[float, float]:
    """Wilson-style CI for the difference of paired proportions.

    Paired-difference CI via the Newcombe-Score-style approximation:
    take the per-direction Wilson and widen for paired structure. For
    the small N typical of robot benches this is conservative; for
    formal reporting use Bonett-Price or scipy. Within ghostloop we
    keep it stdlib so users can run benches in restricted envs.
    """
    if n == 0:
        return 0.0, 0.0
    sim_only = sum(1 for s, r in paired if s and not r)
    real_only = sum(1 for s, r in paired if r and not s)
    delta = (sim_only - real_only) / n
    if n < 2:
        return delta, delta
    # Wilson on each marginal then take the symmetric difference.
    p_lo, p_hi = wilson_ci(sim_only, n)
    q_lo, q_hi = wilson_ci(real_only, n)
    return delta - (p_hi + q_hi - p_lo - q_lo) / 2.0, delta + (p_hi + q_hi - p_lo - q_lo) / 2.0


@dataclass
class Sim2RealBench:
    """Run paired Episode lists against two configurations and produce a report.

    Episodes own their own backend via the ``setup`` callable, so the
    bench takes two Episode lists — typically the same goals + policies
    but with different ``setup`` and / or ``pipeline`` reflecting the
    two regimes (sim vs real, sim vs randomized-sim, calibration vs
    deployment). The lists must be the same length; episodes pair by
    index for the McNemar / Cohen's h statistics.

    Args:
        sim_episodes: episodes targeting the calibration backend.
        real_episodes: episodes targeting the deployment backend.
        sim_label / real_label: friendly labels for the report.
    """

    sim_episodes: list[Episode]
    real_episodes: list[Episode]
    sim_label: str = "sim"
    real_label: str = "real"

    def __post_init__(self) -> None:
        if len(self.sim_episodes) != len(self.real_episodes):
            raise ValueError(
                f"sim and real episode lists must be the same length, "
                f"got {len(self.sim_episodes)} vs {len(self.real_episodes)}"
            )

    def run(self) -> Sim2RealReport:
        runner = EpisodeRunner()
        sim_results = [runner.run(ep) for ep in self.sim_episodes]
        sim_report = summarize(sim_results, run_name=self.sim_label, bench_name="sim2real")
        real_results = [runner.run(ep) for ep in self.real_episodes]
        real_report = summarize(real_results, run_name=self.real_label, bench_name="sim2real")
        # Pair episodes by index.
        paired: list[tuple[bool, bool]] = list(zip(
            (r.passed for r in sim_results),
            (r.passed for r in real_results),
            strict=True,
        ))
        sim_pass = sum(1 for s, _ in paired if s)
        real_pass = sum(1 for _, r in paired if r)
        n = len(paired)
        gap = (sim_pass - real_pass) / n if n else 0.0
        # Use sim_report.rate / real_report.rate as authoritative.
        sim_rate = sim_report.rate
        real_rate = real_report.rate
        gap_ci = _gap_wilson_ci(n, paired)
        # McNemar pairs disagreement counts.
        b = sum(1 for s, r in paired if s and not r)  # sim pass, real fail
        c = sum(1 for s, r in paired if not s and r)  # sim fail, real pass
        mc_p = mcnemar_p(b, c)
        h = cohens_h(sim_rate, real_rate)
        return Sim2RealReport(
            sim_label=self.sim_label,
            real_label=self.real_label,
            sim_report=sim_report,
            real_report=real_report,
            paired_pairs=paired,
            transfer_gap=gap,
            transfer_gap_ci=gap_ci,
            mcnemar_p=mc_p,
            cohens_h=h,
            cohens_label=effect_label(h),
            action_kl=_compute_action_kl(sim_results, real_results),
            sim_violation_rate=_violation_rate(sim_results),
            real_violation_rate=_violation_rate(real_results),
        )
