"""Cross-backend paired benchmark: the same seeded episodes on two backends.

Built for engine-vs-engine questions (MuJoCo vs another simulator, sim vs a
randomized sim, sim vs hardware). Episodes keep their canonical-frame goals,
gates and predicates; each backend is bound through its embodiment's
:class:`~ghostloop.bench.frames.Workspace`, so results pair one to one.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core import PolicyPipeline
from .episode import Episode, EpisodeResult, EpisodeRunner
from .frames import FramedBackend, Workspace, framed_primitive, workspace_for
from .sim2real import Sim2RealReport, _compute_action_kl, _gap_wilson_ci, _violation_rate
from .compare import cohens_h, effect_label, mcnemar_p
from .report import summarize


def bind_episodes(
    episodes: list[Episode],
    spec: str,
    *,
    workspace: Workspace | None = None,
    backend_kwargs: dict[str, Any] | None = None,
) -> tuple[list[Episode], dict[str, Any]]:
    """Retarget episodes at backend ``spec``. Every episode gets a fresh backend.

    Returns the bound episodes and the provenance of the first backend built.
    """
    from ..backends.registry import create_backend

    kwargs = dict(backend_kwargs or {})
    probe = create_backend(spec, **kwargs)
    ws = workspace or workspace_for(probe.embodiment)
    provenance = {**probe.provenance, "embodiment": probe.embodiment, "workspace": ws.to_json()}

    bound = []
    for ep in episodes:
        holder: dict[str, Any] = {}

        def setup(_holder=holder):
            binding = create_backend(spec, **kwargs)
            _holder["primitives"] = binding.primitives
            return binding.backend if ws.is_identity else FramedBackend(binding.backend, ws)

        def primitives(_holder=holder):
            prims = _holder["primitives"]
            return prims if ws.is_identity else [framed_primitive(p, ws) for p in prims]

        clone = copy.copy(ep)
        clone.setup = setup
        clone.primitives = primitives
        clone.metadata = {**ep.metadata, "backend": spec}
        bound.append(clone)
    return bound, provenance


@dataclass
class EpisodePair:
    name: str
    a_passed: bool
    b_passed: bool
    divergence: float | None
    divergence_m: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "episode": self.name,
            "a_passed": self.a_passed,
            "b_passed": self.b_passed,
            "final_position_divergence": self.divergence,
            "final_position_divergence_m": self.divergence_m,
        }


@dataclass
class CrossBackendReport:
    stats: Sim2RealReport
    pairs: list[EpisodePair]
    provenance: dict[str, Any]
    a_results: list[EpisodeResult] = field(repr=False, default_factory=list)
    b_results: list[EpisodeResult] = field(repr=False, default_factory=list)

    def divergence_summary(self) -> dict[str, float | None]:
        vals = [p.divergence for p in self.pairs if p.divergence is not None]
        meters = [p.divergence_m for p in self.pairs if p.divergence_m is not None]
        return {
            "mean": statistics.fmean(vals) if vals else None,
            "max": max(vals) if vals else None,
            "mean_m": statistics.fmean(meters) if meters else None,
            "max_m": max(meters) if meters else None,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "stats": self.stats.to_json(),
            "divergence": self.divergence_summary(),
            "pairs": [p.to_json() for p in self.pairs],
        }

    def render_md(self) -> str:
        prov = self.provenance
        a, b = self.stats.sim_label, self.stats.real_label
        div = self.divergence_summary()
        stats_md = self.stats.render_md().rstrip().replace(
            f"# Sim-to-Real Bench, {a} vs {b}", f"# Cross-backend bench: {a} vs {b}", 1,
        ).replace("**Transfer gap**", "**Pass-rate gap**", 1)
        lines = [stats_md, ""]
        lines.append("## Final-position divergence")
        lines.append("")
        if div["mean"] is None:
            lines.append("(no comparable positions)")
        else:
            lines.append(f"- canonical frame: mean {div['mean']:.4g}, max {div['max']:.4g}")
            if div["mean_m"] is not None:
                lines.append(f"- robot frame: mean {div['mean_m'] * 1000:.2f} mm, max {div['max_m'] * 1000:.2f} mm")
        lines += ["", "## Per-episode", "", f"| Episode | {a} | {b} | divergence |", "|---|:-:|:-:|---:|"]
        for p in self.pairs:
            d = "" if p.divergence is None else f"{p.divergence:.4g}"
            lines.append(f"| {p.name} | {'pass' if p.a_passed else 'FAIL'} | {'pass' if p.b_passed else 'FAIL'} | {d} |")
        lines += ["", "## Provenance", "", "```json", json.dumps(prov, indent=2, sort_keys=True), "```", ""]
        return "\n".join(lines)

    def write(self, out_dir: str | Path) -> Path:
        """Write report.md, report.json and one JSONL trace per episode per side."""
        out = Path(out_dir)
        for side, results in (("a", self.a_results), ("b", self.b_results)):
            trace_dir = out / "traces" / side
            trace_dir.mkdir(parents=True, exist_ok=True)
            for stale in trace_dir.glob("*.jsonl"):
                stale.unlink()
            for res in results:
                res.trace.write_jsonl(str(trace_dir / f"{_safe(res.episode_name)}.jsonl"))
        (out / "report.md").write_text(self.render_md(), encoding="utf-8")
        (out / "report.json").write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return out


def _label(spec: str, kwargs: dict[str, Any] | None) -> str:
    if not kwargs:
        return spec
    return f"{spec}({', '.join(f'{k}={v}' for k, v in sorted(kwargs.items()))})"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_.@" else "_" for c in name)


def _position(res: EpisodeResult, key: str) -> list[float] | None:
    pos = res.final_state.get(key)
    if isinstance(pos, (list, tuple)) and len(pos) == 3:
        return [float(v) for v in pos]
    return None


def cross_backend_compare(
    preset: Callable[..., list[Episode]],
    a: str,
    b: str,
    *,
    seeds: list[int | None] | tuple[int | None, ...] = (None,),
    tolerance: float = 0.01,
    pipeline_factory: Callable[[], PolicyPipeline] | None = None,
    workspace: Workspace | None = None,
    a_kwargs: dict[str, Any] | None = None,
    b_kwargs: dict[str, Any] | None = None,
    preset_name: str | None = None,
) -> CrossBackendReport:
    """Run ``preset`` for every seed on backends ``a`` and ``b`` and pair the results.

    ``tolerance`` is in canonical units; on an embodiment with workspace scale
    0.35, 0.01 canonical is 3.5 mm.
    """
    episodes: list[Episode] = []
    for seed in seeds:
        episodes.extend(preset(seed=seed, tolerance=tolerance))
    if pipeline_factory is not None:
        for ep in episodes:
            ep.pipeline = pipeline_factory()

    a_eps, a_prov = bind_episodes(episodes, a, workspace=workspace, backend_kwargs=a_kwargs)
    b_eps, b_prov = bind_episodes(episodes, b, workspace=workspace, backend_kwargs=b_kwargs)
    if pipeline_factory is not None:
        for ep in b_eps:
            ep.pipeline = pipeline_factory()

    a_label, b_label = _label(a, a_kwargs), _label(b, b_kwargs)
    runner = EpisodeRunner()
    a_results = runner.run_all(a_eps)
    b_results = runner.run_all(b_eps)

    paired = [(ra.passed, rb.passed) for ra, rb in zip(a_results, b_results, strict=True)]
    n = len(paired)
    a_report = summarize(a_results, run_name=a_label, bench_name=preset_name or "cross-backend")
    b_report = summarize(b_results, run_name=b_label, bench_name=preset_name or "cross-backend")
    only_a = sum(1 for x, y in paired if x and not y)
    only_b = sum(1 for x, y in paired if y and not x)
    stats = Sim2RealReport(
        sim_label=a_label,
        real_label=b_label,
        sim_report=a_report,
        real_report=b_report,
        paired_pairs=paired,
        transfer_gap=(only_a - only_b) / n if n else 0.0,
        transfer_gap_ci=_gap_wilson_ci(n, paired),
        mcnemar_p=mcnemar_p(only_a, only_b),
        cohens_h=cohens_h(a_report.rate, b_report.rate),
        cohens_label=effect_label(cohens_h(a_report.rate, b_report.rate)),
        action_kl=_compute_action_kl(a_results, b_results),
        sim_violation_rate=_violation_rate(a_results),
        real_violation_rate=_violation_rate(b_results),
    )

    same_frame = a_prov["workspace"] == b_prov["workspace"]
    pairs = []
    for ra, rb in zip(a_results, b_results):
        pa, pb = _position(ra, "position"), _position(rb, "position")
        ma, mb = _position(ra, "robot_position"), _position(rb, "robot_position")
        pairs.append(EpisodePair(
            name=ra.episode_name,
            a_passed=ra.passed,
            b_passed=rb.passed,
            divergence=math.dist(pa, pb) if pa and pb else None,
            divergence_m=math.dist(ma, mb) if ma and mb and same_frame else None,
        ))

    from .. import __version__

    provenance = {
        "ghostloop_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "preset": preset_name or getattr(preset, "__name__", "custom"),
        "seeds": list(seeds),
        "tolerance_canonical": tolerance,
        "gates": [g.name for g in pipeline_factory().gates] if pipeline_factory else [],
        "n_pairs": n,
        "a": a_prov,
        "b": b_prov,
    }
    return CrossBackendReport(stats=stats, pairs=pairs, provenance=provenance,
                              a_results=a_results, b_results=b_results)
