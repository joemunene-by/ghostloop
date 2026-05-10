"""Energy ledger — joules-per-primitive accounting over a Trace.

Real deployments care about energy. A nightly inspection robot that
drains its battery in 4 hours is useless; one that lasts 14 hours
gets you through the shift. Despite that, no robotics framework I
know of surfaces energy in the trace path. ``EnergyLedger`` does.

Each ``PrimitiveEnergyModel`` declares an estimator function for one
primitive name: given the args and result, how many joules did this
call consume? The estimator can be:

  - constant (a fixed J per call, e.g. "scan = 0.5 J")
  - linear in a numeric arg (e.g. move_to consumes ``k * |delta|``)
  - linear in duration (e.g. continuous-running primitives)
  - whatever closure you want — the API is just a callable

``EnergyEstimator`` aggregates models and assigns a fallback value
to unmodelled primitives. ``EnergyLedger`` walks a Trace and produces
a per-primitive breakdown plus the episode total.

Useful pairings:
  - Sim2RealBench: compare energy budgets sim-vs-real.
  - Property mining: emit "episode never exceeds X joules" candidates.
  - Reward shaping: use total energy as a negative reward term for
    energy-aware RL (RewardShaper has CustomReward — adapt).

Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core import Intent, Result, Trace


EnergyFn = Callable[[Intent, Result], float]


@dataclass
class PrimitiveEnergyModel:
    """One model: name + estimator callable + units.

    For convenience, a few helper constructors:
      - constant(name, joules)
      - linear_in_arg(name, arg, slope, intercept)
      - linear_in_duration(name, watts, duration_arg)
    """

    primitive_name: str
    estimator: EnergyFn
    description: str = ""

    @classmethod
    def constant(cls, name: str, joules: float) -> "PrimitiveEnergyModel":
        def _fn(intent: Intent, result: Result) -> float:
            return float(joules)
        return cls(
            primitive_name=name, estimator=_fn,
            description=f"constant {joules:g} J per call",
        )

    @classmethod
    def linear_in_arg(
        cls, name: str, arg: str, slope: float, intercept: float = 0.0,
    ) -> "PrimitiveEnergyModel":
        def _fn(intent: Intent, result: Result) -> float:
            try:
                v = float(intent.args.get(arg, 0.0))
            except (TypeError, ValueError):
                v = 0.0
            return slope * abs(v) + intercept
        return cls(
            primitive_name=name, estimator=_fn,
            description=(
                f"{slope:g} J per unit |{arg}| + {intercept:g} J base"
            ),
        )

    @classmethod
    def linear_in_duration(
        cls, name: str, watts: float, duration_arg: str = "duration",
    ) -> "PrimitiveEnergyModel":
        def _fn(intent: Intent, result: Result) -> float:
            try:
                t = float(intent.args.get(duration_arg, 0.0))
            except (TypeError, ValueError):
                t = 0.0
            return watts * max(0.0, t)
        return cls(
            primitive_name=name, estimator=_fn,
            description=f"{watts:g} W * {duration_arg}",
        )

    @classmethod
    def linear_in_xyz_step(
        cls, name: str, joules_per_meter: float = 1.0, base: float = 0.0,
    ) -> "PrimitiveEnergyModel":
        """For move_to-style primitives: J = base + k * sqrt(x^2+y^2+z^2)."""
        def _fn(intent: Intent, result: Result) -> float:
            try:
                x = float(intent.args.get("x", 0.0))
                y = float(intent.args.get("y", 0.0))
                z = float(intent.args.get("z", 0.0))
            except (TypeError, ValueError):
                return base
            mag = (x * x + y * y + z * z) ** 0.5
            return base + joules_per_meter * mag
        return cls(
            primitive_name=name, estimator=_fn,
            description=(
                f"{base:g} J base + {joules_per_meter:g} J per metre |xyz|"
            ),
        )


@dataclass
class EnergyEstimator:
    """Holds a registry of PrimitiveEnergyModels with a fallback default."""

    models: dict[str, PrimitiveEnergyModel] = field(default_factory=dict)
    fallback_joules: float = 0.1

    def register(self, model: PrimitiveEnergyModel) -> None:
        self.models[model.primitive_name] = model

    def estimate(self, intent: Intent, result: Result) -> float:
        m = self.models.get(intent.name)
        if m is None:
            return float(self.fallback_joules)
        try:
            return float(m.estimator(intent, result))
        except Exception:  # noqa: BLE001
            return float(self.fallback_joules)


def default_estimator() -> EnergyEstimator:
    """Sane defaults for the primitives shipped with ghostloop."""
    est = EnergyEstimator(fallback_joules=0.05)
    est.register(PrimitiveEnergyModel.linear_in_xyz_step(
        "move_to", joules_per_meter=2.0, base=0.05,
    ))
    est.register(PrimitiveEnergyModel.constant("scan", 0.2))
    est.register(PrimitiveEnergyModel.constant("pick", 1.5))
    est.register(PrimitiveEnergyModel.constant("place", 1.5))
    est.register(PrimitiveEnergyModel.linear_in_duration(
        "wait", watts=0.01,
    ))
    return est


@dataclass
class EnergyLedger:
    """Aggregate energy consumption across a Trace."""

    estimator: EnergyEstimator = field(default_factory=default_estimator)

    def total(self, trace: Trace) -> float:
        return sum(
            self.estimator.estimate(ev.intent, ev.result)
            for ev in trace.events
            # Only count events that actually executed (not denied).
            if ev.decision.action.value == "allow"
        )

    def by_primitive(self, trace: Trace) -> dict[str, float]:
        out: dict[str, float] = {}
        for ev in trace.events:
            if ev.decision.action.value != "allow":
                continue
            j = self.estimator.estimate(ev.intent, ev.result)
            out[ev.intent.name] = out.get(ev.intent.name, 0.0) + j
        return out

    def per_step(self, trace: Trace) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for ev in trace.events:
            if ev.decision.action.value != "allow":
                continue
            j = self.estimator.estimate(ev.intent, ev.result)
            out.append({
                "step": ev.step,
                "primitive": ev.intent.name,
                "joules": round(j, 6),
            })
        return out

    def render_md(self, trace: Trace) -> str:
        total = self.total(trace)
        breakdown = self.by_primitive(trace)
        lines = [
            f"# Energy ledger — episode `{trace.episode_id}`",
            "",
            f"- Total: **{total:.4g} J**",
            f"- Events counted: {sum(1 for e in trace.events if e.decision.action.value == 'allow')}",
            "",
            "## Per primitive",
            "",
            "| Primitive | Joules | % |",
            "|---|---:|---:|",
        ]
        for name, j in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            pct = (j / total * 100) if total > 0 else 0
            lines.append(f"| {name} | {j:.4g} | {pct:.1f}% |")
        return "\n".join(lines) + "\n"
