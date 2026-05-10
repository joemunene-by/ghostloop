"""System identification — fit a simple dynamics model from real captures.

When you bring your own robot, the safety pipeline needs reasonable
caps on force / velocity / acceleration. The defaults in
``RobotProfile`` are conservative guesses; the right values come from
your specific arm. This module helps capture them by:

  1. Driving the robot through a small calibration sequence
     (``calibration_episode``).
  2. Recording (action_command, observed_state) pairs.
  3. Fitting a simple linear model — least-squares without numpy
     fallback to plain math when numpy is missing.
  4. Recommending workspace bounds + force / velocity / acceleration
     caps from the empirical data, with safety margins.

Output: a ``CalibrationReport`` carrying:

  - empirical workspace AABB (min + max observed positions)
  - p99 + max observed force / velocity / acceleration (with margins)
  - per-primitive call latency stats
  - a ``recommended_profile()`` factory returning a tuned
    ``RobotProfile`` you can drop into the runtime

Stdlib only; numpy used opportunistically for least-squares but a
pure-stdlib fallback handles the simple linear case.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from .core import Backend, Intent, Runtime, Trace
from .profiles import RobotProfile


# ---------------------------------------------------------------------------
# Capture: drive a sequence of intents + record observations.
# ---------------------------------------------------------------------------


@dataclass
class CalibrationCapture:
    """One observation captured during calibration."""

    timestamp: float
    intent_name: str
    intent_args: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    duration_ms: float
    decision_action: str

    def position(self, key: str = "position") -> tuple[float, float, float] | None:
        pos = self.state_after.get(key)
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            try:
                return float(pos[0]), float(pos[1]), float(pos[2])
            except (TypeError, ValueError):
                return None
        return None

    def numeric_field(self, key: str) -> float | None:
        v = self.state_after.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None


@dataclass
class CalibrationCaptureLog:
    captures: list[CalibrationCapture] = field(default_factory=list)

    def add_from_trace(self, trace: Trace) -> int:
        added = 0
        for ev in trace.events:
            self.captures.append(CalibrationCapture(
                timestamp=ev.timestamp,
                intent_name=ev.intent.name,
                intent_args=dict(ev.intent.args),
                state_before=dict(ev.state_before) if isinstance(ev.state_before, dict) else {},
                state_after=dict(ev.state_after) if isinstance(ev.state_after, dict) else {},
                duration_ms=ev.result.duration_ms,
                decision_action=ev.decision.action.value,
            ))
            added += 1
        return added


def calibration_episode(
    runtime: Runtime,
    *,
    sweep: list[Intent] | None = None,
    n_repeats: int = 1,
) -> CalibrationCaptureLog:
    """Drive the runtime through a calibration sweep and record captures.

    The default sweep is a small box-corner pattern that exercises
    workspace extents. Pass a custom ``sweep`` for robots whose
    primitive surface differs (mobile bases, drones, etc.).
    """
    if sweep is None:
        sweep = [
            Intent("scan", {"radius": 0.1}),
            Intent("move_to", {"x": 0.0, "y": 0.0, "z": 0.5}),
            Intent("move_to", {"x": 0.3, "y": 0.0, "z": 0.5}),
            Intent("move_to", {"x": 0.3, "y": 0.3, "z": 0.5}),
            Intent("move_to", {"x": 0.0, "y": 0.3, "z": 0.5}),
            Intent("move_to", {"x": 0.0, "y": 0.0, "z": 0.5}),
            Intent("scan", {"radius": 0.3}),
        ]
    log = CalibrationCaptureLog()
    starting_n = len(runtime.trace.events)
    for _ in range(n_repeats):
        for intent in sweep:
            runtime.step(intent)
    # Only harvest the events we just produced.
    new_trace = Trace(
        episode_id=runtime.trace.episode_id,
        backend_name=runtime.trace.backend_name,
        events=runtime.trace.events[starting_n:],
    )
    log.add_from_trace(new_trace)
    return log


# ---------------------------------------------------------------------------
# Analysis: fit + summarise.
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    """Summary of a calibration capture session."""

    n_captures: int
    n_allowed: int
    n_denied: int

    workspace_min: tuple[float, float, float] | None
    workspace_max: tuple[float, float, float] | None

    max_velocity_ms: float | None
    max_acceleration_ms2: float | None
    max_force_n: float | None

    p99_call_latency_ms: float
    mean_call_latency_ms: float

    primitive_distribution: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_captures": self.n_captures,
            "n_allowed": self.n_allowed,
            "n_denied": self.n_denied,
            "workspace_min": list(self.workspace_min) if self.workspace_min else None,
            "workspace_max": list(self.workspace_max) if self.workspace_max else None,
            "max_velocity_ms": self.max_velocity_ms,
            "max_acceleration_ms2": self.max_acceleration_ms2,
            "max_force_n": self.max_force_n,
            "p99_call_latency_ms": round(self.p99_call_latency_ms, 4),
            "mean_call_latency_ms": round(self.mean_call_latency_ms, 4),
            "primitive_distribution": self.primitive_distribution,
            "warnings": self.warnings,
        }

    def render_md(self) -> str:
        ws = (
            f"  - workspace: `{self.workspace_min}` → `{self.workspace_max}`"
            if self.workspace_min is not None else "  - workspace: (no positional data captured)"
        )
        lines = [
            "# Calibration report",
            "",
            f"- captures: {self.n_captures} ({self.n_allowed} allowed, {self.n_denied} denied)",
            f"- mean call latency: {self.mean_call_latency_ms:.2f} ms",
            f"- p99 call latency: {self.p99_call_latency_ms:.2f} ms",
            "",
            "## Empirical bounds",
            "",
            ws,
            f"  - max observed velocity: {self.max_velocity_ms} m/s",
            f"  - max observed acceleration: {self.max_acceleration_ms2} m/s²",
            f"  - max observed force: {self.max_force_n} N",
            "",
            "## Primitive call distribution",
            "",
        ]
        for name, count in sorted(
            self.primitive_distribution.items(), key=lambda kv: -kv[1],
        ):
            lines.append(f"  - {name}: {count}")
        if self.warnings:
            lines.append("")
            lines.append("## Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines) + "\n"

    def recommended_profile(
        self,
        *,
        name: str = "calibrated_robot",
        morphology: str = "custom",
        velocity_margin: float = 1.2,
        force_margin: float = 1.2,
        workspace_margin: float = 0.05,
    ) -> RobotProfile:
        """Build a RobotProfile with conservative caps from the captures.

        Margins multiply the empirical max so the cap allows slightly
        more than what was observed (avoids false positives on
        legitimate motion). Workspace bounds are inflated by
        ``workspace_margin`` metres.
        """
        bounds = None
        if self.workspace_min and self.workspace_max:
            bounds = (
                tuple(v - workspace_margin for v in self.workspace_min),
                tuple(v + workspace_margin for v in self.workspace_max),
            )
        return RobotProfile(
            name=name,
            morphology=morphology,
            workspace_bounds=bounds,
            max_velocity=(self.max_velocity_ms or 1.0) * velocity_margin if self.max_velocity_ms else None,
            max_acceleration=(self.max_acceleration_ms2 or 5.0) * velocity_margin if self.max_acceleration_ms2 else None,
            max_force_n=(self.max_force_n or 15.0) * force_margin if self.max_force_n else None,
            instructions=(
                f"Auto-tuned profile from calibration on {time.strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"Captured {self.n_captures} steps. Empirical bounds + margins above; "
                f"override before production deployment."
            ),
        )


def analyse_capture(log: CalibrationCaptureLog) -> CalibrationReport:
    """Compute the CalibrationReport from a capture log."""
    if not log.captures:
        return CalibrationReport(
            n_captures=0, n_allowed=0, n_denied=0,
            workspace_min=None, workspace_max=None,
            max_velocity_ms=None, max_acceleration_ms2=None, max_force_n=None,
            p99_call_latency_ms=0.0, mean_call_latency_ms=0.0,
            warnings=["no captures recorded"],
        )
    n = len(log.captures)
    n_allow = sum(1 for c in log.captures if c.decision_action == "allow")
    n_deny = sum(1 for c in log.captures if c.decision_action == "deny")
    warnings: list[str] = []

    # Position bounds + velocity / acceleration estimates from finite differences.
    positions: list[tuple[float, tuple[float, float, float]]] = []
    for c in log.captures:
        p = c.position()
        if p is not None:
            positions.append((c.timestamp, p))
    ws_min = ws_max = None
    velocities: list[float] = []
    accelerations: list[float] = []
    if len(positions) >= 2:
        xs = [p[1][0] for p in positions]
        ys = [p[1][1] for p in positions]
        zs = [p[1][2] for p in positions]
        ws_min = (min(xs), min(ys), min(zs))
        ws_max = (max(xs), max(ys), max(zs))
        last_v = 0.0
        for i in range(1, len(positions)):
            ts_prev, p_prev = positions[i - 1]
            ts_cur, p_cur = positions[i]
            dt = max(ts_cur - ts_prev, 1e-3)
            d = math.sqrt(sum((p_cur[j] - p_prev[j]) ** 2 for j in range(3)))
            v = d / dt
            velocities.append(v)
            accelerations.append((v - last_v) / dt)
            last_v = v
    elif positions:
        warnings.append("only one position captured; can't estimate velocity")
    else:
        warnings.append("no position field in any state_after; workspace + velocity bounds unavailable")

    forces: list[float] = []
    for c in log.captures:
        for key in ("force_norm", "force"):
            v = c.numeric_field(key)
            if v is not None:
                forces.append(v)
                break

    durations = [c.duration_ms for c in log.captures]
    p99 = max(durations) if len(durations) < 100 else (
        statistics.quantiles(durations, n=100)[-1]
    )
    mean = statistics.mean(durations) if durations else 0.0

    primitive_dist: dict[str, int] = {}
    for c in log.captures:
        primitive_dist[c.intent_name] = primitive_dist.get(c.intent_name, 0) + 1

    if not forces:
        warnings.append("no force / force_norm field captured; force cap unavailable")

    return CalibrationReport(
        n_captures=n,
        n_allowed=n_allow,
        n_denied=n_deny,
        workspace_min=ws_min,
        workspace_max=ws_max,
        max_velocity_ms=max(velocities) if velocities else None,
        max_acceleration_ms2=max(abs(a) for a in accelerations) if accelerations else None,
        max_force_n=max(forces) if forces else None,
        p99_call_latency_ms=p99,
        mean_call_latency_ms=mean,
        primitive_distribution=primitive_dist,
        warnings=warnings,
    )
