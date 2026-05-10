"""Trace diff: side-by-side comparison of two traces.

Operates on the structured ReplayedEvent stream from replay.load_trace.
The diff is sequential — step N of trace A is compared to step N of
trace B. Three categories per step:

  identical          same intent, same args, same result status
  diverged           same step number, different intent / args / status
  only_a / only_b    one trace has more steps than the other

Used for: regression debugging ("v0.5 vs v0.4 traces of the same
episode"), policy A/B comparisons at the per-step level, post-incident
analysis ("where did the runs diverge").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .replay import ReplayedEvent, load_trace


@dataclass
class StepDiff:
    """One row of the diff table."""

    step: int
    kind: str  # identical | diverged | only_a | only_b
    a: ReplayedEvent | None
    b: ReplayedEvent | None
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        def _ev(e: ReplayedEvent | None) -> dict | None:
            if e is None:
                return None
            return {
                "intent": e.intent_name,
                "args": e.intent_args,
                "status": e.result_status,
            }
        return {
            "step": self.step,
            "kind": self.kind,
            "a": _ev(self.a),
            "b": _ev(self.b),
            "reasons": self.reasons,
        }


@dataclass
class TraceDiff:
    """Result of comparing two traces."""

    a_path: str
    b_path: str
    a_episode_id: str
    b_episode_id: str
    n_a: int
    n_b: int
    steps: list[StepDiff]

    @property
    def n_identical(self) -> int:
        return sum(1 for s in self.steps if s.kind == "identical")

    @property
    def n_diverged(self) -> int:
        return sum(1 for s in self.steps if s.kind == "diverged")

    @property
    def n_only_a(self) -> int:
        return sum(1 for s in self.steps if s.kind == "only_a")

    @property
    def n_only_b(self) -> int:
        return sum(1 for s in self.steps if s.kind == "only_b")

    def first_divergence(self) -> StepDiff | None:
        for s in self.steps:
            if s.kind != "identical":
                return s
        return None

    def render_md(self) -> str:
        first = self.first_divergence()
        first_marker = (
            f"step {first.step} ({first.kind})" if first else "no divergence"
        )
        lines = [
            "# Trace diff",
            "",
            f"A: `{self.a_episode_id}` ({self.n_a} steps, {self.a_path})",
            f"B: `{self.b_episode_id}` ({self.n_b} steps, {self.b_path})",
            "",
            f"Identical: {self.n_identical}  ·  Diverged: {self.n_diverged}  "
            f"·  only-A: {self.n_only_a}  ·  only-B: {self.n_only_b}",
            f"First divergence: {first_marker}",
            "",
            "| Step | Kind | A | B | Reason |",
            "|---:|---|---|---|---|",
        ]
        for s in self.steps:
            a_desc = f"{s.a.intent_name}({s.a.intent_args}) -> {s.a.result_status}" if s.a else "—"
            b_desc = f"{s.b.intent_name}({s.b.intent_args}) -> {s.b.result_status}" if s.b else "—"
            reason = "; ".join(s.reasons) or ""
            marker = {
                "identical": "✓",
                "diverged": "≠",
                "only_a": "←",
                "only_b": "→",
            }[s.kind]
            lines.append(f"| {s.step} | {marker} {s.kind} | {a_desc} | {b_desc} | {reason} |")
        return "\n".join(lines) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "a_path": self.a_path,
            "b_path": self.b_path,
            "a_episode_id": self.a_episode_id,
            "b_episode_id": self.b_episode_id,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_identical": self.n_identical,
            "n_diverged": self.n_diverged,
            "n_only_a": self.n_only_a,
            "n_only_b": self.n_only_b,
            "steps": [s.to_json() for s in self.steps],
        }


def diff_events(
    a_events: list[ReplayedEvent],
    b_events: list[ReplayedEvent],
) -> list[StepDiff]:
    """Step-by-step diff of two event lists. Sequential index alignment."""
    out: list[StepDiff] = []
    n = max(len(a_events), len(b_events))
    for i in range(n):
        a = a_events[i] if i < len(a_events) else None
        b = b_events[i] if i < len(b_events) else None
        if a is None:
            out.append(StepDiff(step=i + 1, kind="only_b", a=None, b=b))
            continue
        if b is None:
            out.append(StepDiff(step=i + 1, kind="only_a", a=a, b=None))
            continue
        reasons: list[str] = []
        if a.intent_name != b.intent_name:
            reasons.append(f"intent {a.intent_name!r} vs {b.intent_name!r}")
        if a.intent_args != b.intent_args:
            reasons.append("args differ")
        if a.result_status != b.result_status:
            reasons.append(f"status {a.result_status!r} vs {b.result_status!r}")
        kind = "identical" if not reasons else "diverged"
        out.append(StepDiff(step=i + 1, kind=kind, a=a, b=b, reasons=reasons))
    return out


def diff_traces(a_path: str | Path, b_path: str | Path) -> TraceDiff:
    """Load two traces from disk and return a TraceDiff."""
    a_header, a_events = load_trace(a_path)
    b_header, b_events = load_trace(b_path)
    return TraceDiff(
        a_path=str(a_path),
        b_path=str(b_path),
        a_episode_id=a_header.episode_id,
        b_episode_id=b_header.episode_id,
        n_a=len(a_events),
        n_b=len(b_events),
        steps=diff_events(a_events, b_events),
    )
