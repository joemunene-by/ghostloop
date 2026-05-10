"""Read trace JSONL back into structured events. The inverse of Trace.write_jsonl."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class TraceHeader:
    """First-line header from a trace JSONL."""

    episode_id: str
    backend: str
    started_at: float
    n_steps: int
    raw: dict[str, Any]


@dataclass
class ReplayedEvent:
    """One step parsed back from a trace JSONL event line."""

    step: int
    timestamp: float
    intent_name: str
    intent_args: dict[str, Any]
    intent_rationale: str
    decision_action: str
    decision_reason: str
    decision_gate_name: str
    result_status: str
    result_message: str
    result_observation: dict[str, Any]
    result_duration_ms: float
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReplayedEvent":
        intent = d.get("intent") or {}
        decision = d.get("decision") or {}
        result = d.get("result") or {}
        return cls(
            step=int(d.get("step", 0)),
            timestamp=float(d.get("timestamp", 0.0)),
            intent_name=intent.get("name", ""),
            intent_args=intent.get("args") or {},
            intent_rationale=intent.get("rationale", ""),
            decision_action=decision.get("action", ""),
            decision_reason=decision.get("reason", ""),
            decision_gate_name=decision.get("gate_name", ""),
            result_status=result.get("status", ""),
            result_message=result.get("message", ""),
            result_observation=result.get("observation") or {},
            result_duration_ms=float(result.get("duration_ms", 0.0)),
            state_before=d.get("state_before") or {},
            state_after=d.get("state_after") or {},
            raw=d,
        )


def load_trace(path: str | Path) -> tuple[TraceHeader, list[ReplayedEvent]]:
    """Read a JSONL trace file completely into memory."""
    p = Path(path)
    header: TraceHeader | None = None
    events: list[ReplayedEvent] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0:
                header = TraceHeader(
                    episode_id=obj.get("episode_id", ""),
                    backend=obj.get("backend", ""),
                    started_at=float(obj.get("started_at", 0.0)),
                    n_steps=int(obj.get("n_steps", 0)),
                    raw=obj,
                )
            else:
                events.append(ReplayedEvent.from_dict(obj))
    if header is None:
        raise ValueError(f"empty trace file: {path}")
    return header, events


def iter_events(path: str | Path) -> Iterator[ReplayedEvent]:
    """Stream events one at a time without loading the whole file (large traces)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i == 0:
                continue  # skip header
            yield ReplayedEvent.from_dict(obj)


def summarize_trace(path: str | Path) -> dict[str, Any]:
    """High-level summary suitable for fleet dashboards / CI gates."""
    header, events = load_trace(path)
    statuses: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    deny_reasons: list[str] = []
    total_duration_ms = 0.0
    for ev in events:
        statuses[ev.result_status] += 1
        intent_counts[ev.intent_name] += 1
        if ev.decision_action == "deny":
            deny_reasons.append(f"{ev.decision_gate_name}: {ev.decision_reason}")
        total_duration_ms += ev.result_duration_ms
    return {
        "episode_id": header.episode_id,
        "backend": header.backend,
        "n_events": len(events),
        "by_status": dict(statuses),
        "by_intent": dict(intent_counts),
        "denied": len(deny_reasons),
        "deny_reasons": deny_reasons,
        "total_duration_ms": round(total_duration_ms, 3),
    }
