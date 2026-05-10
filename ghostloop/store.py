"""SQLite-backed episode + run-report store.

Single-file dependency-free persistence so a fleet operator can ask
"how have my policies performed over time" without standing up Postgres.
sqlite3 is in the stdlib; the schema below is forward-compatible (new
columns get added via ALTER TABLE migrations on open).

Tables:
  episodes      — every Trace ever ingested. Full JSONL stored in a TEXT
                  column for replay; metadata extracted into indexed
                  columns for fast filtering.
  run_reports   — every RunReport ever scored. Per-episode pass/fail in
                  the JSON blob; aggregate rate / Wilson CI in indexed
                  columns for dashboards.
  comparisons   — every PairedComparison ever computed.

The store is content-addressed: episode_id, run_name+bench_name, and
comparison_id are unique. Re-ingesting the same artefact is a no-op.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .core import Trace


SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    started_at REAL NOT NULL,
    n_steps INTEGER NOT NULL,
    n_blocked INTEGER NOT NULL,
    n_errored INTEGER NOT NULL,
    body_jsonl TEXT NOT NULL,
    ingested_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_episodes_backend ON episodes(backend);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at DESC);

CREATE TABLE IF NOT EXISTS run_reports (
    run_id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    bench_name TEXT NOT NULL,
    n INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    rate REAL NOT NULL,
    ci_low REAL NOT NULL,
    ci_high REAL NOT NULL,
    body_json TEXT NOT NULL,
    ingested_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
    UNIQUE(run_name, bench_name, ingested_at)
);
CREATE INDEX IF NOT EXISTS idx_runs_bench ON run_reports(bench_name);
CREATE INDEX IF NOT EXISTS idx_runs_ingested ON run_reports(ingested_at DESC);

CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id TEXT PRIMARY KEY,
    bench_name TEXT NOT NULL,
    a_name TEXT NOT NULL,
    b_name TEXT NOT NULL,
    n INTEGER NOT NULL,
    a_passed INTEGER NOT NULL,
    b_passed INTEGER NOT NULL,
    p_value REAL NOT NULL,
    effect_h REAL NOT NULL,
    significant INTEGER NOT NULL,
    body_json TEXT NOT NULL,
    ingested_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_comparisons_bench ON comparisons(bench_name);
"""


@dataclass
class EpisodeRow:
    episode_id: str
    backend: str
    started_at: float
    n_steps: int
    n_blocked: int
    n_errored: int

    @classmethod
    def from_db(cls, row: sqlite3.Row) -> "EpisodeRow":
        return cls(
            episode_id=row["episode_id"],
            backend=row["backend"],
            started_at=row["started_at"],
            n_steps=row["n_steps"],
            n_blocked=row["n_blocked"],
            n_errored=row["n_errored"],
        )


@dataclass
class RunRow:
    run_id: str
    run_name: str
    bench_name: str
    n: int
    passed: int
    rate: float
    ci_low: float
    ci_high: float

    @classmethod
    def from_db(cls, row: sqlite3.Row) -> "RunRow":
        return cls(
            run_id=row["run_id"],
            run_name=row["run_name"],
            bench_name=row["bench_name"],
            n=row["n"],
            passed=row["passed"],
            rate=row["rate"],
            ci_low=row["ci_low"],
            ci_high=row["ci_high"],
        )


class GhostloopStore:
    """SQLite-backed store for episodes, run reports, and paired comparisons.

    Use as a context manager OR construct, call methods, ``close()``.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def __enter__(self) -> "GhostloopStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Episodes.
    # ------------------------------------------------------------------

    def save_episode(self, trace: Trace) -> str:
        """Persist a trace; returns the episode_id."""
        body = trace.to_json()
        n_blocked = sum(1 for ev in trace.events if ev.result.status.value == "blocked")
        n_errored = sum(1 for ev in trace.events if ev.result.status.value == "error")
        # Body stored as a tiny JSONL: header line + one event line each.
        lines = [json.dumps({
            "episode_id": trace.episode_id,
            "backend": trace.backend_name,
            "started_at": trace.started_at,
            "n_steps": len(trace.events),
        })] + [json.dumps(ev.to_json()) for ev in trace.events]
        body_jsonl = "\n".join(lines)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes
              (episode_id, backend, started_at, n_steps, n_blocked, n_errored, body_jsonl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (trace.episode_id, trace.backend_name, trace.started_at,
             len(trace.events), n_blocked, n_errored, body_jsonl),
        )
        self._conn.commit()
        return trace.episode_id

    def list_episodes(
        self,
        *,
        backend: str | None = None,
        limit: int = 100,
    ) -> list[EpisodeRow]:
        cur = self._conn.cursor()
        if backend:
            cur.execute(
                "SELECT * FROM episodes WHERE backend = ? "
                "ORDER BY started_at DESC LIMIT ?",
                (backend, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
        return [EpisodeRow.from_db(r) for r in cur.fetchall()]

    def load_episode(self, episode_id: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute("SELECT body_jsonl FROM episodes WHERE episode_id = ?",
                    (episode_id,))
        row = cur.fetchone()
        if row is None:
            return None
        lines = row["body_jsonl"].splitlines()
        header = json.loads(lines[0])
        events = [json.loads(line) for line in lines[1:] if line.strip()]
        return {**header, "events": events}

    # ------------------------------------------------------------------
    # Run reports.
    # ------------------------------------------------------------------

    def save_run_report(self, report: "RunReportLike") -> str:
        """Persist a RunReport (anything with .run_name/.bench_name/.to_json/.ci/.passed/.n/.rate).

        Returns a generated run_id. We accept the duck-typed shape so the
        store doesn't need a hard import on bench/.
        """
        import uuid
        run_id = str(uuid.uuid4())
        ci_lo, ci_hi = report.ci
        body_json = json.dumps(report.to_json())
        self._conn.execute(
            """
            INSERT INTO run_reports
              (run_id, run_name, bench_name, n, passed, rate, ci_low, ci_high, body_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, report.run_name, report.bench_name, report.n,
             report.passed, report.rate, ci_lo, ci_hi, body_json),
        )
        self._conn.commit()
        return run_id

    def list_runs(self, *, bench_name: str | None = None, limit: int = 100) -> list[RunRow]:
        cur = self._conn.cursor()
        if bench_name:
            cur.execute(
                "SELECT * FROM run_reports WHERE bench_name = ? "
                "ORDER BY ingested_at DESC LIMIT ?",
                (bench_name, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM run_reports ORDER BY ingested_at DESC LIMIT ?",
                (limit,),
            )
        return [RunRow.from_db(r) for r in cur.fetchall()]

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute("SELECT body_json FROM run_reports WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        return json.loads(row["body_json"]) if row else None

    # ------------------------------------------------------------------
    # Paired comparisons.
    # ------------------------------------------------------------------

    def save_comparison(self, comp: "ComparisonLike") -> str:
        import uuid
        comp_id = str(uuid.uuid4())
        body_json = json.dumps({
            "bench_name": comp.bench_name,
            "a_name": comp.a_name, "b_name": comp.b_name,
            "n": comp.n,
            "a_passed": comp.a_passed, "b_passed": comp.b_passed,
            "both_passed": comp.both_passed,
            "only_a": comp.only_a, "only_b": comp.only_b,
            "neither": comp.neither,
            "p_value": comp.p_value, "effect_h": comp.effect_h,
            "significant": comp.significant,
        })
        self._conn.execute(
            """
            INSERT INTO comparisons
              (comparison_id, bench_name, a_name, b_name, n,
               a_passed, b_passed, p_value, effect_h, significant, body_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (comp_id, comp.bench_name, comp.a_name, comp.b_name, comp.n,
             comp.a_passed, comp.b_passed, comp.p_value, comp.effect_h,
             1 if comp.significant else 0, body_json),
        )
        self._conn.commit()
        return comp_id

    def stats(self) -> dict[str, int]:
        cur = self._conn.cursor()
        out = {}
        for table in ("episodes", "run_reports", "comparisons"):
            cur.execute(f"SELECT COUNT(*) AS c FROM {table}")
            out[table] = cur.fetchone()["c"]
        return out


# Duck-typed protocols — store stays usable even if bench/ ever moves.
class RunReportLike(Protocol):
    run_name: str
    bench_name: str
    n: int
    passed: int
    rate: float
    ci: tuple[float, float]
    def to_json(self) -> dict[str, Any]: ...


class ComparisonLike(Protocol):
    bench_name: str
    a_name: str
    b_name: str
    n: int
    a_passed: int
    b_passed: int
    both_passed: int
    only_a: int
    only_b: int
    neither: int
    p_value: float
    effect_h: float
    significant: bool
