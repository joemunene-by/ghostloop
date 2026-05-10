"""Trace replay tooling.

A Trace JSONL written by ``Trace.write_jsonl()`` is a header line plus one
JSON-per-event. Replay ingests that file and yields a stream of structured
events for analysis, visualisation, regression tests, or fleet ingestion.
"""

from .diff import StepDiff, TraceDiff, diff_events, diff_traces
from .query import QueryError, compile_query, query
from .replay import (
    ReplayedEvent,
    TraceHeader,
    iter_events,
    load_trace,
    summarize_trace,
)

__all__ = [
    "ReplayedEvent",
    "StepDiff",
    "TraceDiff",
    "TraceHeader",
    "QueryError",
    "compile_query",
    "diff_events",
    "diff_traces",
    "iter_events",
    "load_trace",
    "query",
    "summarize_trace",
]
