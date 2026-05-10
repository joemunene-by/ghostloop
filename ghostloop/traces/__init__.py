"""Trace replay tooling.

A Trace JSONL written by ``Trace.write_jsonl()`` is a header line plus one
JSON-per-event. Replay ingests that file and yields a stream of structured
events for analysis, visualisation, regression tests, or fleet ingestion.
"""

from .replay import (
    ReplayedEvent,
    TraceHeader,
    iter_events,
    load_trace,
    summarize_trace,
)

__all__ = [
    "ReplayedEvent",
    "TraceHeader",
    "iter_events",
    "load_trace",
    "summarize_trace",
]
