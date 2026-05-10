"""OpenTelemetry hooks — every Runtime.step emits a span when OTel is configured.

Conditional import: package itself never depends on opentelemetry. If the
user has OTel installed AND has called ``configure_otel(...)`` (or set the
standard OTEL_* env vars), ghostloop emits structured spans for each step
with intent / decision / result attributes plus exception recording.

If OTel isn't installed or hasn't been configured, the hook is a no-op
context manager so the runtime stays fast in dev / sim.

  pip install ghostloop[otel]
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from ..core import Decision, Intent, Result


def otel_available() -> bool:
    try:
        from opentelemetry import trace  # noqa: F401
        return True
    except ImportError:
        return False


_TRACER = None


def configure_otel(service_name: str = "ghostloop") -> None:
    """Attach a Tracer to ghostloop. Call once at process startup.

    Reads OTEL_EXPORTER_OTLP_ENDPOINT etc from env via the standard SDK
    auto-configuration, so any production OTel setup (Honeycomb / Jaeger /
    Grafana Tempo / Datadog) just works.
    """
    global _TRACER
    if not otel_available():
        return
    from opentelemetry import trace  # type: ignore
    _TRACER = trace.get_tracer(service_name)


def _ensure_tracer():
    """Lazy auto-attach if the user has OTel installed but never called configure_otel."""
    global _TRACER
    if _TRACER is None and otel_available():
        from opentelemetry import trace  # type: ignore
        _TRACER = trace.get_tracer("ghostloop")
    return _TRACER


@contextmanager
def step_span(intent: Intent):
    """Emit an OTel span around one runtime.step.

    Usage in a runtime is::

        with step_span(intent) as span:
            ... do the work ...
            record_decision(span, decision)
            record_result(span, result)

    No-op when OTel isn't configured.
    """
    tracer = _ensure_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(f"ghostloop.step:{intent.name}") as span:
        span.set_attribute("ghostloop.intent.name", intent.name)
        span.set_attribute("ghostloop.intent.args.json",
                           __import__("json").dumps(intent.args))
        if intent.rationale:
            span.set_attribute("ghostloop.intent.rationale", intent.rationale)
        try:
            yield span
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            raise


def record_decision(span: Any, decision: Decision) -> None:
    if span is None:
        return
    span.set_attribute("ghostloop.decision.action", decision.action.value)
    span.set_attribute("ghostloop.decision.gate", decision.gate_name)
    if decision.reason:
        span.set_attribute("ghostloop.decision.reason", decision.reason)


def record_result(span: Any, result: Result) -> None:
    if span is None:
        return
    span.set_attribute("ghostloop.result.status", result.status.value)
    span.set_attribute("ghostloop.result.duration_ms", result.duration_ms)
    if result.message:
        span.set_attribute("ghostloop.result.message", result.message)
