"""Telemetry — OpenTelemetry hooks + energy ledger.

Two modules ship under telemetry/:

  - otel.py: OpenTelemetry tracer / span instrumentation (was the
    flat ghostloop/telemetry.py module before v0.10).
  - energy.py: per-primitive joule estimates, episode-level energy
    ledger, integration with traces.
"""

from .energy import (
    EnergyEstimator,
    EnergyLedger,
    PrimitiveEnergyModel,
    default_estimator,
)
from .otel import (
    configure_otel,
    otel_available,
    record_decision,
    record_result,
    step_span,
)

__all__ = [
    "EnergyEstimator",
    "EnergyLedger",
    "PrimitiveEnergyModel",
    "default_estimator",
    "configure_otel",
    "otel_available",
    "record_decision",
    "record_result",
    "step_span",
]
