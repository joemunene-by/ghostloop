"""Sample custom Primitive factories referenced by examples/custom_robot.yaml.

This file shows how a user adds robot-specific actions WITHOUT forking
ghostloop. Each public function is a factory returning a ``Primitive``;
the YAML profile lists the ``module:factory`` paths under
``custom_primitives:`` and ghostloop loads them at runtime.

Two examples shipped:

  - ``dispense_pill(count)`` — sends a pill-dispenser command. Stub
    here; replace the ``backend.apply_action`` payload with your real
    integration (Arduino over serial, ROS 2 service call, vendor SDK).
  - ``alert_nurse(message)`` — fires a nurse-call notification. Wire
    to your hospital paging system / Slack / Twilio etc.

Both demonstrate the contract: a Primitive is just (name, call,
description, arg_schema). The ``call`` is whatever you need it to be.
"""

from __future__ import annotations

from ghostloop.core import Backend, Primitive, Result, ResultStatus


def dispense_pill() -> Primitive:
    """Dispense N pills from the medication tray."""

    def _call(backend: Backend, count: int = 1, drug: str = "unspecified") -> Result:
        # Replace this body with your actual hardware integration. For a
        # ROS 2 setup, call backend.publish on a /pill_dispenser topic.
        # For an Arduino, use pyserial. For a vendor SDK, call its API.
        try:
            count = max(0, min(int(count), 50))
        except (TypeError, ValueError):
            return Result(
                status=ResultStatus.ERROR,
                message=f"dispense_pill: count must be int, got {count!r}",
            )
        # Stub: best-effort apply_action if the backend has one, else
        # record the request for the trace.
        out: dict = {"action": "dispense_pill", "count": count, "drug": drug}
        if hasattr(backend, "apply_action"):
            try:
                backend_out = backend.apply_action(out)
                out["backend_out"] = backend_out
            except Exception as exc:  # noqa: BLE001
                return Result(
                    status=ResultStatus.ERROR,
                    message=f"dispense_pill: {type(exc).__name__}: {exc}",
                )
        return Result(
            status=ResultStatus.OK,
            observation=out,
            message=f"dispensed {count} x {drug}",
        )

    return Primitive(
        name="dispense_pill",
        call=_call,
        description=(
            "Dispense N pills from the medication tray. count must be in "
            "[0, 50]; drug labels the medication for the trace log."
        ),
        arg_schema={
            "count": "int — number of pills (0-50)",
            "drug": "str — drug label / NDC code",
        },
    )


def alert_nurse() -> Primitive:
    """Notify the nurse station with a free-form message."""

    def _call(backend: Backend, message: str = "", room: str = "") -> Result:
        # Replace with your paging integration. Twilio, Slack, PagerDuty,
        # vendor REST API — whatever the hospital uses.
        if not message:
            return Result(
                status=ResultStatus.ERROR,
                message="alert_nurse: message is required",
            )
        return Result(
            status=ResultStatus.OK,
            observation={
                "kind": "nurse_alert",
                "room": room,
                "message": message[:240],  # cap to one notification field
            },
            message=f"nurse alerted ({room or 'global'}): {message[:80]}",
        )

    return Primitive(
        name="alert_nurse",
        call=_call,
        description="Send a notification to the nurse station.",
        arg_schema={
            "message": "str — body of the notification",
            "room": "str — patient room number, optional",
        },
    )
