#!/usr/bin/env python3
"""End-to-end ghostloop demo: a 5-step pick-and-place episode.

Runs against MockBackend so you can ``python examples/pick_and_place.py`` with
zero install beyond the package itself. The same script will work against a
MuJoCoBackend in v0.2 by swapping one constructor.
"""

from __future__ import annotations

import json

from ghostloop import (
    Intent,
    PolicyPipeline,
    PrimitiveRegistry,
    MockBackend,
    Runtime,
)
from ghostloop.policies import DenyListGate, GeofenceGate, RateLimitGate
from ghostloop.primitives import move_to, pick, place, scan


def main() -> None:
    backend = MockBackend()
    registry = PrimitiveRegistry([move_to(), scan(), pick(), place()])
    pipeline = PolicyPipeline(
        gates=[
            DenyListGate(denied=set()),
            RateLimitGate(per_minute=600),
            GeofenceGate(min_corner=(-1.0, -1.0, 0.0), max_corner=(1.0, 1.0, 1.0)),
        ],
    )
    runtime = Runtime(backend=backend, registry=registry, policy_pipeline=pipeline)

    plan = [
        Intent("scan", {"radius": 0.5}, rationale="initial workspace scan"),
        Intent("move_to", {"x": 0.4, "y": 0.2, "z": 0.1}, rationale="approach widget"),
        Intent("pick", {"object_id": "widget-7"}, rationale="acquire widget"),
        Intent("move_to", {"x": -0.4, "y": 0.2, "z": 0.1}, rationale="approach drop zone"),
        Intent("place", {}, rationale="release into drop zone"),
        # Deliberate out-of-fence intent to demonstrate fail-closed safety.
        Intent("move_to", {"x": 5.0, "y": 0.0, "z": 0.0}, rationale="overshoot test"),
    ]

    for intent in plan:
        result = runtime.step(intent)
        marker = {
            "ok": "OK ",
            "error": "ERR",
            "blocked": "BLK",
            "timeout": "TMO",
        }[result.status.value]
        print(f"  [{marker}] {intent.name:<10} -> {result.message}")

    print()
    print(f"episode {runtime.trace.episode_id} complete: "
          f"{len(runtime.trace.events)} steps")
    print(json.dumps(runtime.trace.to_json()["events"][-1], indent=2))


if __name__ == "__main__":
    main()
