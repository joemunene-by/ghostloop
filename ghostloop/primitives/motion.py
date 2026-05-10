"""Mock-backend motion primitives: move_to and scan."""

from __future__ import annotations

from ..core import MockBackend, Primitive, Result, ResultStatus


def _move_to_call(backend: MockBackend, x: float, y: float, z: float) -> Result:
    old = backend.position
    new = (float(x), float(y), float(z))
    backend.position = new
    distance = sum((a - b) ** 2 for a, b in zip(new, old)) ** 0.5
    return Result(
        status=ResultStatus.OK,
        observation={
            "from": list(old),
            "to": list(new),
            "distance": round(distance, 6),
        },
        message=f"moved {distance:.4g} units",
    )


def move_to() -> Primitive:
    """Cartesian move: teleports the mock backend to (x, y, z) instantly.

    Real backends will integrate over time, respect velocity / acceleration
    limits, and return partial-progress observations. The mock keeps it
    deterministic so traces are reproducible.
    """
    return Primitive(
        name="move_to",
        call=_move_to_call,
        description="Move end-effector to a Cartesian target.",
        arg_schema={"x": "float", "y": "float", "z": "float"},
    )


def _scan_call(backend: MockBackend, radius: float = 1.0) -> Result:
    return Result(
        status=ResultStatus.OK,
        observation={
            "center": list(backend.position),
            "radius": float(radius),
            "detections": [],  # mock: no objects
        },
        message=f"scanned {radius:g}m sphere from {backend.position}",
    )


def scan() -> Primitive:
    """Read the workspace within ``radius`` of the current pose.

    Mock returns an empty detection list. Real backends fuse depth + RGB into
    structured detections (label, bbox, pose) — the schema is a list of
    dicts so adding fields later is non-breaking.
    """
    return Primitive(
        name="scan",
        call=_scan_call,
        description="Sense objects within a sphere of the current end-effector pose.",
        arg_schema={"radius": "float (optional, default 1.0)"},
    )
