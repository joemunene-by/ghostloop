"""Bounding-box geofence: deny if the requested target is outside the workspace.

Inspects ``intent.args`` for ``x`` / ``y`` / ``z`` (or ``target`` as a 3-tuple).
If none are present the gate is a no-op for that intent — only motion-with-
explicit-coords gets fenced. Real workspaces are obviously not always
axis-aligned boxes; the abstraction generalises to convex hulls or
joint-space limits when those backends land."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core import Decision, Intent, Primitive


@dataclass
class GeofenceGate:
    """Reject motions whose target falls outside an axis-aligned bounding box.

    ``min_corner`` and ``max_corner`` are inclusive 3D points. Primitives with
    no positional argument pass through transparently — geofencing only
    triggers when the intent declares a target.
    """

    min_corner: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    max_corner: tuple[float, float, float] = (1.0, 1.0, 1.0)
    name: str = "geofence"

    def _extract_target(self, args: dict) -> tuple[float, float, float] | None:
        if "target" in args:
            t: Sequence[float] = args["target"]
            if len(t) == 3:
                return float(t[0]), float(t[1]), float(t[2])
        if all(k in args for k in ("x", "y", "z")):
            return float(args["x"]), float(args["y"]), float(args["z"])
        return None

    def check(self, intent: Intent, primitive: Primitive) -> Decision:
        target = self._extract_target(intent.args)
        if target is None:
            return Decision.allow(self.name, "no target coords in intent")
        for axis, value, lo, hi in zip(
            ("x", "y", "z"),
            target,
            self.min_corner,
            self.max_corner,
            strict=True,
        ):
            if value < lo or value > hi:
                return Decision.deny(
                    self.name,
                    f"target {axis}={value:g} outside workspace [{lo:g},{hi:g}]",
                )
        return Decision.allow(
            self.name,
            f"target {target} inside workspace",
        )
