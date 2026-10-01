"""Canonical bench frame, so one preset runs on any embodiment.

Presets, success predicates and safety gates are written in a canonical frame
(the unit-ish cube the mock robot lives in). A :class:`Workspace` maps that
frame onto a real robot's reachable volume: ``robot = center + scale * canonical``.
Two backends that model the same embodiment use the same workspace, so their
results pair up episode for episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core import Primitive, Result

_XYZ = ("x", "y", "z")


@dataclass(frozen=True)
class Workspace:
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0

    def to_robot(self, p: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        return tuple(c + self.scale * float(v) for c, v in zip(self.center, p))  # type: ignore[return-value]

    def to_canonical(self, p: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        return tuple((float(v) - c) / self.scale for c, v in zip(self.center, p))  # type: ignore[return-value]

    @property
    def is_identity(self) -> bool:
        return self.center == (0.0, 0.0, 0.0) and self.scale == 1.0

    def to_json(self) -> dict[str, Any]:
        return {"center": list(self.center), "scale": self.scale}


IDENTITY = Workspace()

# Canonical [-1, 1] maps to a box in front of each arm that stays clear of the
# base and inside the reach envelope for every catalogue preset.
WORKSPACES: dict[str, Workspace] = {
    "ur5e": Workspace(center=(0.45, 0.0, 0.40), scale=0.35),
    "ur10e": Workspace(center=(0.65, 0.0, 0.50), scale=0.50),
    "panda": Workspace(center=(0.45, 0.0, 0.40), scale=0.30),
}


def workspace_for(embodiment: str | None) -> Workspace:
    if embodiment is None:
        return IDENTITY
    return WORKSPACES.get(embodiment.lower(), IDENTITY)


class FramedBackend:
    """Wraps a backend so ``snapshot()['position']`` is reported in the canonical frame.

    The robot-frame position is kept as ``robot_position``. Attribute access
    falls through to the wrapped backend.
    """

    def __init__(self, inner: Any, workspace: Workspace) -> None:
        self.inner = inner
        self.workspace = workspace
        self.name = inner.name

    def snapshot(self) -> dict[str, Any]:
        snap = dict(self.inner.snapshot())
        pos = snap.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            snap["robot_position"] = list(pos)
            snap["position"] = list(self.workspace.to_canonical(pos))
        return snap

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.inner, attr)


def framed_primitive(prim: Primitive, workspace: Workspace) -> Primitive:
    """Return a copy of ``prim`` that maps canonical x/y/z args to the robot frame."""

    def call(backend: Any, **kwargs: Any) -> Result:
        if all(k in kwargs for k in _XYZ):
            rx, ry, rz = workspace.to_robot([kwargs[k] for k in _XYZ])
            kwargs = {**kwargs, "x": rx, "y": ry, "z": rz}
        inner = backend.inner if isinstance(backend, FramedBackend) else backend
        return prim.call(inner, **kwargs)

    return Primitive(
        name=prim.name,
        call=call,
        description=prim.description,
        arg_schema=dict(prim.arg_schema),
    )
