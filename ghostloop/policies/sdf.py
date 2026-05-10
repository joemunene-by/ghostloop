"""Signed-distance-field (SDF) workspace primitives + convex hull obstacles.

The v0.5 WorkspaceModel composes axis-aligned boxes and spheres. Real
deployments often need:

  - Convex polyhedra (polytope obstacles, robot bodies as obstacles to
    each other in fleet ops).
  - Half-spaces (floor / ceiling / wall planes — semi-infinite).
  - SDF-style queries: "what's my signed distance to the nearest
    obstacle?" — useful for proximity-based slow-down policies, not
    just hard accept/reject.

This module adds those obstacle types and exposes a ``signed_distance``
function across the whole workspace. Negative distance = inside an
obstacle (or outside outer bounds); positive = clearance to nearest.
Pure stdlib math; no numpy required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .workspace import AxisAlignedBox, Sphere, WorkspaceModel


Point3 = tuple[float, float, float]


@dataclass
class HalfSpace:
    """Half-space obstacle: ``ax + by + cz + d >= 0`` is INSIDE the obstacle.

    Useful for floors (normal=(0,0,1), offset=0 -> below z=0 is forbidden),
    ceilings, walls. ``inflation`` extends the forbidden region outward
    by N units along the normal.
    """

    normal: Point3                 # plane normal (need not be unit)
    offset: float = 0.0            # plane equation constant (n . p + offset >= 0)
    inflation: float = 0.0
    label: str = ""

    def contains(self, p: Point3) -> bool:
        nx, ny, nz = self.normal
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm == 0:
            return False
        signed = (nx * p[0] + ny * p[1] + nz * p[2] + self.offset) / norm
        return signed >= -self.inflation


@dataclass
class ConvexPolytope:
    """Convex region defined as the INTERSECTION of half-space safe-sides.

    Each face's ``normal`` points OUTWARD from the polytope interior.
    A point is inside iff for every face ``n . p + offset <= 0``
    (the point is on the inward side of every face plane).
    """

    faces: list[HalfSpace]
    label: str = ""

    def contains(self, p: Point3) -> bool:
        if not self.faces:
            return False
        for face in self.faces:
            nx, ny, nz = face.normal
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm == 0:
                return False
            signed = (nx * p[0] + ny * p[1] + nz * p[2] + face.offset) / norm
            # Inside the polytope iff signed <= -inflation on every face.
            if signed > -face.inflation:
                return False
        return True


# ---------------------------------------------------------------------------
# Signed-distance helpers — return shortest signed distance to nearest obstacle.
# Negative inside an obstacle, positive outside. Returns +inf for "no obstacle".
# ---------------------------------------------------------------------------


def _sd_sphere(p: Point3, s: Sphere) -> float:
    cx, cy, cz = s.center
    d = math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)
    return d - (s.radius + s.inflation)


def _sd_box(p: Point3, b: AxisAlignedBox) -> float:
    """SDF for an axis-aligned box (with inflation), Inigo Quilez's formula."""
    inflate = b.inflation
    qx = abs(p[0] - (b.min_corner[0] + b.max_corner[0]) / 2.0) - (
        (b.max_corner[0] - b.min_corner[0]) / 2.0 + inflate
    )
    qy = abs(p[1] - (b.min_corner[1] + b.max_corner[1]) / 2.0) - (
        (b.max_corner[1] - b.min_corner[1]) / 2.0 + inflate
    )
    qz = abs(p[2] - (b.min_corner[2] + b.max_corner[2]) / 2.0) - (
        (b.max_corner[2] - b.min_corner[2]) / 2.0 + inflate
    )
    outside = math.sqrt(max(qx, 0) ** 2 + max(qy, 0) ** 2 + max(qz, 0) ** 2)
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside


def _sd_halfspace(p: Point3, h: HalfSpace) -> float:
    nx, ny, nz = h.normal
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0:
        return float("inf")
    signed_inside = (nx * p[0] + ny * p[1] + nz * p[2] + h.offset) / norm
    # ``signed_inside`` is positive when inside the obstacle. Convert to the
    # ghostloop convention: negative inside, positive outside.
    return -(signed_inside + h.inflation)


def _sd_polytope(p: Point3, poly: ConvexPolytope) -> float:
    """Exact SDF for a convex polytope (intersection of half-space safe-sides).

    Each face's normal points OUT of the polytope, so ``n . p + offset``
    is positive when ``p`` is outside that face. The polytope SDF is
    the maximum of those per-face signed values: negative inside (every
    face reports a negative offset -> max is the least-negative -> the
    closest face), positive outside (at least one face reports positive
    -> max is the most-outside face).
    """
    if not poly.faces:
        return float("inf")
    sds: list[float] = []
    for face in poly.faces:
        nx, ny, nz = face.normal
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm == 0:
            continue
        signed = (nx * p[0] + ny * p[1] + nz * p[2] + face.offset) / norm
        sds.append(signed + face.inflation)
    if not sds:
        return float("inf")
    return max(sds)


def signed_distance(
    p: Point3,
    workspace: WorkspaceModel,
    extras: Sequence = (),
) -> float:
    """Signed distance to the nearest obstacle in the workspace.

    Negative inside an obstacle / outside outer bounds. Positive outside
    every obstacle (clearance to the nearest). Combine with ``extras``
    to mix in HalfSpace / ConvexPolytope obstacles not stored on the
    base WorkspaceModel.
    """
    distances: list[float] = []
    # Outer bounds — negative when outside the box.
    bx_lo, by_lo, bz_lo = workspace.bounds_min
    bx_hi, by_hi, bz_hi = workspace.bounds_max
    inside_box_dx = min(p[0] - bx_lo, bx_hi - p[0])
    inside_box_dy = min(p[1] - by_lo, by_hi - p[1])
    inside_box_dz = min(p[2] - bz_lo, bz_hi - p[2])
    distances.append(min(inside_box_dx, inside_box_dy, inside_box_dz))
    # Internal obstacles — negative inside.
    for obs in workspace.obstacles:
        if isinstance(obs, Sphere):
            distances.append(_sd_sphere(p, obs))
        elif isinstance(obs, AxisAlignedBox):
            distances.append(_sd_box(p, obs))
    # Extras (HalfSpaces, ConvexPolytopes).
    for obs in extras:
        if isinstance(obs, HalfSpace):
            distances.append(_sd_halfspace(p, obs))
        elif isinstance(obs, ConvexPolytope):
            distances.append(_sd_polytope(p, obs))
    return min(distances) if distances else float("inf")
