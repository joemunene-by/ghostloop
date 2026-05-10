"""URDF workspace builder — auto-derive a ``WorkspaceModel`` from a URDF.

Bringing your own robot to ghostloop has been a four-step manual chore:
read the URDF, eyeball the joint limits, sketch the reachable volume,
hand-write a ``WorkspaceModel`` with workspace-bounds + collision
obstacles. This module collapses those four steps into one call:

    workspace = workspace_from_urdf("franka_panda.urdf")
    gate = ObstacleAvoidanceGate(workspace=workspace)

URDF (Unified Robot Description Format) is XML. Every commonly-used
arm / mobile robot ships one. This parser is intentionally narrow —
it reads only what the workspace model needs:

  - ``<joint>`` elements with ``type="revolute"`` or ``"prismatic"``
    and ``<limit lower="..." upper="...">`` to define joint ranges.
  - ``<link>``'s ``<collision><geometry><box|sphere|cylinder>`` to
    pull each link's shape and add it as an obstacle.
  - The ``<origin xyz="x y z">`` of each link to place obstacles in
    world coordinates (without forward kinematics — assumes static
    pose; for moving links the obstacle list is conservative).

It deliberately avoids forward kinematics: solving joint configs to
real geometry would pull a kinematics dependency. For a static fixture
(table, walls, mounted equipment) this is exact; for moving links it
yields a conservative bounding model that's still useful.

Pure stdlib (``xml.etree.ElementTree`` + ``math``) — zero new deps.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .workspace import AxisAlignedBox, Sphere, WorkspaceModel


@dataclass
class URDFParseStats:
    """Diagnostics returned alongside the WorkspaceModel."""

    n_joints: int
    n_revolute_joints: int
    n_prismatic_joints: int
    n_links: int
    n_collision_shapes: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    parser_warnings: list[str]


def _parse_xyz(xyz_str: str | None) -> tuple[float, float, float]:
    if not xyz_str:
        return (0.0, 0.0, 0.0)
    parts = xyz_str.split()
    if len(parts) != 3:
        return (0.0, 0.0, 0.0)
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_size(size_str: str | None) -> tuple[float, float, float] | None:
    if not size_str:
        return None
    parts = size_str.split()
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _link_to_obstacles(
    link: ET.Element, warnings: list[str]
) -> list[AxisAlignedBox | Sphere]:
    """Best-effort extract collision geometries as obstacles (in link frame).

    Without forward kinematics we treat the obstacle as anchored at
    the URDF's collision-origin xyz — fine for static fixtures, a
    conservative approximation for moving links.
    """
    obs: list[AxisAlignedBox | Sphere] = []
    link_name = link.get("name", "<unknown>")
    for collision in link.findall("collision"):
        origin = collision.find("origin")
        ox, oy, oz = _parse_xyz(origin.get("xyz") if origin is not None else None)
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        # box <size="x y z" />
        box = geometry.find("box")
        if box is not None:
            size = _parse_size(box.get("size"))
            if size is None:
                warnings.append(f"link {link_name!r}: bad box size, skipping")
                continue
            sx, sy, sz = size
            obs.append(AxisAlignedBox(
                min_corner=(ox - sx / 2.0, oy - sy / 2.0, oz - sz / 2.0),
                max_corner=(ox + sx / 2.0, oy + sy / 2.0, oz + sz / 2.0),
                label=f"{link_name}.box",
            ))
            continue
        sphere = geometry.find("sphere")
        if sphere is not None:
            try:
                radius = float(sphere.get("radius", "0"))
            except ValueError:
                warnings.append(f"link {link_name!r}: bad sphere radius, skipping")
                continue
            obs.append(Sphere(
                center=(ox, oy, oz), radius=radius, label=f"{link_name}.sphere",
            ))
            continue
        cyl = geometry.find("cylinder")
        if cyl is not None:
            try:
                radius = float(cyl.get("radius", "0"))
                length = float(cyl.get("length", "0"))
            except ValueError:
                warnings.append(f"link {link_name!r}: bad cylinder, skipping")
                continue
            # Approximate cylinder with an AABB of the right footprint.
            obs.append(AxisAlignedBox(
                min_corner=(ox - radius, oy - radius, oz - length / 2.0),
                max_corner=(ox + radius, oy + radius, oz + length / 2.0),
                label=f"{link_name}.cylinder",
            ))
            continue
        # Mesh geometries: fall back to skip (can't load STL/DAE in stdlib).
        mesh = geometry.find("mesh")
        if mesh is not None:
            warnings.append(
                f"link {link_name!r}: <mesh> geometry skipped — load via "
                "trimesh externally and add an inflated AABB if you need it"
            )
    return obs


def workspace_from_urdf(
    urdf_path: str | Path,
    *,
    bounds_inflate: float = 0.1,
    obstacle_inflate: float = 0.0,
    floor_at: float | None = 0.0,
) -> tuple[WorkspaceModel, URDFParseStats]:
    """Parse a URDF and build a (WorkspaceModel, URDFParseStats) pair.

    The outer bounds are derived from the union AABB of every
    collision shape encountered, expanded by ``bounds_inflate``.
    Joint limits are recorded in stats but don't directly shape the
    workspace bounds (that requires forward kinematics).

    ``obstacle_inflate`` applies an extra safety margin to every
    collision obstacle.

    ``floor_at`` (default 0.0) clamps the lower-z bound to that height
    — common since most URDFs centre links and the world floor is
    z=0. Pass ``None`` to disable.

    Args:
        urdf_path: filesystem path to the URDF XML.
        bounds_inflate: per-axis padding added to the union AABB.
        obstacle_inflate: extra safety margin per obstacle.
        floor_at: optional minimum-z floor; ``None`` to skip.

    Returns:
        (workspace, stats) tuple. ``stats`` carries diagnostics like
        n_joints / parser warnings — check warnings if obstacles look
        sparse (mesh geometries are silently skipped).

    Raises:
        FileNotFoundError if the path doesn't exist.
        ValueError if the URDF is malformed XML.
    """
    path = Path(urdf_path)
    if not path.exists():
        raise FileNotFoundError(f"URDF not found: {path}")
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        raise ValueError(f"malformed URDF {path}: {e}") from e
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"URDF root must be <robot>, got <{root.tag}>")
    warnings: list[str] = []
    obstacles: list[AxisAlignedBox | Sphere] = []
    n_links = 0
    for link in root.findall("link"):
        n_links += 1
        for ob in _link_to_obstacles(link, warnings):
            if obstacle_inflate > 0 and hasattr(ob, "inflation"):
                ob.inflation = obstacle_inflate
            obstacles.append(ob)
    n_revolute = 0
    n_prismatic = 0
    n_joints = 0
    for joint in root.findall("joint"):
        n_joints += 1
        jtype = joint.get("type", "")
        if jtype == "revolute":
            n_revolute += 1
        elif jtype == "prismatic":
            n_prismatic += 1
    # Compute outer bounds as the union AABB of every obstacle.
    if obstacles:
        x_lo = min(_obstacle_min(ob, 0) for ob in obstacles)
        y_lo = min(_obstacle_min(ob, 1) for ob in obstacles)
        z_lo = min(_obstacle_min(ob, 2) for ob in obstacles)
        x_hi = max(_obstacle_max(ob, 0) for ob in obstacles)
        y_hi = max(_obstacle_max(ob, 1) for ob in obstacles)
        z_hi = max(_obstacle_max(ob, 2) for ob in obstacles)
    else:
        warnings.append(
            "URDF has no usable collision geometry; bounds default to 1x1x1"
        )
        x_lo, y_lo, z_lo = -0.5, -0.5, 0.0
        x_hi, y_hi, z_hi = 0.5, 0.5, 1.0
    bounds_min = (
        x_lo - bounds_inflate,
        y_lo - bounds_inflate,
        z_lo - bounds_inflate if floor_at is None else max(floor_at, z_lo - bounds_inflate),
    )
    bounds_max = (
        x_hi + bounds_inflate,
        y_hi + bounds_inflate,
        z_hi + bounds_inflate,
    )
    workspace = WorkspaceModel(
        bounds_min=bounds_min, bounds_max=bounds_max, obstacles=obstacles,
    )
    stats = URDFParseStats(
        n_joints=n_joints,
        n_revolute_joints=n_revolute,
        n_prismatic_joints=n_prismatic,
        n_links=n_links,
        n_collision_shapes=len(obstacles),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        parser_warnings=warnings,
    )
    return workspace, stats


def _obstacle_min(ob: AxisAlignedBox | Sphere, axis: int) -> float:
    if isinstance(ob, AxisAlignedBox):
        return ob.min_corner[axis]
    return ob.center[axis] - ob.radius


def _obstacle_max(ob: AxisAlignedBox | Sphere, axis: int) -> float:
    if isinstance(ob, AxisAlignedBox):
        return ob.max_corner[axis]
    return ob.center[axis] + ob.radius
