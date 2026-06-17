"""Safe-action projection — repair denied actions instead of just rejecting them.

The fail-closed pipeline blocks unsafe actions, which is the right
default for hard constraints. But for many cases it's better to
*repair* the action: an autonomous picker reaching outside the bin
should snap to the nearest valid pose rather than refusing to move at
all. This module ships two projectors that take an Intent the safety
pipeline would deny and produce a corrected Intent that satisfies the
constraint.

  ``project_to_workspace(intent, workspace)``
      For motion intents with x/y/z targets, projects the target into
      the workspace bounding box AND outside any sphere obstacles.
      Pure analytic — clamping to AABB + radial pushout for spheres.
      Returns either the original intent (already valid) or a new
      Intent with corrected args + a ``"projected_from": orig_target``
      field for auditability.

      When the optional native extension ``ghostloop._rt_safety`` is
      built AND the workspace carries only Sphere obstacles, the
      analytic geometry runs in the allocation-free Rust core instead
      of pure Python. The result is numerically identical (verified to
      1e-9 per axis in tests/test_rt_safety_equivalence.py); the
      audit/return shape is unchanged. Any workspace with a box
      obstacle, or any environment where the native module is not
      built, transparently uses the pure-Python path below.

  ``project_to_sdf(intent, workspace, extras=())``
      Numerical: gradient-descent on the SDF until distance >= 0.
      Slower than the analytic version; handles convex polytopes /
      half-spaces / arbitrary extras. Falls back to the analytic
      projector if the SDF isn't available.

Stdlib math; no numpy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from ..core import Intent
from .sdf import HalfSpace, ConvexPolytope, signed_distance
from .workspace import (
    AxisAlignedBox,
    Sphere,
    WorkspaceModel,
)


# Optional allocation-free Rust fast path for the sphere-only analytic
# projection. Guarded so ghostloop runs unchanged when the native extension
# has not been built (the pure-Python path below is always available).
try:
    from .. import _rt_safety as _RT_SAFETY  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - any import/ABI failure must fall back cleanly.
    _RT_SAFETY = None


Point3 = tuple[float, float, float]


def _extract_target(args: dict) -> Point3 | None:
    if "target" in args:
        t: Sequence[float] = args["target"]
        if len(t) == 3:
            return float(t[0]), float(t[1]), float(t[2])
    if all(k in args for k in ("x", "y", "z")):
        return float(args["x"]), float(args["y"]), float(args["z"])
    return None


def _set_target(args: dict, p: Point3) -> dict:
    args = dict(args)
    if "target" in args:
        args["target"] = list(p)
        return args
    args["x"], args["y"], args["z"] = float(p[0]), float(p[1]), float(p[2])
    return args


def _project_point_native(
    target: Point3, workspace: WorkspaceModel,
) -> Point3 | None:
    """Run the sphere-only analytic projection in the native Rust core.

    Returns the projected point, or ``None`` if the native path does not
    apply (module not built, or the workspace has a non-sphere obstacle).
    The numerics match the pure-Python branch below bit-for-bit: the Rust
    core uses f64 and the identical operation order (clamp to bounds, then
    sequential radial push-out per sphere, degenerate ``dist < 1e-9`` sets
    ``p.x = center.x + forbidden_r`` and continues).
    """
    if _RT_SAFETY is None:
        return None
    spheres = []
    for ob in workspace.obstacles:
        if not isinstance(ob, Sphere):
            # Box (or any future non-sphere) obstacle: the native core does
            # not handle these analytically; defer to pure Python.
            return None
        spheres.append(
            (
                float(ob.center[0]),
                float(ob.center[1]),
                float(ob.center[2]),
                float(ob.radius),
                float(ob.inflation),
            )
        )
    bmin = (
        float(workspace.bounds_min[0]),
        float(workspace.bounds_min[1]),
        float(workspace.bounds_min[2]),
    )
    bmax = (
        float(workspace.bounds_max[0]),
        float(workspace.bounds_max[1]),
        float(workspace.bounds_max[2]),
    )
    try:
        out = _RT_SAFETY.project_to_workspace(target, bmin, bmax, spheres)
    except Exception:  # noqa: BLE001 - never let the fast path break a repair.
        return None
    return float(out[0]), float(out[1]), float(out[2])


def project_to_workspace(
    intent: Intent, workspace: WorkspaceModel,
) -> Intent:
    """Analytic projection: clamp to AABB + radial pushout from sphere obstacles.

    Returns the input ``intent`` unchanged if already valid; otherwise
    returns a new Intent with corrected target args and an extra
    ``args["projected_from"] = original_target`` for auditability.

    AxisAlignedBox obstacles are NOT pushed out of analytically (the
    closest-point-on-box-exterior is geometry that's annoying to
    compute robustly without numpy). For boxes use ``project_to_sdf``.

    When ``ghostloop._rt_safety`` is built and the workspace has only
    Sphere obstacles, the analytic geometry runs in the native Rust core;
    the result and the ``projected_from`` audit field are identical to the
    pure-Python computation. Otherwise the pure-Python path runs unchanged.
    """
    target = _extract_target(intent.args)
    if target is None:
        return intent

    native = _project_point_native(target, workspace)
    if native is not None:
        out_target = native
    else:
        p = list(target)
        # Clamp to outer bounds.
        for i in range(3):
            p[i] = max(workspace.bounds_min[i], min(workspace.bounds_max[i], p[i]))
        # Radial pushout from each sphere.
        for ob in workspace.obstacles:
            if not isinstance(ob, Sphere):
                continue
            cx, cy, cz = ob.center
            dx, dy, dz = p[0] - cx, p[1] - cy, p[2] - cz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            forbidden_r = ob.radius + ob.inflation
            if dist < forbidden_r:
                if dist < 1e-9:
                    # Degenerate: pick an arbitrary direction.
                    p[0] = cx + forbidden_r
                    continue
                scale = forbidden_r / dist
                p[0] = cx + dx * scale
                p[1] = cy + dy * scale
                p[2] = cz + dz * scale
        out_target = (p[0], p[1], p[2])

    if out_target == target:
        return intent
    new_args = _set_target(intent.args, out_target)
    new_args["projected_from"] = list(target)
    return Intent(name=intent.name, args=new_args, rationale=intent.rationale)


def project_to_sdf(
    intent: Intent,
    workspace: WorkspaceModel,
    *,
    extras: Sequence[Any] = (),
    margin: float = 0.0,
    n_steps: int = 50,
    learning_rate: float = 0.05,
) -> Intent:
    """Gradient-descent projection: numerical SDF push-out.

    Slower than the analytic projector but handles AABB obstacles,
    half-spaces, and convex polytopes. Walks the target up the SDF
    gradient until ``signed_distance >= margin`` or ``n_steps``
    exhausted.

    ``margin`` is an additional safety buffer: distance to nearest
    obstacle must end up at LEAST ``margin`` units. Defaults to 0
    (just-on-the-boundary). Use a positive margin to back off.
    """
    target = _extract_target(intent.args)
    if target is None:
        return intent
    p = list(target)
    for _ in range(n_steps):
        d = signed_distance(tuple(p), workspace, extras=extras)
        if d >= margin:
            break
        # Estimate gradient via finite differences.
        eps = 1e-3
        grad = [0.0, 0.0, 0.0]
        for axis in range(3):
            forward = list(p)
            forward[axis] += eps
            backward = list(p)
            backward[axis] -= eps
            d_f = signed_distance(tuple(forward), workspace, extras=extras)
            d_b = signed_distance(tuple(backward), workspace, extras=extras)
            grad[axis] = (d_f - d_b) / (2 * eps)
        norm = math.sqrt(sum(g * g for g in grad))
        if norm < 1e-9:
            break
        # Move along the gradient (which points away from obstacles).
        for axis in range(3):
            p[axis] += learning_rate * grad[axis] / norm
    out = (p[0], p[1], p[2])
    if out == target:
        return intent
    new_args = _set_target(intent.args, out)
    new_args["projected_from"] = list(target)
    return Intent(name=intent.name, args=new_args, rationale=intent.rationale)
