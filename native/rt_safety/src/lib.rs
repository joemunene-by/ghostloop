//! rt_safety: an allocation-free, real-time-grade port of ghostloop's
//! safe-projection envelope.
//!
//! This crate re-implements, bit-for-bit, the sphere-only fast path of
//! ghostloop's pure-Python safety geometry:
//!
//!   ghostloop/policies/workspace.py     -> `in_envelope`
//!       (WorkspaceModel.violates for sphere-only workspaces:
//!        outer-bounds AABB test + Sphere.contains inflated-radius test).
//!
//!   ghostloop/policies/safe_projection.py -> `project_to_workspace`
//!       (project_to_workspace: clamp target to the bounds AABB, then for
//!        each sphere in order, if the point is inside the inflated radius,
//!        push it radially out to exactly `radius + inflation`).
//!
//! Fidelity guarantees that make the native path a drop-in for the Python:
//!   - `f64` throughout, matching CPython float.
//!   - the *same* arithmetic order as the Python (`dx*dx + dy*dy + dz*dz`,
//!     then `sqrt`, then compare against `radius + inflation`, then
//!     `scale = forbidden_r / dist`). No fused-multiply-add, no reordering.
//!   - the clamp is `max(lo, min(hi, p))`, identical to Python's
//!     `max(bounds_min, min(bounds_max, p))`.
//!   - the degenerate `dist < 1e-9` branch sets `p[0] = center.x + forbidden_r`
//!     and continues, leaving `p[1]` / `p[2]` untouched, exactly as Python.
//!   - the bounds test uses strict `<` / `>` and the sphere test uses `<=`,
//!     matching `violates` and `Sphere.contains` respectively.
//!
//! Allocation discipline: the hot path (`project`, `in_envelope`) takes the
//! spheres as a borrowed slice and operates on fixed `[f64; 3]` stack arrays.
//! No `Vec`, no boxing, no locks, no I/O. The PyO3 wrappers do the only
//! allocation, and only to marshal the Python `list` of spheres into a small
//! stack buffer (`MAX_SPHERES`) before calling the pure core.

#![allow(clippy::needless_range_loop)]

use pyo3::prelude::*;
use pyo3::types::PyList;

/// Cartesian degrees of freedom. ghostloop's workspace is strictly 3D.
pub const DOF: usize = 3;

/// Maximum spheres marshalled onto the stack per call. Real ghostloop
/// workspaces carry a handful of obstacles; this cap keeps the marshalling
/// buffer fixed-size and the obstacle loop's worst case bounded. Workspaces
/// with more spheres than this fall back to Python (the shim only dispatches
/// to native when the count fits).
pub const MAX_SPHERES: usize = 64;

/// A spherical forbidden region. `forbidden_r` is the already-summed
/// `radius + inflation`, so the core never has to recombine them and the
/// arithmetic order is fixed at the marshalling boundary.
#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: [f64; DOF],
    pub forbidden_r: f64,
}

/// Faithful port of `WorkspaceModel.violates` for a sphere-only workspace.
///
/// Returns `true` iff the point is *valid* (inside the outer bounds AND
/// outside every inflated sphere): i.e. `violates(p) is None` in Python.
///
/// Bounds test mirrors `violates`: a point exactly on a face is in-bounds
/// (`p < lo` / `p > hi` are the only rejections). Sphere test mirrors
/// `Sphere.contains`: `dist <= radius + inflation` is *inside* (a violation),
/// so a point exactly on the inflated surface is treated as contained, which
/// is why the projector pushes to *exactly* `forbidden_r` and the validity
/// check must therefore use the same `<=` boundary the projector lands on.
///
/// NOTE: box obstacles are intentionally unsupported here. The Python shim
/// only routes sphere-only workspaces to this function; anything with an
/// `AxisAlignedBox` obstacle stays on the pure-Python path.
#[inline]
pub fn in_envelope(
    p: &[f64; DOF],
    bounds_min: &[f64; DOF],
    bounds_max: &[f64; DOF],
    spheres: &[Sphere],
) -> bool {
    for i in 0..DOF {
        if p[i] < bounds_min[i] || p[i] > bounds_max[i] {
            return false;
        }
    }
    for s in spheres {
        // Same accumulation order as Sphere.contains:
        // sqrt(sum((p[i]-c[i])**2)).
        let dx = p[0] - s.center[0];
        let dy = p[1] - s.center[1];
        let dz = p[2] - s.center[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d <= s.forbidden_r {
            return false;
        }
    }
    true
}

/// Faithful port of `project_to_workspace`'s analytic geometry.
///
/// Operation order, replicated exactly from safe_projection.py:
///   1. Clamp every axis into `[bounds_min, bounds_max]` via
///      `max(lo, min(hi, p))`.
///   2. For each sphere *in input order*, recompute the displacement from the
///      *current* (possibly already-pushed) point. If `dist < forbidden_r`:
///        - degenerate `dist < 1e-9`: set `p[0] = center.x + forbidden_r`,
///          leave `p[1]`/`p[2]`, and continue to the next sphere;
///        - otherwise scale all three axes by `forbidden_r / dist` about the
///          centre.
/// There is no re-clamp and no second pass, matching the Python exactly.
#[inline]
pub fn project_to_workspace(
    target: &[f64; DOF],
    bounds_min: &[f64; DOF],
    bounds_max: &[f64; DOF],
    spheres: &[Sphere],
) -> [f64; DOF] {
    let mut p = *target;

    // (1) Clamp to outer bounds: max(lo, min(hi, p)).
    for i in 0..DOF {
        p[i] = f64::max(bounds_min[i], f64::min(bounds_max[i], p[i]));
    }

    // (2) Radial push-out from each sphere, in order, against the live point.
    for s in spheres {
        let cx = s.center[0];
        let cy = s.center[1];
        let cz = s.center[2];
        let dx = p[0] - cx;
        let dy = p[1] - cy;
        let dz = p[2] - cz;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let forbidden_r = s.forbidden_r;
        if dist < forbidden_r {
            if dist < 1e-9 {
                // Degenerate: arbitrary direction along +x; leave y, z.
                p[0] = cx + forbidden_r;
                continue;
            }
            let scale = forbidden_r / dist;
            p[0] = cx + dx * scale;
            p[1] = cy + dy * scale;
            p[2] = cz + dz * scale;
        }
    }

    p
}

// --------------------------------------------------------------------------
// PyO3 marshalling boundary.
//
// Each sphere is passed from Python as a 5-tuple
// (cx, cy, cz, radius, inflation); we fold radius+inflation into forbidden_r
// here, at the boundary, so the order of that single add is fixed and the
// hot core never sees the split.
// --------------------------------------------------------------------------

fn marshal_spheres(spheres: &Bound<'_, PyList>) -> PyResult<([Sphere; MAX_SPHERES], usize)> {
    let n = spheres.len();
    if n > MAX_SPHERES {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rt_safety: {} spheres exceeds MAX_SPHERES={}",
            n, MAX_SPHERES
        )));
    }
    let mut buf = [Sphere { center: [0.0; DOF], forbidden_r: 0.0 }; MAX_SPHERES];
    for (i, item) in spheres.iter().enumerate() {
        let cx: f64 = item.get_item(0)?.extract()?;
        let cy: f64 = item.get_item(1)?.extract()?;
        let cz: f64 = item.get_item(2)?.extract()?;
        let radius: f64 = item.get_item(3)?.extract()?;
        let inflation: f64 = item.get_item(4)?.extract()?;
        buf[i] = Sphere {
            center: [cx, cy, cz],
            // Same combination as Python: forbidden_r = ob.radius + ob.inflation.
            forbidden_r: radius + inflation,
        };
    }
    Ok((buf, n))
}

/// Python: `_rt_safety.in_envelope(p, bounds_min, bounds_max, spheres) -> bool`.
///
/// `p`, `bounds_min`, `bounds_max` are 3-tuples of float; `spheres` is a list
/// of `(cx, cy, cz, radius, inflation)` tuples. Returns `True` iff `p` is a
/// valid workspace point (equivalent to `WorkspaceModel.violates(p) is None`).
#[pyfunction]
#[pyo3(name = "in_envelope")]
fn py_in_envelope(
    p: (f64, f64, f64),
    bounds_min: (f64, f64, f64),
    bounds_max: (f64, f64, f64),
    spheres: &Bound<'_, PyList>,
) -> PyResult<bool> {
    let (buf, n) = marshal_spheres(spheres)?;
    let pa = [p.0, p.1, p.2];
    let bmin = [bounds_min.0, bounds_min.1, bounds_min.2];
    let bmax = [bounds_max.0, bounds_max.1, bounds_max.2];
    Ok(in_envelope(&pa, &bmin, &bmax, &buf[..n]))
}

/// Python: `_rt_safety.project_to_workspace(p, bounds_min, bounds_max, spheres)
/// -> (f64, f64, f64)`. Numerically identical to the analytic geometry in
/// `safe_projection.project_to_workspace`.
#[pyfunction]
#[pyo3(name = "project_to_workspace")]
fn py_project_to_workspace(
    p: (f64, f64, f64),
    bounds_min: (f64, f64, f64),
    bounds_max: (f64, f64, f64),
    spheres: &Bound<'_, PyList>,
) -> PyResult<(f64, f64, f64)> {
    let (buf, n) = marshal_spheres(spheres)?;
    let target = [p.0, p.1, p.2];
    let bmin = [bounds_min.0, bounds_min.1, bounds_min.2];
    let bmax = [bounds_max.0, bounds_max.1, bounds_max.2];
    let out = project_to_workspace(&target, &bmin, &bmax, &buf[..n]);
    Ok((out[0], out[1], out[2]))
}

/// The Python module: `ghostloop._rt_safety`.
#[pymodule]
fn _rt_safety(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_in_envelope, m)?)?;
    m.add_function(wrap_pyfunction!(py_project_to_workspace, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> ([f64; DOF], [f64; DOF]) {
        ([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0])
    }

    #[test]
    fn clamps_out_of_bounds() {
        let (lo, hi) = unit_box();
        let out = project_to_workspace(&[5.0, -9.0, 0.5], &lo, &hi, &[]);
        assert_eq!(out, [1.0, -1.0, 0.5]);
    }

    #[test]
    fn pushes_out_of_sphere() {
        let (lo, hi) = unit_box();
        let s = Sphere { center: [0.0, 0.0, 0.5], forbidden_r: 0.25 };
        // Inside the sphere, off-centre.
        let out = project_to_workspace(&[0.05, 0.0, 0.5], &lo, &hi, &[s]);
        let d = ((out[0]) * (out[0]) + (out[2] - 0.5) * (out[2] - 0.5)).sqrt();
        assert!((d - 0.25).abs() < 1e-12, "landed at {:?}", out);
        assert!(in_envelope(&out, &lo, &hi, &[s]));
    }

    #[test]
    fn degenerate_center_pushes_along_x() {
        let (lo, hi) = unit_box();
        let s = Sphere { center: [0.0, 0.0, 0.5], forbidden_r: 0.25 };
        let out = project_to_workspace(&[0.0, 0.0, 0.5], &lo, &hi, &[s]);
        assert_eq!(out, [0.25, 0.0, 0.5]);
    }

    #[test]
    fn valid_point_is_in_envelope() {
        let (lo, hi) = unit_box();
        let s = Sphere { center: [0.0, 0.0, 0.5], forbidden_r: 0.25 };
        assert!(in_envelope(&[0.8, 0.8, 0.8], &lo, &hi, &[s]));
        assert!(!in_envelope(&[0.0, 0.0, 0.5], &lo, &hi, &[s]));
        assert!(!in_envelope(&[2.0, 0.0, 0.5], &lo, &hi, &[s]));
    }
}
