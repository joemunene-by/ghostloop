"""MuJoCoBackend: real physics via Google DeepMind's MuJoCo engine.

Conditional import — the package itself imports cleanly without ``mujoco``
on the system. ``MuJoCoBackend(...)`` raises ImportError with installation
guidance only at construction time.

The minimum supported workflow: load any MJCF / URDF model, expose the
end-effector pose via ``snapshot()``, drive joints via velocity control or
direct qpos setting (escape hatch). Primitives (motion / manipulation) bind
to a specific MuJoCoBackend instance and call its low-level helpers.

For v0.2 we ship the backend + primitives; pre-canned models from the
MuJoCo Menagerie (Franka Panda, UR5e, Stretch) are documented in the
README rather than vendored — keeps the package small and avoids licence
cross-contamination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..core import Primitive, Result, ResultStatus


def mujoco_available() -> bool:
    """True iff ``import mujoco`` succeeds in this environment."""
    try:
        import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


_MUJOCO_INSTALL_HINT = (
    "MuJoCoBackend requires the mujoco package.\n"
    "  pip install mujoco\n"
    "or  pip install ghostloop[mujoco]\n"
    "Models: https://github.com/google-deepmind/mujoco_menagerie"
)


@dataclass
class MuJoCoBackend:
    """A MuJoCo-backed Backend.

    Holds the (model, data) pair plus the name of the body whose position we
    treat as the end-effector. Steps the simulator on every call to
    ``advance(dt)`` so primitives can integrate over time.

    Args:
        model_path: path to an MJCF (.xml) or URDF (.urdf) model file.
        end_effector: body name whose pose surfaces in ``snapshot()``.
        timestep: per-step integration delta in seconds (default 0.002).
        name: friendly name for traces.
        site: optional site name used as the end-effector point instead of
            the body origin (e.g. ``"attachment_site"`` on the UR5e).
        keyframe: optional keyframe name or index to reset into on load.
        control: ``"ik"`` (Cartesian targets via damped least-squares IK),
            ``"naive"`` (write x/y/z into the first three qpos entries), or
            ``"auto"`` (naive when the first three joints are slide joints).
        gravity_compensation: cancel bias forces while advancing in IK mode,
            so position actuators track the IK solution without sag.
        ik_tolerance: residual (m) above which move_to refuses to move.
        settle_timeout: extra sim time (s) move_to may spend waiting for
            joint velocities to drop below ``settle_velocity`` (rad/s or m/s).
    """

    model_path: str
    end_effector: str = "end_effector"
    timestep: float = 0.002
    name: str = "mujoco"
    site: str | None = None
    keyframe: str | int | None = None
    control: str = "auto"
    gravity_compensation: bool = True
    ik_tolerance: float = 0.005
    ik_max_iters: int = 200
    settle_timeout: float = 3.0
    settle_velocity: float = 1e-3

    # Initialised in __post_init__.
    _mj: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _data: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError as e:
            raise ImportError(_MUJOCO_INSTALL_HINT) from e
        if self.control not in ("auto", "ik", "naive"):
            raise ValueError(f"control must be 'auto', 'ik' or 'naive', got {self.control!r}")
        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_path(self.model_path)
        self._model.opt.timestep = self.timestep
        self._data = mujoco.MjData(self._model)
        if self.keyframe is not None:
            key_id = (
                self.keyframe if isinstance(self.keyframe, int)
                else mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe)
            )
            if key_id < 0 or key_id >= self._model.nkey:
                raise ValueError(f"keyframe {self.keyframe!r} not found in {self.model_path}")
            mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
        mujoco.mj_forward(self._model, self._data)
        if self.control == "auto":
            self.control = "naive" if self._leading_slide_joints() else "ik"

    @classmethod
    def from_menagerie(cls, robot: str, menagerie_root: str | None = None, **kwargs: Any) -> "MuJoCoBackend":
        """Load a MuJoCo Menagerie robot with IK-ready defaults (``ur5e``, ``panda``, ...)."""
        from .menagerie import ROBOT_DEFAULTS, resolve_model
        defaults = dict(ROBOT_DEFAULTS.get(robot.lower(), {}))
        defaults.update(kwargs)
        defaults.setdefault("name", f"mujoco:{robot.lower()}")
        return cls(model_path=resolve_model(robot, menagerie_root), **defaults)

    # ------------------------------------------------------------------
    # Backend Protocol surface.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        ee_pos = self._end_effector_position()
        return {
            "backend": self.name,
            "model": self.model_path,
            "time": float(self._data.time),
            "end_effector": self.site or self.end_effector,
            "position": list(ee_pos),
            "qpos": [float(q) for q in self._data.qpos],
        }

    def describe(self) -> dict[str, Any]:
        """Static provenance for bench reports: engine version + control settings."""
        return {
            "engine": "mujoco",
            "engine_version": getattr(self._mj, "__version__", "unknown"),
            **_model_provenance(self.model_path),
            "end_effector": self.site or self.end_effector,
            "control": self.control,
            "gravity_compensation": self.gravity_compensation and self.control == "ik",
            "timestep": self.timestep,
            "ik_tolerance": self.ik_tolerance,
            "settle_timeout": self.settle_timeout,
            "settle_velocity": self.settle_velocity,
        }

    # ------------------------------------------------------------------
    # Low-level helpers used by primitives.
    # ------------------------------------------------------------------

    def advance(self, duration: float) -> None:
        """Step the sim forward by ``duration`` seconds (multiple integration steps)."""
        n = max(1, int(round(duration / self.timestep)))
        compensate = self.gravity_compensation and self.control == "ik"
        for _ in range(n):
            if compensate:
                self._data.qfrc_applied[:] = self._data.qfrc_bias
            self._mj.mj_step(self._model, self._data)
        if compensate:
            self._data.qfrc_applied[:] = 0.0

    def solve_ik(self, target: tuple[float, float, float]) -> tuple[list[float], float]:
        """Position-only damped least-squares IK from the current configuration.

        Runs on a scratch MjData so the live simulation is untouched. Returns
        ``(qpos, residual_m)``; callers decide whether the residual is acceptable.
        """
        import numpy as np

        mj, model = self._mj, self._model
        scratch = mj.MjData(model)
        scratch.qpos[:] = self._data.qpos
        mj.mj_kinematics(model, scratch)
        mj.mj_comPos(model, scratch)
        goal = np.asarray(target, dtype=float)
        jac = np.zeros((3, model.nv))
        limited = model.jnt_limited.astype(bool) & (model.jnt_type >= int(mj.mjtJoint.mjJNT_SLIDE))
        qadr = model.jnt_qposadr[limited]
        lo, hi = model.jnt_range[limited, 0], model.jnt_range[limited, 1]
        damping = 1e-2
        err = goal - self._point(scratch)
        for _ in range(self.ik_max_iters):
            if float(np.linalg.norm(err)) < self.ik_tolerance * 0.02:
                break
            self._point_jacobian(scratch, jac)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), err)
            norm = float(np.linalg.norm(dq))
            if norm > 0.3:
                dq *= 0.3 / norm
            mj.mj_integratePos(model, scratch.qpos, dq, 1.0)
            scratch.qpos[qadr] = np.clip(scratch.qpos[qadr], lo, hi)
            mj.mj_kinematics(model, scratch)
            mj.mj_comPos(model, scratch)
            err = goal - self._point(scratch)
        return [float(q) for q in scratch.qpos], float(np.linalg.norm(err))

    def command_qpos(self, qpos: list[float]) -> None:
        """Send a joint-space target: position actuators get it as ctrl,
        otherwise the configuration is snapped kinematically."""
        model = self._model
        driven = False
        for a in range(model.nu):
            if int(model.actuator_trntype[a]) != int(self._mj.mjtTrn.mjTRN_JOINT):
                continue
            j = int(model.actuator_trnid[a, 0])
            value = qpos[int(model.jnt_qposadr[j])]
            if model.actuator_ctrllimited[a]:
                lo, hi = model.actuator_ctrlrange[a]
                value = min(max(value, lo), hi)
            self._data.ctrl[a] = value
            driven = True
        if not driven:
            self.set_qpos(qpos)

    def max_joint_speed(self) -> float:
        return float(max((abs(v) for v in self._data.qvel), default=0.0))

    def _leading_slide_joints(self) -> bool:
        types = list(self._model.jnt_type[:3])
        return len(types) == 3 and all(int(t) == int(self._mj.mjtJoint.mjJNT_SLIDE) for t in types)

    def _site_id(self) -> int:
        return self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_SITE, self.site)

    def _body_id(self) -> int:
        return self._mj.mj_name2id(self._model, self._mj.mjtObj.mjOBJ_BODY, self.end_effector)

    def _point(self, data: Any):
        if self.site:
            return data.site_xpos[self._site_id()].copy()
        return data.xpos[self._body_id()].copy()

    def _point_jacobian(self, data: Any, jac: Any) -> None:
        if self.site:
            self._mj.mj_jacSite(self._model, data, jac, None, self._site_id())
        else:
            self._mj.mj_jacBody(self._model, data, jac, None, self._body_id())

    def set_qpos(self, qpos: list[float]) -> None:
        """Snap joint positions directly (escape hatch; no dynamics)."""
        if len(qpos) != self._model.nq:
            raise ValueError(
                f"qpos length {len(qpos)} != model.nq {self._model.nq}"
            )
        self._data.qpos[:] = qpos
        self._mj.mj_forward(self._model, self._data)

    def set_actuator(self, idx: int, value: float) -> None:
        """Set actuator control input (typical use: velocity or torque)."""
        self._data.ctrl[idx] = value

    def _end_effector_position(self) -> tuple[float, float, float]:
        if self.site:
            if self._site_id() < 0:
                return (0.0, 0.0, 0.0)
            pos = self._data.site_xpos[self._site_id()]
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        try:
            body_id = self._body_id()
        except Exception:
            return (0.0, 0.0, 0.0)
        if body_id < 0:
            return (0.0, 0.0, 0.0)
        pos = self._data.xpos[body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))


def _model_provenance(model_path: str) -> dict[str, Any]:
    import subprocess
    from pathlib import Path

    path = Path(model_path).resolve()
    for parent in path.parents:
        if parent.name == "mujoco_menagerie" or ((parent / ".git").exists() and (parent / "universal_robots_ur5e").is_dir()):
            out: dict[str, Any] = {"model": f"menagerie:{path.relative_to(parent).as_posix()}"}
            try:
                rev = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                )
                if rev.returncode == 0:
                    out["menagerie_commit"] = rev.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
            return out
    return {"model": path.name}


# ----------------------------------------------------------------------
# Primitives bound to MuJoCoBackend.
# ----------------------------------------------------------------------


def _move_to_call(
    backend: MuJoCoBackend,
    x: float,
    y: float,
    z: float,
    duration: float = 1.0,
) -> Result:
    """Cartesian move_to. Arms use IK + joint-target control; models whose
    first three joints are slide joints keep the original qpos-snap path."""
    if backend.control == "ik":
        return _ik_move_to(backend, float(x), float(y), float(z), float(duration))
    if backend._model.nq < 3:
        return Result(
            status=ResultStatus.ERROR,
            message=f"model only has {backend._model.nq} qpos; need >= 3 for naive move_to",
        )
    target = list(backend._data.qpos)
    target[0], target[1], target[2] = float(x), float(y), float(z)
    try:
        backend.set_qpos(target)
        backend.advance(duration)
    except Exception as exc:  # noqa: BLE001
        return Result(status=ResultStatus.ERROR, message=str(exc))
    pos = backend._end_effector_position()
    return Result(
        status=ResultStatus.OK,
        observation={
            "target": [float(x), float(y), float(z)],
            "achieved": list(pos),
            "duration_s": float(duration),
        },
        message=f"advanced {duration:.3g}s toward {(x, y, z)}",
    )


def _ik_move_to(backend: MuJoCoBackend, x: float, y: float, z: float, duration: float) -> Result:
    if not all(math.isfinite(v) for v in (x, y, z, duration)):
        return Result(status=ResultStatus.ERROR, message="non-finite move_to argument")
    qpos, residual = backend.solve_ik((x, y, z))
    if residual > backend.ik_tolerance:
        return Result(
            status=ResultStatus.ERROR,
            observation={"target": [x, y, z], "ik_residual_m": residual},
            message=f"unreachable: IK residual {residual:.3f}m > {backend.ik_tolerance:g}m; not moving",
        )
    try:
        backend.command_qpos(qpos)
        backend.advance(duration)
        settled_after = 0.0
        while settled_after < backend.settle_timeout and backend.max_joint_speed() > backend.settle_velocity:
            backend.advance(0.05)
            settled_after += 0.05
    except Exception as exc:  # noqa: BLE001
        return Result(status=ResultStatus.ERROR, message=str(exc))
    pos = backend._end_effector_position()
    tracking = math.dist(pos, (x, y, z))
    return Result(
        status=ResultStatus.OK,
        observation={
            "target": [x, y, z],
            "achieved": list(pos),
            "ik_residual_m": residual,
            "tracking_error_m": tracking,
            "duration_s": duration + settled_after,
        },
        message=f"reached {(x, y, z)} within {tracking * 1000:.1f}mm after {duration + settled_after:.3g}s",
    )


def move_to() -> Primitive:
    """MuJoCo-bound move_to. Same name as the MockBackend version so policies
    are backend-agnostic at the Intent layer.
    """
    return Primitive(
        name="move_to",
        call=_move_to_call,
        description="Move end-effector toward (x, y, z) over `duration` seconds.",
        arg_schema={
            "x": "float",
            "y": "float",
            "z": "float",
            "duration": "float (optional, default 1.0)",
        },
    )


def _scan_call(backend: MuJoCoBackend, radius: float = 1.0) -> Result:
    """Read every body within ``radius`` of the end-effector — a stand-in
    for vision/depth pipelines until v0.6 lands camera primitives.
    """
    ex, ey, ez = backend._end_effector_position()
    detections: list[dict[str, Any]] = []
    n_bodies = backend._model.nbody
    for i in range(n_bodies):
        bx, by, bz = backend._data.xpos[i]
        d2 = (bx - ex) ** 2 + (by - ey) ** 2 + (bz - ez) ** 2
        if d2 <= radius * radius:
            try:
                body_name = backend._mj.mj_id2name(
                    backend._model,
                    backend._mj.mjtObj.mjOBJ_BODY,
                    i,
                )
            except Exception:
                body_name = f"body_{i}"
            detections.append({
                "id": int(i),
                "name": body_name,
                "position": [float(bx), float(by), float(bz)],
                "distance": float(d2 ** 0.5),
            })
    return Result(
        status=ResultStatus.OK,
        observation={
            "center": [ex, ey, ez],
            "radius": float(radius),
            "detections": detections,
        },
        message=f"scanned {radius:g}m sphere; {len(detections)} detection(s)",
    )


def scan() -> Primitive:
    return Primitive(
        name="scan",
        call=_scan_call,
        description="List bodies in the current MuJoCo scene within `radius` of the end-effector.",
        arg_schema={"radius": "float (optional, default 1.0)"},
    )
