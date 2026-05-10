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
    """

    model_path: str
    end_effector: str = "end_effector"
    timestep: float = 0.002
    name: str = "mujoco"

    # Initialised in __post_init__.
    _mj: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _data: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import mujoco  # type: ignore
        except ImportError as e:
            raise ImportError(_MUJOCO_INSTALL_HINT) from e
        self._mj = mujoco
        self._model = mujoco.MjModel.from_xml_path(self.model_path)
        self._model.opt.timestep = self.timestep
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # Backend Protocol surface.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        ee_pos = self._end_effector_position()
        return {
            "backend": self.name,
            "model": self.model_path,
            "time": float(self._data.time),
            "end_effector": self.end_effector,
            "position": list(ee_pos),
            "qpos": list(self._data.qpos),
        }

    # ------------------------------------------------------------------
    # Low-level helpers used by primitives.
    # ------------------------------------------------------------------

    def advance(self, duration: float) -> None:
        """Step the sim forward by ``duration`` seconds (multiple integration steps)."""
        n = max(1, int(round(duration / self.timestep)))
        for _ in range(n):
            self._mj.mj_step(self._model, self._data)

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
        try:
            body_id = self._mj.mj_name2id(
                self._model, self._mj.mjtObj.mjOBJ_BODY, self.end_effector
            )
        except Exception:
            return (0.0, 0.0, 0.0)
        if body_id < 0:
            return (0.0, 0.0, 0.0)
        pos = self._data.xpos[body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))


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
    """Naive position-target proxy: snaps the first 3 qpos entries to (x,y,z),
    then advances the sim. Real-robot deployments will replace this with an
    IK solver + joint-velocity controller, but for sim demos against models
    whose first joint chain is the end-effector this is enough to validate
    the agent loop end-to-end without hand-rolling IK.
    """
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
