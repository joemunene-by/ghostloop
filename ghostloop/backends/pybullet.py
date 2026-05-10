"""PyBulletBackend: Bullet Physics backend, no-MuJoCo path.

Conditional import — same pattern as MuJoCoBackend, the package itself
imports cleanly without ``pybullet``. PyBullet is BSD-licensed, ships
on PyPI as a single wheel for every major OS, and supports URDF natively.
For users who can't install MuJoCo (Windows GPU drivers, NIH licensing
review queues, etc.) this gives an equivalent runtime backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import Primitive, Result, ResultStatus


def pybullet_available() -> bool:
    """True iff ``import pybullet`` succeeds in this environment."""
    try:
        import pybullet  # noqa: F401
        return True
    except ImportError:
        return False


_PYBULLET_INSTALL_HINT = (
    "PyBulletBackend requires the pybullet package.\n"
    "  pip install pybullet\n"
    "or  pip install ghostloop[pybullet]\n"
    "URDF models: https://github.com/bulletphysics/bullet3/tree/master/data"
)


@dataclass
class PyBulletBackend:
    """A Bullet-physics-backed Backend.

    Loads any URDF, exposes the end-effector link's world pose via snapshot(),
    drives joints via DIRECT mode (no GUI by default; flip ``gui=True`` for
    visualisation during development).
    """

    model_path: str
    end_effector_link: int = -1  # -1 = base link
    gui: bool = False
    timestep: float = 1 / 240
    name: str = "pybullet"

    _pb: Any = field(default=None, init=False, repr=False)
    _client: int = field(default=-1, init=False, repr=False)
    _body_id: int = field(default=-1, init=False, repr=False)
    _n_joints: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import pybullet  # type: ignore
            import pybullet_data  # type: ignore
        except ImportError as e:
            raise ImportError(_PYBULLET_INSTALL_HINT) from e
        self._pb = pybullet
        mode = pybullet.GUI if self.gui else pybullet.DIRECT
        self._client = pybullet.connect(mode)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setTimeStep(self.timestep, physicsClientId=self._client)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self._client)
        self._body_id = pybullet.loadURDF(
            self.model_path, physicsClientId=self._client,
        )
        self._n_joints = pybullet.getNumJoints(self._body_id, physicsClientId=self._client)

    def __del__(self) -> None:
        try:
            if self._pb is not None and self._client >= 0:
                self._pb.disconnect(physicsClientId=self._client)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Backend Protocol surface.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        if self.end_effector_link < 0:
            pos, orn = self._pb.getBasePositionAndOrientation(
                self._body_id, physicsClientId=self._client,
            )
        else:
            link_state = self._pb.getLinkState(
                self._body_id, self.end_effector_link, physicsClientId=self._client,
            )
            pos = link_state[0]
            orn = link_state[1]
        joint_states = self._pb.getJointStates(
            self._body_id, list(range(self._n_joints)),
            physicsClientId=self._client,
        )
        joints = [s[0] for s in joint_states]
        return {
            "backend": self.name,
            "model": self.model_path,
            "end_effector_link": self.end_effector_link,
            "position": list(pos),
            "orientation": list(orn),
            "joints": joints,
            "n_joints": self._n_joints,
        }

    # ------------------------------------------------------------------
    # Low-level helpers used by primitives.
    # ------------------------------------------------------------------

    def advance(self, duration: float) -> None:
        n = max(1, int(round(duration / self.timestep)))
        for _ in range(n):
            self._pb.stepSimulation(physicsClientId=self._client)

    def set_joint_position(self, joint_idx: int, position: float) -> None:
        self._pb.resetJointState(
            self._body_id, joint_idx, position,
            physicsClientId=self._client,
        )

    def control_joint(self, joint_idx: int, target: float, max_force: float = 100.0) -> None:
        self._pb.setJointMotorControl2(
            self._body_id, joint_idx,
            controlMode=self._pb.POSITION_CONTROL,
            targetPosition=target,
            force=max_force,
            physicsClientId=self._client,
        )

    def teleport_base(self, x: float, y: float, z: float) -> None:
        _, orn = self._pb.getBasePositionAndOrientation(
            self._body_id, physicsClientId=self._client,
        )
        self._pb.resetBasePositionAndOrientation(
            self._body_id, [x, y, z], orn,
            physicsClientId=self._client,
        )


# ----------------------------------------------------------------------
# Primitives bound to PyBulletBackend.
# ----------------------------------------------------------------------


def _move_to_call(
    backend: PyBulletBackend,
    x: float,
    y: float,
    z: float,
    duration: float = 1.0,
) -> Result:
    """Naive base-pose teleport + sim advance.

    Real arms will compute IK to the link target instead. This shape works
    for free-flying mobile robots and as a smoke test for the runtime layer
    before swapping in a proper IK solver in v0.4.
    """
    try:
        backend.teleport_base(float(x), float(y), float(z))
        backend.advance(duration)
    except Exception as exc:  # noqa: BLE001
        return Result(status=ResultStatus.ERROR, message=str(exc))
    snap = backend.snapshot()
    return Result(
        status=ResultStatus.OK,
        observation={
            "target": [float(x), float(y), float(z)],
            "achieved": snap["position"],
            "duration_s": float(duration),
        },
        message=f"advanced {duration:.3g}s toward {(x, y, z)}",
    )


def move_to() -> Primitive:
    return Primitive(
        name="move_to",
        call=_move_to_call,
        description="Move base toward (x, y, z) over `duration` seconds (PyBullet).",
        arg_schema={
            "x": "float",
            "y": "float",
            "z": "float",
            "duration": "float (optional, default 1.0)",
        },
    )


def _scan_call(backend: PyBulletBackend, radius: float = 1.0) -> Result:
    """Read all bodies within ``radius`` of our base — proxy for vision."""
    snap = backend.snapshot()
    cx, cy, cz = snap["position"]
    detections: list[dict[str, Any]] = []
    n_bodies = backend._pb.getNumBodies(physicsClientId=backend._client)
    for i in range(n_bodies):
        try:
            pos, _ = backend._pb.getBasePositionAndOrientation(
                i, physicsClientId=backend._client,
            )
        except Exception:
            continue
        d = ((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2 + (pos[2] - cz) ** 2) ** 0.5
        if d <= radius:
            detections.append({
                "body_id": int(i),
                "position": list(pos),
                "distance": float(d),
            })
    return Result(
        status=ResultStatus.OK,
        observation={"center": [cx, cy, cz], "radius": float(radius), "detections": detections},
        message=f"scanned {radius:g}m sphere; {len(detections)} detection(s)",
    )


def scan() -> Primitive:
    return Primitive(
        name="scan",
        call=_scan_call,
        description="List bodies within `radius` of the base (PyBullet).",
        arg_schema={"radius": "float (optional, default 1.0)"},
    )
