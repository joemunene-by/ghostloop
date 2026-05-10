"""Cross-morphology primitive library — actions for every kind of robot.

The v0.1 motion + manipulation primitives covered arms (move_to / pick /
place / scan). Real fleets are wider: a mobile base wants ``drive`` /
``stop`` / ``goto``; a quadruped wants ``sit`` / ``stand`` / ``walk_to``;
a drone wants ``takeoff`` / ``land`` / ``fly_to``; a humanoid wants
``wave`` / ``look_at`` / ``point_at``. This module catalogues them.

Every primitive here is backend-agnostic — the call dispatches through
``backend.apply_action(...)`` (Gymnasium-style) when present, otherwise
records the intent in the snapshot so policies + traces still capture
the behaviour. Rebind to your real backend by writing a thin Primitive
factory that talks to your hardware (e.g. ROS 2 publisher, native SDK)
and passing it to the registry instead of these defaults.

Categories shipped:

  - ``MOBILE_BASE``       drive / stop / goto / rotate
  - ``QUADRUPED``         sit / stand / lie_down / walk_to / rotate
  - ``HUMANOID``          wave / look_at / point_at / nod
  - ``AERIAL``            takeoff / land / fly_to / hover
  - ``DEXTEROUS``         set_joint / set_gripper
  - ``SENSING``           sense / scan_360 / take_photo / read_battery
  - ``GENERIC``           emit_event / wait

Mix and match: a Stretch RE3 (mobile base + arm) pulls from MOBILE_BASE
+ DEXTEROUS + manipulation; Spot pulls from QUADRUPED + SENSING.
"""

from __future__ import annotations

from typing import Any

from ..core import Backend, Primitive, Result, ResultStatus


# ---------------------------------------------------------------------------
# Helper that records "we did X" in the result observation. Real Primitive
# factories override this to actually publish to ROS 2 / native SDK / etc.
# ---------------------------------------------------------------------------


def _emit_intent(backend: Backend, action: str, **payload: Any) -> Result:
    """Default action: forward to backend.apply_action if present, else
    record-and-pass-through. Real-hardware primitives override this."""
    if hasattr(backend, "apply_action"):
        try:
            out = backend.apply_action({"action": action, **payload})
            return Result(
                status=ResultStatus.OK,
                observation={"action": action, **payload, "backend_out": out},
                message=f"{action} -> {payload}",
            )
        except Exception as exc:  # noqa: BLE001
            return Result(
                status=ResultStatus.ERROR,
                message=f"{action} failed: {type(exc).__name__}: {exc}",
            )
    return Result(
        status=ResultStatus.OK,
        observation={"action": action, **payload, "noop_backend": True},
        message=f"{action}({payload}) — backend has no apply_action; recorded only",
    )


# ---------------------------------------------------------------------------
# Mobile-base primitives.
# ---------------------------------------------------------------------------


def drive() -> Primitive:
    """Issue a velocity command to a mobile base."""
    def _call(backend: Backend, linear_x: float = 0.0, angular_z: float = 0.0) -> Result:
        return _emit_intent(
            backend, "drive", linear_x=linear_x, angular_z=angular_z,
        )
    return Primitive(
        name="drive", call=_call,
        description="Drive a mobile base. Linear m/s + angular rad/s.",
        arg_schema={
            "linear_x": "float — forward velocity m/s",
            "angular_z": "float — yaw velocity rad/s",
        },
    )


def stop() -> Primitive:
    """Zero-velocity / E-stop on a mobile base. Wired in many safety pipelines as the fallback action."""
    def _call(backend: Backend) -> Result:
        return _emit_intent(backend, "stop")
    return Primitive(
        name="stop", call=_call,
        description="Stop motion immediately (zero Twist).",
        arg_schema={},
    )


def goto() -> Primitive:
    """Navigate to a 2D pose. Real backends route through MoveIt / Nav2."""
    def _call(backend: Backend, x: float, y: float, theta: float = 0.0) -> Result:
        return _emit_intent(backend, "goto", x=x, y=y, theta=theta)
    return Primitive(
        name="goto", call=_call,
        description="Navigate to (x, y, theta) — yaw in radians.",
        arg_schema={
            "x": "float — target x metres",
            "y": "float — target y metres",
            "theta": "float — target yaw radians (default 0)",
        },
    )


def rotate() -> Primitive:
    """Rotate in place by the given yaw delta."""
    def _call(backend: Backend, dtheta: float) -> Result:
        return _emit_intent(backend, "rotate", dtheta=dtheta)
    return Primitive(
        name="rotate", call=_call,
        description="Rotate in place by dtheta radians.",
        arg_schema={"dtheta": "float — yaw delta radians"},
    )


# ---------------------------------------------------------------------------
# Quadruped primitives (Spot-style).
# ---------------------------------------------------------------------------


def sit() -> Primitive:
    def _call(backend: Backend) -> Result:
        return _emit_intent(backend, "sit")
    return Primitive(
        name="sit", call=_call,
        description="Quadruped: sit down. Reduces battery draw.",
        arg_schema={},
    )


def stand() -> Primitive:
    def _call(backend: Backend) -> Result:
        return _emit_intent(backend, "stand")
    return Primitive(
        name="stand", call=_call,
        description="Quadruped: stand up to nominal stance.",
        arg_schema={},
    )


def lie_down() -> Primitive:
    def _call(backend: Backend) -> Result:
        return _emit_intent(backend, "lie_down")
    return Primitive(
        name="lie_down", call=_call,
        description="Quadruped: lie down for charging / shutdown.",
        arg_schema={},
    )


def walk_to() -> Primitive:
    def _call(backend: Backend, x: float, y: float, theta: float = 0.0) -> Result:
        return _emit_intent(backend, "walk_to", x=x, y=y, theta=theta)
    return Primitive(
        name="walk_to", call=_call,
        description="Quadruped: walk to (x, y, theta).",
        arg_schema={
            "x": "float — target x metres",
            "y": "float — target y metres",
            "theta": "float — target yaw radians",
        },
    )


# ---------------------------------------------------------------------------
# Humanoid primitives (social interaction).
# ---------------------------------------------------------------------------


def wave() -> Primitive:
    def _call(backend: Backend, hand: str = "right") -> Result:
        return _emit_intent(backend, "wave", hand=hand)
    return Primitive(
        name="wave", call=_call,
        description="Humanoid: wave a hand (right by default).",
        arg_schema={"hand": "str — 'right' or 'left'"},
    )


def look_at() -> Primitive:
    def _call(backend: Backend, x: float, y: float, z: float) -> Result:
        return _emit_intent(backend, "look_at", x=x, y=y, z=z)
    return Primitive(
        name="look_at", call=_call,
        description="Orient the head / camera toward (x, y, z) in world frame.",
        arg_schema={
            "x": "float", "y": "float", "z": "float",
        },
    )


def point_at() -> Primitive:
    def _call(backend: Backend, x: float, y: float, z: float, hand: str = "right") -> Result:
        return _emit_intent(backend, "point_at", x=x, y=y, z=z, hand=hand)
    return Primitive(
        name="point_at", call=_call,
        description="Humanoid: point a hand at world coordinates.",
        arg_schema={
            "x": "float", "y": "float", "z": "float",
            "hand": "str — 'right' or 'left'",
        },
    )


def nod() -> Primitive:
    def _call(backend: Backend, direction: str = "yes") -> Result:
        return _emit_intent(backend, "nod", direction=direction)
    return Primitive(
        name="nod", call=_call,
        description="Humanoid: nod 'yes' (vertical) or 'no' (horizontal).",
        arg_schema={"direction": "str — 'yes' or 'no'"},
    )


# ---------------------------------------------------------------------------
# Aerial / drone primitives.
# ---------------------------------------------------------------------------


def takeoff() -> Primitive:
    def _call(backend: Backend, altitude: float = 1.0) -> Result:
        return _emit_intent(backend, "takeoff", altitude=altitude)
    return Primitive(
        name="takeoff", call=_call,
        description="Drone: takeoff to target altitude (m).",
        arg_schema={"altitude": "float — target altitude metres"},
    )


def land() -> Primitive:
    def _call(backend: Backend) -> Result:
        return _emit_intent(backend, "land")
    return Primitive(
        name="land", call=_call,
        description="Drone: land at current xy. Recommended fallback in HITL.",
        arg_schema={},
    )


def fly_to() -> Primitive:
    def _call(backend: Backend, x: float, y: float, z: float, yaw: float = 0.0) -> Result:
        return _emit_intent(backend, "fly_to", x=x, y=y, z=z, yaw=yaw)
    return Primitive(
        name="fly_to", call=_call,
        description="Drone: fly to a 3D pose (x, y, z, yaw).",
        arg_schema={
            "x": "float", "y": "float", "z": "float",
            "yaw": "float — heading radians",
        },
    )


def hover() -> Primitive:
    def _call(backend: Backend, seconds: float = 1.0) -> Result:
        return _emit_intent(backend, "hover", seconds=seconds)
    return Primitive(
        name="hover", call=_call,
        description="Drone: hold position for N seconds.",
        arg_schema={"seconds": "float — hold duration"},
    )


# ---------------------------------------------------------------------------
# Dexterous / per-joint primitives.
# ---------------------------------------------------------------------------


def set_joint() -> Primitive:
    def _call(backend: Backend, joint_name: str, angle: float, duration: float = 1.0) -> Result:
        return _emit_intent(
            backend, "set_joint",
            joint_name=joint_name, angle=angle, duration=duration,
        )
    return Primitive(
        name="set_joint", call=_call,
        description="Set a single joint's angle (radians) over duration (s).",
        arg_schema={
            "joint_name": "str",
            "angle": "float — radians",
            "duration": "float — seconds",
        },
    )


def set_gripper() -> Primitive:
    def _call(backend: Backend, state: str = "open", force: float = 0.0) -> Result:
        return _emit_intent(backend, "set_gripper", state=state, force=force)
    return Primitive(
        name="set_gripper", call=_call,
        description="Open or close a gripper. Force optional (Newtons).",
        arg_schema={
            "state": "str — 'open' / 'close' / 'half'",
            "force": "float — clamp force in Newtons",
        },
    )


# ---------------------------------------------------------------------------
# Sensing primitives.
# ---------------------------------------------------------------------------


def sense() -> Primitive:
    def _call(backend: Backend, modality: str = "rgb") -> Result:
        snap = backend.snapshot()
        return Result(
            status=ResultStatus.OK,
            observation={"modality": modality, "snapshot": snap},
            message=f"sense({modality})",
        )
    return Primitive(
        name="sense", call=_call,
        description=(
            "Read a sensor by modality. modality is 'rgb' / 'depth' / "
            "'lidar' / 'odom' / 'force_torque' / 'imu'."
        ),
        arg_schema={"modality": "str"},
    )


def scan_360() -> Primitive:
    def _call(backend: Backend) -> Result:
        return _emit_intent(backend, "scan_360")
    return Primitive(
        name="scan_360", call=_call,
        description="Rotate in place while sampling sensors. Useful for mapping.",
        arg_schema={},
    )


def take_photo() -> Primitive:
    def _call(backend: Backend, camera: str = "default") -> Result:
        return _emit_intent(backend, "take_photo", camera=camera)
    return Primitive(
        name="take_photo", call=_call,
        description="Capture a frame from a named camera.",
        arg_schema={"camera": "str — camera name (default 'default')"},
    )


def read_battery() -> Primitive:
    def _call(backend: Backend) -> Result:
        snap = backend.snapshot()
        battery = snap.get("battery") or snap.get("battery_pct")
        return Result(
            status=ResultStatus.OK,
            observation={"battery": battery, "snapshot": snap},
            message=f"battery: {battery}",
        )
    return Primitive(
        name="read_battery", call=_call,
        description="Read battery percentage from the backend snapshot.",
        arg_schema={},
    )


# ---------------------------------------------------------------------------
# Generic primitives.
# ---------------------------------------------------------------------------


def wait() -> Primitive:
    """Pause for N seconds. Common in HITL flows where the operator needs time."""
    import time as _time

    def _call(backend: Backend, seconds: float = 1.0) -> Result:
        seconds = max(0.0, min(seconds, 30.0))
        _time.sleep(seconds)
        return Result(
            status=ResultStatus.OK,
            observation={"waited_s": seconds},
            message=f"waited {seconds:.2f}s",
        )
    return Primitive(
        name="wait", call=_call,
        description="Block for N seconds (max 30). Use sparingly.",
        arg_schema={"seconds": "float — duration"},
    )


def emit_event() -> Primitive:
    """Append a structured event to the trace without dispatching to a backend."""
    def _call(backend: Backend, kind: str = "note", message: str = "") -> Result:
        return Result(
            status=ResultStatus.OK,
            observation={"kind": kind, "message": message},
            message=f"event[{kind}]: {message}",
        )
    return Primitive(
        name="emit_event", call=_call,
        description="Record a labelled event in the trace. Useful for milestones.",
        arg_schema={
            "kind": "str — event class label",
            "message": "str — free-form description",
        },
    )


# ---------------------------------------------------------------------------
# Catalogue: name groups for quick assembly.
# ---------------------------------------------------------------------------


MOBILE_BASE = [drive, stop, goto, rotate]
QUADRUPED   = [sit, stand, lie_down, walk_to, rotate, stop]
HUMANOID    = [wave, look_at, point_at, nod]
AERIAL      = [takeoff, land, fly_to, hover]
DEXTEROUS   = [set_joint, set_gripper]
SENSING     = [sense, scan_360, take_photo, read_battery]
GENERIC     = [wait, emit_event]

# Map category name -> list of factories. ``RobotProfile`` consumes this.
CATEGORIES = {
    "mobile_base": MOBILE_BASE,
    "quadruped":   QUADRUPED,
    "humanoid":    HUMANOID,
    "aerial":      AERIAL,
    "dexterous":   DEXTEROUS,
    "sensing":     SENSING,
    "generic":     GENERIC,
}


def primitives_for_categories(categories: list[str]) -> list[Primitive]:
    """Build a list of Primitive instances from category names.

    Categories not in the catalogue are silently ignored — passing
    user-typed strings is safe.
    """
    out: list[Primitive] = []
    seen: set[str] = set()
    for cat in categories:
        for factory in CATEGORIES.get(cat, []):
            prim = factory()
            if prim.name in seen:
                continue
            seen.add(prim.name)
            out.append(prim)
    return out
