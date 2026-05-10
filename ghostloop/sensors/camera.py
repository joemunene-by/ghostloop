"""Camera abstraction + MockCamera + capture_camera primitive.

Designed so a backend exposes ``backend.cameras: dict[str, Camera]`` and
the ``capture_camera`` primitive does the lookup by name. The Camera
protocol is small (capture / intrinsics / name) so backends can wrap
MuJoCo-rendered offscreen views, PyBullet's ``getCameraImage``, ROS 2's
``image_raw`` topics, or real RealSense / RGB-D devices uniformly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..core import Backend, Primitive, Result, ResultStatus


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics: fx, fy, cx, cy plus image dimensions.

    Stored alongside every CameraFrame so downstream CV pipelines can
    deproject pixels into 3D rays without out-of-band metadata files.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def to_json(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height,
            "fx": round(self.fx, 6), "fy": round(self.fy, 6),
            "cx": round(self.cx, 6), "cy": round(self.cy, 6),
        }


@dataclass
class CameraFrame:
    """One snapshot from a Camera.

    ``rgb`` and ``depth`` are intentionally typed as ``Any`` so backends
    can return numpy arrays, PIL Images, or raw bytes — whatever native
    type the underlying renderer produces. Downstream CameraProcessors
    are responsible for normalising. The metadata fields below ARE
    canonical and JSON-serialisable so the trace can record them safely.
    """

    name: str
    timestamp: float
    intrinsics: CameraIntrinsics
    rgb: Any = None  # opaque (numpy / PIL / bytes / None)
    depth: Any = None  # opaque
    rgb_shape: tuple[int, ...] | None = None
    depth_shape: tuple[int, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """JSON-safe view of the frame metadata (no opaque rgb/depth payload)."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "intrinsics": self.intrinsics.to_json(),
            "rgb_shape": list(self.rgb_shape) if self.rgb_shape else None,
            "depth_shape": list(self.depth_shape) if self.depth_shape else None,
            "extra": self.extra,
        }


class Camera(Protocol):
    """Anything that produces CameraFrames on demand."""

    name: str

    def capture(self) -> CameraFrame: ...


class CameraProcessor(Protocol):
    """A pluggable post-processor — object detector, depth refiner, etc.

    Implementations transform a CameraFrame into a structured observation
    dict (label, bbox, pose, ...). Stays out-of-band from the runtime so
    you can swap CV stacks without touching the agent layer.
    """

    name: str

    def process(self, frame: CameraFrame) -> dict[str, Any]: ...


@dataclass
class MockCamera:
    """In-memory deterministic camera for tests + sim demos.

    Generates a tiny 'gradient' RGB block (no numpy dep — a 2D list of
    (r,g,b) tuples) plus a constant depth field, so downstream code that
    expects rgb_shape / depth_shape can validate without a real renderer.
    """

    name: str = "mock_cam"
    width: int = 16
    height: int = 12
    detections: list[dict[str, Any]] = field(default_factory=list)
    intrinsics: CameraIntrinsics | None = None

    def __post_init__(self) -> None:
        if self.intrinsics is None:
            self.intrinsics = CameraIntrinsics(
                width=self.width, height=self.height,
                fx=float(self.width), fy=float(self.height),
                cx=self.width / 2.0, cy=self.height / 2.0,
            )

    def capture(self) -> CameraFrame:
        rgb = [
            [(int(255 * x / max(self.width - 1, 1)),
              int(255 * y / max(self.height - 1, 1)),
              128)
             for x in range(self.width)]
            for y in range(self.height)
        ]
        depth = [[1.0 for _ in range(self.width)] for _ in range(self.height)]
        return CameraFrame(
            name=self.name,
            timestamp=time.time(),
            intrinsics=self.intrinsics,  # type: ignore[arg-type]
            rgb=rgb,
            depth=depth,
            rgb_shape=(self.height, self.width, 3),
            depth_shape=(self.height, self.width),
            extra={"detections": list(self.detections)},
        )


def _capture_camera_call(backend: Backend, camera: str = "default") -> Result:
    """Look up ``camera`` on backend.cameras (or backend.cameras[default]) and capture.

    Falls back to a MockCamera-with-mock-detections when the backend has
    no cameras configured — keeps demos / tests usable without sim setup.
    """
    cameras = getattr(backend, "cameras", None)
    cam: Camera | None = None
    if isinstance(cameras, dict):
        cam = cameras.get(camera) or cameras.get("default")
    if cam is None:
        cam = MockCamera(name=camera)
    try:
        frame = cam.capture()
    except Exception as exc:  # noqa: BLE001
        return Result(
            status=ResultStatus.ERROR,
            message=f"camera {camera!r} capture failed: {exc}",
        )
    return Result(
        status=ResultStatus.OK,
        observation={"frame": frame.metadata()},
        message=f"captured frame from {camera!r}",
    )


def capture_camera() -> Primitive:
    """``capture_camera(camera="default")`` — sensor primitive for any backend
    that exposes a ``cameras: dict[str, Camera]`` attribute. Returns frame
    metadata (dimensions, intrinsics, timestamp, detections from extras)
    in the observation; the opaque RGB/depth payload lives on the frame
    object and is not serialised into the trace.
    """
    return Primitive(
        name="capture_camera",
        call=_capture_camera_call,
        description="Capture an RGB(+depth) frame from a named camera on the backend.",
        arg_schema={"camera": "str (optional, default 'default')"},
    )
