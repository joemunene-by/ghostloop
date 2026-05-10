"""Sensor abstractions for ghostloop backends.

A ``Camera`` produces ``CameraFrame``s with RGB, optional depth, intrinsics,
and a timestamp. Backends opt-in by attaching one or more cameras; the
``capture_camera`` primitive reads from a named camera and returns the
frame's metadata in the result observation. Real CV pipelines (object
detection, segmentation, depth estimation) plug in via ``CameraProcessor``.
"""

from .camera import (
    Camera,
    CameraFrame,
    CameraIntrinsics,
    CameraProcessor,
    MockCamera,
    capture_camera,
)

__all__ = [
    "Camera",
    "CameraFrame",
    "CameraIntrinsics",
    "CameraProcessor",
    "MockCamera",
    "capture_camera",
]
