"""RGB-D fusion + lightweight object detection — v1.0 perception layer.

The v0.5 sensors module shipped a ``Camera`` Protocol with intrinsics
+ an opaque ``rgb`` / ``depth`` payload. This module adds the
processing layer on top:

  - ``deproject_depth(frame)`` — convert a depth image + intrinsics
    into a 3D point cloud (numpy if available, list-of-tuples
    fallback).
  - ``RGBDFusion`` — combine an RGB frame and a depth frame from the
    same camera into a single ``FusedFrame`` carrying the cloud + a
    per-pixel colour map.
  - ``BlobDetector`` — colour-threshold blob-finder that produces
    ``Detection`` records (bounding box + centroid in pixel + 3D
    space). Lightweight, stdlib-only fallback that works without
    OpenCV / mediapipe.
  - ``CameraProcessorPipeline`` — chain processors that operate on
    a ``CameraFrame`` and emit ``Detection``s.

The detection layer is deliberately simple: a colour-threshold
blob-finder. For production CV install ``ghostloop[perception]``
(adds OpenCV) and write your own ``Detector`` callable plugged into
the same pipeline. The Detection dataclass + the pipeline shape are
the stable contract; the detector implementation is swap-in.

Pure stdlib by default; numpy if available for vectorised path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .camera import CameraFrame, CameraIntrinsics


def _numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class Detection:
    """One detected object in a CameraFrame.

    Carries pixel-space bounding box + centroid in the source image AND
    (when depth + intrinsics are available) a 3D centroid via
    ray-deprojection. Downstream policies / properties / reward shapers
    consume detections without caring how they were produced.
    """

    label: str
    score: float                              # 0..1 confidence
    bbox: tuple[int, int, int, int]            # (x_min, y_min, x_max, y_max)
    centroid_px: tuple[float, float]
    centroid_3d: tuple[float, float, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "bbox": list(self.bbox),
            "centroid_px": [round(self.centroid_px[0], 2), round(self.centroid_px[1], 2)],
            "centroid_3d": (
                [round(c, 4) for c in self.centroid_3d]
                if self.centroid_3d is not None else None
            ),
            "extra": self.extra,
        }


@dataclass
class FusedFrame:
    """RGB-D fusion output: per-pixel colour + 3D point + intrinsics."""

    name: str
    timestamp: float
    intrinsics: CameraIntrinsics
    points: Any = None        # numpy (H,W,3) if numpy else list of (x,y,z)
    colors: Any = None        # numpy (H,W,3) uint8 OR list of (r,g,b)
    valid_mask: Any = None    # numpy (H,W) bool OR list of bool
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": round(self.timestamp, 6),
            "intrinsics": self.intrinsics.to_json(),
            "n_valid_points": _count_valid(self.valid_mask),
            "extra": self.extra,
        }


def _count_valid(mask: Any) -> int:
    if mask is None:
        return 0
    if hasattr(mask, "sum"):
        try:
            return int(mask.sum())
        except Exception:  # noqa: BLE001
            pass
    try:
        return sum(1 for v in mask if v)
    except TypeError:
        return 0


# ---------------------------------------------------------------------------
# Depth deprojection
# ---------------------------------------------------------------------------


def deproject_depth(
    frame: CameraFrame,
    *,
    valid_min: float = 1e-3,
    valid_max: float = 100.0,
) -> FusedFrame:
    """Convert a depth image + intrinsics into a 3D point cloud.

    Backend-agnostic — accepts numpy arrays (H,W) or lists of lists.
    Each pixel (u, v) deprojects to:

        x = (u - cx) * Z / fx
        y = (v - cy) * Z / fy
        z = Z                                    [where Z = depth[v, u]]

    Returns a ``FusedFrame`` with points + colors + a valid_mask
    (depths within [valid_min, valid_max]).

    Args:
        frame: source CameraFrame (must carry depth + intrinsics).
        valid_min/valid_max: depth values outside this range are
            marked invalid (sensor noise, missing returns, sky pixels).
    """
    intr = frame.intrinsics
    if frame.depth is None:
        raise ValueError(f"camera {frame.name!r} produced no depth payload")
    if _numpy_available():
        return _deproject_numpy(frame, valid_min, valid_max)
    return _deproject_pure(frame, valid_min, valid_max)


def _deproject_numpy(
    frame: CameraFrame, valid_min: float, valid_max: float,
) -> FusedFrame:
    import numpy as np
    intr = frame.intrinsics
    depth = np.asarray(frame.depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(
            f"depth must be 2D (H,W), got shape {depth.shape}"
        )
    H, W = depth.shape
    if (W, H) != (intr.width, intr.height):
        # Don't refuse — many backends return depth at intrinsics
        # resolution but tagging is loose. Just trust the intrinsics.
        pass
    u = np.arange(W).reshape(1, W).astype(np.float32)
    v = np.arange(H).reshape(H, 1).astype(np.float32)
    Z = depth
    X = (u - intr.cx) * Z / intr.fx
    Y = (v - intr.cy) * Z / intr.fy
    points = np.stack([X, Y, Z], axis=-1)
    valid = (Z >= valid_min) & (Z <= valid_max) & np.isfinite(Z)
    colors = None
    if frame.rgb is not None:
        try:
            colors = np.asarray(frame.rgb, dtype=np.uint8)
        except Exception:  # noqa: BLE001
            colors = None
    return FusedFrame(
        name=frame.name, timestamp=frame.timestamp, intrinsics=intr,
        points=points, colors=colors, valid_mask=valid,
    )


def _deproject_pure(
    frame: CameraFrame, valid_min: float, valid_max: float,
) -> FusedFrame:
    intr = frame.intrinsics
    depth = frame.depth
    points: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int] | None] = []
    valid: list[bool] = []
    rgb = frame.rgb if isinstance(frame.rgb, (list, tuple)) else None
    for v_idx, row in enumerate(depth):
        rgb_row = rgb[v_idx] if rgb and v_idx < len(rgb) else None
        for u_idx, Z in enumerate(row):
            try:
                Z_f = float(Z)
            except (TypeError, ValueError):
                points.append((0.0, 0.0, 0.0))
                colors.append(None)
                valid.append(False)
                continue
            X = (u_idx - intr.cx) * Z_f / intr.fx
            Y = (v_idx - intr.cy) * Z_f / intr.fy
            points.append((X, Y, Z_f))
            valid.append(valid_min <= Z_f <= valid_max)
            if rgb_row is not None and u_idx < len(rgb_row):
                colors.append(tuple(rgb_row[u_idx]))
            else:
                colors.append(None)
    return FusedFrame(
        name=frame.name, timestamp=frame.timestamp, intrinsics=intr,
        points=points, colors=colors, valid_mask=valid,
    )


# ---------------------------------------------------------------------------
# Lightweight blob detector
# ---------------------------------------------------------------------------


@dataclass
class ColorTarget:
    """One target colour for the blob detector."""

    label: str
    rgb_min: tuple[int, int, int]
    rgb_max: tuple[int, int, int]


class Detector(Protocol):
    """Pluggable detector contract: CameraFrame -> list[Detection]."""

    def detect(self, frame: CameraFrame) -> list[Detection]: ...


@dataclass
class BlobDetector:
    """Colour-threshold blob finder. Stdlib + optional numpy.

    For each ``ColorTarget``, finds connected pixel clusters whose RGB
    fits ``(rgb_min, rgb_max)`` componentwise, returns one Detection
    per cluster. Slow without numpy (O(WH) Python loop) but adequate
    for low-res sim cameras + tests; install OpenCV or torchvision
    and write a real detector for real cameras.
    """

    targets: list[ColorTarget]
    min_area_px: int = 16

    def detect(self, frame: CameraFrame) -> list[Detection]:
        if frame.rgb is None:
            return []
        if _numpy_available():
            return self._detect_numpy(frame)
        return self._detect_pure(frame)

    def _detect_numpy(self, frame: CameraFrame) -> list[Detection]:
        import numpy as np
        rgb = np.asarray(frame.rgb)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            return []
        out: list[Detection] = []
        for target in self.targets:
            r_min, g_min, b_min = target.rgb_min
            r_max, g_max, b_max = target.rgb_max
            mask = (
                (rgb[..., 0] >= r_min) & (rgb[..., 0] <= r_max)
                & (rgb[..., 1] >= g_min) & (rgb[..., 1] <= g_max)
                & (rgb[..., 2] >= b_min) & (rgb[..., 2] <= b_max)
            )
            if not mask.any():
                continue
            ys, xs = np.where(mask)
            if xs.size < self.min_area_px:
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cx_px = float(xs.mean())
            cy_px = float(ys.mean())
            centroid_3d = self._lookup_3d(frame, cx_px, cy_px)
            out.append(Detection(
                label=target.label,
                score=min(1.0, xs.size / (mask.size or 1) * 50),
                bbox=(x_min, y_min, x_max, y_max),
                centroid_px=(cx_px, cy_px),
                centroid_3d=centroid_3d,
                extra={"area_px": int(xs.size)},
            ))
        return out

    def _detect_pure(self, frame: CameraFrame) -> list[Detection]:
        rgb = frame.rgb
        if not isinstance(rgb, (list, tuple)) or not rgb:
            return []
        H = len(rgb)
        W = len(rgb[0]) if rgb[0] else 0
        out: list[Detection] = []
        for target in self.targets:
            r_min, g_min, b_min = target.rgb_min
            r_max, g_max, b_max = target.rgb_max
            xs: list[int] = []
            ys: list[int] = []
            for y in range(H):
                for x in range(W):
                    px = rgb[y][x]
                    if not (
                        r_min <= px[0] <= r_max
                        and g_min <= px[1] <= g_max
                        and b_min <= px[2] <= b_max
                    ):
                        continue
                    xs.append(x)
                    ys.append(y)
            if len(xs) < self.min_area_px:
                continue
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx_px = sum(xs) / len(xs)
            cy_px = sum(ys) / len(ys)
            centroid_3d = self._lookup_3d(frame, cx_px, cy_px)
            out.append(Detection(
                label=target.label,
                score=min(1.0, len(xs) / max(1, W * H) * 50),
                bbox=(x_min, y_min, x_max, y_max),
                centroid_px=(cx_px, cy_px),
                centroid_3d=centroid_3d,
                extra={"area_px": len(xs)},
            ))
        return out

    def _lookup_3d(
        self, frame: CameraFrame, cx_px: float, cy_px: float,
    ) -> tuple[float, float, float] | None:
        if frame.depth is None:
            return None
        intr = frame.intrinsics
        u, v = int(round(cx_px)), int(round(cy_px))
        try:
            if _numpy_available():
                import numpy as np
                Z = float(np.asarray(frame.depth)[v, u])
            else:
                Z = float(frame.depth[v][u])
        except (IndexError, TypeError, ValueError):
            return None
        if not math.isfinite(Z) or Z <= 0:
            return None
        X = (u - intr.cx) * Z / intr.fx
        Y = (v - intr.cy) * Z / intr.fy
        return (X, Y, Z)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class CameraProcessorPipeline:
    """Chain detectors. Returns the union of every detector's outputs."""

    detectors: list[Detector] = field(default_factory=list)

    def process(self, frame: CameraFrame) -> list[Detection]:
        out: list[Detection] = []
        for d in self.detectors:
            try:
                out.extend(d.detect(frame))
            except Exception:  # noqa: BLE001
                # Detector errors should not crash the pipeline; just
                # skip that detector for this frame.
                continue
        return out
