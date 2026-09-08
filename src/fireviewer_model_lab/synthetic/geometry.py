"""Deterministic pinhole projection used for exact synthetic annotations."""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence


Vector3 = tuple[float, float, float]


def _vector(values: Sequence[float]) -> Vector3:
    if len(values) != 3:
        raise ValueError("a three-dimensional vector is required")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("vectors must contain finite values")
    return result  # type: ignore[return-value]


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(value: Vector3) -> Vector3:
    length = math.sqrt(_dot(value, value))
    if length <= 1e-9:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(component / length for component in value)  # type: ignore[return-value]


def camera_contract(
    *,
    position: Sequence[float],
    look_at: Sequence[float],
    width: int,
    height: int,
    focal_length_mm: float = 24.0,
    horizontal_aperture_mm: float = 20.955,
) -> dict[str, object]:
    position_vector = _vector(position)
    look_at_vector = _vector(look_at)
    forward = _normalize(_subtract(look_at_vector, position_vector))
    right = _normalize(_cross(forward, (0.0, 0.0, 1.0)))
    up = _normalize(_cross(right, forward))
    vertical_aperture_mm = horizontal_aperture_mm * height / width
    fx = width * focal_length_mm / horizontal_aperture_mm
    fy = height * focal_length_mm / vertical_aperture_mm
    return {
        "model": "pinhole_usd_camera",
        "position": list(position_vector),
        "look_at": list(look_at_vector),
        "axis": list(forward),
        "right_axis": list(right),
        "up_axis": list(up),
        "intrinsics": {
            "width_px": width,
            "height_px": height,
            "fx_px": round(fx, 9),
            "fy_px": round(fy, 9),
            "cx_px": width / 2.0,
            "cy_px": height / 2.0,
            "focal_length_mm": focal_length_mm,
            "horizontal_aperture_mm": horizontal_aperture_mm,
            "vertical_aperture_mm": vertical_aperture_mm,
        },
    }


def project_point(
    point: Sequence[float], camera: dict[str, object]
) -> dict[str, float]:
    world = _vector(point)
    position = _vector(camera["position"])  # type: ignore[arg-type]
    axis = _vector(camera["axis"])  # type: ignore[arg-type]
    right = _vector(camera["right_axis"])  # type: ignore[arg-type]
    up = _vector(camera["up_axis"])  # type: ignore[arg-type]
    relative = _subtract(world, position)
    depth = _dot(relative, axis)
    if depth <= 1e-6:
        raise ValueError("world point is behind the camera")
    intrinsics = camera["intrinsics"]
    if not isinstance(intrinsics, dict):
        raise ValueError("camera intrinsics are absent")
    width = float(intrinsics["width_px"])
    height = float(intrinsics["height_px"])
    x_px = float(intrinsics["cx_px"]) + float(intrinsics["fx_px"]) * _dot(relative, right) / depth
    y_px = float(intrinsics["cy_px"]) - float(intrinsics["fy_px"]) * _dot(relative, up) / depth
    return {
        "x_px": round(x_px, 6),
        "y_px": round(y_px, 6),
        "x_normalized": round(x_px / width, 9),
        "y_normalized": round(y_px / height, 9),
        "depth_m": round(depth, 6),
    }


def project_aabb(
    minimum: Sequence[float],
    maximum: Sequence[float],
    camera: dict[str, object],
) -> dict[str, float]:
    lower = _vector(minimum)
    upper = _vector(maximum)
    if any(a >= b for a, b in zip(lower, upper)):
        raise ValueError("AABB bounds must have positive extent")
    projected = [
        project_point(corner, camera)
        for corner in itertools.product(
            (lower[0], upper[0]),
            (lower[1], upper[1]),
            (lower[2], upper[2]),
        )
    ]
    x_values = [item["x_normalized"] for item in projected]
    y_values = [item["y_normalized"] for item in projected]
    box = {
        "x_min": max(0.0, min(x_values)),
        "y_min": max(0.0, min(y_values)),
        "x_max": min(1.0, max(x_values)),
        "y_max": min(1.0, max(y_values)),
    }
    if box["x_min"] >= box["x_max"] or box["y_min"] >= box["y_max"]:
        raise ValueError("projected AABB falls outside the image")
    return {key: round(value, 9) for key, value in box.items()}


def assert_visible(points: Iterable[dict[str, float]], *, margin: float = 0.01) -> None:
    for point in points:
        if not (
            margin <= point["x_normalized"] <= 1.0 - margin
            and margin <= point["y_normalized"] <= 1.0 - margin
        ):
            raise RuntimeError("synthetic annotation projects outside the review image")
