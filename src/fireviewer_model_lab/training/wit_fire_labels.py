"""Derive conservative fire labels from WIT-UAS thermal observations.

WIT-UAS annotates crew assets rather than fire. Those boxes are still valuable:
they define regions that must not be promoted to fire merely because a person or
vehicle is hot in LWIR imagery. The functions below operate on decoded, registered
thermal arrays; archive- and ROS-specific decoding stays in the acquisition layer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalFireLabel:
    mask: np.ndarray
    fire_base_xy: tuple[float, float] | None
    abstain: bool
    quality: str


def calibrate_hot_threshold(
    pre_ignition_frames: np.ndarray,
    *,
    quantile: float = 0.999,
    mad_multiplier: float = 8.0,
) -> float:
    """Calibrate a native-unit threshold from source-specific negative frames."""

    values = np.asarray(pre_ignition_frames, dtype=np.float64)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("pre-ignition thermal frames must have shape [T, H, W]")
    if not np.isfinite(values).all():
        raise ValueError("pre-ignition thermal frames contain non-finite values")
    if not 0.5 < quantile < 1.0:
        raise ValueError("thermal calibration quantile must be between 0.5 and 1")
    if mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be positive")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_limit = median + mad_multiplier * max(mad, np.finfo(np.float64).eps)
    return max(float(np.quantile(values, quantile)), robust_limit)


def exclusion_mask(
    shape: tuple[int, int],
    boxes_xyxy: Iterable[tuple[float, float, float, float]],
    *,
    margin_pixels: int,
) -> np.ndarray:
    if margin_pixels < 0:
        raise ValueError("asset exclusion margin cannot be negative")
    height, width = shape
    excluded = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        left = max(0, int(np.floor(min(x1, x2))) - margin_pixels)
        right = min(width, int(np.ceil(max(x1, x2))) + margin_pixels)
        top = max(0, int(np.floor(min(y1, y2))) - margin_pixels)
        bottom = min(height, int(np.ceil(max(y1, y2))) + margin_pixels)
        if right > left and bottom > top:
            excluded[top:bottom, left:right] = True
    return excluded


def temporal_vote(aligned_masks: np.ndarray, *, minimum_votes: int) -> np.ndarray:
    masks = np.asarray(aligned_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("aligned masks must have shape [T, H, W]")
    if not 1 <= minimum_votes <= masks.shape[0]:
        raise ValueError("minimum_votes must be within the temporal window")
    return np.count_nonzero(masks, axis=0) >= minimum_votes


def fire_base_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("fire mask must be two-dimensional")
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    bottom = int(ys.max())
    band_start = max(int(ys.min()), bottom - max(1, int(binary.shape[0] * 0.02)))
    band_x = xs[ys >= band_start]
    return float(np.median(band_x)), float(bottom)


def derive_wit_fire_label(
    thermal_frame: np.ndarray,
    *,
    hot_threshold: float,
    asset_boxes_xyxy: Iterable[tuple[float, float, float, float]] = (),
    asset_margin_pixels: int = 4,
    teacher_mask: np.ndarray | None = None,
    temporal_support_mask: np.ndarray | None = None,
    minimum_pixels: int = 16,
) -> ThermalFireLabel:
    """Create a weak fire mask only where all required signals agree.

    A hot LWIR region alone is never sufficient: sun-heated terrain, engines,
    people, and equipment can all produce the same threshold response. The
    default WIT admission rule therefore requires both independent semantic
    support and temporal persistence.
    """

    thermal = np.asarray(thermal_frame)
    if thermal.ndim != 2 or not np.isfinite(thermal).all():
        raise ValueError("thermal_frame must be a finite two-dimensional array")
    if not np.isfinite(hot_threshold):
        raise ValueError("hot_threshold must be finite")
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")

    candidate = thermal >= hot_threshold
    excluded = exclusion_mask(thermal.shape, asset_boxes_xyxy, margin_pixels=asset_margin_pixels)
    candidate &= ~excluded
    evidence = ["thermal_calibrated", "asset_excluded"]
    if teacher_mask is None or temporal_support_mask is None:
        candidate[:] = False
        missing = []
        if teacher_mask is None:
            missing.append("teacher_consensus")
        if temporal_support_mask is None:
            missing.append("temporal_consensus")
        return ThermalFireLabel(
            mask=candidate,
            fire_base_xy=None,
            abstain=True,
            quality="sensor_derived_weak:missing_" + "+".join(missing),
        )

    teacher = np.asarray(teacher_mask, dtype=bool)
    if teacher.shape != thermal.shape:
        raise ValueError("teacher mask shape does not match thermal frame")
    candidate &= teacher
    evidence.append("teacher_consensus")

    temporal = np.asarray(temporal_support_mask, dtype=bool)
    if temporal.shape != thermal.shape:
        raise ValueError("temporal support shape does not match thermal frame")
    candidate &= temporal
    evidence.append("temporal_consensus")

    if int(np.count_nonzero(candidate)) < minimum_pixels:
        candidate[:] = False
        return ThermalFireLabel(
            mask=candidate,
            fire_base_xy=None,
            abstain=True,
            quality="sensor_derived_weak:" + "+".join(evidence),
        )
    return ThermalFireLabel(
        mask=candidate,
        fire_base_xy=fire_base_from_mask(candidate),
        abstain=False,
        quality="sensor_derived_weak:" + "+".join(evidence),
    )
