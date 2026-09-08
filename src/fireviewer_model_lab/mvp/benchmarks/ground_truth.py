from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from hashlib import sha256
from math import isfinite
from typing import Any

from pydantic import AnyHttpUrl, Field, model_validator

from fireviewer_contracts.contracts import Sha256HexV2, StrictModel
from fireviewer_contracts.mvp.contracts.common import (
    is_timezone_aware,
    validate_lon_lat,
)
from fireviewer_geolocation.mvp.localization import haversine_m

MAX_GROUND_TRUTH_BYTES = 64 * 1024 * 1024


class GroundTruthParseError(ValueError):
    """Raised when an observed-event layer is not safe benchmark ground truth."""


class ObservedEventGroundTruth(StrictModel):
    source_url: AnyHttpUrl
    retrieved_at: datetime
    content_sha256: Sha256HexV2
    payload_size_bytes: int = Field(ge=1, le=MAX_GROUND_TRUTH_BYTES)
    feature_count: int = Field(ge=1)
    bbox_wgs84: tuple[float, float, float, float]
    center_wgs84: tuple[float, float]
    radius_m: float = Field(gt=0, le=1_000_000)
    source_area_sum: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_summary(self) -> ObservedEventGroundTruth:
        if not is_timezone_aware(self.retrieved_at):
            raise ValueError("ground-truth retrieval time must include a timezone")
        min_lon, min_lat, max_lon, max_lat = self.bbox_wgs84
        validate_lon_lat((min_lon, min_lat), label="ground-truth minimum corner")
        validate_lon_lat((max_lon, max_lat), label="ground-truth maximum corner")
        validate_lon_lat(self.center_wgs84, label="ground-truth center")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("ground-truth bbox must be ordered")
        return self


def _coordinate_pairs(value: object) -> Iterator[tuple[float, float]]:
    if not isinstance(value, list):
        raise GroundTruthParseError("GeoJSON coordinates must be arrays")
    if len(value) >= 2 and all(
        isinstance(coordinate, int | float) and not isinstance(coordinate, bool)
        for coordinate in value[:2]
    ):
        longitude, latitude = float(value[0]), float(value[1])
        if not isfinite(longitude) or not isfinite(latitude):
            raise GroundTruthParseError("GeoJSON coordinates must be finite")
        try:
            validate_lon_lat((longitude, latitude), label="observed-event coordinate")
        except ValueError as exc:
            raise GroundTruthParseError(str(exc)) from exc
        yield longitude, latitude
        return
    for child in value:
        yield from _coordinate_pairs(child)


def _feature_coordinates(feature: dict[str, Any]) -> Iterator[tuple[float, float]]:
    if feature.get("type") != "Feature":
        raise GroundTruthParseError("observed-event items must be GeoJSON Features")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise GroundTruthParseError("observed-event feature has no geometry")
    if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise GroundTruthParseError("observed-event geometry must be Polygon or MultiPolygon")
    yield from _coordinate_pairs(geometry.get("coordinates"))


def summarize_observed_event_geojson(
    payload: bytes,
    *,
    source_url: str,
    retrieved_at: datetime,
) -> ObservedEventGroundTruth:
    if not payload:
        raise GroundTruthParseError("observed-event payload is empty")
    if len(payload) > MAX_GROUND_TRUTH_BYTES:
        raise GroundTruthParseError("observed-event payload exceeds the 64 MiB safety cap")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundTruthParseError("observed-event payload is not valid JSON") from exc
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection":
        raise GroundTruthParseError("observed-event payload must be a FeatureCollection")
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise GroundTruthParseError("observed-event layer contains no features")

    min_lon = 180.0
    min_lat = 90.0
    max_lon = -180.0
    max_lat = -90.0
    area_sum = 0.0
    area_count = 0
    for raw_feature in features:
        if not isinstance(raw_feature, dict):
            raise GroundTruthParseError("observed-event layer contains a non-object feature")
        coordinate_count = 0
        for longitude, latitude in _feature_coordinates(raw_feature):
            min_lon = min(min_lon, longitude)
            min_lat = min(min_lat, latitude)
            max_lon = max(max_lon, longitude)
            max_lat = max(max_lat, latitude)
            coordinate_count += 1
        if coordinate_count < 4:
            raise GroundTruthParseError("observed-event polygon has too few coordinates")
        properties = raw_feature.get("properties")
        if isinstance(properties, dict):
            area = properties.get("area")
            if (
                isinstance(area, int | float)
                and not isinstance(area, bool)
                and isfinite(float(area))
                and area > 0
            ):
                area_sum += float(area)
                area_count += 1

    if min_lon >= max_lon or min_lat >= max_lat:
        raise GroundTruthParseError("observed-event layer has a degenerate extent")
    center = ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
    radius_m = max(
        haversine_m(center, corner)
        for corner in (
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
        )
    )
    return ObservedEventGroundTruth.model_validate(
        {
            "source_url": source_url,
            "retrieved_at": retrieved_at,
            "content_sha256": sha256(payload).hexdigest(),
            "payload_size_bytes": len(payload),
            "feature_count": len(features),
            "bbox_wgs84": (min_lon, min_lat, max_lon, max_lat),
            "center_wgs84": center,
            "radius_m": radius_m,
            "source_area_sum": area_sum if area_count else None,
        }
    )
