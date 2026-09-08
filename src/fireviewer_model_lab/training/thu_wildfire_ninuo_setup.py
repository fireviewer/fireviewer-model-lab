"""Adapt the THU-Wildfire Ninuo subset as a quarantined temporal corpus.

Ninuo contains a useful active-fire time series with aligned RGB, infrared and
radiometric thermal products.  It must not be represented as a true multiview
corpus: the released subset alternates several UAVs at nearly the same nadir
pose.  This module verifies that distinction and fails closed for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from fireviewer_model_lab.training.spatial_training_setup import SetupError, _sha256_file, _write_json

SOURCE_ID = "thu_wildfire_ninuo_v1"
EVENT_ID = "thu-wildfire:ninuo:2025-02-14"
EXPECTED_FRAMES = 389
EXPECTED_ANNOTATIONS = 125
MINIMUM_MULTIVIEW_BASELINE_METERS = 5.0
MINIMUM_INDEPENDENT_EVENTS = 3
GPS_IFD_TAG = 34853
MODEL_TRANSFORMATION_TAG = 34264
XMP_PATTERN = re.compile(rb"(?:drone-dji|Camera):([A-Za-z0-9_]+)=\"([^\"]*)\"")

MODALITY_DIRECTORIES = {
    "optical": "Active-fire/Image/Optical",
    "infrared_jpg": "Active-fire/Image/InfraredJPG",
    "optical_projected": "Active-fire/Image/Optical_projected",
    "thermal_radiometric": "Active-fire/Image/ThermalTIFF",
    "thermal_projected": "Active-fire/Image/ThermalTIFF_projected",
}


def _parse_dji_xmp(payload: bytes) -> dict[str, str]:
    return {
        key.decode("utf-8", errors="strict"): value.decode("utf-8", errors="strict")
        for key, value in XMP_PATTERN.findall(payload)
    }


def _required_float(metadata: dict[str, str], key: str, *, frame_id: int) -> float:
    raw = metadata.get(key)
    if raw is None:
        raise SetupError(f"frame {frame_id:06d} is missing DJI XMP field {key}")
    try:
        return float(raw)
    except ValueError as exc:
        raise SetupError(f"frame {frame_id:06d} has invalid DJI XMP field {key}") from exc


def _camera_id(serial: str) -> str:
    digest = hashlib.sha256(serial.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"dji-m30t-{digest[:12]}"


def _timestamp_and_pose(image_path: Path, *, frame_id: int) -> tuple[str, dict[str, Any]]:
    with Image.open(image_path) as image:
        exif = image.getexif()
        raw_timestamp = exif.get(306)
        if not isinstance(raw_timestamp, str):
            raise SetupError(f"frame {frame_id:06d} is missing EXIF DateTime")
        try:
            captured_at = datetime.strptime(raw_timestamp, "%Y:%m:%d %H:%M:%S")
        except ValueError as exc:
            raise SetupError(f"frame {frame_id:06d} has invalid EXIF DateTime") from exc
        gps_ifd = exif.get_ifd(GPS_IFD_TAG) if GPS_IFD_TAG in exif else {}
        gps_status = gps_ifd.get(9)

    metadata = _parse_dji_xmp(image_path.read_bytes())
    serial = metadata.get("DroneSerialNumber")
    if not serial:
        raise SetupError(f"frame {frame_id:06d} is missing DroneSerialNumber")
    pose = {
        "camera_id": _camera_id(serial),
        "camera_model": metadata.get("DroneModel", "DJI M30T"),
        "gps_status": metadata.get("GpsStatus") or gps_status,
        "rtk_flag": metadata.get("RtkFlag"),
        "source_shifted_latitude": _required_float(metadata, "GpsLatitude", frame_id=frame_id),
        "source_shifted_longitude": _required_float(metadata, "GpsLongitude", frame_id=frame_id),
        "absolute_altitude_m": _required_float(metadata, "AbsoluteAltitude", frame_id=frame_id),
        "relative_altitude_m": _required_float(metadata, "RelativeAltitude", frame_id=frame_id),
        "gimbal_roll_deg": _required_float(metadata, "GimbalRollDegree", frame_id=frame_id),
        "gimbal_yaw_deg": _required_float(metadata, "GimbalYawDegree", frame_id=frame_id),
        "gimbal_pitch_deg": _required_float(metadata, "GimbalPitchDegree", frame_id=frame_id),
        "flight_x_speed_mps": _required_float(metadata, "FlightXSpeed", frame_id=frame_id),
        "flight_y_speed_mps": _required_float(metadata, "FlightYSpeed", frame_id=frame_id),
        "flight_z_speed_mps": _required_float(metadata, "FlightZSpeed", frame_id=frame_id),
        "coordinates_are_source_shifted": True,
    }
    return captured_at.isoformat(), pose


def _geotransform(image_path: Path, *, frame_id: int) -> dict[str, Any]:
    with Image.open(image_path) as image:
        transform = image.tag_v2.get(MODEL_TRANSFORMATION_TAG)
        if transform is None or len(transform) != 16:
            raise SetupError(f"frame {frame_id:06d} projected TIFF has no 4x4 transform")
        width, height = image.size
        transform_values = [float(value) for value in transform]
        center_x = (
            transform_values[0] * (width / 2)
            + transform_values[1] * (height / 2)
            + transform_values[3]
        )
        center_y = (
            transform_values[4] * (width / 2)
            + transform_values[5] * (height / 2)
            + transform_values[7]
        )
    return {
        "width": width,
        "height": height,
        "model_transformation": transform_values,
        "source_shifted_center": [center_x, center_y],
        "coordinates_are_source_shifted": True,
    }


def _file_record(path: Path, source_root: Path, *, verify_hashes: bool) -> dict[str, Any]:
    if not path.is_file():
        raise SetupError(f"required THU-Wildfire file is missing: {path}")
    return {
        "relpath": path.relative_to(source_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path) if verify_hashes else None,
    }


def _haversine_meters(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_lat, left_lon = left
    right_lat, right_lon = right
    radius = 6_371_000.0
    left_phi = math.radians(left_lat)
    right_phi = math.radians(right_lat)
    delta_phi = math.radians(right_lat - left_lat)
    delta_lambda = math.radians(right_lon - left_lon)
    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(left_phi) * math.cos(right_phi) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(value))


def _summarize_geometry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positions_by_camera: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        pose = row["camera_pose"]
        positions_by_camera[pose["camera_id"]].append(
            (pose["source_shifted_latitude"], pose["source_shifted_longitude"])
        )

    centroids = {
        camera_id: (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for camera_id, points in positions_by_camera.items()
    }
    baselines = [
        _haversine_meters(centroids[left], centroids[right])
        for index, left in enumerate(sorted(centroids))
        for right in sorted(centroids)[index + 1 :]
    ]
    relative_altitudes = [float(row["camera_pose"]["relative_altitude_m"]) for row in rows]
    maximum_baseline = max(baselines, default=0.0)
    median_altitude = median(relative_altitudes)
    return {
        "camera_count": len(centroids),
        "camera_ids": sorted(centroids),
        "camera_centroids_source_shifted": {
            key: [value[0], value[1]] for key, value in sorted(centroids.items())
        },
        "maximum_camera_centroid_baseline_m": maximum_baseline,
        "median_relative_altitude_m": median_altitude,
        "baseline_to_altitude_ratio": maximum_baseline / median_altitude,
        "minimum_required_baseline_m": MINIMUM_MULTIVIEW_BASELINE_METERS,
    }


def _manifest_sha256(path: Path) -> str:
    return _sha256_file(path)


def prepare(
    source_root: Path,
    output: Path,
    *,
    license_id: str | None = None,
    verify_hashes: bool = True,
    expected_frames: int = EXPECTED_FRAMES,
    expected_annotations: int = EXPECTED_ANNOTATIONS,
) -> dict[str, Any]:
    """Verify Ninuo and write a non-training quarantine manifest."""

    source_root = source_root.resolve()
    output = output.resolve()
    if source_root.name != "Ninuo":
        raise SetupError("THU-Wildfire source root must be the extracted Ninuo directory")

    modality_paths = {
        name: source_root / relative for name, relative in MODALITY_DIRECTORIES.items()
    }
    for name, directory in modality_paths.items():
        if not directory.is_dir():
            raise SetupError(f"THU-Wildfire modality directory is missing: {name}")
    annotation_root = source_root / "Active-fire/Image/Annotation"
    if not annotation_root.is_dir():
        raise SetupError("THU-Wildfire annotation directory is missing")

    optical_frames = sorted(modality_paths["optical"].glob("*.jpg"))
    annotations = {int(path.stem): path for path in annotation_root.glob("*.png")}
    if len(optical_frames) != expected_frames:
        raise SetupError(
            f"THU-Wildfire frame count mismatch: {len(optical_frames)} != {expected_frames}"
        )
    if len(annotations) != expected_annotations:
        raise SetupError(
            f"THU-Wildfire annotation count mismatch: {len(annotations)} != {expected_annotations}"
        )

    rows: list[dict[str, Any]] = []
    previous_timestamp: datetime | None = None
    for optical_path in optical_frames:
        frame_id = int(optical_path.stem)
        stem = f"{frame_id:06d}"
        captured_at, pose = _timestamp_and_pose(optical_path, frame_id=frame_id)
        timestamp = datetime.fromisoformat(captured_at)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise SetupError(f"THU-Wildfire timestamp is not monotonic at frame {stem}")
        delta_seconds = (
            None if previous_timestamp is None else (timestamp - previous_timestamp).total_seconds()
        )
        previous_timestamp = timestamp

        files = {
            "optical": _file_record(optical_path, source_root, verify_hashes=verify_hashes),
            "infrared_jpg": _file_record(
                modality_paths["infrared_jpg"] / f"{stem}.jpg",
                source_root,
                verify_hashes=verify_hashes,
            ),
            "optical_projected": _file_record(
                modality_paths["optical_projected"] / f"{stem}.tif",
                source_root,
                verify_hashes=verify_hashes,
            ),
            "thermal_radiometric": _file_record(
                modality_paths["thermal_radiometric"] / f"{stem}.tif",
                source_root,
                verify_hashes=verify_hashes,
            ),
            "thermal_projected": _file_record(
                modality_paths["thermal_projected"] / f"{stem}.tif",
                source_root,
                verify_hashes=verify_hashes,
            ),
        }
        annotation_path = annotations.get(frame_id)
        annotation = (
            None
            if annotation_path is None
            else {
                **_file_record(annotation_path, source_root, verify_hashes=verify_hashes),
                "format": "binary_png_mask",
                "class_name": "active_fire",
                "source_annotation_provenance": "not_declared_in_released_subset",
            }
        )
        rows.append(
            {
                "schema_version": 1,
                "source_id": SOURCE_ID,
                "event_id": EVENT_ID,
                "sequence_id": EVENT_ID,
                "sample_id": f"{SOURCE_ID}:{stem}",
                "frame_id": frame_id,
                "captured_at": captured_at,
                "delta_seconds_from_previous": delta_seconds,
                "camera_pose": pose,
                "optical_georeference": _geotransform(
                    modality_paths["optical_projected"] / f"{stem}.tif",
                    frame_id=frame_id,
                ),
                "files": files,
                "annotation": annotation,
                "split": "quarantine",
                "split_group": EVENT_ID,
                "license": license_id or "UNVERIFIED",
                "training_membership": False,
                "temporal_fire_evolution": True,
                "true_multiview": False,
                "production_promotion_gate": False,
            }
        )

    geometry = _summarize_geometry(rows)
    timestamps = [datetime.fromisoformat(row["captured_at"]) for row in rows]
    intervals = [
        (timestamps[index] - timestamps[index - 1]).total_seconds()
        for index in range(1, len(timestamps))
    ]
    blockers: list[str] = []
    if not license_id:
        blockers.append("declared_license_missing")
    blockers.append("single_event_no_leakage_safe_train_validation_test_split")
    if geometry["maximum_camera_centroid_baseline_m"] < MINIMUM_MULTIVIEW_BASELINE_METERS:
        blockers.append("viewpoint_baseline_below_multiview_requirement")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    temporary_manifest = manifest_path.with_suffix(".jsonl.partial")
    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary_manifest, manifest_path)

    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_root": str(source_root),
        "manifest": str(manifest_path),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "files_verified_with_sha256": verify_hashes,
        "declared_license": license_id,
        "event_count": 1,
        "minimum_independent_events": MINIMUM_INDEPENDENT_EVENTS,
        "frames": len(rows),
        "annotated_frames": sum(row["annotation"] is not None for row in rows),
        "start": timestamps[0].isoformat(),
        "end": timestamps[-1].isoformat(),
        "duration_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
        "median_interval_seconds": median(intervals),
        "maximum_interval_seconds": max(intervals),
        "geometry": geometry,
        "quarantine_manifest_ready": True,
        "temporal_cross_modal_data_present": True,
        "temporal_training_ready": False,
        "true_multiview_training_ready": False,
        "recommended_role": "temporal_cross_modal_auxiliary_only",
        "blocking_reasons": blockers,
        "limitations": [
            "released subset contains indexed images rather than the complete videos",
            "all geographic coordinates are source-shifted",
            "annotation provenance is not declared in the released subset",
            "four UAVs relay through nearly the same nadir viewpoint",
        ],
    }
    _write_json(output / "preflight-report.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--license-id")
    parser.add_argument("--no-verify-files", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = prepare(
        args.source_root,
        args.output,
        license_id=args.license_id,
        verify_hashes=not args.no_verify_files,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.action == "preflight" and not report["true_multiview_training_ready"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
