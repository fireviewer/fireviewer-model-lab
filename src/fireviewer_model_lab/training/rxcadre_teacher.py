"""Prepare bounded RxCADRE validation assets with an immutable DINO teacher."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from fireviewer_model_lab.training.legacy_dinov3_teacher import load_published_teacher

SOURCE_REVISION = "RDS-2018-0033:JPG_L2F_CA4-CA5"
CAMERAS = ("L2F_CA4", "L2F_CA5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes campaign root: {path}")
    return path.relative_to(root).as_posix()


def _capture_time(path: Path) -> datetime:
    with Image.open(path) as image:
        exif = image.getexif()
        value = exif.get(36867) or exif.get(306)
    if not value:
        raise ValueError(f"RxCADRE image has no capture timestamp: {path.name}")
    return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")


def select_by_interval(
    observations: list[tuple[datetime, Path]], interval_seconds: int
) -> list[tuple[datetime, Path]]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    selected: list[tuple[datetime, Path]] = []
    last: datetime | None = None
    for timestamp, path in sorted(observations, key=lambda item: (item[0], item[1].name)):
        if last is None or (timestamp - last).total_seconds() >= interval_seconds:
            selected.append((timestamp, path))
            last = timestamp
    return selected


def _filter_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask, dtype=np.uint8)
    for component in range(1, count):
        if int(statistics[component, cv2.CC_STAT_AREA]) >= minimum_pixels:
            kept[labels == component] = 255
    return kept


def _resize_dimensions(width: int, height: int, maximum_edge: int) -> tuple[int, int]:
    scale = min(1.0, maximum_edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _save(path: Path, image: Image.Image, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    image.save(temporary, **kwargs)
    os.replace(temporary, path)


def _anchor(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return None
    height, width = mask.shape
    return float(xs.mean()) / width, float(ys.mean()) / height


def _pair_records(
    left: list[dict[str, Any]], right: list[dict[str, Any]], max_delta_seconds: int
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    right_times = [float(row["timestamp_epoch"]) for row in right]
    candidates: list[tuple[float, int, int]] = []
    for left_index, row in enumerate(left):
        timestamp = float(row["timestamp_epoch"])
        insertion = bisect.bisect_left(right_times, timestamp)
        for right_index in (insertion - 1, insertion):
            if 0 <= right_index < len(right):
                delta = abs(right_times[right_index] - timestamp)
                if delta <= max_delta_seconds:
                    candidates.append((delta, left_index, right_index))
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for delta, left_index, right_index in sorted(candidates):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        pairs.append((left[left_index], right[right_index], delta))
    return sorted(pairs, key=lambda pair: float(pair[0]["timestamp_epoch"]))


def build_rxcadre_teacher_corpus(
    *,
    campaign_root: Path,
    source_root: Path,
    output_root: Path,
    model_path: Path,
    model_revision: str,
    teacher_checkpoint: Path,
    teacher_repository_revision: str,
    interval_seconds: int = 60,
    maximum_edge: int = 896,
    batch_size: int = 12,
    probability_threshold: float = 0.5,
    minimum_component_pixels: int = 4,
    pair_max_delta_seconds: int = 20,
    pre_fire_margin_seconds: int = 300,
    negative_limit_per_camera: int = 12,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RxCADRE teacher inference")
    if batch_size <= 0 or not 0.0 < probability_threshold < 1.0:
        raise ValueError("invalid RxCADRE teacher parameters")
    campaign_root = campaign_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    _relative(campaign_root, source_root)
    _relative(campaign_root, output_root)
    device = torch.device("cuda")
    teacher = load_published_teacher(
        model_id=str(model_path.resolve()),
        revision=model_revision,
        checkpoint=teacher_checkpoint.resolve(),
        device=device,
    )
    selected_by_camera: dict[str, list[tuple[datetime, Path]]] = {}
    for camera in CAMERAS:
        paths = sorted((source_root / camera).glob("*.JPG"))
        observations = [(_capture_time(path), path) for path in paths]
        selected_by_camera[camera] = select_by_interval(observations, interval_seconds)
    work = [
        (camera, timestamp, path)
        for camera in CAMERAS
        for timestamp, path in selected_by_camera[camera]
    ]
    rows: list[dict[str, Any]] = []
    skipped_teacher_empty = 0
    negative_candidates: dict[str, list[tuple[datetime, Path]]] = {camera: [] for camera in CAMERAS}
    records_by_camera: dict[str, list[dict[str, Any]]] = {camera: [] for camera in CAMERAS}
    for start in range(0, len(work), batch_size):
        items = work[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        canonical: list[tuple[Image.Image, int, int]] = []
        for _, _, path in items:
            with Image.open(path) as opened:
                rgb = opened.convert("RGB")
                width, height = _resize_dimensions(rgb.width, rgb.height, maximum_edge)
                resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
                teacher_image = rgb.resize((224, 224), Image.Resampling.BILINEAR)
            tensor = (
                torch.from_numpy(np.asarray(teacher_image, dtype=np.float32).copy())
                .permute(2, 0, 1)
                .div_(255.0)
            )
            tensors.append(tensor)
            canonical.append((resized, width, height))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = teacher(torch.stack(tensors).to(device))
            probabilities = outputs["segmentation_logits"].sigmoid().float().cpu().numpy()
        for item, prepared, probability in zip(items, canonical, probabilities, strict=True):
            camera, timestamp, source_path = item
            image, width, height = prepared
            mask_224 = _filter_components(
                probability[0] >= probability_threshold, minimum_component_pixels
            )
            mask_image = Image.fromarray(mask_224).resize((width, height), Image.Resampling.NEAREST)
            mask_array = np.asarray(mask_image, dtype=np.uint8)
            anchor = _anchor(mask_array)
            if anchor is None:
                skipped_teacher_empty += 1
                negative_candidates[camera].append((timestamp, source_path))
                continue
            sample_stem = f"{camera.lower()}-{timestamp.strftime('%Y%m%d-%H%M%S')}"
            image_path = output_root / "images" / camera.lower() / f"{sample_stem}.jpg"
            mask_path = output_root / "masks" / camera.lower() / f"{sample_stem}.png"
            _save(image_path, image, quality=92, optimize=True, subsampling=0)
            _save(mask_path, mask_image)
            row = {
                "sample_id": f"rxcadre:{sample_stem}",
                "source_id": "RxCADRE 2012 L2F CA4-CA5",
                "source_revision": SOURCE_REVISION,
                "split": "validation",
                "split_group": "rxcadre:l2f:2012-11-11",
                "image_relpath": _relative(campaign_root, image_path),
                "image_sha256": _sha256(image_path),
                "mask_relpath": _relative(campaign_root, mask_path),
                "mask_sha256": _sha256(mask_path),
                "mask_quality": "immutable_dinov3_boreal_teacher_weak",
                "annotation_strength": "weak",
                "sample_validation_status": "teacher_generated_weak",
                "anchor_points": [{"kind": "smoke_centroid", "x": anchor[0], "y": anchor[1]}],
                "visual_abstention_reason": None,
                "license": "CC-BY-4.0",
                "redistribution_allowed": True,
                "is_operational_incident": False,
                "camera": camera,
                "capture_time": timestamp.isoformat(),
                "timestamp_epoch": timestamp.timestamp(),
                "source_filename": source_path.name,
                "source_sha256": _sha256(source_path),
                "teacher_repository_revision": teacher_repository_revision,
            }
            rows.append(row)
            records_by_camera[camera].append(row)
    negative_rows = 0
    for camera in CAMERAS:
        positives = records_by_camera[camera]
        if not positives:
            continue
        first_positive = min(float(row["timestamp_epoch"]) for row in positives)
        eligible = [
            item
            for item in sorted(negative_candidates[camera], key=lambda item: item[0])
            if item[0].timestamp() <= first_positive - pre_fire_margin_seconds
        ]
        if len(eligible) > negative_limit_per_camera:
            indices = np.linspace(0, len(eligible) - 1, negative_limit_per_camera, dtype=int)
            eligible = [eligible[int(index)] for index in sorted(set(indices.tolist()))]
        for timestamp, source_path in eligible:
            with Image.open(source_path) as opened:
                rgb = opened.convert("RGB")
                width, height = _resize_dimensions(rgb.width, rgb.height, maximum_edge)
                image = rgb.resize((width, height), Image.Resampling.LANCZOS)
            empty_mask = Image.new("L", (width, height), 0)
            sample_stem = f"{camera.lower()}-{timestamp.strftime('%Y%m%d-%H%M%S')}-prefire"
            image_path = output_root / "images" / camera.lower() / f"{sample_stem}.jpg"
            mask_path = output_root / "masks" / camera.lower() / f"{sample_stem}.png"
            _save(image_path, image, quality=92, optimize=True, subsampling=0)
            _save(mask_path, empty_mask)
            rows.append(
                {
                    "sample_id": f"rxcadre:{sample_stem}",
                    "source_id": "RxCADRE 2012 L2F CA4-CA5",
                    "source_revision": SOURCE_REVISION,
                    "split": "validation",
                    "split_group": "rxcadre:l2f:2012-11-11",
                    "image_relpath": _relative(campaign_root, image_path),
                    "image_sha256": _sha256(image_path),
                    "mask_relpath": _relative(campaign_root, mask_path),
                    "mask_sha256": _sha256(mask_path),
                    "mask_quality": "temporal_pre_fire_negative_weak",
                    "annotation_strength": "weak",
                    "sample_validation_status": "teacher_generated_weak",
                    "anchor_points": [],
                    "visual_abstention_reason": "pre_fire_before_first_positive_with_margin",
                    "license": "CC-BY-4.0",
                    "redistribution_allowed": True,
                    "is_operational_incident": False,
                    "camera": camera,
                    "capture_time": timestamp.isoformat(),
                    "timestamp_epoch": timestamp.timestamp(),
                    "source_filename": source_path.name,
                    "source_sha256": _sha256(source_path),
                    "teacher_repository_revision": teacher_repository_revision,
                }
            )
            negative_rows += 1
            skipped_teacher_empty -= 1
    rows.sort(key=lambda row: str(row["sample_id"]))
    segmentation_manifest = output_root / "segmentation-manifest.jsonl"
    segmentation_manifest.parent.mkdir(parents=True, exist_ok=True)
    segmentation_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    pair_rows: list[dict[str, Any]] = []
    left = sorted(records_by_camera[CAMERAS[0]], key=lambda row: row["timestamp_epoch"])
    right = sorted(records_by_camera[CAMERAS[1]], key=lambda row: row["timestamp_epoch"])
    for pair_index, (source, target, delta) in enumerate(
        _pair_records(left, right, pair_max_delta_seconds)
    ):
        pair_rows.append(
            {
                "pair_id": f"rxcadre:l2f:{pair_index:05d}",
                "split": "validation",
                "split_group": "rxcadre:l2f:2012-11-11",
                "source_id": "RxCADRE 2012 L2F CA4-CA5",
                "source_revision": SOURCE_REVISION,
                "source_image_relpath": source["image_relpath"],
                "source_transient_mask_relpath": source["mask_relpath"],
                "map_image_relpath": target["image_relpath"],
                "map_transient_mask_relpath": target["mask_relpath"],
                "capture_delta_seconds": delta,
                "license": "CC-BY-4.0",
                "purpose": "roma_pycolmap_validation_only",
            }
        )
    pair_manifest = output_root / "roma-validation-pairs.jsonl"
    pair_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in pair_rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "dataset_family": "rxcadre-l2f-shared-v1",
        "source_revision": SOURCE_REVISION,
        "segmentation_manifest": _relative(campaign_root, segmentation_manifest),
        "segmentation_manifest_sha256": _sha256(segmentation_manifest),
        "roma_validation_manifest": _relative(campaign_root, pair_manifest),
        "roma_validation_manifest_sha256": _sha256(pair_manifest),
        "rows": len(rows),
        "positive_rows": len(rows) - negative_rows,
        "abstention_rows": negative_rows,
        "teacher_empty_rows_excluded": skipped_teacher_empty,
        "camera_counts": {camera: len(records_by_camera[camera]) for camera in CAMERAS},
        "roma_validation_pairs": len(pair_rows),
        "parameters": {
            "interval_seconds": interval_seconds,
            "maximum_edge": maximum_edge,
            "probability_threshold": probability_threshold,
            "minimum_component_pixels_at_224": minimum_component_pixels,
            "pair_max_delta_seconds": pair_max_delta_seconds,
            "pre_fire_margin_seconds": pre_fire_margin_seconds,
            "negative_limit_per_camera": negative_limit_per_camera,
        },
        "teacher": {
            "repository_revision": teacher_repository_revision,
            "checkpoint_sha256": _sha256(teacher_checkpoint),
            "base_model_revision": model_revision,
        },
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
