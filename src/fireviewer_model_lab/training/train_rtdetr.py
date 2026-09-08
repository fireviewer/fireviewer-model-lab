from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fireviewer_model_lab.training.corpus_pipeline import CLASS_NAMES, sha256_bytes, validate_manifest

BASE_MODEL = "PekingU/rtdetr_v2_r50vd"
BASE_MODEL_REVISION = "282494075698cab9faa1096ae26856890030c817"
DEFAULT_OUTPUT = Path("data/training/rtdetr-v2-r50")
DETECTOR_ROLES = {"detector_training", "detector_critical_test"}
TRAINING_PROFILES = {
    "media_filter_v1": {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]},
    "operational_four_class_v1": dict(CLASS_NAMES),
}
DEFAULT_GLOBAL_IMAGE_SIZE = 768
DEFAULT_MULTISCALE_SIZES = (640, 768, 896, 960)
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 0.25
DEFAULT_MAX_POSITIVE_TILES = 1
DEFAULT_NEGATIVE_TILE_FRACTION = 0.05
OBJECT_SIZE_THRESHOLDS_PX = (4.0, 8.0, 16.0, 32.0)
OBJECT_SIZE_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


@dataclass(frozen=True)
class LoadedRecord:
    record: dict[str, Any]
    corpus_root: Path


@dataclass(frozen=True)
class DetectorModelSpec:
    family: str
    model_id: str
    revision: str
    output_prefix: str
    default_output: Path


RTDETR_MODEL_SPEC = DetectorModelSpec(
    family="RT-DETRv2-R50",
    model_id=BASE_MODEL,
    revision=BASE_MODEL_REVISION,
    output_prefix="FW_RTDETR",
    default_output=DEFAULT_OUTPUT,
)


def _resolve_resume_checkpoint(
    output_dir: Path,
    requested: str | None,
) -> str | None:
    if requested is None:
        return None
    if requested != "auto":
        checkpoint = Path(requested).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Training checkpoint is missing: {checkpoint}")
        return str(checkpoint)
    candidates: list[tuple[int, Path]] = []
    for checkpoint in output_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, checkpoint))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1].resolve())


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile from an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _percentile_map(values: list[float]) -> dict[str, float]:
    return {
        f"p{percentile:02d}": round(_percentile(values, percentile), 3)
        for percentile in OBJECT_SIZE_PERCENTILES
    }


def _summarize_object_sizes(values: list[tuple[float, float]]) -> dict[str, Any]:
    if not values:
        return {
            "objects": 0,
            "min_dimension_percentiles_px": {},
            "width_percentiles_px": {},
            "height_percentiles_px": {},
            "area_percentiles_px2": {},
            "under_threshold_counts": {},
            "under_threshold_ratios": {},
        }
    widths = [value[0] for value in values]
    heights = [value[1] for value in values]
    minimum_dimensions = [min(value) for value in values]
    areas = [width * height for width, height in values]
    total = len(values)
    return {
        "objects": total,
        "min_dimension_percentiles_px": _percentile_map(minimum_dimensions),
        "width_percentiles_px": _percentile_map(widths),
        "height_percentiles_px": _percentile_map(heights),
        "area_percentiles_px2": _percentile_map(areas),
        "under_threshold_counts": {
            f"lt_{int(threshold)}px": sum(value < threshold for value in minimum_dimensions)
            for threshold in OBJECT_SIZE_THRESHOLDS_PX
        },
        "under_threshold_ratios": {
            f"lt_{int(threshold)}px": round(
                sum(value < threshold for value in minimum_dimensions) / total, 6
            )
            for threshold in OBJECT_SIZE_THRESHOLDS_PX
        },
    }


def build_object_size_audit(
    records: list[LoadedRecord],
    *,
    allowed_class_names: frozenset[str],
    target_sizes: tuple[int, ...] = DEFAULT_MULTISCALE_SIZES,
) -> dict[str, Any]:
    sizes = tuple(sorted(set(target_sizes)))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("target_sizes must contain positive image sizes")
    by_size: dict[int, list[tuple[float, float]]] = {size: [] for size in sizes}
    by_class: dict[int, dict[str, list[tuple[float, float]]]] = {
        size: defaultdict(list) for size in sizes
    }
    source_dimensions: Counter[str] = Counter()
    for loaded in records:
        record = loaded.record
        if str(record.get("corpus_role")) != "detector_training" or str(
            record.get("split")
        ) not in {"train", "validation"}:
            continue
        width = int(record["width"])
        height = int(record["height"])
        source_dimensions[f"{width}x{height}"] += 1
        for annotation in record["annotations"]:
            class_name = str(annotation["class_name"])
            if class_name not in allowed_class_names:
                continue
            _x, _y, box_width, box_height = (float(value) for value in annotation["bbox_xywh"])
            for size in sizes:
                scale = min(size / width, size / height)
                scaled_size = (box_width * scale, box_height * scale)
                by_size[size].append(scaled_size)
                by_class[size][class_name].append(scaled_size)
    return {
        "resize_policy": "preserve_aspect_ratio_then_square_pad",
        "included_roles": ["detector_training"],
        "included_splits": ["train", "validation"],
        "target_sizes": list(sizes),
        "source_dimension_unique_count": len(source_dimensions),
        "source_dimension_counts_top_20": dict(source_dimensions.most_common(20)),
        "sizes": {
            str(size): {
                "all_classes": _summarize_object_sizes(by_size[size]),
                "by_class": {
                    class_name: _summarize_object_sizes(values)
                    for class_name, values in sorted(by_class[size].items())
                },
            }
            for size in sizes
        },
    }


def _read_manifest(path: Path, *, verify_files: bool) -> list[LoadedRecord]:
    validate_manifest(path, output_dir=path.parent, verify_files=verify_files)
    loaded: list[LoadedRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest {path} line {line_number} is not an object")
            loaded.append(LoadedRecord(value, path.parent.resolve()))
    return loaded


def load_records(manifests: list[Path], *, verify_files: bool) -> list[LoadedRecord]:
    if not manifests:
        raise ValueError("At least one --manifest is required")
    records: list[LoadedRecord] = []
    seen_digests: dict[str, Path] = {}
    for manifest in manifests:
        for loaded in _read_manifest(manifest.resolve(), verify_files=verify_files):
            digest = str(loaded.record["sha256"])
            if digest in seen_digests:
                raise ValueError(
                    f"Duplicate image digest across manifests: {digest} "
                    f"({seen_digests[digest]} and {manifest})"
                )
            seen_digests[digest] = manifest
            records.append(loaded)
    return records


def build_preflight_report(
    records: list[LoadedRecord],
    *,
    model_spec: DetectorModelSpec = RTDETR_MODEL_SPEC,
    profile: str = "operational_four_class_v1",
    global_image_size: int = DEFAULT_GLOBAL_IMAGE_SIZE,
    multiscale_sizes: tuple[int, ...] = DEFAULT_MULTISCALE_SIZES,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: float = DEFAULT_TILE_OVERLAP,
    max_positive_tiles_per_image: int = DEFAULT_MAX_POSITIVE_TILES,
    negative_tile_fraction: float = DEFAULT_NEGATIVE_TILE_FRACTION,
) -> dict[str, Any]:
    if profile not in TRAINING_PROFILES:
        raise ValueError(f"Unknown training profile: {profile}")
    if global_image_size not in multiscale_sizes:
        raise ValueError("global_image_size must be present in multiscale_sizes")
    if tile_size < max(multiscale_sizes):
        raise ValueError("tile_size must be at least the largest multiscale size")
    if not 0 <= tile_overlap < 1:
        raise ValueError("tile_overlap must be in [0, 1)")
    if max_positive_tiles_per_image < 0:
        raise ValueError("max_positive_tiles_per_image must be non-negative")
    if not 0 <= negative_tile_fraction <= 1:
        raise ValueError("negative_tile_fraction must be in [0, 1]")
    role_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_splits: dict[str, set[str]] = defaultdict(set)
    consent_kinds: Counter[str] = Counter()
    detector_records: list[dict[str, Any]] = []
    critical_records: list[dict[str, Any]] = []
    digest_splits: dict[str, str] = {}
    near_duplicate_links: list[tuple[str, str]] = []
    training_errors: list[str] = []
    deployment_errors: list[str] = []

    for loaded in records:
        record = loaded.record
        role = str(record.get("corpus_role", "missing"))
        role_counts[role] += 1
        consent = record.get("consent_basis", {})
        consent_kinds[str(consent.get("kind", "missing"))] += 1
        if role not in DETECTOR_ROLES:
            continue
        detector_records.append(record)
        split = str(record["split"])
        digest_splits[str(record["sha256"])] = split
        near_duplicate = record.get("near_duplicate_of")
        if near_duplicate:
            near_duplicate_links.append((str(near_duplicate), split))
        split_counts[split] += 1
        group_splits[str(record["split_group"])].add(split)
        annotations = record["annotations"]
        if not annotations:
            negative_counts[split] += 1
        for annotation in annotations:
            class_counts[split][str(annotation["class_name"])] += 1
        if role == "detector_critical_test":
            critical_records.append(record)

    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        training_errors.append(f"split_group_leakage:{len(leaking_groups)}")
    missing_near_duplicate_references = sum(
        reference not in digest_splits for reference, _split in near_duplicate_links
    )
    cross_split_near_duplicates = sum(
        reference in digest_splits and digest_splits[reference] != split
        for reference, split in near_duplicate_links
    )
    if missing_near_duplicate_references:
        training_errors.append(
            f"near_duplicate_reference_missing:{missing_near_duplicate_references}"
        )
    if cross_split_near_duplicates:
        training_errors.append(f"cross_split_near_duplicates:{cross_split_near_duplicates}")

    required_classes = set(TRAINING_PROFILES[profile].values())
    for split in ("train", "validation"):
        if split_counts[split] == 0:
            training_errors.append(f"missing_split:{split}")
        missing = sorted(required_classes - set(class_counts[split]))
        if missing:
            training_errors.append(f"missing_classes:{split}:{','.join(missing)}")
        if negative_counts[split] == 0:
            training_errors.append(f"missing_negative_rows:{split}")

    if split_counts["test"] == 0:
        deployment_errors.append("missing_split:test")
    if not critical_records:
        deployment_errors.append("missing_detector_critical_test")
    else:
        critical_classes = {
            str(annotation["class_name"])
            for record in critical_records
            for annotation in record["annotations"]
        }
        missing_critical = sorted(required_classes - critical_classes)
        if missing_critical:
            deployment_errors.append(f"missing_classes:critical_test:{','.join(missing_critical)}")
        if not any(not record["annotations"] for record in critical_records):
            deployment_errors.append("missing_negative_rows:critical_test")
        invalid_critical_samples = sum(
            str(record.get("sample_validation_status", "candidate_unreviewed"))
            != "double_validated"
            for record in critical_records
        )
        invalid_critical_annotations = sum(
            str(annotation.get("validation_status", "")) != "double_validated"
            for record in critical_records
            for annotation in record["annotations"]
        )
        if invalid_critical_samples:
            deployment_errors.append(
                f"critical_samples_not_double_validated:{invalid_critical_samples}"
            )
        if invalid_critical_annotations:
            deployment_errors.append(
                f"critical_annotations_not_double_validated:{invalid_critical_annotations}"
            )

    invalid_training_samples = sum(
        str(record.get("sample_validation_status", "candidate_unreviewed"))
        not in {"source_provided", "double_validated"}
        for record in detector_records
        if record["corpus_role"] == "detector_training"
    )
    if invalid_training_samples:
        training_errors.append(f"training_samples_not_approved:{invalid_training_samples}")

    object_size_audit = build_object_size_audit(
        records,
        allowed_class_names=frozenset(required_classes),
        target_sizes=multiscale_sizes,
    )
    global_stats = object_size_audit["sizes"][str(global_image_size)]["all_classes"]
    resolution_warnings: list[str] = []
    under_8_ratio = float(global_stats["under_threshold_ratios"].get("lt_8px", 0.0))
    if under_8_ratio > 0.05:
        resolution_warnings.append(f"global_view_objects_under_8px:{under_8_ratio:.6f}")
    if int(global_stats["objects"]) == 0:
        resolution_warnings.append("object_size_audit_empty")

    errors = [*training_errors, *deployment_errors]

    return {
        "schema_version": 1,
        "base_model": model_spec.model_id,
        "base_model_revision": model_spec.revision,
        "training_profile": profile,
        "required_classes": sorted(required_classes),
        "input_rows": len(records),
        "detector_rows": len(detector_records),
        "role_counts": dict(sorted(role_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "negative_counts": dict(sorted(negative_counts.items())),
        "class_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(class_counts.items())
        },
        "consent_kinds": dict(sorted(consent_kinds.items())),
        "split_leakage_groups": len(leaking_groups),
        "cross_split_near_duplicates": cross_split_near_duplicates,
        "missing_near_duplicate_references": missing_near_duplicate_references,
        "critical_test_rows": len(critical_records),
        "resolution_plan": {
            "global_image_size": global_image_size,
            "multiscale_sizes": list(multiscale_sizes),
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "tile_stride": round(tile_size * (1 - tile_overlap)),
            "max_positive_tiles_per_image": max_positive_tiles_per_image,
            "negative_tile_fraction": negative_tile_fraction,
        },
        "object_size_audit": object_size_audit,
        "resolution_warnings": resolution_warnings,
        "errors": errors,
        "training_errors": training_errors,
        "deployment_errors": deployment_errors,
        "training_ready": not training_errors,
        "deployment_ready": not errors,
    }


def _tile_axis_starts(length: int, *, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def generate_tile_windows(
    width: int,
    height: int,
    *,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_TILE_OVERLAP,
) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("Image and tile dimensions must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    return [
        (x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in _tile_axis_starts(height, tile_size=tile_size, overlap=overlap)
        for x in _tile_axis_starts(width, tile_size=tile_size, overlap=overlap)
    ]


def _clip_annotation_to_window(
    annotation: dict[str, Any],
    window: tuple[int, int, int, int],
    *,
    minimum_visibility: float = 0.5,
) -> tuple[dict[str, Any], float] | None:
    window_x0, window_y0, window_x1, window_y1 = window
    x, y, width, height = (float(value) for value in annotation["bbox_xywh"])
    box_x1 = x + width
    box_y1 = y + height
    clipped_x0 = max(x, window_x0)
    clipped_y0 = max(y, window_y0)
    clipped_x1 = min(box_x1, window_x1)
    clipped_y1 = min(box_y1, window_y1)
    clipped_width = clipped_x1 - clipped_x0
    clipped_height = clipped_y1 - clipped_y0
    if clipped_width < 2 or clipped_height < 2:
        return None
    original_area = width * height
    visible_ratio = clipped_width * clipped_height / original_area
    if visible_ratio < minimum_visibility:
        return None
    clipped = dict(annotation)
    clipped["bbox_xywh"] = [
        clipped_x0 - window_x0,
        clipped_y0 - window_y0,
        clipped_width,
        clipped_height,
    ]
    return clipped, visible_ratio


def _objects(annotations: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "id": list(range(len(annotations))),
        "area": [
            float(annotation["bbox_xywh"][2]) * float(annotation["bbox_xywh"][3])
            for annotation in annotations
        ],
        "bbox": [[float(value) for value in annotation["bbox_xywh"]] for annotation in annotations],
        "category": [int(annotation["class_id"]) for annotation in annotations],
    }


def _dataset_rows(
    records: list[LoadedRecord],
    *,
    allowed_class_ids: frozenset[int],
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: float = DEFAULT_TILE_OVERLAP,
    max_positive_tiles_per_image: int = DEFAULT_MAX_POSITIVE_TILES,
    negative_tile_fraction: float = DEFAULT_NEGATIVE_TILE_FRACTION,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if max_positive_tiles_per_image < 0:
        raise ValueError("max_positive_tiles_per_image must be non-negative")
    if not 0 <= negative_tile_fraction <= 1:
        raise ValueError("negative_tile_fraction must be in [0, 1]")
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    view_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for loaded in records:
        record = loaded.record
        role = str(record["corpus_role"])
        if role not in DETECTOR_ROLES:
            continue
        split = "critical_test" if role == "detector_critical_test" else str(record["split"])
        image_id = counters[split]
        counters[split] += 1
        annotations = [
            annotation
            for annotation in record["annotations"]
            if int(annotation["class_id"]) in allowed_class_ids
        ]
        image_path = str((loaded.corpus_root / str(record["image_relpath"])).resolve())
        rows_by_split[split].append(
            {
                "image": image_path,
                "image_id": image_id,
                "view_kind": "global",
                "crop_xyxy": None,
                "objects": _objects(annotations),
            }
        )
        view_counts[split]["global"] += 1
        if split != "train":
            continue
        width = int(record["width"])
        height = int(record["height"])
        if width <= tile_size and height <= tile_size:
            continue
        windows = generate_tile_windows(
            width,
            height,
            tile_size=tile_size,
            overlap=tile_overlap,
        )
        tile_candidates: list[
            tuple[tuple[int, int, int, int], list[dict[str, Any]], tuple[float, ...]]
        ] = []
        if annotations:
            for window in windows:
                clipped_with_visibility = [
                    clipped
                    for annotation in annotations
                    if (clipped := _clip_annotation_to_window(annotation, window)) is not None
                ]
                if not clipped_with_visibility:
                    continue
                clipped_annotations = [item[0] for item in clipped_with_visibility]
                visibility = sum(item[1] for item in clipped_with_visibility)
                smallest_target = min(
                    min(
                        float(annotation["bbox_xywh"][2]),
                        float(annotation["bbox_xywh"][3]),
                    )
                    for annotation in clipped_annotations
                )
                score = (len(clipped_annotations), visibility, -smallest_target)
                tile_candidates.append((window, clipped_annotations, score))
            tile_candidates.sort(
                key=lambda item: (
                    item[2],
                    -item[0][1],
                    -item[0][0],
                ),
                reverse=True,
            )
            tile_candidates = tile_candidates[:max_positive_tiles_per_image]
        elif windows:
            digest_fraction = int(str(record["sha256"])[:8], 16) / 0xFFFFFFFF
            if digest_fraction < negative_tile_fraction:
                index = int(str(record["sha256"])[8:16], 16) % len(windows)
                tile_candidates = [(windows[index], [], (0.0, 0.0, 0.0))]
        for window, tile_annotations, _score in tile_candidates:
            tile_image_id = counters[split]
            counters[split] += 1
            rows_by_split[split].append(
                {
                    "image": image_path,
                    "image_id": tile_image_id,
                    "view_kind": "positive_tile" if tile_annotations else "negative_tile",
                    "crop_xyxy": list(window),
                    "objects": _objects(tile_annotations),
                }
            )
            view_counts[split]["positive_tile" if tile_annotations else "negative_tile"] += 1
    return rows_by_split, {
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "max_positive_tiles_per_image": max_positive_tiles_per_image,
        "negative_tile_fraction": negative_tile_fraction,
        "view_counts": {
            split: dict(sorted(counts.items())) for split, counts in sorted(view_counts.items())
        },
    }


def _format_coco(
    image_id: int,
    categories: list[int],
    areas: list[float],
    boxes: list[list[float]],
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "annotations": [
            {
                "image_id": image_id,
                "category_id": category,
                "iscrowd": 0,
                "area": area,
                "bbox": box,
            }
            for category, area, box in zip(categories, areas, boxes, strict=True)
        ],
    }


def _transform_batch(
    examples: dict[str, Any],
    *,
    transforms: dict[int, Any],
    target_sizes: tuple[int, ...],
    randomize_size: bool,
    image_processor: Any,
) -> Any:
    import numpy as np

    # Training scale selection is intentionally pseudo-random and seeded through set_seed().
    target_size = (
        random.choice(target_sizes)  # noqa: S311
        if randomize_size
        else target_sizes[0]
    )
    transform = transforms[target_size]
    images: list[Any] = []
    annotations: list[dict[str, Any]] = []
    crop_windows = examples.get("crop_xyxy", [None] * len(examples["image_id"]))
    for image_id, image, objects, crop_window in zip(
        examples["image_id"],
        examples["image"],
        examples["objects"],
        crop_windows,
        strict=True,
    ):
        image_array = np.asarray(image.convert("RGB"))
        if crop_window is not None:
            x0, y0, x1, y1 = (int(value) for value in crop_window)
            image_array = image_array[y0:y1, x0:x1]
        transformed = transform(
            image=image_array,
            bboxes=list(objects["bbox"]),
            category=list(objects["category"]),
        )
        boxes = [[float(value) for value in box] for box in transformed["bboxes"]]
        categories = [int(value) for value in transformed["category"]]
        areas = [box[2] * box[3] for box in boxes]
        images.append(transformed["image"])
        annotations.append(_format_coco(int(image_id), categories, areas, boxes))
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    result.pop("pixel_mask", None)
    return result


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }


def _center_to_corners(boxes: Any, image_size: Any) -> Any:
    import torch
    from transformers.image_transforms import center_to_corners_format

    corners = center_to_corners_format(boxes)
    height, width = [int(value) for value in image_size]
    return corners * torch.tensor([[width, height, width, height]])


def _prediction_logits_and_boxes(batch: Any) -> tuple[Any, Any]:
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unexpected RT-DETR prediction batch type: {type(batch).__name__}")
    if len(batch) >= 3:
        return batch[1], batch[2]
    if len(batch) == 2:
        return batch[0], batch[1]
    raise ValueError(f"Unexpected RT-DETR prediction tuple length: {len(batch)}")


def _metric_batch(
    predictions: Any,
    targets: Any,
    *,
    image_processor: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numpy as np
    import torch

    if isinstance(targets, dict):
        targets = [targets]
    batch_sizes: list[list[int]] = []
    metric_targets: list[dict[str, Any]] = []
    for target in targets:
        original_size = torch.as_tensor(target["orig_size"]).detach().cpu().numpy()
        original_size = np.atleast_1d(original_size).flatten()
        if not original_size.size:
            raise ValueError("RT-DETR target orig_size is empty")
        height = int(original_size[0])
        width = int(original_size[1]) if original_size.size >= 2 else height
        batch_sizes.append([height, width])
        metric_targets.append(
            {
                "boxes": _center_to_corners(
                    torch.as_tensor(target["boxes"]).detach().cpu(),
                    (height, width),
                ),
                "labels": torch.as_tensor(target["class_labels"]).detach().cpu(),
            }
        )

    logits, boxes = _prediction_logits_and_boxes(predictions)
    logits = torch.as_tensor(logits).detach()
    boxes = torch.as_tensor(boxes).detach()
    target_sizes = torch.tensor(batch_sizes, device=logits.device)
    output = SimpleNamespace(logits=logits, pred_boxes=boxes)
    processed = image_processor.post_process_object_detection(
        output,
        threshold=0.0,
        target_sizes=target_sizes,
    )
    metric_predictions: list[dict[str, Any]] = []
    for prediction in processed:
        scores = prediction["scores"].detach().cpu()
        boxes = prediction["boxes"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        # COCO mAP uses at most 100 detections per image. Trimming before
        # torchmetrics stores its state preserves the exact evaluated contract
        # while bounding validation memory for large corpora.
        if scores.numel() > 100:
            keep = torch.topk(scores, k=100).indices
            scores = scores[keep]
            boxes = boxes[keep]
            labels = labels[keep]
        metric_predictions.append(
            {
                "scores": scores,
                "boxes": boxes,
                "labels": labels,
            }
        )
    return metric_predictions, metric_targets


def _format_map_metrics(
    raw: dict[str, Any],
    *,
    class_names: dict[int, str] = CLASS_NAMES,
) -> dict[str, float]:
    classes = raw.pop("classes")
    per_class_map = raw.pop("map_per_class")
    per_class_mar = raw.pop("mar_100_per_class")
    if classes.ndim == 0:
        classes = classes.unsqueeze(0)
        per_class_map = per_class_map.unsqueeze(0)
        per_class_mar = per_class_mar.unsqueeze(0)
    for class_id, class_map, class_mar in zip(classes, per_class_map, per_class_mar, strict=True):
        class_name = class_names[int(class_id.item())]
        raw[f"map_{class_name}"] = class_map
        raw[f"mar_100_{class_name}"] = class_mar
    return {key: round(float(value.item()), 4) for key, value in raw.items()}


class StreamingDetectionMetrics:
    """Accumulate final COCO metrics on CPU without retaining eval tensors on CUDA."""

    def __init__(
        self,
        *,
        image_processor: Any,
        class_names: dict[int, str] = CLASS_NAMES,
    ) -> None:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        self.image_processor = image_processor
        self.class_names = class_names
        self.metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True).cpu()

    def __call__(
        self,
        evaluation: Any,
        *,
        compute_result: bool = False,
    ) -> dict[str, float]:
        metric_predictions, metric_targets = _metric_batch(
            evaluation.predictions,
            evaluation.label_ids,
            image_processor=self.image_processor,
        )
        self.metric.update(metric_predictions, metric_targets)
        if not compute_result:
            return {}
        try:
            return _format_map_metrics(
                self.metric.compute(),
                class_names=self.class_names,
            )
        finally:
            self.metric.reset()


def _checkpoint_digest(output_dir: Path) -> str:
    weights = output_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError("Training output does not contain model.safetensors")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    value = digest.hexdigest()
    (output_dir / "model.safetensors.sha256").write_text(
        f"{value}  model.safetensors\n", encoding="ascii"
    )
    return value


def _select_smoke_records(
    records: list[LoadedRecord],
    *,
    allowed_class_ids: frozenset[int],
) -> list[LoadedRecord]:
    selected: list[LoadedRecord] = []
    for split in ("train", "validation"):
        split_records = [
            loaded
            for loaded in records
            if str(loaded.record.get("corpus_role")) == "detector_training"
            and str(loaded.record.get("split")) == split
        ]
        for class_id in sorted(allowed_class_ids):
            candidate = next(
                (
                    loaded
                    for loaded in split_records
                    if any(
                        int(annotation["class_id"]) == class_id
                        for annotation in loaded.record["annotations"]
                    )
                ),
                None,
            )
            if candidate is None:
                raise ValueError(f"Smoke run is missing class {class_id} in split {split}")
            selected.append(candidate)
        negative = next(
            (loaded for loaded in split_records if not loaded.record["annotations"]),
            None,
        )
        if negative is None:
            raise ValueError(f"Smoke run is missing a negative row in split {split}")
        selected.append(negative)
    return selected


def _select_benchmark_records(
    records: list[LoadedRecord],
    *,
    allowed_class_ids: frozenset[int],
    limit: int = 128,
) -> list[LoadedRecord]:
    if limit <= 0:
        raise ValueError("benchmark record limit must be positive")
    groups: dict[tuple[str, tuple[int, ...]], list[LoadedRecord]] = defaultdict(list)
    for loaded in records:
        record = loaded.record
        if (
            str(record.get("corpus_role")) != "detector_training"
            or str(record.get("split")) != "train"
        ):
            continue
        present_classes = tuple(
            sorted(
                {
                    int(annotation["class_id"])
                    for annotation in record["annotations"]
                    if int(annotation["class_id"]) in allowed_class_ids
                }
            )
        )
        groups[(str(loaded.corpus_root.resolve()), present_classes)].append(loaded)
    if not groups:
        raise ValueError("Benchmark run has no detector training records")
    ordered_groups: list[list[LoadedRecord]] = []
    for key in sorted(groups):
        group = sorted(
            groups[key],
            key=lambda loaded: (
                int(loaded.record["width"]) * int(loaded.record["height"]),
                str(loaded.record["sha256"]),
            ),
            reverse=True,
        )
        ordered_groups.append(group)
    selected: list[LoadedRecord] = []
    index = 0
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1
    present = {
        int(annotation["class_id"])
        for loaded in selected
        for annotation in loaded.record["annotations"]
        if int(annotation["class_id"]) in allowed_class_ids
    }
    missing = sorted(allowed_class_ids - present)
    if missing:
        raise ValueError(f"Benchmark run is missing classes: {missing}")
    return selected


def _build_optimizer_plan(
    *,
    train_views: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_optimizer_steps: int,
) -> dict[str, int]:
    if train_views <= 0:
        raise ValueError("Training plan must contain at least one train view")
    positive_values = {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "max_optimizer_steps": max_optimizer_steps,
    }
    invalid = sorted(name for name, value in positive_values.items() if value <= 0)
    if invalid:
        raise ValueError(f"Optimizer-plan values must be positive: {', '.join(invalid)}")
    micro_batches_per_epoch = math.ceil(train_views / batch_size)
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
    planned_max_optimizer_steps = optimizer_steps_per_epoch * epochs
    if planned_max_optimizer_steps > max_optimizer_steps:
        raise ValueError(
            "Training plan exceeds the optimizer-step safety limit: "
            f"{planned_max_optimizer_steps} > {max_optimizer_steps}. "
            "Reduce duplicate views or epochs; do not bypass this guard without "
            "recording and reviewing a new runtime benchmark."
        )
    return {
        "train_views": train_views,
        "micro_batches_per_epoch": micro_batches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "planned_max_optimizer_steps": planned_max_optimizer_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
    }


def _trainer_step_budget(
    *,
    smoke_mode: bool,
    benchmark_mode: bool,
    benchmark_steps: int,
    planned_max_optimizer_steps: int,
) -> int:
    if smoke_mode:
        return 1
    if benchmark_mode:
        return benchmark_steps
    return planned_max_optimizer_steps


def run_training(
    records: list[LoadedRecord],
    manifests: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
    class_names: dict[int, str],
    *,
    model_spec: DetectorModelSpec = RTDETR_MODEL_SPEC,
) -> None:
    def report_stage(stage: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "runtime-stage.txt").write_text(stage + "\n", encoding="ascii")
        print(f"{model_spec.output_prefix}_STAGE={stage}", flush=True)

    report_stage("imports_start")
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    report_stage("transformers_imported")
    import torch

    report_stage("torch_imported")
    from datasets import Dataset, DatasetDict, Image

    report_stage("datasets_imported")
    import albumentations as A

    report_stage("imports_complete")

    if not torch.cuda.is_available():
        raise RuntimeError(f"{model_spec.family} training requires a CUDA GPU")
    report_stage("cuda_ready")

    class ProcessTreeMemoryGuard(TrainerCallback):
        def __init__(self, limit_bytes: int) -> None:
            import psutil

            self.psutil = psutil
            self.process = psutil.Process()
            self.limit_bytes = limit_bytes
            self.peak_uss_bytes = 0

        def sample(self) -> None:
            processes = [self.process, *self.process.children(recursive=True)]
            current_uss_bytes = 0
            for process in processes:
                try:
                    current_uss_bytes += int(process.memory_full_info().uss)
                except (
                    OSError,
                    RuntimeError,
                    self.psutil.AccessDenied,
                    self.psutil.NoSuchProcess,
                    self.psutil.ZombieProcess,
                ):
                    continue
            self.peak_uss_bytes = max(self.peak_uss_bytes, current_uss_bytes)
            if current_uss_bytes > self.limit_bytes:
                raise RuntimeError(
                    "Training exceeded the host-RAM safety limit: "
                    f"{current_uss_bytes} > {self.limit_bytes} bytes"
                )

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.sample()
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.sample()
            return control

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    precision_bf16 = torch.cuda.is_bf16_supported()
    smoke_mode = args.command == "smoke"
    benchmark_mode = args.command == "benchmark"
    if smoke_mode:
        run_records = _select_smoke_records(records, allowed_class_ids=frozenset(class_names))
    elif benchmark_mode:
        run_records = _select_benchmark_records(
            records,
            allowed_class_ids=frozenset(class_names),
            limit=args.benchmark_records,
        )
    else:
        run_records = records
    report_stage("records_selected")

    rows_by_split, tile_report = _dataset_rows(
        run_records,
        allowed_class_ids=frozenset(class_names),
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        max_positive_tiles_per_image=args.max_positive_tiles_per_image,
        negative_tile_fraction=args.negative_tile_fraction,
    )
    report_stage("views_built")
    optimizer_plan = _build_optimizer_plan(
        train_views=len(rows_by_split["train"]),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_optimizer_steps=(
            max(args.max_optimizer_steps, args.benchmark_steps)
            if smoke_mode or benchmark_mode
            else args.max_optimizer_steps
        ),
    )
    effective_batch_size = optimizer_plan["effective_batch_size"]
    optimizer_steps_per_epoch = optimizer_plan["optimizer_steps_per_epoch"]
    planned_max_optimizer_steps = optimizer_plan["planned_max_optimizer_steps"]
    datasets: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        dataset = Dataset.from_list(rows).cast_column("image", Image())
        datasets[split] = dataset
    dataset_dict = DatasetDict(datasets)
    report_stage("datasets_materialized")

    id2label = dict(class_names)
    label2id = {name: identifier for identifier, name in id2label.items()}
    pretrained = {
        "revision": model_spec.revision,
        "trust_remote_code": False,
        "cache_dir": str(args.cache_dir),
    }
    report_stage("image_processor_load_start")
    image_processor = AutoImageProcessor.from_pretrained(
        model_spec.model_id,
        do_resize=False,
        do_pad=False,
        use_fast=True,
        **pretrained,
    )
    report_stage("image_processor_loaded")
    deployment_image_processor = AutoImageProcessor.from_pretrained(
        model_spec.model_id,
        do_resize=True,
        size={
            "max_height": args.global_image_size,
            "max_width": args.global_image_size,
        },
        do_pad=True,
        pad_size={
            "height": args.global_image_size,
            "width": args.global_image_size,
        },
        use_fast=True,
        **pretrained,
    )
    report_stage("deployment_processor_loaded")
    model = AutoModelForObjectDetection.from_pretrained(
        model_spec.model_id,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        **pretrained,
    )
    report_stage("model_loaded")
    supports_gradient_checkpointing = bool(getattr(model, "supports_gradient_checkpointing", False))
    if args.gradient_checkpointing and not supports_gradient_checkpointing:
        raise ValueError(
            f"{model.__class__.__name__} does not support gradient checkpointing; "
            "rerun with --no-gradient-checkpointing"
        )

    bbox_parameters = A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=4)

    def build_transform(size: int, *, augment: bool) -> Any:
        transforms: list[Any] = []
        if augment:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.35),
                    A.OneOf(
                        [
                            A.MotionBlur(blur_limit=5, p=1.0),
                            A.GaussianBlur(blur_limit=5, p=1.0),
                        ],
                        p=0.08,
                    ),
                    A.HueSaturationValue(p=0.08),
                ]
            )
        transforms.extend(
            [
                A.LongestMaxSize(max_size=size, interpolation=1, p=1.0),
                A.PadIfNeeded(
                    min_height=size,
                    min_width=size,
                    border_mode=0,
                    fill=0,
                    position="center",
                    p=1.0,
                ),
            ]
        )
        return A.Compose(transforms, bbox_params=bbox_parameters)

    train_transforms = {size: build_transform(size, augment=True) for size in args.multiscale_sizes}
    eval_transforms = {
        args.global_image_size: build_transform(args.global_image_size, augment=False)
    }
    dataset_dict["train"] = dataset_dict["train"].with_transform(
        partial(
            _transform_batch,
            transforms=train_transforms,
            target_sizes=args.multiscale_sizes,
            randomize_size=True,
            image_processor=image_processor,
        )
    )
    for split in ("validation", "test", "critical_test"):
        if split in dataset_dict:
            dataset_dict[split] = dataset_dict[split].with_transform(
                partial(
                    _transform_batch,
                    transforms=eval_transforms,
                    target_sizes=(args.global_image_size,),
                    randomize_size=False,
                    image_processor=image_processor,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _resolve_resume_checkpoint(
        output_dir,
        args.resume_from_checkpoint,
    )
    preflight_report = output_dir / "preflight-report.json"
    if not preflight_report.is_file():
        raise FileNotFoundError(
            f"Preflight report must be materialized before {model_spec.family} training starts"
        )
    provenance = {
        "schema_version": 1,
        "model_family": model_spec.family,
        "base_model": model_spec.model_id,
        "base_model_revision": model_spec.revision,
        "preflight_report_sha256": sha256_bytes(preflight_report.read_bytes()),
        "manifest_sha256": {
            str(path.resolve()): sha256_bytes(path.resolve().read_bytes()) for path in manifests
        },
        "class_names": class_names,
        "training_profile": args.profile,
        "run_mode": ("smoke" if smoke_mode else "benchmark" if benchmark_mode else "train"),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "precision": "bf16" if precision_bf16 else "fp16",
        "global_image_size": args.global_image_size,
        "multiscale_sizes": list(args.multiscale_sizes),
        "tile_report": tile_report,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "generated_train_views_per_epoch": len(rows_by_split["train"]),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "planned_max_optimizer_steps": planned_max_optimizer_steps,
        "max_optimizer_steps_guard": args.max_optimizer_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "resume_from_checkpoint": resume_checkpoint,
        "evaluation_policy": (
            "bounded_smoke_after_training"
            if smoke_mode
            else "disabled"
            if benchmark_mode
            else "full_validation_test_once_after_training"
        ),
        "gradient_checkpointing": args.gradient_checkpointing,
        "model_supports_gradient_checkpointing": supports_gradient_checkpointing,
        "seed": args.seed,
    }
    (output_dir / "training-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_stage("provenance_written")

    if not smoke_mode and not benchmark_mode:
        print(
            f"{model_spec.output_prefix}_TRAINING_PLAN="
            + json.dumps(
                {
                    "epochs_max": args.epochs,
                    "generated_train_views_per_epoch": len(rows_by_split["train"]),
                    "effective_batch_size": effective_batch_size,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "planned_max_optimizer_steps": planned_max_optimizer_steps,
                    "checkpoint_steps": args.checkpoint_steps,
                    "evaluation_policy": "full_validation_test_once_after_training",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if smoke_mode or benchmark_mode else args.epochs,
        # Keep the executable Trainer contract aligned with the preflight plan.
        # Relying on Trainer's implicit epoch-to-step conversion made the
        # displayed duration diverge from the guarded optimizer-step budget on
        # the real Windows run.
        max_steps=_trainer_step_budget(
            smoke_mode=smoke_mode,
            benchmark_mode=benchmark_mode,
            benchmark_steps=args.benchmark_steps,
            planned_max_optimizer_steps=planned_max_optimizer_steps,
        ),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=(1 if smoke_mode else args.gradient_accumulation_steps),
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,
        bf16=precision_bf16,
        fp16=not precision_bf16,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused",
        eval_strategy="no",
        save_strategy="no" if smoke_mode or benchmark_mode else "steps",
        save_steps=args.checkpoint_steps,
        logging_strategy="steps",
        logging_steps=1 if smoke_mode or benchmark_mode else 25,
        load_best_model_at_end=False,
        save_total_limit=4,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
        dataloader_num_workers=args.workers,
        dataloader_persistent_workers=args.workers > 0,
        report_to="none",
        push_to_hub=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    resource_monitor = ProcessTreeMemoryGuard(limit_bytes=round(args.max_host_ram_gb * 1024**3))
    callbacks: list[Any] = [resource_monitor]
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset_dict["train"],
        eval_dataset=None if benchmark_mode else dataset_dict["validation"],
        processing_class=image_processor,
        data_collator=_collate,
        compute_metrics=StreamingDetectionMetrics(
            image_processor=image_processor,
            class_names=class_names,
        ),
        callbacks=callbacks,
    )
    report_stage("trainer_ready")
    torch.cuda.reset_peak_memory_stats()
    report_stage("training_start")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_result.metrics.update(
        {
            "peak_process_tree_uss_bytes": resource_monitor.peak_uss_bytes,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    )
    if smoke_mode:
        evaluation_metrics = trainer.evaluate(metric_key_prefix="smoke_eval")
        smoke_metrics = {
            **train_result.metrics,
            **evaluation_metrics,
            "cuda_device": torch.cuda.get_device_name(0),
            "precision": "bf16" if precision_bf16 else "fp16",
            "selected_source_records": len(run_records),
            "generated_train_views": len(rows_by_split["train"]),
            "evaluated_validation_views": len(rows_by_split["validation"]),
        }
        (output_dir / "smoke-metrics.json").write_text(
            json.dumps(smoke_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(smoke_metrics, ensure_ascii=False, sort_keys=True))
        return
    if benchmark_mode:
        benchmark_metrics = {
            **train_result.metrics,
            "cuda_device": torch.cuda.get_device_name(0),
            "precision": "bf16" if precision_bf16 else "fp16",
            "selected_source_records": len(run_records),
            "generated_train_views": len(rows_by_split["train"]),
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        (output_dir / "benchmark-metrics.json").write_text(
            json.dumps(benchmark_metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(benchmark_metrics, ensure_ascii=False, sort_keys=True))
        return
    trainer.save_model(str(output_dir))
    deployment_image_processor.save_pretrained(str(output_dir))
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    # Release optimizer and training graph allocations before the single final
    # validation/test pass. StreamingDetectionMetrics keeps only bounded CPU
    # state and Trainer clears each CUDA batch immediately.
    torch.cuda.empty_cache()
    for split in ("validation", "test", "critical_test"):
        if split in dataset_dict:
            metrics = trainer.evaluate(eval_dataset=dataset_dict[split], metric_key_prefix=split)
            trainer.save_metrics(split, metrics)
    digest = _checkpoint_digest(output_dir)
    print(f"{model_spec.output_prefix}_CHECKPOINT_PATH={output_dir.resolve()}")
    print(f"{model_spec.output_prefix}_CHECKPOINT_SHA256=sha256:{digest}")


def _parse_multiscale_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "multiscale sizes must be comma-separated integers"
        ) from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError(
            "multiscale sizes must contain at least one positive integer"
        )
    return sizes


def build_parser(
    model_spec: DetectorModelSpec = RTDETR_MODEL_SPEC,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Preflight and train FireWarning {model_spec.family}"
    )
    parser.add_argument(
        "command",
        choices=("preflight", "plan", "benchmark", "smoke", "train"),
    )
    parser.add_argument("--profile", choices=tuple(TRAINING_PROFILES), default="media_filter_v1")
    parser.add_argument("--require-deployment-ready", action="store_true")
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=model_spec.default_output)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/huggingface-cache"))
    parser.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--global-image-size",
        "--image-size",
        dest="global_image_size",
        type=int,
        default=DEFAULT_GLOBAL_IMAGE_SIZE,
        help="Fixed global-view and evaluation side length; --image-size remains an alias.",
    )
    parser.add_argument(
        "--multiscale-sizes",
        type=_parse_multiscale_sizes,
        default=DEFAULT_MULTISCALE_SIZES,
        help="Comma-separated square training sizes (default: 640,768,896,960).",
    )
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--tile-overlap", type=float, default=DEFAULT_TILE_OVERLAP)
    parser.add_argument(
        "--max-positive-tiles-per-image",
        type=int,
        default=DEFAULT_MAX_POSITIVE_TILES,
    )
    parser.add_argument(
        "--negative-tile-fraction",
        type=float,
        default=DEFAULT_NEGATIVE_TILE_FRACTION,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-optimizer-steps", type=int, default=100_000)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--max-host-ram-gb", type=float, default=10.0)
    parser.add_argument("--benchmark-records", type=int, default=128)
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Checkpoint directory to resume, or 'auto' to use the highest "
            "checkpoint-N directory in --output. Auto starts fresh when none exists."
        ),
    )
    return parser


def main_for_model(model_spec: DetectorModelSpec) -> None:
    args = build_parser(model_spec).parse_args()
    positive_integer_args = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_optimizer_steps": args.max_optimizer_steps,
        "checkpoint_steps": args.checkpoint_steps,
        "benchmark_records": args.benchmark_records,
        "benchmark_steps": args.benchmark_steps,
    }
    invalid = sorted(name for name, value in positive_integer_args.items() if value <= 0)
    if invalid:
        raise ValueError(f"These arguments must be positive: {', '.join(invalid)}")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.max_host_ram_gb <= 0:
        raise ValueError("max_host_ram_gb must be positive")
    records = load_records(args.manifest, verify_files=args.verify_files)
    report = build_preflight_report(
        records,
        model_spec=model_spec,
        profile=args.profile,
        global_image_size=args.global_image_size,
        multiscale_sizes=args.multiscale_sizes,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        max_positive_tiles_per_image=args.max_positive_tiles_per_image,
        negative_tile_fraction=args.negative_tile_fraction,
    )
    report_json = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight-report.json").write_text(report_json, encoding="utf-8")
    print(report_json, end="")
    if args.command == "preflight":
        gate = "deployment_ready" if args.require_deployment_ready else "training_ready"
        if not report[gate]:
            raise SystemExit(2)
        return
    if not report["training_ready"]:
        raise RuntimeError("Training gate failed; resolve every preflight error before training")
    if args.command == "plan":
        rows_by_split, tile_report = _dataset_rows(
            records,
            allowed_class_ids=frozenset(TRAINING_PROFILES[args.profile]),
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_positive_tiles_per_image=args.max_positive_tiles_per_image,
            negative_tile_fraction=args.negative_tile_fraction,
        )
        plan = {
            **tile_report,
            "source_records": len(records),
            "generated_views": {split: len(rows) for split, rows in sorted(rows_by_split.items())},
        }
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (args.output / "training-view-plan.json").write_text(plan_json, encoding="utf-8")
        print(plan_json, end="")
        return
    run_training(
        records,
        args.manifest,
        args.output,
        args,
        TRAINING_PROFILES[args.profile],
        model_spec=model_spec,
    )


def main() -> None:
    main_for_model(RTDETR_MODEL_SPEC)


if __name__ == "__main__":
    main()
