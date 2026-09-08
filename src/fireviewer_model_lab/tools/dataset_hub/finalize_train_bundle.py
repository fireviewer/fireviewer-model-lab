from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import tarfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from dataset_archive_validation import (
    _copy_stream,
    _safe_relative_path,
    _zip_info,
    iter_files,
    load_archive_manifest,
    sha256_file,
    sha256_stream,
    validate_and_extract_shards,
    validate_firewarning_dataset,
)

SCHEMA_VERSION = 1
PACKAGE_FORMAT = "firewarning-train-bundle-zip-v1"
_VERIFIED_TRAIN_BUNDLE_DIGESTS: dict[tuple[Path, int, int], str] = {}
ALREADY_COMPRESSED_SUFFIXES = {
    ".7z",
    ".avi",
    ".gz",
    ".jpg",
    ".jpeg",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".npz",
    ".png",
    ".tar",
    ".tif",
    ".tiff",
    ".webm",
    ".webp",
    ".zip",
}
ALARMOD_DATASET = "alarmod/forest_fire"
ALARMOD_REVISION = "374d506829827673fd8aee7a30cdc414a05071ff"
ALARMOD_LICENSE = "GPL-3.0"
ALARMOD_CLASS_IDS = {
    "smoke_visible": 0,
    "flame_visible": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate source archives and emit one self-contained ZIP64 per training run."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported train bundle specification version")
    train_id = str(spec.get("train_id", ""))
    if not train_id or PurePosixPath(train_id).name != train_id:
        raise ValueError(f"Invalid train_id: {train_id!r}")
    if not spec.get("sources"):
        raise ValueError("A train bundle requires at least one source")
    if not spec.get("entrypoints"):
        raise ValueError("A train bundle requires at least one training entrypoint")
    return spec


def _resolve_under(root: Path, relative: str) -> Path:
    safe = _safe_relative_path(relative)
    candidate = root.joinpath(*safe.parts).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Path escapes source root: {relative}")
    return candidate


def _extract_source(
    *, source: dict[str, Any], source_root: Path, bundle_root: Path
) -> tuple[dict[str, Any], Path]:
    source_id = str(source["source_id"])
    mount_path = _safe_relative_path(str(source["mount_path"]))
    destination = bundle_root.joinpath(*mount_path.parts)
    destination.mkdir(parents=True, exist_ok=False)
    kind = str(source.get("kind", "archive_manifest"))
    if kind == "receipt_tar_gz_set":
        report = _extract_receipt_tar_gz_set(
            source=source,
            source_root=source_root,
            destination=destination,
        )
        return report, destination
    if kind == "train_bundle_subtree":
        report = _extract_train_bundle_subtree(
            source=source,
            source_root=source_root,
            destination=destination,
        )
        transformer = source.get("transformer")
        if transformer is not None:
            report["transformation"] = _apply_source_transformer(
                dataset_root=destination,
                transformer=str(transformer),
            )
        if "validator" in source:
            validator = str(source["validator"])
            report["validator"] = validator
            report["dataset_validation"] = _validate_source_dataset(
                dataset_root=destination,
                validator=validator,
            )
        return report, destination
    if kind == "materialized_directory":
        report = _extract_materialized_directory(
            source=source,
            source_root=source_root,
            destination=destination,
        )
        return report, destination
    if kind != "archive_manifest":
        raise ValueError(f"Unsupported source kind for {source_id}: {kind}")

    manifest_path = _resolve_under(source_root, str(source["archive_manifest"]))
    archive = load_archive_manifest(manifest_path, str(source["dataset_id"]))
    archive_report = validate_and_extract_shards(
        archive_manifest=archive,
        manifest_dir=manifest_path.parent,
        dataset_root=destination,
    )

    transformer = source.get("transformer")
    transform_report = (
        _apply_source_transformer(dataset_root=destination, transformer=str(transformer))
        if transformer is not None
        else None
    )

    validator = str(source.get("validator", "archive_only"))
    data_report = _validate_source_dataset(dataset_root=destination, validator=validator)

    return (
        {
            "source_id": source_id,
            "dataset_id": source["dataset_id"],
            "mount_path": mount_path.as_posix(),
            "purpose": source.get("purpose", []),
            "license": source.get("license"),
            "validator": validator,
            "archive_validation": archive_report,
            "transformation": transform_report,
            "dataset_validation": data_report,
        },
        destination,
    )


def _extract_materialized_directory(
    *,
    source: dict[str, Any],
    source_root: Path,
    destination: Path,
) -> dict[str, Any]:
    source_id = str(source["source_id"])
    source_path = _resolve_under(source_root, str(source["dataset_id"]))
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)

    copied_files = 0
    for path in iter_files(source_path):
        relative = path.relative_to(source_path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        copied_files += 1
        if copied_files % 10_000 == 0:
            print(
                f"materialized source linked source={source_id} files={copied_files}",
                flush=True,
            )

    expected_files = int(source["expected_files"])
    if copied_files != expected_files:
        raise ValueError(
            f"Materialized source file-count drift for {source_id}: "
            f"{copied_files} != {expected_files}"
        )
    validation = _validate_prithvi_materialized_dataset(
        destination,
        expected_samples=int(source["expected_samples"]),
        expected_source_counts={
            str(name): int(count) for name, count in source["expected_source_counts"].items()
        },
    )
    return {
        "source_id": source_id,
        "dataset_id": source["dataset_id"],
        "mount_path": str(source["mount_path"]),
        "purpose": source.get("purpose", []),
        "licenses": source.get("licenses", {}),
        "validator": str(source["validator"]),
        "copied_files": copied_files,
        "dataset_validation": validation,
    }


def _validate_prithvi_materialized_dataset(
    dataset_root: Path,
    *,
    expected_samples: int,
    expected_source_counts: dict[str, int],
) -> dict[str, Any]:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required to validate a materialized Prithvi bundle"
        ) from exc

    report_path = dataset_root / "materialization-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_splits = {
        str(split): int(count) for split, count in report["combined_split_counts"].items()
    }
    observed_splits: dict[str, int] = {}
    seen_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    sample_ids_by_source: dict[str, list[str]] = defaultdict(list)

    for split in ("train", "validation", "test"):
        split_path = dataset_root / "splits" / f"{split}.txt"
        sample_ids = [
            value.strip()
            for value in split_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"Duplicate sample ids in Prithvi split {split}")
        overlap = seen_ids.intersection(sample_ids)
        if overlap:
            raise ValueError(f"Prithvi split leakage: {len(overlap)} duplicate ids")
        seen_ids.update(sample_ids)
        observed_splits[split] = len(sample_ids)
        if observed_splits[split] != expected_splits.get(split):
            raise ValueError(
                f"Prithvi split-count drift for {split}: "
                f"{observed_splits[split]} != {expected_splits.get(split)}"
            )

        for _sample_index, sample_id in enumerate(sample_ids, start=1):
            source_name = "hls" if sample_id.startswith("hls_") else "eo4"
            if source_name == "eo4" and not sample_id.startswith("eo4_"):
                raise ValueError(f"Unsupported Prithvi sample id: {sample_id}")
            image_path = dataset_root / "data" / f"{sample_id}_merged.tif"
            mask_path = dataset_root / "data" / f"{sample_id}.mask.tif"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing Prithvi pair for {sample_id}")
            source_counts[source_name] += 1
            sample_ids_by_source[source_name].append(sample_id)

    if len(seen_ids) != expected_samples:
        raise ValueError(f"Prithvi sample-count drift: {len(seen_ids)} != {expected_samples}")
    if source_counts != Counter(expected_source_counts):
        raise ValueError(f"Prithvi source-count drift: {dict(source_counts)}")
    if report["normalization"]["eo4_audit"].get("compatible_with_hls_normalization") is not True:
        raise ValueError("EO4 reflectance is incompatible with HLS normalization")

    sampled_shapes: dict[str, list[list[int]]] = {}
    for source_name, source_sample_ids in sorted(sample_ids_by_source.items()):
        stride = max(1, len(source_sample_ids) // 64)
        selected_ids = source_sample_ids[::stride][:64]
        shapes: list[list[int]] = []
        for sample_id in selected_ids:
            image_path = dataset_root / "data" / f"{sample_id}_merged.tif"
            mask_path = dataset_root / "data" / f"{sample_id}.mask.tif"
            with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
                if image.count != 6:
                    raise ValueError(f"Prithvi image must have six bands: {sample_id}")
                if mask.count != 1:
                    raise ValueError(f"Prithvi mask must have one band: {sample_id}")
                if (image.height, image.width) != (mask.height, mask.width):
                    raise ValueError(f"Prithvi image/mask shape drift: {sample_id}")
                if image.height <= 0 or image.width <= 0:
                    raise ValueError(f"Prithvi empty raster: {sample_id}")
                if source_name == "hls" and (image.height, image.width) != (512, 512):
                    raise ValueError(f"HLS raster outside 512x512 contract: {sample_id}")
                shapes.append([image.height, image.width])
        sampled_shapes[source_name] = shapes

    return {
        "samples": len(seen_ids),
        "split_counts": observed_splits,
        "source_counts": dict(sorted(source_counts.items())),
        "raster_header_samples": {
            source_name: {
                "count": len(shapes),
                "shape_min": [min(shape[index] for shape in shapes) for index in (0, 1)],
                "shape_max": [max(shape[index] for shape in shapes) for index in (0, 1)],
            }
            for source_name, shapes in sampled_shapes.items()
        },
        "pair_inventory_verified": True,
        "runtime_tensor_contract": {
            "height": 512,
            "width": 512,
            "policy": "constant_pad_then_crop_without_geospatial_resize",
            "padding_mask_value": -1,
        },
        "reflectance_compatible": True,
        "split_leakage": 0,
    }


def _apply_source_transformer(*, dataset_root: Path, transformer: str) -> dict[str, Any]:
    if transformer == "alarmod_to_firewarning_detection_v1":
        return _normalize_alarmod_detection(dataset_root)
    if transformer == "hls_split_group_from_source_record_id_v1":
        return _normalize_hls_split_groups(dataset_root)
    if transformer == "firewarning_detection_to_image_triage_v1":
        return _derive_image_triage_manifest(dataset_root)
    if transformer == "boreal_yolo_to_firewarning_detection_v1":
        return _normalize_boreal_detection(dataset_root)
    raise ValueError(f"Unsupported transformer: {transformer}")


def _validate_source_dataset(*, dataset_root: Path, validator: str) -> dict[str, Any]:
    if validator == "firewarning_image_manifest":
        return validate_firewarning_dataset(dataset_root)
    if validator == "hls_burn_scars_manifest":
        return _validate_hls_burn_scars(dataset_root)
    if validator == "archive_only":
        return {"files_verified": True, "semantic_validation": "not_requested"}
    raise ValueError(f"Unsupported validator: {validator}")


def _extract_train_bundle_subtree(
    *, source: dict[str, Any], source_root: Path, destination: Path
) -> dict[str, Any]:
    archive_path = _resolve_under(source_root, str(source["bundle_zip"]))
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    expected_size = int(source["bundle_size_bytes"])
    expected_sha256 = str(source["bundle_sha256"])
    archive_stat = archive_path.stat()
    if archive_stat.st_size != expected_size:
        raise ValueError(f"Train bundle size mismatch: {archive_path}")
    digest_key = (archive_path, archive_stat.st_size, archive_stat.st_mtime_ns)
    observed_sha256 = _VERIFIED_TRAIN_BUNDLE_DIGESTS.get(digest_key)
    if observed_sha256 is None:
        observed_sha256 = sha256_file(archive_path)
        _VERIFIED_TRAIN_BUNDLE_DIGESTS[digest_key] = observed_sha256
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Train bundle SHA-256 mismatch: {archive_path}")

    bundle_train_id = str(source["bundle_train_id"])
    if PurePosixPath(bundle_train_id).name != bundle_train_id:
        raise ValueError(f"Invalid bundle_train_id: {bundle_train_id!r}")
    subtree = _safe_relative_path(str(source["subtree"]))
    prefix = f"{bundle_train_id}/{subtree.as_posix()}/"
    extracted_files = 0
    extracted_bytes = 0
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path, mode="r", allowZip64=True) as archive:
        for info in archive.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise ValueError(f"Encrypted train bundle entry: {info.filename}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic link in train bundle: {info.filename}")
            relative = _safe_relative_path(info.filename.removeprefix(prefix))
            key = relative.as_posix()
            if key in seen:
                raise ValueError(f"Duplicate train bundle subtree entry: {key}")
            seen.add(key)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source_stream, target.open("wb") as output:
                copied = _copy_stream(source_stream, output)
            if copied != info.file_size:
                raise ValueError(f"Train bundle entry size mismatch: {info.filename}")
            os.chmod(target, 0o644)
            extracted_files += 1
            extracted_bytes += copied
    if extracted_files == 0:
        raise ValueError(f"Train bundle subtree is empty: {prefix}")
    return {
        "source_id": source["source_id"],
        "dataset_id": source["dataset_id"],
        "mount_path": str(source["mount_path"]),
        "purpose": source.get("purpose", []),
        "license": source.get("license"),
        "validator": "verified_train_bundle_subtree",
        "archive_validation": {
            "bundle_train_id": bundle_train_id,
            "bundle_sha256": expected_sha256,
            "subtree": subtree.as_posix(),
            "files": extracted_files,
            "bytes": extracted_bytes,
        },
        "dataset_validation": {
            "files_verified": True,
            "semantic_validation": "performed_by_bundle_validator",
        },
    }


def _derive_image_triage_manifest(dataset_root: Path) -> dict[str, Any]:
    source_manifest = dataset_root / "manifest.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    output_manifest = dataset_root / "triage-manifest.jsonl"
    output_report = dataset_root / "triage-report.json"
    allowed_annotation_classes = {"flame_visible", "smoke_visible"}
    rows = 0
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    seen_samples: set[str] = set()
    temporary = output_manifest.with_suffix(".jsonl.partial")
    with (
        source_manifest.open(encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8", newline="\n") as output,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in seen_samples:
                raise ValueError(f"Duplicate triage sample_id at line {line_number}: {sample_id}")
            seen_samples.add(sample_id)
            annotations = row.get("annotations", [])
            if not isinstance(annotations, list):
                raise ValueError(f"Invalid annotations at line {line_number}")
            annotation_classes = {str(item["class_name"]) for item in annotations}
            unsupported = annotation_classes - allowed_annotation_classes
            if unsupported:
                raise ValueError(
                    f"Unsupported triage annotation classes at line {line_number}: "
                    f"{sorted(unsupported)}"
                )
            if not annotations:
                negative_tags = set(str(item) for item in row.get("negative_tags", []))
                if "no_target_visible" not in negative_tags and row.get("negative") is not True:
                    raise ValueError(f"Unproven normal triage row at line {line_number}")
                labels = ["normal"]
                primary_class = "normal"
            else:
                labels = []
                if "flame_visible" in annotation_classes:
                    labels.append("fire")
                if "smoke_visible" in annotation_classes:
                    labels.append("smoke")
                primary_class = "fire_and_smoke" if len(labels) == 2 else labels[0]
            split = str(row["split"])
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unsupported triage split at line {line_number}: {split}")
            split_group = str(row["split_group"])
            split_groups[split_group].add(split)
            split_counts[split] += 1
            label_counts[primary_class] += 1
            triage_row = {
                "consent_basis": row.get("consent_basis"),
                "derived_from": "source_provided_detection_annotations",
                "height": int(row["height"]),
                "image_relpath": str(row["image_relpath"]),
                "label_strength": "source_provided_bbox_presence",
                "labels": labels,
                "license": row.get("license"),
                "primary_class": primary_class,
                "sample_id": sample_id,
                "sha256": str(row["sha256"]),
                "source_id": str(row["source_id"]),
                "source_record_id": row.get("source_record_id"),
                "split": split,
                "split_group": split_group,
                "width": int(row["width"]),
            }
            output.write(json.dumps(triage_row, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    os.replace(temporary, output_manifest)
    leaking_groups = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leaking_groups:
        raise ValueError(f"Image triage split-group leakage: {len(leaking_groups)} groups")
    report = {
        "transformer": "firewarning_detection_to_image_triage_v1",
        "rows": rows,
        "split_counts": dict(sorted(split_counts.items())),
        "primary_class_counts": dict(sorted(label_counts.items())),
        "split_groups": len(split_groups),
        "split_group_leakage": 0,
        "manifest": "triage-manifest.jsonl",
        "manifest_sha256": sha256_file(output_manifest),
    }
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _normalize_alarmod_detection(dataset_root: Path) -> dict[str, Any]:
    manifest = dataset_root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    upstream_manifest = dataset_root / "upstream-manifest.jsonl"
    manifest.replace(upstream_manifest)
    image_root = dataset_root / "images"
    image_root.mkdir()
    canonical_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    positive_rows = 0
    annotation_count = 0
    seen_digests: set[str] = set()

    with upstream_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_image = dataset_root.joinpath(
                *_safe_relative_path(str(row["image_relpath"])).parts
            )
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            digest = str(row["sha256"])
            if sha256_file(source_image) != digest:
                raise ValueError(f"Alarmod image digest mismatch at line {line_number}")
            if digest in seen_digests:
                raise ValueError(f"Alarmod duplicate image digest at line {line_number}: {digest}")
            seen_digests.add(digest)
            extension = source_image.suffix.lower() or ".jpg"
            canonical_image = image_root / f"{digest}{extension}"
            source_image.replace(canonical_image)
            width = int(row["width"])
            height = int(row["height"])
            annotations: list[dict[str, Any]] = []
            for annotation in row["annotations"]:
                center_x, center_y, box_width, box_height = (
                    float(value) for value in annotation["bbox_xywh_normalized"]
                )
                left, top, absolute_width, absolute_height = _alarmod_box_to_absolute(
                    center_x=center_x,
                    center_y=center_y,
                    box_width=box_width,
                    box_height=box_height,
                    image_width=width,
                    image_height=height,
                    line_number=line_number,
                )
                class_name = str(annotation["class_name"])
                canonical_class = {
                    "fire": "flame_visible",
                    "flame": "flame_visible",
                    "smoke": "smoke_visible",
                }.get(class_name)
                if canonical_class is None:
                    raise ValueError(
                        f"Unsupported Alarmod class at line {line_number}: {class_name}"
                    )
                annotations.append(
                    {
                        "bbox_xywh": [left, top, absolute_width, absolute_height],
                        "class_name": canonical_class,
                        "point_xy_normalized": annotation.get("point_xy_normalized"),
                        "source_class_name": class_name,
                    }
                )
            if annotations:
                positive_rows += 1
                annotation_count += len(annotations)
            split = str(row["split"])
            split_counts[split] += 1
            canonical_rows.append(
                _upgrade_alarmod_detection_row(
                    {
                        "annotations": annotations,
                        "corpus_role": "detector_training",
                        "height": height,
                        "image_relpath": canonical_image.relative_to(dataset_root).as_posix(),
                        "negative": not annotations,
                        "sample_id": row["sample_id"],
                        "sha256": digest,
                        "source_asset": row.get("source_asset"),
                        "source_id": row["source_id"],
                        "source_record_id": row.get("source_record_id"),
                        "split": split,
                        "split_group": row["split_group"],
                        "width": width,
                    }
                )
            )

    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in canonical_rows
        ),
        encoding="utf-8",
    )
    return {
        "transformer": "alarmod_to_firewarning_detection_v1",
        "rows": len(canonical_rows),
        "positive_rows": positive_rows,
        "negative_rows": len(canonical_rows) - positive_rows,
        "annotation_count": annotation_count,
        "split_counts": dict(sorted(split_counts.items())),
        "content_addressed_images": len(seen_digests),
        "upstream_manifest": "upstream-manifest.jsonl",
        "canonical_manifest": "manifest.jsonl",
    }


def _normalize_boreal_detection(dataset_root: Path) -> dict[str, Any]:
    from PIL import Image

    manifest = dataset_root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    upstream_manifest = dataset_root / "upstream-manifest.jsonl"
    manifest.replace(upstream_manifest)
    image_root = dataset_root / "images"
    image_root.mkdir()
    canonical_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    seen_digests: set[str] = set()
    annotation_count = 0

    with upstream_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            upstream = json.loads(line)
            artifact_path = dataset_root.joinpath(
                *_safe_relative_path(str(upstream["artifact"]["path"])).parts
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            image_ref = artifact["image"]
            label_ref = artifact["annotation"]
            source_image = dataset_root.joinpath(*_safe_relative_path(str(image_ref["path"])).parts)
            source_label = dataset_root.joinpath(*_safe_relative_path(str(label_ref["path"])).parts)
            if not source_image.is_file() or not source_label.is_file():
                raise FileNotFoundError(
                    f"Boreal image or label missing at manifest line {line_number}"
                )
            digest = str(image_ref["sha256"])
            if sha256_file(source_image) != digest:
                raise ValueError(f"Boreal image digest mismatch at line {line_number}")
            if digest in seen_digests:
                raise ValueError(f"Duplicate Boreal image digest at line {line_number}")
            seen_digests.add(digest)
            with Image.open(source_image) as image:
                width, height = image.size

            annotations: list[dict[str, Any]] = []
            for label_line in source_label.read_text(encoding="utf-8").splitlines():
                if not label_line.strip():
                    continue
                parts = label_line.split()
                if len(parts) != 5 or parts[0] != "0":
                    raise ValueError(f"Unsupported Boreal YOLO label at line {line_number}")
                center_x, center_y, box_width, box_height = (float(value) for value in parts[1:])
                left, top, absolute_width, absolute_height = _alarmod_box_to_absolute(
                    center_x=center_x,
                    center_y=center_y,
                    box_width=box_width,
                    box_height=box_height,
                    image_width=width,
                    image_height=height,
                    line_number=line_number,
                )
                annotations.append(
                    {
                        "annotated_at": None,
                        "annotator_id": "boreal-source-label",
                        "bbox_xywh": [left, top, absolute_width, absolute_height],
                        "class_id": 0,
                        "class_name": "smoke_visible",
                        "occlusion": "unknown",
                        "origin": "boreal-forest-fire-detection-v1:human_bounding_box",
                        "validation_status": "source_provided",
                        "visibility": "unknown",
                    }
                )
            expected_boxes = int(artifact["box_count"])
            if len(annotations) != expected_boxes:
                raise ValueError(f"Boreal box-count drift at line {line_number}")
            if not annotations and not bool(artifact.get("negative")):
                raise ValueError(f"Unproven Boreal negative at line {line_number}")

            extension = source_image.suffix.lower() or ".jpg"
            canonical_image = image_root / f"{digest}{extension}"
            source_image.replace(canonical_image)
            artifact["image"]["path"] = canonical_image.relative_to(dataset_root).as_posix()
            artifact_path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            split = str(artifact["split"])
            split_counts[split] += 1
            annotation_count += len(annotations)
            canonical_rows.append(
                {
                    "annotations": annotations,
                    "candidate_classes": [],
                    "consent_basis": {
                        "kind": "source_license",
                        "reference": "boreal-forest-fire-detection-v1:CC-BY-4.0",
                    },
                    "corpus_role": "detector_training",
                    "height": height,
                    "image_relpath": canonical_image.relative_to(dataset_root).as_posix(),
                    "license": "CC-BY-4.0",
                    "near_duplicate_of": None,
                    "negative": not annotations,
                    "negative_tags": [] if annotations else ["no_target_visible"],
                    "sample_id": upstream["sample_id"],
                    "sample_validation_status": "source_provided",
                    "sha256": digest,
                    "source_id": upstream["source_id"],
                    "source_record_id": upstream.get("source_record_id"),
                    "split": split,
                    "split_group": upstream["split_group"],
                    "width": width,
                }
            )

    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in canonical_rows
        ),
        encoding="utf-8",
    )
    return {
        "transformer": "boreal_yolo_to_firewarning_detection_v1",
        "rows": len(canonical_rows),
        "annotation_count": annotation_count,
        "positive_rows": sum(bool(row["annotations"]) for row in canonical_rows),
        "negative_rows": sum(not row["annotations"] for row in canonical_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "content_addressed_images": len(seen_digests),
        "upstream_manifest": "upstream-manifest.jsonl",
        "canonical_manifest": "manifest.jsonl",
    }


def _upgrade_alarmod_detection_row(row: dict[str, Any]) -> dict[str, Any]:
    source_asset = row.get("source_asset")
    if not isinstance(source_asset, dict):
        source_asset = {}
    dataset = str(source_asset.get("dataset") or ALARMOD_DATASET)
    revision = str(source_asset.get("revision") or ALARMOD_REVISION)
    license_id = str(source_asset.get("license") or ALARMOD_LICENSE)
    annotations: list[dict[str, Any]] = []
    for annotation in row["annotations"]:
        canonical = dict(annotation)
        class_name = str(canonical["class_name"])
        expected_class_id = ALARMOD_CLASS_IDS.get(class_name)
        if expected_class_id is None:
            raise ValueError(f"Unsupported canonical Alarmod class: {class_name}")
        if "class_id" in canonical and int(canonical["class_id"]) != expected_class_id:
            raise ValueError(f"Alarmod class id/name mismatch: {canonical}")
        canonical.update(
            {
                "annotated_at": canonical.get("annotated_at"),
                "annotator_id": canonical.get("annotator_id", "alarmod-source-label"),
                "class_id": expected_class_id,
                "occlusion": canonical.get("occlusion", "unknown"),
                "origin": canonical.get("origin", f"{dataset}:annotations"),
                "validation_status": canonical.get("validation_status", "source_provided"),
                "visibility": canonical.get("visibility", "unknown"),
            }
        )
        annotations.append(canonical)
    upgraded = dict(row)
    upgraded.update(
        {
            "annotations": annotations,
            "candidate_classes": upgraded.get("candidate_classes", []),
            "consent_basis": upgraded.get(
                "consent_basis",
                {"kind": "source_license", "reference": f"{dataset}@{revision}"},
            ),
            "license": upgraded.get("license", license_id),
            "near_duplicate_of": upgraded.get("near_duplicate_of"),
            "negative_tags": upgraded.get(
                "negative_tags",
                [] if annotations else ["no_target_visible"],
            ),
            "sample_validation_status": upgraded.get(
                "sample_validation_status",
                "source_provided",
            ),
        }
    )
    return upgraded


def repair_alarmod_detection_manifest(source: Path, output: Path) -> dict[str, Any]:
    if source.resolve() == output.resolve():
        raise ValueError("Alarmod repair output must not overwrite the source manifest")
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    rows = 0
    annotations = 0
    with source.open(encoding="utf-8") as handle, temporary.open("x", encoding="utf-8") as sink:
        for line in handle:
            if not line.strip():
                continue
            upgraded = _upgrade_alarmod_detection_row(json.loads(line))
            sink.write(json.dumps(upgraded, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
            annotations += len(upgraded["annotations"])
    os.replace(temporary, output)
    return {
        "rows": rows,
        "annotations": annotations,
        "source_manifest_sha256": sha256_file(source),
        "output_manifest_sha256": sha256_file(output),
    }


def _alarmod_box_to_absolute(
    *,
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
    line_number: int,
) -> list[float]:
    """Convert rounded YOLO coordinates while rejecting materially invalid boxes."""
    values = (center_x, center_y, box_width, box_height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite Alarmod box at line {line_number}")
    if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
        raise ValueError(f"Alarmod box center outside image at line {line_number}")
    if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
        raise ValueError(f"Invalid Alarmod box size at line {line_number}")

    left = (center_x - box_width / 2.0) * image_width
    top = (center_y - box_height / 2.0) * image_height
    right = (center_x + box_width / 2.0) * image_width
    bottom = (center_y + box_height / 2.0) * image_height
    rounding_tolerance_pixels = 0.01
    if (
        left < -rounding_tolerance_pixels
        or top < -rounding_tolerance_pixels
        or right > image_width + rounding_tolerance_pixels
        or bottom > image_height + rounding_tolerance_pixels
    ):
        raise ValueError(f"Alarmod box materially exceeds image at line {line_number}")

    clipped_left = max(0.0, left)
    clipped_top = max(0.0, top)
    clipped_right = min(float(image_width), right)
    clipped_bottom = min(float(image_height), bottom)
    clipped_width = clipped_right - clipped_left
    clipped_height = clipped_bottom - clipped_top
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        raise ValueError(f"Alarmod box is empty after rounding repair at line {line_number}")
    return [clipped_left, clipped_top, clipped_width, clipped_height]


def _normalize_hls_split_groups(dataset_root: Path) -> dict[str, Any]:
    """Add a stable leakage group to legacy HLS rows without changing their split."""
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    normalized_rows: list[dict[str, Any]] = []
    derived_rows = 0
    preserved_rows = 0
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split_group = row.get("split_group")
            if split_group:
                preserved_rows += 1
            else:
                source_record_id = row.get("source_record_id")
                if not source_record_id:
                    raise ValueError(
                        "HLS row cannot derive split_group without source_record_id "
                        f"at line {line_number}"
                    )
                row["split_group"] = f"hls-scene:{source_record_id}"
                derived_rows += 1
            normalized_rows.append(row)
    if not normalized_rows:
        raise ValueError("HLS manifest is empty")
    temporary = manifest_path.with_suffix(".jsonl.partial")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in normalized_rows
        ),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return {
        "transformer": "hls_split_group_from_source_record_id_v1",
        "rows": len(normalized_rows),
        "derived_rows": derived_rows,
        "preserved_rows": preserved_rows,
        "derivation": "hls-scene:{source_record_id}",
    }


def _validate_hls_burn_scars(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = 0
    split_counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    seen_images: set[str] = set()
    seen_masks: set[str] = set()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {
                "image_relpath",
                "mask_relpath",
                "mask_sha256",
                "sha256",
                "split",
                "split_group",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(f"HLS manifest line {line_number} misses {sorted(missing)}")
            image_relative = _safe_relative_path(str(row["image_relpath"]))
            mask_relative = _safe_relative_path(str(row["mask_relpath"]))
            image = dataset_root.joinpath(*image_relative.parts)
            mask = dataset_root.joinpath(*mask_relative.parts)
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"HLS pair missing at line {line_number}")
            if sha256_file(image) != str(row["sha256"]):
                raise ValueError(f"HLS image digest mismatch at line {line_number}")
            if sha256_file(mask) != str(row["mask_sha256"]):
                raise ValueError(f"HLS mask digest mismatch at line {line_number}")
            if str(row["sha256"]) in seen_images or str(row["mask_sha256"]) in seen_masks:
                raise ValueError(f"Duplicate HLS image or mask at line {line_number}")
            seen_images.add(str(row["sha256"]))
            seen_masks.add(str(row["mask_sha256"]))
            split = str(row["split"])
            split_counts[split] += 1
            split_groups[str(row["split_group"])].add(split)
            rows += 1
    leaking = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"HLS split leakage detected in {len(leaking)} groups")
    return {
        "rows": rows,
        "split_counts": dict(sorted(split_counts.items())),
        "split_leakage_groups": 0,
        "images_verified": len(seen_images),
        "masks_verified": len(seen_masks),
        "files_verified": True,
    }


def _extract_receipt_tar_gz_set(
    *, source: dict[str, Any], source_root: Path, destination: Path
) -> dict[str, Any]:
    receipt_pattern = str(source["receipt_glob"])
    safe_pattern = receipt_pattern.replace("\\", "/")
    if safe_pattern.startswith("/") or ".." in PurePosixPath(safe_pattern).parts:
        raise ValueError(f"Unsafe receipt_glob: {receipt_pattern}")
    receipts = sorted(source_root.glob(receipt_pattern))
    if not receipts:
        raise FileNotFoundError(f"No receipts match {receipt_pattern}")

    seen_members: set[str] = set()
    seen_digests: set[str] = set()
    split_counts: Counter[str] = Counter()
    canonical_rows: list[dict[str, Any]] = []
    archive_reports: list[dict[str, Any]] = []
    metadata_root = destination / "shard-metadata"
    extracted_bytes = 0

    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        archive_path = receipt_path.parent / str(receipt["archive"]["path"])
        manifest_path = receipt_path.parent / str(receipt["manifest"]["path"])
        for path, expected in (
            (archive_path, receipt["archive"]),
            (manifest_path, receipt["manifest"]),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(expected["size_bytes"]):
                raise ValueError(f"Receipt size mismatch: {path}")
            if sha256_file(path) != str(expected["sha256"]):
                raise ValueError(f"Receipt SHA-256 mismatch: {path}")

        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != int(receipt["samples"]):
            raise ValueError(f"Receipt sample count mismatch: {receipt_path}")
        rows_by_member = {str(row["source_member"]): row for row in rows}
        if len(rows_by_member) != len(rows):
            raise ValueError(f"Duplicate source_member in {manifest_path}")

        archive_member_count = 0
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsupported EO4 tar member: {member.name}")
                relative = _safe_relative_path(member.name)
                member_name = relative.as_posix()
                if member_name in seen_members:
                    raise ValueError(f"Duplicate EO4 member across shards: {member_name}")
                row = rows_by_member.pop(member_name, None)
                if row is None:
                    raise ValueError(f"EO4 member absent from manifest: {member_name}")
                if member.size != int(row["size_bytes"]):
                    raise ValueError(f"EO4 member size mismatch: {member_name}")
                digest = str(row["sha256"])
                if digest in seen_digests:
                    raise ValueError(f"Duplicate EO4 payload digest: {digest}")
                source_stream = archive.extractfile(member)
                if source_stream is None:
                    raise ValueError(f"Unable to read EO4 member: {member_name}")
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source_stream, target.open("wb") as output:
                    copied = _copy_stream(source_stream, output)
                if copied != member.size or sha256_file(target) != digest:
                    raise ValueError(f"EO4 extracted payload mismatch: {member_name}")
                os.chmod(target, 0o644)
                seen_members.add(member_name)
                seen_digests.add(digest)
                split_counts[str(row["split"])] += 1
                canonical_rows.append(row)
                extracted_bytes += copied
                archive_member_count += 1
        if rows_by_member:
            raise ValueError(f"EO4 manifest contains absent members: {sorted(rows_by_member)[:3]}")

        metadata_split = metadata_root / str(receipt["split"])
        metadata_split.mkdir(parents=True, exist_ok=True)
        shutil.copy2(receipt_path, metadata_split / receipt_path.name)
        shutil.copy2(manifest_path, metadata_split / manifest_path.name)
        archive_reports.append(
            {
                "archive": archive_path.name,
                "archive_sha256": receipt["archive"]["sha256"],
                "manifest": manifest_path.name,
                "manifest_sha256": receipt["manifest"]["sha256"],
                "samples": archive_member_count,
                "split": receipt["split"],
            }
        )

    expected_samples = int(source["expected_samples"])
    if len(canonical_rows) != expected_samples:
        raise ValueError(
            f"EO4 materialization is incomplete: {len(canonical_rows)} != {expected_samples}"
        )
    canonical_rows.sort(key=lambda row: (str(row["split"]), int(row["official_ordinal"])))
    (destination / "manifest.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in canonical_rows
        ),
        encoding="utf-8",
    )
    return {
        "source_id": source["source_id"],
        "dataset_id": source["dataset_id"],
        "mount_path": str(source["mount_path"]),
        "purpose": source.get("purpose", []),
        "license": source.get("license"),
        "validator": "eo4_receipt_tar_gz_set",
        "archive_validation": {
            "shards": archive_reports,
            "samples": len(canonical_rows),
            "source_bytes": extracted_bytes,
        },
        "dataset_validation": {
            "rows": len(canonical_rows),
            "split_counts": dict(sorted(split_counts.items())),
            "unique_payload_sha256": len(seen_digests),
            "files_verified": True,
            "crs_and_variable_evidence_preserved": True,
        },
    }


def _iter_manifest_digests(dataset_root: Path) -> Iterable[tuple[str, str, str]]:
    manifest = dataset_root / "manifest.jsonl"
    if not manifest.is_file():
        return
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            digest = row.get("sha256")
            if digest:
                yield str(digest), str(row.get("split", "unknown")), f"{manifest}:{line_number}"


def _validate_cross_source_images(source_roots: dict[str, Path]) -> dict[str, Any]:
    owners: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for source_id, root in source_roots.items():
        for digest, split, location in _iter_manifest_digests(root):
            owners[digest].append((source_id, split, location))

    duplicates = {digest: rows for digest, rows in owners.items() if len(rows) > 1}
    leaking = {
        digest: rows
        for digest, rows in duplicates.items()
        if len({split for _, split, _ in rows}) > 1
    }
    if leaking:
        first_digest, rows = next(iter(leaking.items()))
        raise ValueError(
            "Cross-source split leakage for image "
            f"{first_digest}: {[(source, split) for source, split, _ in rows]}"
        )
    if duplicates:
        first_digest, rows = next(iter(duplicates.items()))
        raise ValueError(
            "Cross-source duplicate image requires explicit de-duplication: "
            f"{first_digest} in {[source for source, _, _ in rows]}"
        )
    return {
        "unique_image_sha256": len(owners),
        "cross_source_exact_duplicates": 0,
        "cross_source_split_leakage": 0,
    }


def _validate_cross_view_registration_bundle(
    *, bundle_root: Path, validator: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = _resolve_under(bundle_root, str(validator["manifest"]))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_rows = int(validator["expected_rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"Cross-view registration row mismatch: {len(rows)} != {expected_rows}")

    sample_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    verified_digests: dict[Path, str] = {}
    for line_number, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate registration sample_id at line {line_number}: {sample_id}")
        sample_ids.add(sample_id)
        if row.get("operational_incident") is not False or "critical_lot" in row:
            raise ValueError(f"Operational or critical registration row at line {line_number}")
        split = str(row["split"])
        split_group = str(row["split_group"])
        split_counts[split] += 1
        split_groups[split_group].add(split)
        source_counts[str(row["source_id"])] += 1
        if not isinstance(row.get("ground_truth"), dict):
            raise ValueError(f"Missing registration ground truth at line {line_number}")

        for view_name in ("source_view", "map_view"):
            view = row[view_name]
            relative = str(view["image_relpath"])
            path = _resolve_under(bundle_root, relative)
            if not path.is_file():
                raise FileNotFoundError(f"Missing registration image at line {line_number}: {path}")
            expected_sha256 = str(view["sha256"])
            observed_sha256 = verified_digests.get(path)
            if observed_sha256 is None:
                observed_sha256 = sha256_file(path)
                verified_digests[path] = observed_sha256
            if observed_sha256 != expected_sha256:
                raise ValueError(
                    f"Registration image SHA-256 mismatch at line {line_number}: {relative}"
                )

    leaking_groups = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leaking_groups:
        raise ValueError(f"Cross-view registration split leakage: {len(leaking_groups)} groups")
    expected_splits = {
        str(key): int(value) for key, value in validator["expected_split_counts"].items()
    }
    if dict(sorted(split_counts.items())) != dict(sorted(expected_splits.items())):
        raise ValueError("Cross-view registration split counts differ from specification")
    return {
        "kind": "cross_view_registration_v1",
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "split_group_leakage": 0,
        "verified_image_files": len(verified_digests),
        "operational_incidents": 0,
        "critical_lot_rows": 0,
    }


def _validate_image_triage_bundle(
    *, bundle_root: Path, validator: dict[str, Any]
) -> dict[str, Any]:
    rows = 0
    split_counts: Counter[str] = Counter()
    primary_class_counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    seen_samples: set[tuple[str, str]] = set()
    allowed_primary = {"fire", "fire_and_smoke", "normal", "smoke"}
    for relative in validator["manifests"]:
        manifest_path = _resolve_under(bundle_root, str(relative))
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["source_id"]), str(row["sample_id"]))
                if key in seen_samples:
                    raise ValueError(f"Duplicate triage sample at {manifest_path}:{line_number}")
                seen_samples.add(key)
                primary_class = str(row["primary_class"])
                labels = row.get("labels")
                if primary_class not in allowed_primary or not isinstance(labels, list):
                    raise ValueError(f"Invalid triage label at {manifest_path}:{line_number}")
                expected_labels = {
                    "fire": ["fire"],
                    "fire_and_smoke": ["fire", "smoke"],
                    "normal": ["normal"],
                    "smoke": ["smoke"],
                }[primary_class]
                if labels != expected_labels:
                    raise ValueError(f"Inconsistent triage labels at {manifest_path}:{line_number}")
                image_path = manifest_path.parent / _safe_relative_path(str(row["image_relpath"]))
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                split = str(row["split"])
                split_group = str(row["split_group"])
                split_counts[split] += 1
                primary_class_counts[primary_class] += 1
                split_groups[f"{row['source_id']}:{split_group}"].add(split)
                rows += 1
    expected_rows = int(validator["expected_rows"])
    if rows != expected_rows:
        raise ValueError(f"Image triage row mismatch: {rows} != {expected_rows}")
    expected_splits = {
        str(key): int(value) for key, value in validator["expected_split_counts"].items()
    }
    if dict(sorted(split_counts.items())) != dict(sorted(expected_splits.items())):
        raise ValueError("Image triage split counts differ from specification")
    expected_classes = {
        str(key): int(value) for key, value in validator["expected_primary_class_counts"].items()
    }
    if dict(sorted(primary_class_counts.items())) != dict(sorted(expected_classes.items())):
        raise ValueError("Image triage class counts differ from specification")
    leaking_groups = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leaking_groups:
        raise ValueError(f"Image triage split leakage: {len(leaking_groups)} groups")
    return {
        "kind": "image_triage_v1",
        "rows": rows,
        "split_counts": dict(sorted(split_counts.items())),
        "primary_class_counts": dict(sorted(primary_class_counts.items())),
        "split_groups": len(split_groups),
        "split_group_leakage": 0,
        "verified_image_paths": rows,
    }


def _validate_bundle_contracts(
    *, bundle_root: Path, validators: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for validator in validators:
        kind = str(validator["kind"])
        if kind == "cross_view_registration_v1":
            reports.append(
                _validate_cross_view_registration_bundle(
                    bundle_root=bundle_root,
                    validator=validator,
                )
            )
        elif kind == "image_triage_v1":
            reports.append(
                _validate_image_triage_bundle(
                    bundle_root=bundle_root,
                    validator=validator,
                )
            )
        else:
            raise ValueError(f"Unsupported bundle validator: {kind}")
    return reports


def _generated_files(
    *, spec: dict[str, Any], source_reports: list[dict[str, Any]], integrity: dict[str, Any]
) -> dict[str, bytes]:
    train_manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_format": PACKAGE_FORMAT,
        "train_id": spec["train_id"],
        "training_ready": bool(spec.get("training_ready", False)),
        "blocking_reasons": spec.get("blocking_reasons", []),
        "promotion_ready": bool(spec.get("promotion_ready", False)),
        "promotion_blocking_reasons": spec.get("promotion_blocking_reasons", []),
        "contract_revision": spec.get("contract_revision"),
        "model_contract": spec.get("model_contract"),
        "entrypoints": spec["entrypoints"],
        "sources": source_reports,
        "integrity": integrity,
        "excluded_evaluation_sets": spec.get("excluded_evaluation_sets", []),
        "publication_policy": "private_training_only",
    }
    readme = (
        f"# {spec['train_id']}\n\n"
        "Bundle FireWarning autonome pour un objectif d'entraînement unique.\n\n"
        "- `TRAIN_BUNDLE.json` décrit les sources, les licences et les commandes.\n"
        "- `PAYLOAD_CHECKSUMS.sha256` couvre tous les fichiers sources.\n"
        "- les lots critiques et incidents d'évaluation sont exclus du train ;\n"
        "- extraire le ZIP dans un dossier vide avant d'exécuter un entrypoint.\n"
    ).encode()
    return {
        "README.md": readme,
        "TRAIN_BUNDLE.json": (
            json.dumps(train_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }


def _build_zip(
    *, bundle_root: Path, train_id: str, output_path: Path, generated: dict[str, bytes]
) -> dict[str, str]:
    generated = dict(generated)
    payload_digests: dict[str, str] = {}
    expected: dict[str, str] = {}
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with zipfile.ZipFile(partial, mode="w", allowZip64=True) as archive:
        for file_index, path in enumerate(iter_files(bundle_root), start=1):
            relative = path.relative_to(bundle_root).as_posix()
            entry = f"{train_id}/{relative}"
            compressed = path.suffix.lower() not in ALREADY_COMPRESSED_SUFFIXES
            info = _zip_info(entry, compressed=compressed)
            digest = hashlib.sha256()
            with (
                path.open("rb") as source,
                archive.open(info, mode="w", force_zip64=True) as target,
            ):
                while chunk := source.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    target.write(chunk)
            payload_digests[relative] = digest.hexdigest()
            expected[entry] = payload_digests[relative]
            if file_index % 10_000 == 0:
                print(f"train-bundle ZIP write files={file_index}", flush=True)
        generated["PAYLOAD_CHECKSUMS.sha256"] = "".join(
            f"{digest}  {relative}\n" for relative, digest in payload_digests.items()
        ).encode("utf-8")
        for relative, content in sorted(generated.items()):
            entry = f"{train_id}/{relative}"
            archive.writestr(_zip_info(entry, compressed=True), content)
            expected[entry] = hashlib.sha256(content).hexdigest()
    os.replace(partial, output_path)
    return expected


def _validate_zip(path: Path, expected: dict[str, str], train_id: str) -> dict[str, Any]:
    seen: set[str] = set()
    with zipfile.ZipFile(path, mode="r", allowZip64=True) as archive:
        for file_index, info in enumerate(archive.infolist(), start=1):
            relative = _safe_relative_path(info.filename)
            if relative.parts[0] != train_id:
                raise ValueError(f"ZIP entry outside train root: {info.filename}")
            if info.filename in seen:
                raise ValueError(f"Duplicate ZIP entry: {info.filename}")
            seen.add(info.filename)
            with archive.open(info, "r") as stream:
                # Reading every entry to EOF verifies its ZIP CRC while the
                # SHA-256 check below verifies the uncompressed payload.
                if sha256_stream(stream) != expected.get(info.filename):
                    raise ValueError(f"ZIP entry SHA-256 mismatch: {info.filename}")
            if file_index % 10_000 == 0:
                print(f"train-bundle ZIP validation files={file_index}", flush=True)
    if seen != set(expected):
        raise ValueError("ZIP entry inventory mismatch")
    return {
        "zip_sha256": sha256_file(path),
        "zip_size_bytes": path.stat().st_size,
        "entry_count": len(seen),
        "crc_verified": True,
        "entry_sha256_verified": True,
        "single_train_root": train_id,
    }


def finalize_train_bundle(
    *, spec_path: Path, source_root: Path, work_dir: Path, output_dir: Path, force: bool
) -> dict[str, Any]:
    spec = _load_spec(spec_path)
    train_id = str(spec["train_id"])
    bundle_root = work_dir / train_id
    if bundle_root.exists():
        if not force:
            raise FileExistsError(bundle_root)
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_reports: list[dict[str, Any]] = []
    owned_image_roots: dict[str, Path] = {}
    mount_paths: set[str] = set()
    for source in spec["sources"]:
        mount_path = str(source["mount_path"])
        if mount_path in mount_paths:
            raise ValueError(f"Duplicate mount_path: {mount_path}")
        mount_paths.add(mount_path)
        print(
            f"train-bundle source start train={train_id} source={source['source_id']}",
            flush=True,
        )
        report, extracted_root = _extract_source(
            source=source, source_root=source_root, bundle_root=bundle_root
        )
        source_reports.append(report)
        if report["validator"] == "firewarning_image_manifest":
            owned_image_roots[str(source["source_id"])] = extracted_root
        print(
            f"train-bundle source ready train={train_id} source={source['source_id']}",
            flush=True,
        )

    integrity = _validate_cross_source_images(owned_image_roots)
    integrity["bundle_validations"] = _validate_bundle_contracts(
        bundle_root=bundle_root,
        validators=list(spec.get("bundle_validators", [])),
    )
    generated = _generated_files(spec=spec, source_reports=source_reports, integrity=integrity)
    output_path = output_dir / f"{train_id}.zip"
    if output_path.exists() and not force:
        raise FileExistsError(output_path)
    expected = _build_zip(
        bundle_root=bundle_root,
        train_id=train_id,
        output_path=output_path,
        generated=generated,
    )
    print(f"train-bundle ZIP written train={train_id} path={output_path}", flush=True)
    zip_report = _validate_zip(output_path, expected, train_id)
    print(f"train-bundle ZIP verified train={train_id}", flush=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "package_format": PACKAGE_FORMAT,
        "train_id": train_id,
        "source_validation": source_reports,
        "integrity": integrity,
        "zip_validation": zip_report,
    }
    (output_dir / f"{train_id}.validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"{train_id}.zip.sha256").write_text(
        f"{zip_report['zip_sha256']}  {train_id}.zip\n", encoding="ascii"
    )
    return report


def main() -> int:
    args = parse_args()
    report = finalize_train_bundle(
        spec_path=args.spec.resolve(),
        source_root=args.source_root.resolve(),
        work_dir=args.work_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
