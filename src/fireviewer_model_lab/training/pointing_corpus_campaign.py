"""Isolated, fail-closed preparation of the FireViewer semantic-pointing corpus.

The campaign intentionally has two different input layers:

* an existing explicit-point seed, retained for revalidation;
* a pointing-specific COCO box reservoir, retained only as annotation context.

Boxes are never converted into strong point ground truth. Detection-training and
independent-benchmark resources are rejected before any output is produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_ANCHORS = frozenset({"fire_base", "smoke_column_base"})
SPLIT_ALIASES = {"valid": "validation", "val": "validation"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL input: {path}")
    return rows


def _read_sha256_sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid checksum line at {path}:{line_number}")
        digest, raw_relative = parts
        relative = raw_relative.lstrip("*").replace("\\", "/")
        posix = PurePosixPath(relative)
        try:
            valid_digest = len(digest) == 64 and int(digest, 16) >= 0
        except ValueError:
            valid_digest = False
        if not valid_digest or posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"invalid checksum entry at {path}:{line_number}")
        normalized = posix.as_posix()
        if normalized in checksums:
            raise ValueError(f"duplicate checksum path at {path}:{line_number}: {normalized}")
        checksums[normalized] = digest.lower()
    if not checksums:
        raise ValueError(f"empty checksum input: {path}")
    return checksums


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _normal_split(value: Any) -> str:
    split = str(value or "").strip().lower()
    return SPLIT_ALIASES.get(split, split)


def _normal_coco_image_relpath(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid COCO image path: {raw!r}")
    if len(path.parts) == 1:
        return f"images/{path.name}"
    if path.parts[0] != "images":
        raise ValueError(f"unexpected COCO image path: {raw!r}")
    return path.as_posix()


def _issue(
    issues: list[dict[str, str]],
    severity: str,
    code: str,
    sample_id: str,
    detail: str,
) -> None:
    issues.append(
        {
            "severity": severity,
            "code": code,
            "sample_id": sample_id,
            "detail": detail,
        }
    )


def load_registry(path: Path) -> dict[str, Any]:
    registry = _read_json(path)
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported pointing corpus campaign registry")
    boundary = registry.get("corpus_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("registry has no corpus boundary")
    serialized_sources = json.dumps(
        {
            "seed_sources": registry.get("seed_sources"),
            "new_source_candidates": registry.get("new_source_candidates"),
        },
        sort_keys=True,
    ).lower()
    for marker in boundary.get("forbidden_references", []):
        if str(marker).lower() in serialized_sources:
            raise ValueError(f"forbidden corpus reference in source registry: {marker}")
    if boundary.get("box_to_point_policy") != "never_promote_box_bottom_center_to_ground_truth":
        raise ValueError("box-to-point isolation policy is not fail-closed")
    _validate_professional_extension_plan(registry)
    return registry


def _validate_professional_extension_plan(registry: dict[str, Any]) -> None:
    gates = registry.get("quality_gates")
    plan = registry.get("professional_extension_plan")
    if not isinstance(gates, dict) or not isinstance(plan, dict):
        raise ValueError("registry has no enforceable professional extension plan")
    families = plan.get("ready_source_families")
    capacity = plan.get("target_capacity_after_strict_admission")
    annotation = plan.get("annotation_quality")
    if not isinstance(families, list) or not families:
        raise ValueError("professional extension plan has no ready source families")
    if not isinstance(capacity, dict):
        raise ValueError("professional extension plan has no ready-data capacity target")
    if not isinstance(annotation, dict):
        raise ValueError("professional extension plan has no annotation policy")
    if annotation.get("box_center_conversion_allowed") is not False:
        raise ValueError("professional extension plan permits box-to-point conversion")
    if annotation.get("human_annotation_allowed") is not False:
        raise ValueError("professional extension plan permits human annotation")
    if annotation.get("sagemaker_ground_truth_job_allowed") is not False:
        raise ValueError("professional extension plan permits SageMaker Ground Truth")
    if annotation.get("point_source") != "strict_automated_source_provided_mask_or_sensor_geometry_only":
        raise ValueError("professional extension plan permits non-deterministic point sources")
    if "automated_silver_extension" in plan:
        raise ValueError("professional extension plan still permits pseudo-label generation")
    if plan.get("publication_allowed") is not False or plan.get("training_allowed") is not False:
        raise ValueError("professional extension plan is prematurely enabled")

    if plan.get("status") != "blocked_insufficient_ready_licensed_sources":
        if len(set(families)) < int(gates["source_families_min"]):
            raise ValueError("professional extension plan has insufficient source families")
        comparisons = {
            "unique_source_images_min": "unique_source_images_min",
            "unique_point_images_min": "strict_automated_validated_point_images_min",
            "fire_base_points_min": "fire_base_points_min",
            "smoke_column_base_points_min": "smoke_column_base_points_min",
            "explicit_negative_images_min": "explicit_negative_images_min",
        }
        for planned_name, gate_name in comparisons.items():
            if int(capacity.get(planned_name, 0)) < int(gates[gate_name]):
                raise ValueError(
                    f"professional extension capacity below gate: {planned_name} < {gate_name}"
                )

    for source in registry.get("new_source_candidates", []):
        if not isinstance(source, dict):
            raise ValueError("invalid new source candidate")
        rights = str(source.get("media_license") or "").casefold()
        status = str(source.get("admission_status") or "").casefold()
        if "unknown" in rights and "blocked" not in status:
            raise ValueError(
                f"source with unknown rights is not blocked: {source.get('source_id')}"
            )


def assert_input_isolation(registry: dict[str, Any], *values: str) -> None:
    markers = [
        str(value).lower() for value in registry["corpus_boundary"].get("forbidden_references", [])
    ]
    for value in values:
        lowered = value.lower()
        for marker in markers:
            if marker in lowered:
                raise ValueError(f"forbidden non-pointing corpus input: {value}")


def audit_point_seed(
    rows: list[dict[str, Any]],
    *,
    source_s3_prefix: str,
    materialized_hashes: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]], set[str]]:
    issues: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    source_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    source_hashes: set[str] = set()
    split_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    canonical_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, 1):
        sample_id = str(row.get("sample_id") or f"row-{index}")
        if sample_id in sample_ids:
            _issue(issues, "error", "duplicate_sample_id", sample_id, sample_id)
        sample_ids.add(sample_id)
        split = _normal_split(row.get("split"))
        split_counts[split] += 1
        if split not in ALLOWED_SPLITS:
            _issue(issues, "error", "invalid_split", sample_id, split)

        source = row.get("source")
        if not isinstance(source, dict):
            _issue(issues, "error", "missing_source", sample_id, "source must be an object")
            continue
        source_id = str(source.get("source_id") or "")
        source_sha = str(source.get("source_sha256") or "").lower()
        image_relpath = str(source.get("image_relpath") or "")
        materialized_sha = materialized_hashes.get(image_relpath, "")
        split_group = str(row.get("split_group") or "")
        width = int(source.get("width") or 0)
        height = int(source.get("height") or 0)
        if not source_id or not source_sha or not image_relpath or not split_group:
            _issue(
                issues,
                "error",
                "incomplete_provenance",
                sample_id,
                "required source field missing",
            )
        if width <= 0 or height <= 0:
            _issue(issues, "error", "invalid_dimensions", sample_id, f"{width}x{height}")
        if image_relpath in image_paths:
            _issue(issues, "error", "duplicate_materialized_path", sample_id, image_relpath)
        if not materialized_sha:
            _issue(
                issues,
                "error",
                "missing_materialized_sha256",
                sample_id,
                image_relpath,
            )
        image_paths.add(image_relpath)
        source_splits[source_id].add(split)
        group_splits[split_group].add(split)
        if source_sha:
            source_hashes.add(source_sha)

        variant = str(row.get("variant") or "")
        variant_counts[variant] += 1
        targets = row.get("targets")
        if not isinstance(targets, list):
            _issue(issues, "error", "targets_not_list", sample_id, type(targets).__name__)
            targets = []
        seen_anchors: set[str] = set()
        valid_targets: list[dict[str, Any]] = []
        for target_index, target in enumerate(targets):
            if not isinstance(target, dict):
                _issue(
                    issues,
                    "error",
                    "invalid_target",
                    sample_id,
                    f"target index {target_index}",
                )
                continue
            anchor = str(target.get("semantic_anchor") or "")
            if anchor not in ALLOWED_ANCHORS:
                _issue(issues, "error", "unknown_semantic_anchor", sample_id, anchor)
                continue
            if anchor in seen_anchors:
                _issue(issues, "error", "duplicate_semantic_anchor", sample_id, anchor)
            seen_anchors.add(anchor)
            normalized = target.get("point_normalized")
            pixel = target.get("point_pixel")
            if not isinstance(normalized, list) or len(normalized) != 2:
                _issue(issues, "error", "invalid_normalized_point", sample_id, anchor)
                continue
            try:
                x_norm, y_norm = float(normalized[0]), float(normalized[1])
            except (TypeError, ValueError):
                _issue(issues, "error", "invalid_normalized_point", sample_id, anchor)
                continue
            if not (math.isfinite(x_norm) and math.isfinite(y_norm)) or not (
                0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0
            ):
                _issue(issues, "error", "point_out_of_bounds", sample_id, anchor)
                continue
            if isinstance(pixel, list) and len(pixel) == 2 and width > 0 and height > 0:
                try:
                    x_px, y_px = float(pixel[0]), float(pixel[1])
                except (TypeError, ValueError):
                    _issue(issues, "error", "invalid_pixel_point", sample_id, anchor)
                else:
                    if abs(x_px / width - x_norm) > 0.005 or abs(y_px / height - y_norm) > 0.005:
                        _issue(issues, "error", "point_coordinate_mismatch", sample_id, anchor)
            target_counts[anchor] += 1
            valid_targets.append(
                {
                    "semantic_anchor": anchor,
                    "point_normalized": [x_norm, y_norm],
                    "point_pixel": pixel,
                    "point_origin": target.get("point_origin"),
                }
            )

        if variant == "clean":
            canonical_rows.append(
                {
                    "schema_version": 2,
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "source_family": "fireviewer-pointing-ground-v1",
                    "source_sha256": source_sha,
                    "source_image_sha256": materialized_sha,
                    "split": split,
                    "split_group": split_group,
                    "image_s3_uri": f"{source_s3_prefix.rstrip('/')}/{image_relpath}",
                    "width": width,
                    "height": height,
                    "targets": valid_targets,
                    "annotation_strength": "marker_derived_pending_strict_automated_validation",
                    "upstream_training_eligible": bool(row.get("training_eligible")),
                    "admission_status": "excluded_until_rights_and_strict_validation_pass",
                    "corpus_disposition": "excluded_unvalidated",
                    "exclusion_reasons": ["rights_not_cleared", "strict_validation_pending"],
                    "strict_keep": False,
                    "strict_validation_status": "excluded_unvalidated",
                    "validation_profile": "fireviewer_pointing_strict_automated_v1",
                    "training_eligible": False,
                }
            )

    leaking_sources = sorted(key for key, splits in source_splits.items() if len(splits) > 1)
    leaking_groups = sorted(key for key, splits in group_splits.items() if len(splits) > 1)
    for source_id in leaking_sources:
        _issue(
            issues,
            "error",
            "source_cross_split",
            source_id,
            sorted(source_splits[source_id]).__repr__(),
        )
    for group in leaking_groups:
        _issue(issues, "error", "group_cross_split", group, sorted(group_splits[group]).__repr__())

    canonical_source_counts = Counter(row["source_id"] for row in canonical_rows)
    for source_id, count in canonical_source_counts.items():
        if count != 1:
            _issue(issues, "error", "clean_view_count_not_one", source_id, str(count))

    report = {
        "materialized_rows": len(rows),
        "unique_source_images": len(source_splits),
        "canonical_clean_rows": len(canonical_rows),
        "materialized_hashes_resolved": sum(
            bool(row["source_image_sha256"]) for row in canonical_rows
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "target_counts_all_views": dict(sorted(target_counts.items())),
        "source_cross_split": leaking_sources,
        "group_cross_split": leaking_groups,
        "errors": sum(issue["severity"] == "error" for issue in issues),
    }
    return report, canonical_rows, issues, source_hashes


def audit_box_reservoir(
    rows: list[dict[str, Any]], *, source_s3_prefix: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]], set[str]]:
    issues: list[dict[str, str]] = []
    sample_ids: set[str] = set()
    hashes: set[str] = set()
    hash_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    invalid_geometry = 0

    for index, row in enumerate(rows, 1):
        digest = str(row.get("sha256") or "").lower()
        sample_id = digest or str(row.get("image_id") or f"row-{index}")
        if sample_id in sample_ids:
            _issue(issues, "error", "duplicate_sample_id", sample_id, sample_id)
        sample_ids.add(sample_id)
        split = _normal_split(row.get("split"))
        if split not in ALLOWED_SPLITS:
            _issue(issues, "error", "invalid_split", sample_id, split)
        split_counts[split] += 1
        width = int(row.get("width") or 0)
        height = int(row.get("height") or 0)
        if width <= 0 or height <= 0:
            _issue(issues, "error", "invalid_dimensions", sample_id, f"{width}x{height}")
        if not digest:
            _issue(issues, "error", "missing_sha256", sample_id, "")
        elif digest in hashes:
            _issue(issues, "error", "duplicate_image_identity", sample_id, digest)
        hashes.add(digest)
        hash_splits[digest].add(split)
        source_group = str(row.get("split_group_id") or row.get("source_group_id") or "")
        if not source_group:
            _issue(issues, "error", "missing_split_group", sample_id, "")
        group_splits[source_group].add(split)
        source_family = str(row.get("source_family") or row.get("source_dataset") or "unknown")
        source_counts[source_family] += 1
        scene_counts[str(row.get("scene_bin") or "unknown")] += 1

        objects = row.get("objects")
        raw_boxes = objects.get("bbox", []) if isinstance(objects, dict) else []
        raw_categories = objects.get("category", []) if isinstance(objects, dict) else []
        if not isinstance(raw_boxes, list) or not isinstance(raw_categories, list):
            _issue(issues, "error", "invalid_objects", sample_id, "bbox/category must be lists")
            raw_boxes, raw_categories = [], []
        if len(raw_boxes) != len(raw_categories):
            _issue(issues, "error", "box_category_length_mismatch", sample_id, "")
        boxes: list[dict[str, Any]] = []
        for box_index, (raw_box, raw_category) in enumerate(
            zip(raw_boxes, raw_categories, strict=False)
        ):
            try:
                x, y, box_width, box_height = (float(value) for value in raw_box)
                category = int(raw_category)
            except (TypeError, ValueError):
                invalid_geometry += 1
                _issue(issues, "error", "invalid_box", sample_id, str(box_index))
                continue
            if category not in {0, 1}:
                _issue(issues, "error", "invalid_category", sample_id, str(category))
                continue
            values = (x, y, box_width, box_height)
            valid = all(math.isfinite(value) for value in values)
            valid = valid and x >= 0 and y >= 0 and box_width > 0 and box_height > 0
            valid = valid and x + box_width <= width + 1 and y + box_height <= height + 1
            if not valid:
                invalid_geometry += 1
                _issue(issues, "error", "invalid_box_geometry", sample_id, str(raw_box))
                continue
            boxes.append(
                {
                    "category_id": category,
                    "class_name": "fire" if category == 0 else "smoke",
                    "bbox_xywh": [x, y, box_width, box_height],
                    "annotation_role": "visual_context_only_not_point_ground_truth",
                }
            )

        is_negative = bool(row.get("is_negative"))
        if is_negative and boxes:
            _issue(issues, "error", "negative_has_boxes", sample_id, str(len(boxes)))
        if not is_negative and not boxes:
            _issue(issues, "error", "positive_has_no_boxes", sample_id, "")
        file_name = str(row.get("file_name") or row.get("artifact_image") or "")
        if not file_name:
            _issue(issues, "error", "missing_image_relpath", sample_id, "")
            continue
        try:
            image_relpath = _normal_coco_image_relpath(file_name)
        except ValueError as exc:
            _issue(issues, "error", "invalid_image_relpath", sample_id, str(exc))
            continue
        storage_split = "valid" if split == "validation" else split
        image_s3_uri = f"{source_s3_prefix.rstrip('/')}/coco/{storage_split}/{image_relpath}"
        candidates.append(
            {
                "schema_version": 2,
                "sample_id": f"pointing-v8:{sample_id}",
                "source_image_sha256": digest,
                "source_id": str(row.get("source_dataset") or source_family),
                "source_family": source_family,
                "source_record_id": str(row.get("source_record_id") or ""),
                "source_revision": str(row.get("source_revision") or ""),
                "split": split,
                "split_group": source_group,
                "image_s3_uri": image_s3_uri,
                "width": width,
                "height": height,
                "existing_boxes": boxes,
                "scene_bin": str(row.get("scene_bin") or "unknown"),
                "lighting": str(row.get("lighting_review") or "unknown"),
                "visibility": str(row.get("visibility_review") or "unknown"),
                "is_negative_candidate": is_negative,
                "point_annotation_status": "missing_explicit_point_annotation",
                "box_to_point_conversion": "prohibited",
                "corpus_disposition": "excluded_unvalidated",
                "exclusion_reasons": ["missing_explicit_point_annotation"],
                "strict_keep": False,
                "strict_validation_status": "excluded_unvalidated",
                "validation_profile": "fireviewer_pointing_strict_automated_v1",
                "training_eligible": False,
                "license": str(row.get("license") or "unknown"),
            }
        )

    leaking_hashes = sorted(key for key, splits in hash_splits.items() if len(splits) > 1)
    leaking_groups = sorted(key for key, splits in group_splits.items() if len(splits) > 1)
    for digest in leaking_hashes:
        _issue(
            issues,
            "error",
            "exact_hash_cross_split",
            digest,
            sorted(hash_splits[digest]).__repr__(),
        )
    for group in leaking_groups:
        _issue(issues, "error", "group_cross_split", group, sorted(group_splits[group]).__repr__())

    report = {
        "rows": len(rows),
        "candidate_rows": len(candidates),
        "split_counts": dict(sorted(split_counts.items())),
        "source_family_counts": dict(sorted(source_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "negative_candidates": sum(row["is_negative_candidate"] for row in candidates),
        "positive_candidates": sum(not row["is_negative_candidate"] for row in candidates),
        "invalid_geometry": invalid_geometry,
        "exact_hash_cross_split": leaking_hashes,
        "group_cross_split": leaking_groups,
        "errors": sum(issue["severity"] == "error" for issue in issues),
        "point_ground_truth_rows": 0,
    }
    return report, candidates, issues, hashes


def evaluate_seed_gates(
    registry: dict[str, Any],
    seed_report: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    gates = registry["quality_gates"]
    targets = Counter(
        target["semantic_anchor"] for row in seed_rows for target in row.get("targets", [])
    )
    split_counts = Counter(row["split"] for row in seed_rows)
    values: dict[str, int | float] = {
        "unique_source_images_min": int(seed_report["unique_source_images"]),
        "strict_automated_validated_point_images_min": 0,
        "fire_base_points_min": targets["fire_base"],
        "smoke_column_base_points_min": targets["smoke_column_base"],
        "explicit_negative_images_min": sum(not row.get("targets") for row in seed_rows),
        "source_families_min": len({row["source_family"] for row in seed_rows}),
        "validation_images_min": split_counts["validation"],
        "test_images_min": split_counts["test"],
        "unknown_semantics_max": sum(
            issue["code"] == "unknown_semantic_anchor" for issue in issues
        ),
        "invalid_geometry_max": sum(
            issue["code"] in {"point_out_of_bounds", "point_coordinate_mismatch"}
            for issue in issues
        ),
        "exact_cross_split_duplicates_max": sum(
            issue["code"] in {"source_cross_split"} for issue in issues
        ),
        "split_group_leaks_max": sum(issue["code"] == "group_cross_split" for issue in issues),
        "unknown_or_incompatible_rights_max": sum(
            "rights_not_cleared" in row.get("exclusion_reasons", []) for row in seed_rows
        ),
    }
    # Stage 1 emits only the canonical clean view for each source image. The
    # upstream augmentation share is retained in the seed report, but those
    # materialized variants do not enter the rebuilt corpus.
    values["materialized_augmentation_fraction_max"] = 0.0
    source_counts = Counter(row["source_family"] for row in seed_rows)
    total = sum(source_counts.values()) or 1
    shares = sorted((count / total for count in source_counts.values()), reverse=True)
    values["largest_source_share_max"] = shares[0] if shares else 0.0
    values["top_three_source_share_max"] = sum(shares[:3])
    values["low_light_positive_images_min"] = 0
    values["small_or_faint_positive_images_min"] = 0

    checks: list[dict[str, Any]] = []
    for gate, threshold in gates.items():
        actual = values.get(gate, 0)
        if gate.endswith("_min"):
            passed = actual >= threshold
            rule = "min"
        elif gate.endswith("_max"):
            passed = actual <= threshold
            rule = "max"
        else:
            raise ValueError(f"quality gate must end in _min or _max: {gate}")
        checks.append(
            {
                "gate": gate,
                "rule": rule,
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }
        )
    return checks


def run_stage1(
    *,
    registry_path: Path,
    hf_root: Path,
    v8_root: Path,
    output: Path,
    hf_s3_prefix: str,
    v8_s3_prefix: str,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    assert_input_isolation(
        registry,
        str(hf_root),
        str(v8_root),
        hf_s3_prefix,
        v8_s3_prefix,
    )
    hf_manifest = hf_root / "manifest.jsonl"
    v8_manifest = v8_root / "selection_manifest.jsonl"
    seed_report, seed_rows, seed_issues, seed_hashes = audit_point_seed(
        _read_jsonl(hf_manifest),
        source_s3_prefix=hf_s3_prefix,
        materialized_hashes=_read_sha256_sums(hf_root / "checksums.sha256"),
    )
    reservoir_report, candidates, reservoir_issues, reservoir_hashes = audit_box_reservoir(
        _read_jsonl(v8_manifest), source_s3_prefix=v8_s3_prefix
    )
    overlap = sorted(seed_hashes & reservoir_hashes)
    cross_issues = [
        {
            "severity": "error",
            "code": "seed_reservoir_exact_overlap",
            "sample_id": digest,
            "detail": "same source identity in point seed and box reservoir",
        }
        for digest in overlap
    ]
    overlap_set = set(overlap)
    overlap_clean_candidates = [
        row for row in candidates if row["source_image_sha256"] not in overlap_set
    ]
    forbidden_families = {
        str(value).casefold()
        for value in registry["corpus_boundary"].get("forbidden_source_families", [])
    }
    quarantined_candidates = [
        row
        for row in overlap_clean_candidates
        if str(row["source_family"]).casefold() in forbidden_families
    ]
    clean_candidates = [
        row
        for row in overlap_clean_candidates
        if str(row["source_family"]).casefold() not in forbidden_families
    ]
    all_issues = seed_issues + reservoir_issues + cross_issues
    all_issues.extend(
        {
            "severity": "warning",
            "code": "forbidden_source_family_quarantined",
            "sample_id": str(row["sample_id"]),
            "detail": str(row["source_family"]),
        }
        for row in quarantined_candidates
    )
    gate_checks = evaluate_seed_gates(registry, seed_report, seed_rows, all_issues)
    all_gates_pass = all(check["passed"] for check in gate_checks)

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "seed_points_pending_strict_validation.jsonl", seed_rows)
    _write_jsonl(output / "box_reservoir_inventory.jsonl", clean_candidates)
    _write_jsonl(output / "automatic_dispositions_stage1.jsonl", seed_rows + clean_candidates)
    with (output / "issues.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("severity", "code", "sample_id", "detail"))
        writer.writeheader()
        writer.writerows(all_issues)

    summary = {
        "schema_version": 1,
        "campaign_id": registry["campaign_id"],
        "corpus_boundary": registry["corpus_boundary"],
        "inputs": {
            "point_seed": hf_s3_prefix,
            "pointing_box_reservoir": v8_s3_prefix,
            "detection_corpus_used": False,
            "independent_benchmark_used": False,
        },
        "point_seed": seed_report,
        "box_reservoir": reservoir_report,
        "seed_reservoir_exact_overlap": len(overlap),
        "quarantined_source_candidates": {
            "rows": len(quarantined_candidates),
            "source_family_counts": dict(
                sorted(Counter(row["source_family"] for row in quarantined_candidates).items())
            ),
        },
        "strict_automation": {
            **registry["strict_automation"],
            "rows_evaluated": len(seed_rows) + len(clean_candidates),
            "rows_admitted_stage1": 0,
            "rows_excluded_stage1": len(seed_rows) + len(clean_candidates),
            "box_derived_points": 0,
        },
        "quality_gates": gate_checks,
        "satisfactory": all_gates_pass,
        "publication_allowed": False,
        "publication_blockers": (
            [check["gate"] for check in gate_checks if not check["passed"]]
            + ["strict_pixel_validation_pending", "no_strictly_validated_point_source"]
        ),
        "next_action": "run_full_pixel_audit_then_compute_strict_automatic_dispositions",
        "issues": {
            "errors": sum(issue["severity"] == "error" for issue in all_issues),
            "warnings": sum(issue["severity"] == "warning" for issue in all_issues),
        },
    }
    _write_json(output / "audit_summary.json", summary)
    _write_json(
        output / "SATISFACTION_STATUS.json",
        {
            "campaign_id": registry["campaign_id"],
            "satisfactory": all_gates_pass,
            "publication_allowed": False,
            "failed_gates": [check for check in gate_checks if not check["passed"]],
            "validation_profile": registry["strict_automation"]["validation_profile"],
            "reviews_admitted": False,
            "repeat_until_satisfactory": True,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the isolated FireViewer pointing corpus")
    parser.add_argument("stage1", choices=("stage1",))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--hf-root", type=Path, required=True)
    parser.add_argument("--v8-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-s3-prefix", required=True)
    parser.add_argument("--v8-s3-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_stage1(
        registry_path=args.registry,
        hf_root=args.hf_root,
        v8_root=args.v8_root,
        output=args.output,
        hf_s3_prefix=args.hf_s3_prefix,
        v8_s3_prefix=args.v8_s3_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
