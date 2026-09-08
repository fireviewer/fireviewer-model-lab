"""Compose the pinned detection base with strict multi-task annotation overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from training.dinov3_corpus_identity import (
    benchmark_matches,
    load_benchmark_denylist,
    resolve_namespaced_source_identity,
    resolve_source_identity,
    validate_canonical_event_splits,
    validate_composed_row_identities,
    validate_source_identity_contract,
)

DETECTION_CLASSES = frozenset({"flame_visible", "smoke_visible"})
SPLITS = frozenset({"train", "validation", "test"})
WEAK_MARKERS = ("weak", "teacher", "pseudo")
BBOX_POINT_MARKERS = ("bbox", "box_center", "box_bottom_center", "bounding_box")
DEFAULT_BENCHMARK_DENYLIST = (
    Path(__file__).with_name("registries") / "dinov3-independent-benchmark-denylist-v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path.name}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _safe_file(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe overlay payload path: {relative}")
    root = root.resolve()
    path = (root / Path(*posix.parts)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"overlay payload escapes source root: {relative}")
    return path


def _valid_sha(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe repository path: {value}")
    return path.as_posix()


def _detection_row(
    source: dict[str, Any], *, repository: str, revision: str, manifest_path: str
) -> tuple[dict[str, Any], bool]:
    sample_id = str(source.get("sample_id") or "")
    if not sample_id:
        raise ValueError("detection base row has no sample_id")
    if source.get("sample_validation_status") != "strict_automated_validated":
        raise ValueError(f"detection row is not strict: {sample_id}")
    if source.get("validation_profile") != "fireviewer_detection_strict_automated_v1":
        raise ValueError(f"detection validation profile drift: {sample_id}")
    split = str(source.get("split") or "")
    split_group = str(source.get("split_group") or "")
    digest = str(source.get("sha256") or "").lower()
    if split not in SPLITS or not split_group or not _valid_sha(digest):
        raise ValueError(f"invalid detection identity contract: {sample_id}")
    license_name = str(source.get("license") or "")
    if not license_name or not isinstance(source.get("consent_basis"), dict):
        raise ValueError(f"detection rights contract missing: {sample_id}")
    annotations = source.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError(f"detection annotations are not a list: {sample_id}")
    classes: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"invalid detection annotation: {sample_id}")
        class_name = str(annotation.get("class_name") or "")
        if class_name not in DETECTION_CLASSES:
            raise ValueError(f"unknown detection class {class_name!r}: {sample_id}")
        classes.add(class_name)
    negative = not annotations
    if negative:
        negative_tags = source.get("negative_tags")
        if not isinstance(negative_tags, list) or "no_target_visible" not in negative_tags:
            raise ValueError(f"empty detection row is not an explicit negative: {sample_id}")
    elif source.get("negative") is True:
        raise ValueError(f"positive detection row is marked negative: {sample_id}")
    image_relative = _safe_relative(str(source.get("image_relpath") or ""))
    extension = PurePosixPath(image_relative).suffix
    extension = extension.casefold() if extension else ".jpg"
    source_id = str(source.get("source_id") or "unknown")
    source_identity = resolve_namespaced_source_identity(
        lineage_root_id=f"hf-dataset:{repository}",
        source_event_key=f"{source_id}:{split_group}",
    )
    source_identity["source_split_group"] = split_group
    row: dict[str, Any] = {
        "schema_version": 2,
        "sample_id": sample_id,
        "image_sha256": digest,
        "image_extension": extension,
        "image_locator": {
            "kind": "hf_dataset_row",
            "repository": repository,
            "revision": revision,
            "split": split,
            "sample_id": sample_id,
            "sha256": digest,
            "path": image_relative,
        },
        "source_id": source_id,
        "source_record_id": str(source.get("source_record_id") or ""),
        "source_manifest_path": manifest_path,
        "split": split,
        "split_group": f"event:{source_identity['canonical_event_id']}",
        "license": license_name,
        "consent_basis": source["consent_basis"],
        "sample_validation_status": "strict_automated_validated",
        "validation_profile": "fireviewer_multitask_composition_v1",
        "detection_validation_profile": source["validation_profile"],
        "detection_validation_run_id": str(source.get("validation_run_id") or ""),
        "detection_annotation_classes": sorted(classes),
        "detection_annotation_count": len(annotations),
        "anchor_points": [],
        "bbox_to_point": False,
        "sample_weight": 1.0,
        "presence_supervised": True,
        "presence_targets": {
            "flame_visible": "flame_visible" in classes,
            "smoke_visible": "smoke_visible" in classes,
        },
        "presence_provenance": "pinned_strict_detection_class_annotations",
        "overlay_sources": [],
        **source_identity,
    }
    if negative:
        row.update(
            {
                "annotation_strength": "negative",
                "segmentation_supervised": True,
                "point_supervised": True,
                "abstention_supervised": True,
                "mask_encoding": "implicit_zero_from_explicit_negative",
                "mask_quality": "strict_explicit_detection_negative_zero",
                "point_derivation": "explicit_absence_no_point",
                "visual_abstention_reason": "no_fire_or_smoke_visible",
            }
        )
    else:
        row.update(
            {
                "annotation_strength": "strong_presence_only",
                "segmentation_supervised": False,
                "point_supervised": False,
                "abstention_supervised": False,
                "point_derivation": "none_detection_boxes_ignored",
                "visual_abstention_reason": None,
            }
        )
    return row, negative


def _artifact_locator(
    row: dict[str, Any], root: Path, *, kind: str, required: bool
) -> dict[str, str] | None:
    rel_field = f"{kind}_relpath"
    sha_field = f"{kind}_sha256"
    uri_field = f"{kind}_s3_uri"
    relative, digest, uri = (
        row.get(rel_field),
        str(row.get(sha_field) or "").lower(),
        row.get(uri_field),
    )
    if not relative and not digest and not uri and not required:
        return None
    if (
        not relative
        or not _valid_sha(digest)
        or not isinstance(uri, str)
        or not uri.startswith("s3://")
    ):
        raise ValueError(f"incomplete {kind} locator for {row.get('sample_id')}")
    path = _safe_file(root, str(relative))
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"{kind} payload rehash failed for {row.get('sample_id')}")
    extension = PurePosixPath(str(relative)).suffix.casefold()
    return {"kind": "s3_object", "uri": uri, "sha256": digest, "extension": extension}


def _validated_overlay_rows(
    *, root: Path, contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests = list(root.rglob(str(contract["manifest_filename"])))
    reports = list(root.rglob(str(contract["report_filename"])))
    if len(manifests) != 1 or len(reports) != 1:
        raise FileNotFoundError(f"overlay contract files missing for {contract['name']}")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    rows = _read_jsonl(manifests[0])
    manifest_sha256 = _sha256(manifests[0])
    report_sha256 = _sha256(reports[0])
    if (
        contract.get("manifest_sha256") != manifest_sha256
        or contract.get("report_sha256") != report_sha256
    ):
        raise ValueError(f"overlay receipt hash drift for {contract['name']}")
    minimum = int(contract["strict_rows_min"])
    maximum = int(contract["strict_rows_max"])
    if not minimum <= len(rows) <= maximum:
        raise ValueError(f"overlay row count drift for {contract['name']}: {len(rows)}")
    report_count_field = str(
        contract.get("report_row_count_field") or "strict_automated_validated_rows"
    )
    if int(report.get(report_count_field, -1)) != len(rows):
        raise ValueError(f"overlay report count mismatch for {contract['name']}")
    required_report_values = contract.get(
        "report_required_values",
        {
            "reviews_admitted": False,
            "publication_allowed": False,
            "source_gate_passed": True,
        },
    )
    if not isinstance(required_report_values, dict) or any(
        report.get(key) != expected for key, expected in required_report_values.items()
    ):
        raise ValueError(f"overlay source gate state invalid for {contract['name']}")
    error_fields = contract.get(
        "report_error_fields",
        ["gate_errors", "decode_or_payload_errors", "split_group_leakage"],
    )
    if not isinstance(error_fields, list):
        raise ValueError(f"overlay report error fields invalid for {contract['name']}")
    for key in error_fields:
        if report.get(key):
            raise ValueError(f"overlay report has {key} for {contract['name']}")
    return rows, {
        "name": contract["name"],
        "manifest_sha256": manifest_sha256,
        "report_sha256": report_sha256,
        "rows": len(rows),
        "source_gate_passed": True,
    }


def _profile_targets(
    row: dict[str, Any], profile: str
) -> tuple[dict[str, Any], dict[str, str] | None]:
    points = row.get("anchor_points")
    if not isinstance(points, list):
        raise ValueError(f"overlay anchors missing for {row.get('sample_id')}")
    for point in points:
        if not isinstance(point, dict):
            raise ValueError(f"invalid overlay point for {row.get('sample_id')}")
        x, y = float(point["x"]), float(point["y"])
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"overlay point out of bounds for {row.get('sample_id')}")
    derivation = str(row.get("point_derivation") or row.get("mask_to_point_conversion") or "")
    if "bbox" in derivation.casefold() or row.get("bbox_to_point"):
        raise ValueError(f"bbox-derived overlay point rejected: {row.get('sample_id')}")
    if profile == "smoke_segmentation_and_ground_point":
        if not points or not all(
            str(point.get("kind", "")) == "smoke_column_base" for point in points
        ):
            raise ValueError(f"Boreal smoke point contract failed: {row.get('sample_id')}")
        targets = {"flame_visible": False, "smoke_visible": True}
    elif profile == "fire_segmentation_and_ground_point":
        if not points or not all(str(point.get("kind", "")) == "fire_base" for point in points):
            raise ValueError(f"Camp Swift fire point contract failed: {row.get('sample_id')}")
        targets = {"flame_visible": True, "smoke_visible": False}
    elif profile == "industrial_segmentation_presence_abstention_no_point":
        if points or row.get("ground_point_eligible") is not False:
            raise ValueError(f"KIT fake point contract failed: {row.get('sample_id')}")
        targets = {"flame_visible": True, "smoke_visible": False}
    else:
        raise ValueError(f"unknown overlay task profile: {profile}")
    if not profile.endswith("no_point") and not derivation:
        raise ValueError(f"overlay point provenance missing for {row.get('sample_id')}")
    annotation_strength = str(row.get("annotation_strength") or "")
    annotation_provenance = str(row.get("annotation_provenance") or "")
    mask_quality = str(row.get("mask_quality") or "")
    if not annotation_strength or not annotation_provenance or not mask_quality:
        raise ValueError(f"overlay annotation contract incomplete: {row.get('sample_id')}")
    if profile.endswith("no_point"):
        update = {
            "segmentation_supervised": True,
            "point_supervised": False,
            "presence_supervised": True,
            "abstention_supervised": True,
            "presence_targets": targets,
            "presence_provenance": "strict_human_flame_mask",
            "anchor_points": [],
            "point_derivation": "none",
            "visual_abstention_reason": str(row.get("visual_abstention_reason") or ""),
            "annotation_strength": annotation_strength,
            "annotation_provenance": annotation_provenance,
            "mask_quality": mask_quality,
            "sample_weight": 0.5,
        }
    else:
        update = {
            "segmentation_supervised": True,
            "point_supervised": True,
            "presence_supervised": True,
            "abstention_supervised": True,
            "presence_targets": targets,
            "presence_provenance": str(row.get("annotation_provenance") or "strict_mask"),
            "anchor_points": points,
            "point_derivation": derivation,
            "visual_abstention_reason": None,
            "annotation_strength": annotation_strength,
            "annotation_provenance": annotation_provenance,
            "mask_quality": mask_quality,
            "sample_weight": 2.0,
        }
    return update, None


def compose_multitask(
    *,
    registry_path: Path,
    detection_manifests: dict[str, Path],
    overlay_roots: dict[str, Path],
    output_dir: Path,
    benchmark_denylist_path: Path,
    detection_receipts: list[dict[str, Any]] | None = None,
    control_roots: dict[str, Path] | None = None,
) -> dict[str, Any]:
    registry_sha256 = _sha256(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 2:
        raise ValueError("unsupported composition registry schema")
    identities = validate_source_identity_contract(registry)
    benchmark_boundary = registry.get("benchmark_boundary")
    if not isinstance(benchmark_boundary, dict):
        raise ValueError("benchmark boundary contract is missing")
    benchmark_denylist = load_benchmark_denylist(benchmark_denylist_path, benchmark_boundary)
    detection = registry["detection_base"]
    expected_manifest_names = set(detection["manifest_paths"])
    if set(detection_manifests) != expected_manifest_names:
        raise ValueError("detection manifest set differs from the registry")
    if not isinstance(detection_receipts, list):
        raise ValueError("detection download receipts are mandatory")
    receipts_by_name = {str(receipt.get("name") or ""): receipt for receipt in detection_receipts}
    if set(receipts_by_name) != expected_manifest_names:
        raise ValueError("detection receipt set differs from the registry")
    for name, path in detection_manifests.items():
        receipt = receipts_by_name[name]
        if (
            receipt.get("repository_path") != detection["manifest_paths"][name]
            or receipt.get("revision") != detection["revision"]
            or receipt.get("sha256") != _sha256(path)
            or int(receipt.get("bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(f"detection receipt contract failed: {name}")
    rows_by_sha: dict[str, dict[str, Any]] = {}
    detection_records: dict[tuple[str, str], dict[str, Any]] = {}
    sample_ids: set[str] = set()
    detection_split_counts: Counter[str] = Counter()
    detection_source_counts: Counter[str] = Counter()
    detection_positive_rows = 0
    detection_negative_rows = 0
    for name, path in sorted(detection_manifests.items()):
        manifest_path = str(detection["manifest_paths"][name])
        for source in _read_jsonl(path):
            row, negative = _detection_row(
                source,
                repository=str(detection["repository"]),
                revision=str(detection["revision"]),
                manifest_path=manifest_path,
            )
            digest, sample_id = str(row["image_sha256"]), str(row["sample_id"])
            if digest in rows_by_sha or sample_id in sample_ids:
                raise ValueError(f"duplicate detection identity: {sample_id}:{digest}")
            rows_by_sha[digest] = row
            record_key = (str(row["source_id"]), str(row["source_record_id"]))
            if record_key[1]:
                if record_key in detection_records:
                    raise ValueError(f"duplicate detection source record: {record_key}")
                detection_records[record_key] = row
            sample_ids.add(sample_id)
            detection_split_counts[str(row["split"])] += 1
            detection_source_counts[str(row["source_id"])] += 1
            detection_negative_rows += int(negative)
            detection_positive_rows += int(not negative)

    control_roots = control_roots or {}
    control_contracts = registry.get("control_sources", [])
    expected_control_names = {str(contract["name"]) for contract in control_contracts}
    if set(control_roots) != expected_control_names:
        raise ValueError("control source set differs from the registry")
    control_receipts: list[dict[str, Any]] = []
    control_rows_verified = 0
    for contract in control_contracts:
        name = str(contract["name"])
        root = control_roots[name]
        cleaning = contract.get("detection_cleaning_evidence")
        if not isinstance(cleaning, dict):
            raise ValueError(f"negative control cleaning evidence missing: {name}")
        source_rows = int(cleaning["source_rows"])
        exact_removed = int(cleaning["source_exact_duplicates_removed"])
        post_source_rows = int(cleaning["post_source_dedup_rows"])
        strict_kept = int(cleaning["final_strict_kept_rows"])
        strict_not_admitted = int(cleaning["final_strict_not_admitted_rows"])
        detection_source_id = str(contract.get("detection_source_id") or "")
        if (
            source_rows - exact_removed != post_source_rows
            or strict_kept + strict_not_admitted != post_source_rows
            or detection_source_counts[detection_source_id] != strict_kept
        ):
            raise ValueError(f"negative control cleaning evidence drift: {name}")
        source_rows, receipt = _validated_overlay_rows(root=root, contract=contract)
        exact_sha_matches = 0
        source_record_identity_matches = 0
        not_admitted_by_detection_base = 0
        seen_control_records: set[str] = set()
        for source in source_rows:
            sample_id = str(source.get("sample_id") or "")
            digest = str(source.get("image_sha256") or "").lower()
            if (
                source.get("sample_validation_status") != "strict_automated_validated"
                or source.get("strict_keep") is not True
                or source.get("training_eligible") is not True
                or source.get("negative") is not True
                or source.get("source_annotations_exactly_empty") is not True
                or source.get("source_id") != contract["source_id"]
                or source.get("source_revision") != contract["source_revision"]
                or source.get("reviews_admitted") is not False
                or not _valid_sha(digest)
            ):
                raise ValueError(f"negative control row contract failed: {name}:{sample_id}")
            _artifact_locator(source, root, kind="image", required=True)
            _artifact_locator(source, root, kind="mask", required=True)
            base_row = rows_by_sha.get(digest)
            if base_row is not None:
                exact_sha_matches += 1
            else:
                record_field = str(contract.get("control_record_field") or "source_record_id")
                source_record_id = str(source.get(record_field) or "")
                if not source_record_id or source_record_id in seen_control_records:
                    raise ValueError(f"invalid negative control source record: {sample_id}")
                seen_control_records.add(source_record_id)
                base_row = detection_records.get((detection_source_id, source_record_id))
                source_record_identity_matches += int(base_row is not None)
            if base_row is None:
                not_admitted_by_detection_base += 1
            elif base_row.get("annotation_strength") != "negative" or any(
                base_row.get("presence_targets", {}).values()
            ):
                raise ValueError(f"negative control conflicts with detection base: {sample_id}")
            control_rows_verified += 1
        admitted_matches = exact_sha_matches + source_record_identity_matches
        if admitted_matches < int(contract.get("strict_base_matches_min", len(source_rows))):
            raise ValueError(f"negative control overlap is too small: {name}:{admitted_matches}")
        if not_admitted_by_detection_base > int(
            contract.get("not_admitted_by_detection_base_max", 0)
        ):
            raise ValueError(
                "negative control has too many rows outside the canonical detection base: "
                f"{name}:{not_admitted_by_detection_base}"
            )
        receipt["role"] = "detection_negative_overlap_control_only"
        receipt["all_rows_accounted_without_append"] = True
        receipt["exact_sha_matches"] = exact_sha_matches
        receipt["source_record_identity_matches"] = source_record_identity_matches
        receipt["not_admitted_by_detection_base"] = not_admitted_by_detection_base
        receipt["detection_cleaning_evidence"] = cleaning
        control_receipts.append(receipt)

    overlay_receipts: list[dict[str, Any]] = []
    overlay_rows_merged = 0
    overlay_rows_added = 0
    overlay_group_splits: dict[str, set[str]] = defaultdict(set)
    for contract in registry["overlay_sources"]:
        name = str(contract["name"])
        if name not in overlay_roots:
            raise ValueError(f"missing overlay input: {name}")
        root = overlay_roots[name]
        source_rows, receipt = _validated_overlay_rows(root=root, contract=contract)
        overlay_receipts.append(receipt)
        artifact_root = root / str(contract.get("artifact_root_subdir") or "")
        field_aliases = contract.get("row_field_aliases", {})
        field_defaults = contract.get("row_field_defaults", {})
        if not isinstance(field_aliases, dict) or not isinstance(field_defaults, dict):
            raise ValueError(f"overlay row normalization contract invalid: {name}")
        artifact_s3_prefix = contract.get("artifact_s3_prefix")
        if artifact_s3_prefix is not None and (
            not isinstance(artifact_s3_prefix, str)
            or not artifact_s3_prefix.startswith("s3://")
            or artifact_s3_prefix.endswith("/")
        ):
            raise ValueError(f"overlay artifact S3 prefix invalid: {name}")
        for raw_source in source_rows:
            source = dict(raw_source)
            for target, origin in field_aliases.items():
                if target not in source and origin in source:
                    source[target] = source[origin]
            for target, value in field_defaults.items():
                source.setdefault(target, value)
            if artifact_s3_prefix is not None:
                for artifact_kind in ("image", "mask", "valid_mask"):
                    relative = source.get(f"{artifact_kind}_relpath")
                    if not relative:
                        continue
                    expected_uri = f"{artifact_s3_prefix}/{relative}"
                    existing_uri = source.get(f"{artifact_kind}_s3_uri")
                    if existing_uri is not None and existing_uri != expected_uri:
                        raise ValueError(
                            f"overlay {artifact_kind} S3 URI drift: {name}:"
                            f"{source.get('sample_id')}"
                        )
                    source[f"{artifact_kind}_s3_uri"] = expected_uri
            sample_id = str(source.get("sample_id") or "")
            digest = str(
                source.get("image_sha256") or source.get("source_image_sha256") or ""
            ).lower()
            split = str(source.get("split") or "")
            split_group = str(source.get("split_group") or "")
            if (
                source.get("sample_validation_status") != "strict_automated_validated"
                or source.get("strict_keep") is not True
                or source.get("training_eligible") is not True
                or source.get("source_id") != contract["source_id"]
                or source.get("source_revision") != contract["source_revision"]
                or source.get("reviews_admitted") is not False
                or (
                    source.get("source_family") is not None
                    and source.get("source_family") != contract.get("source_family")
                )
                or not _valid_sha(digest)
                or split not in SPLITS
                or not split_group
            ):
                raise ValueError(f"overlay row contract failed: {name}:{sample_id}")
            serialized_source = json.dumps(source, sort_keys=True).casefold()
            if any(marker in serialized_source for marker in WEAK_MARKERS):
                raise ValueError(f"weak overlay row rejected: {name}:{sample_id}")
            if any(marker in serialized_source for marker in BBOX_POINT_MARKERS):
                raise ValueError(f"bbox-derived overlay point rejected: {name}:{sample_id}")
            if (
                contract.get("redistribution_allowed") is not True
                or source.get("license") != contract.get("license")
                or source.get("redistribution_allowed", True) is not True
            ):
                raise ValueError(f"overlay rights contract failed: {name}:{sample_id}")
            exact_fields = {
                "annotation_strength": "allowed_annotation_strengths",
                "annotation_provenance": "allowed_annotation_provenances",
                "mask_quality": "allowed_mask_qualities",
                "validation_profile": "allowed_validation_profiles",
            }
            for field, policy_field in exact_fields.items():
                allowed = contract.get(policy_field)
                if not isinstance(allowed, list) or source.get(field) not in allowed:
                    raise ValueError(f"overlay {field} contract failed: {name}:{sample_id}")
            derivation = str(
                source.get("point_derivation") or source.get("mask_to_point_conversion") or ""
            )
            allowed_derivations = contract.get("allowed_point_derivations")
            if not isinstance(allowed_derivations, list) or derivation not in allowed_derivations:
                raise ValueError(f"overlay point derivation contract failed: {name}:{sample_id}")
            image_locator = _artifact_locator(source, artifact_root, kind="image", required=True)
            mask_locator = _artifact_locator(source, artifact_root, kind="mask", required=True)
            valid_locator = _artifact_locator(
                source, artifact_root, kind="valid_mask", required=False
            )
            update, _ = _profile_targets(source, str(contract["task_profile"]))
            source_identity = resolve_source_identity(
                source,
                binding_kind="overlay",
                binding_name=name,
                identities=identities,
            )
            overlay_reference = {
                "name": name,
                "sample_id": sample_id,
                "source_id": source["source_id"],
                "source_family": contract["source_family"],
                "source_revision": source["source_revision"],
                "license": source["license"],
                "redistribution_allowed": True,
                "consent_basis": {
                    "kind": "source_license",
                    "reference": (
                        f"{source['source_id']}@{source['source_revision']}:"
                        f"{source['license']}"
                    ),
                },
                "rights_components": contract.get(
                    "rights_components",
                    [{"asset": "image_and_labels", "license": source["license"]}],
                ),
                "manifest_sha256": receipt["manifest_sha256"],
                **source_identity,
            }
            if digest in rows_by_sha:
                row = rows_by_sha[digest]
                if row["split"] != split:
                    raise ValueError(
                        f"detection and overlay split conflict: {sample_id}:{row['split']}!={split}"
                    )
                if row["overlay_sources"]:
                    raise ValueError(f"multiple overlay sources require reconciliation: {digest}")
                if row["presence_targets"] != update["presence_targets"]:
                    raise ValueError(f"detection and overlay presence conflict: {sample_id}")
                if row["annotation_strength"] == "negative":
                    raise ValueError(f"negative detection row has positive overlay: {sample_id}")
                if row.get("mask_locator") and row["mask_locator"] != mask_locator:
                    raise ValueError(f"multiple incompatible masks for {digest}")
                row.update(update)
                row["mask_locator"] = mask_locator
                if valid_locator is not None:
                    row["valid_mask_locator"] = valid_locator
                row["overlay_sources"].append(overlay_reference)
                row["overlay_original_split"] = source["split"]
                row["overlay_original_split_group"] = source["split_group"]
                overlay_rows_merged += 1
            else:
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate composed sample id: {name}:{sample_id}")
                row = {
                    "schema_version": 2,
                    "sample_id": sample_id,
                    "image_sha256": digest,
                    "image_extension": PurePosixPath(
                        str(source.get("image_relpath") or "image.jpg")
                    ).suffix.casefold(),
                    "image_locator": image_locator,
                    "mask_locator": mask_locator,
                    "source_id": str(source["source_id"]),
                    "source_record_id": str(source.get("source_record_id") or sample_id),
                    "split": split,
                    "split_group": f"event:{source_identity['canonical_event_id']}",
                    "license": str(source["license"]),
                    "consent_basis": overlay_reference["consent_basis"],
                    "redistribution_allowed": True,
                    "sample_validation_status": "strict_automated_validated",
                    "validation_profile": "fireviewer_multitask_composition_v1",
                    "bbox_to_point": False,
                    "overlay_sources": [overlay_reference],
                    **source_identity,
                    **update,
                }
                if valid_locator is not None:
                    row["valid_mask_locator"] = valid_locator
                rows_by_sha[digest] = row
                sample_ids.add(sample_id)
                overlay_rows_added += 1
            overlay_group = f"event:{source_identity['canonical_event_id']}"
            overlay_group_splits[overlay_group].add(str(row["split"]))

    rows = sorted(rows_by_sha.values(), key=lambda row: (str(row["split"]), str(row["sample_id"])))
    for row in rows:
        row["campaign_id"] = registry["campaign_id"]
        row["composition_registry_sha256"] = registry_sha256
    validate_composed_row_identities(rows, registry=registry, identities=identities)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[str(row["split_group"])].add(str(row["split"]))
    for group, splits in overlay_group_splits.items():
        groups[group].update(splits)
    leaking_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    leaking_events = validate_canonical_event_splits(rows)
    task_counts = Counter()
    point_source_family_counts: Counter[str] = Counter()
    for row in rows:
        for task in ("segmentation", "point", "presence", "abstention"):
            if row[f"{task}_supervised"]:
                task_counts[f"{task}_supervised_rows"] += 1
        if row["point_supervised"] and row["anchor_points"]:
            task_counts["point_positive_rows"] += 1
            task_counts[f"point_positive_{row['split']}_rows"] += 1
            point_overlay_sources = {
                str(source["source_family_id"])
                for source in row.get("overlay_sources", [])
                if source.get("source_family_id")
            }
            if len(point_overlay_sources) != 1:
                raise ValueError(f"point row has ambiguous source family: {row['sample_id']}")
            point_source_family_counts.update(point_overlay_sources)
            point_kinds = {str(point.get("kind") or "") for point in row["anchor_points"]}
            for kind in ("fire_base", "smoke_column_base"):
                if kind in point_kinds:
                    task_counts[f"{kind}_rows"] += 1
        if row["annotation_strength"] == "negative":
            task_counts["explicit_negative_rows"] += 1
    hard = registry["hard_gates"]
    hard_errors: list[str] = []
    actual_detection_rows = detection_positive_rows + detection_negative_rows
    expected_checks = (
        ("detection_rows", actual_detection_rows, int(hard["detection_rows_exact"])),
        (
            "detection_positive_rows",
            detection_positive_rows,
            int(hard["detection_positive_rows_exact"]),
        ),
        (
            "detection_explicit_negative_rows",
            detection_negative_rows,
            int(hard["detection_explicit_negative_rows_exact"]),
        ),
    )
    for label, actual, expected in expected_checks:
        if actual != expected:
            hard_errors.append(f"{label}:{actual}!={expected}")
    if dict(sorted(detection_split_counts.items())) != detection["split_counts"]:
        hard_errors.append(f"detection_split_counts:{dict(sorted(detection_split_counts.items()))}")
    if leaking_groups:
        hard_errors.append(f"split_group_leakage:{len(leaking_groups)}")
    if leaking_events:
        hard_errors.append(f"canonical_event_leakage:{len(leaking_events)}")
    bbox_rows = sum(
        bool(row.get("bbox_to_point"))
        or (
            bool(row.get("point_supervised"))
            and any(
                marker in str(row.get("point_derivation", "")).casefold()
                for marker in BBOX_POINT_MARKERS
            )
        )
        for row in rows
    )
    if bbox_rows > int(hard["bbox_derived_point_rows_max"]):
        hard_errors.append(f"bbox_derived_point_rows:{bbox_rows}")
    weak_rows = sum(
        any(marker in str(row.get("annotation_strength", "")).casefold() for marker in WEAK_MARKERS)
        for row in rows
    )
    if weak_rows > int(hard["weak_or_teacher_generated_rows_max"]):
        hard_errors.append(f"weak_or_teacher_generated_rows:{weak_rows}")
    forbidden = tuple(registry["benchmark_boundary"]["forbidden_references"])
    benchmark_rows = sum(
        any(marker in json.dumps(row, sort_keys=True).casefold() for marker in forbidden)
        for row in rows
    )
    if benchmark_rows > int(hard["independent_benchmark_rows_max"]):
        hard_errors.append(f"independent_benchmark_rows:{benchmark_rows}")
    benchmark_hash_samples = sorted(
        str(row["sample_id"])
        for row in rows
        if benchmark_matches(raw_sha256=str(row["image_sha256"]), denylist=benchmark_denylist)[
            "raw_sha256"
        ]
    )
    if len(benchmark_hash_samples) > int(hard["benchmark_hash_matches_max"]):
        hard_errors.append(f"benchmark_hash_matches:{len(benchmark_hash_samples)}")
    unknown_rights_rows = sum(
        not str(row.get("license") or "")
        or not isinstance(row.get("consent_basis"), dict)
        or not str(row.get("consent_basis", {}).get("reference") or "")
        for row in rows
    )
    if unknown_rights_rows > int(hard["unknown_rights_rows_max"]):
        hard_errors.append(f"unknown_rights_rows:{unknown_rights_rows}")

    quality = registry["quality_gates"]
    quality_deficits: dict[str, dict[str, int | float]] = {}
    point_positive_rows = task_counts["point_positive_rows"]
    point_family_shares = sorted(
        (
            count / point_positive_rows
            for count in point_source_family_counts.values()
            if point_positive_rows
        ),
        reverse=True,
    )
    quality_checks = {
        "pilot_point_positive_rows": task_counts["point_positive_rows"],
        "professional_point_positive_rows": task_counts["point_positive_rows"],
        "professional_fire_base_rows": task_counts["fire_base_rows"],
        "professional_smoke_column_base_rows": task_counts["smoke_column_base_rows"],
        "professional_point_validation_rows": task_counts["point_positive_validation_rows"],
        "professional_point_test_rows": task_counts["point_positive_test_rows"],
        "point_source_families": len(point_source_family_counts),
        "presence_supervised_rows": task_counts["presence_supervised_rows"],
        "explicit_negative_rows": task_counts["explicit_negative_rows"],
    }
    for label, actual in quality_checks.items():
        expected = int(quality[f"{label}_min"])
        if actual < expected:
            quality_deficits[label] = {"actual": int(actual), "minimum": expected}
    share_checks = {
        "largest_point_source_share": point_family_shares[0] if point_family_shares else 0.0,
        "top_three_point_source_share": sum(point_family_shares[:3]),
    }
    for label, actual in share_checks.items():
        maximum = float(quality[f"{label}_max"])
        if actual > maximum:
            quality_deficits[label] = {"actual": actual, "maximum": maximum}
    integrity_passed = not hard_errors
    pilot_ready = integrity_passed and "pilot_point_positive_rows" not in quality_deficits
    professional_ready = integrity_passed and not quality_deficits
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "composition_candidate_manifest.jsonl"
    _write_jsonl(manifest, rows)
    report = {
        "schema_version": 2,
        "campaign_id": registry["campaign_id"],
        "composition_registry_sha256": registry_sha256,
        "source_identity_contract_sha256": identities.contract_sha256,
        "benchmark_denylist_sha256": benchmark_denylist.sha256,
        "detection_repository": detection["repository"],
        "detection_revision": detection["revision"],
        "detection_rows": actual_detection_rows,
        "detection_positive_rows": detection_positive_rows,
        "detection_explicit_negative_rows": detection_negative_rows,
        "detection_split_counts": dict(sorted(detection_split_counts.items())),
        "composition_rows": len(rows),
        "overlay_rows_merged_into_detection_base": overlay_rows_merged,
        "overlay_rows_added": overlay_rows_added,
        "control_rows_verified_without_append": control_rows_verified,
        "task_counts": dict(sorted(task_counts.items())),
        "point_source_families": sorted(point_source_family_counts),
        "point_source_family_counts": dict(sorted(point_source_family_counts.items())),
        "point_source_family_shares": {
            family: count / point_positive_rows
            for family, count in sorted(point_source_family_counts.items())
        }
        if point_positive_rows
        else {},
        "split_group_leakage": leaking_groups,
        "canonical_event_leakage": leaking_events,
        "bbox_derived_point_rows": bbox_rows,
        "weak_or_teacher_generated_rows": weak_rows,
        "independent_benchmark_rows": benchmark_rows,
        "benchmark_hash_matches": len(benchmark_hash_samples),
        "benchmark_hash_match_samples": benchmark_hash_samples[:100],
        "unknown_rights_rows": unknown_rights_rows,
        "hard_gate_errors": hard_errors,
        "hard_gates": hard,
        "integrity_gates_passed": integrity_passed,
        "quality_gates": quality,
        "quality_gate_deficits": quality_deficits,
        "pilot_corpus_ready": pilot_ready,
        "professional_corpus_ready": professional_ready,
        "training_ready": False,
        "training_blockers": [
            "composition_payload_hydration_and_global_image_rehash_pending",
            "gpu_finite_loss_smoke_pending",
        ]
        + ([] if professional_ready else ["professional_corpus_quality_gates_failed"]),
        "publication_allowed": False,
        "hf_replacement_allowed": False,
        "reviews_admitted": False,
        "detection_manifest_receipts": detection_receipts,
        "control_receipts": control_receipts,
        "overlay_receipts": overlay_receipts,
        "manifest": manifest.name,
        "manifest_sha256": _sha256(manifest),
    }
    integrity_receipt = output_dir / "composition_integrity_receipt.json"
    if integrity_passed:
        _write_json(
            integrity_receipt,
            {
                "schema_version": 2,
                "campaign_id": registry["campaign_id"],
                "composition_registry_sha256": registry_sha256,
                "source_identity_contract_sha256": identities.contract_sha256,
                "benchmark_denylist_sha256": benchmark_denylist.sha256,
                "manifest_sha256": report["manifest_sha256"],
                "composition_rows": len(rows),
                "detection_revision": detection["revision"],
                "detection_manifest_sha256": {
                    item["name"]: item["sha256"] for item in detection_receipts
                },
                "overlay_manifest_sha256": {
                    item["name"]: item["manifest_sha256"] for item in overlay_receipts
                },
                "control_manifest_sha256": {
                    item["name"]: item["manifest_sha256"] for item in control_receipts
                },
                "integrity_gates_passed": True,
                "pilot_corpus_ready": pilot_ready,
                "professional_corpus_ready": professional_ready,
                "publication_allowed": False,
            },
        )
        report["composition_integrity_receipt"] = integrity_receipt.name
        report["composition_integrity_receipt_sha256"] = _sha256(integrity_receipt)
    else:
        report["composition_integrity_receipt"] = None
        report["composition_integrity_receipt_sha256"] = None
    _write_json(output_dir / "composition_report.json", report)
    return report


def _download_detection_manifests(
    registry: dict[str, Any], work_dir: Path
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    detection = registry["detection_base"]
    repository, revision = str(detection["repository"]), str(detection["revision"])
    output: dict[str, Path] = {}
    receipts: list[dict[str, Any]] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for name, relative in sorted(detection["manifest_paths"].items()):
        encoded = quote(str(relative), safe="/")
        url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{encoded}"
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
            raise ValueError("detection manifest URL escaped the pinned Hugging Face repository")
        destination = work_dir / f"{name}.jsonl"
        partial = destination.with_suffix(".partial")
        request = urllib.request.Request(  # noqa: S310 - URL is pinned and checked above.
            url,
            headers={"User-Agent": "FireViewer-DINOv3-Composition/1.0"},
        )
        digest = hashlib.sha256()
        downloaded = 0
        try:
            with (
                urllib.request.urlopen(request, timeout=180) as response,  # noqa: S310
                partial.open("wb") as target,
            ):
                resolved = urlsplit(response.geturl())
                if resolved.scheme != "https":
                    raise ValueError("detection manifest redirected away from HTTPS")
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > 256 * 1024**2:
                        raise ValueError(f"detection manifest too large: {name}")
                    digest.update(chunk)
                    target.write(chunk)
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        output[name] = destination
        receipts.append(
            {
                "name": name,
                "repository": repository,
                "repository_path": relative,
                "revision": revision,
                "bytes": downloaded,
                "sha256": digest.hexdigest(),
            }
        )
    return output, receipts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--benchmark-denylist", type=Path, default=DEFAULT_BENCHMARK_DENYLIST)
    parser.add_argument("--overlay", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--control", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    overlays: dict[str, Path] = {}
    for value in args.overlay:
        if "=" not in value:
            raise ValueError(f"invalid overlay argument: {value}")
        name, path = value.split("=", 1)
        if not name or name in overlays:
            raise ValueError(f"invalid duplicate overlay name: {name}")
        overlays[name] = Path(path)
    controls: dict[str, Path] = {}
    for value in args.control:
        if "=" not in value:
            raise ValueError(f"invalid control argument: {value}")
        name, path = value.split("=", 1)
        if not name or name in controls:
            raise ValueError(f"invalid duplicate control name: {name}")
        controls[name] = Path(path)
    manifests, receipts = _download_detection_manifests(
        registry, args.work_dir / "detection-manifests"
    )
    report = compose_multitask(
        registry_path=args.registry,
        detection_manifests=manifests,
        overlay_roots=overlays,
        output_dir=args.output_dir,
        benchmark_denylist_path=args.benchmark_denylist,
        detection_receipts=receipts,
        control_roots=controls,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
