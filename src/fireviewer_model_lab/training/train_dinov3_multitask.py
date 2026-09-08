"""Fail-closed preparation gate for full DINOv3 FireViewer fine-tuning.

Every row declares which of segmentation, point, presence and abstention is
actually supervised. Missing labels are never interpreted as negatives. This
allows the immutable detection corpus to supervise presence while strong
pointing overlays supervise localization without inventing labels from boxes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from fireviewer_model_lab.training.challenger_training import ALL_SPLITS, APPROVED_SAMPLE_STATUSES
from fireviewer_model_lab.training.dinov3_corpus_identity import (
    BenchmarkDenylist,
    benchmark_matches,
    decoded_pixel_sha256,
    load_benchmark_denylist,
    phash64_imagehash_v1,
    validate_canonical_event_splits,
    validate_composed_row_identities,
    validate_source_identity_contract,
)

TRAINABLE_SAMPLE_STATUSES = APPROVED_SAMPLE_STATUSES | frozenset(
    {
        "strict_automated_validated",
    }
)
SUPERVISION_FIELDS = (
    "segmentation_supervised",
    "point_supervised",
    "presence_supervised",
    "abstention_supervised",
)
PRESENCE_LABELS = ("flame_visible", "smoke_visible")
ALLOWED_ANNOTATION_STRENGTHS = frozenset(
    {"negative", "strong", "sensor_derived_strong", "strong_presence_only"}
)
ALLOWED_POINT_KINDS = frozenset({"fire_base", "smoke_column_base"})
FORBIDDEN_BENCHMARK_MARKERS = (
    "benchdata",
    "fireviewer_bench",
    "independent-benchmark",
    "benchmark-independent",
)
REQUIRED_QUALITY_GATES = frozenset(
    {
        "pilot_point_positive_rows_min",
        "professional_point_positive_rows_min",
        "professional_fire_base_rows_min",
        "professional_smoke_column_base_rows_min",
        "professional_point_validation_rows_min",
        "professional_point_test_rows_min",
        "point_source_families_min",
        "largest_point_source_share_max",
        "top_three_point_source_share_max",
        "presence_supervised_rows_min",
        "explicit_negative_rows_min",
    }
)
REQUIRED_HARD_GATES = frozenset(
    {
        "detection_rows_exact",
        "detection_positive_rows_exact",
        "detection_explicit_negative_rows_exact",
        "one_logical_row_per_image_sha256",
        "bbox_derived_point_rows_max",
        "weak_or_teacher_generated_rows_max",
        "payload_hash_mismatches_max",
        "split_group_leaks_max",
        "perceptual_near_duplicate_pairs_max",
        "independent_benchmark_rows_max",
        "benchmark_hash_matches_max",
        "unknown_rights_rows_max",
    }
)

DEFAULT_POINTING_ROOT = Path(
    os.environ.get(
        "FIREVIEWER_POINTING_DATASET_ROOT",
        "data/datasets/pointing-rebuild-required",
    )
)
DEFAULT_OUTPUT = Path("data/training/dinov3-multitask-v1")
DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DEFAULT_MULTITASK_MANIFEST = Path("data/training/dinov3-boreal-multitask-v1/manifest.jsonl")
DEFAULT_COMPOSITION_REGISTRY = Path("training/registries/dinov3-multitask-composition-v1.json")
DEFAULT_BENCHMARK_DENYLIST = Path(
    "training/registries/dinov3-independent-benchmark-denylist-v1.json"
)
DEFAULT_DATA_ROOT = Path(
    "data/training/wildfire-smoke-segmentation-v1/"
    "wildfire-smoke-segmentation-v1/sources/boreal-forest-fire-segmentation-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _read_jsonl_path(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _registry_contract_errors(registry: Any) -> list[str]:
    if not isinstance(registry, dict) or registry.get("schema_version") != 2:
        return ["composition_registry_contract_invalid"]
    errors: list[str] = []
    required_top = {
        "campaign_id",
        "detection_base",
        "control_sources",
        "overlay_sources",
        "hard_gates",
        "quality_gates",
        "benchmark_boundary",
        "source_identity_contract",
    }
    if not required_top.issubset(registry):
        errors.append("composition_registry_sections_missing")
        return errors
    detection = registry.get("detection_base")
    required_detection = {
        "repository",
        "revision",
        "validation_profile",
        "validation_run_id",
        "rows",
        "positive_rows",
        "explicit_negative_rows",
        "split_counts",
        "manifest_paths",
    }
    if (
        not isinstance(detection, dict)
        or not required_detection.issubset(detection)
        or len(str(detection.get("revision") or "")) != 40
        or not isinstance(detection.get("split_counts"), dict)
        or set(detection["split_counts"]) != ALL_SPLITS
        or not isinstance(detection.get("manifest_paths"), dict)
        or not detection["manifest_paths"]
    ):
        errors.append("composition_registry_detection_contract_invalid")
    hard = registry.get("hard_gates")
    if not isinstance(hard, dict) or not REQUIRED_HARD_GATES.issubset(hard):
        errors.append("composition_registry_hard_gates_invalid")
    quality = registry.get("quality_gates")
    if not isinstance(quality, dict) or not REQUIRED_QUALITY_GATES.issubset(quality):
        errors.append("composition_registry_quality_gates_invalid")
    overlays = registry.get("overlay_sources")
    controls = registry.get("control_sources")
    if not isinstance(overlays, list) or not overlays:
        errors.append("composition_registry_overlay_contracts_invalid")
    else:
        required_overlay = {
            "name",
            "source_id",
            "source_family",
            "source_family_id",
            "source_revision",
            "manifest_sha256",
            "report_sha256",
            "strict_rows_min",
            "strict_rows_max",
            "license",
            "redistribution_allowed",
            "task_profile",
        }
        names: set[str] = set()
        for contract in overlays:
            name = str(contract.get("name") or "") if isinstance(contract, dict) else ""
            if (
                not isinstance(contract, dict)
                or not required_overlay.issubset(contract)
                or not name
                or name in names
                or not _valid_sha256(contract.get("manifest_sha256"))
                or not _valid_sha256(contract.get("report_sha256"))
                or contract.get("redistribution_allowed") is not True
            ):
                errors.append("composition_registry_overlay_contracts_invalid")
                break
            names.add(name)
    if not isinstance(controls, list) or not controls:
        errors.append("composition_registry_control_contracts_invalid")
    else:
        required_control = {
            "name",
            "source_id",
            "source_revision",
            "manifest_sha256",
            "report_sha256",
            "strict_rows_min",
            "strict_rows_max",
            "role",
        }
        names = set()
        for contract in controls:
            name = str(contract.get("name") or "") if isinstance(contract, dict) else ""
            if (
                not isinstance(contract, dict)
                or not required_control.issubset(contract)
                or not name
                or name in names
                or not _valid_sha256(contract.get("manifest_sha256"))
                or not _valid_sha256(contract.get("report_sha256"))
            ):
                errors.append("composition_registry_control_contracts_invalid")
                break
            names.add(name)
    boundary = registry.get("benchmark_boundary")
    if (
        not isinstance(boundary, dict)
        or boundary.get("independent_benchmark_must_remain_separate") is not True
        or not isinstance(boundary.get("forbidden_references"), list)
        or not boundary["forbidden_references"]
        or not _valid_sha256(boundary.get("denylist_sha256"))
        or boundary.get("denylist_schema_version") != 1
        or not _valid_sha256(boundary.get("provenance_guard_sha256"))
        or not isinstance(boundary.get("provenance_guard_rows"), int)
        or int(boundary.get("provenance_guard_rows", 0)) <= 0
        or not isinstance(boundary.get("denylist_entry_counts"), dict)
        or boundary.get("decoded_pixel_hash_algorithm") != "rgb8-wh-be32-sha256-v1"
        or boundary.get("perceptual_hash_algorithm") != "imagehash-phash64-hash-size-8-v1"
        or not isinstance(boundary.get("phash_hamming_distance_max"), int)
    ):
        errors.append("composition_registry_benchmark_boundary_invalid")
    try:
        validate_source_identity_contract(registry)
    except (KeyError, TypeError, ValueError):
        errors.append("composition_registry_source_identity_contract_invalid")
    return sorted(set(errors))


def _composition_accounting(
    rows: list[dict[str, Any]],
    registry: dict[str, Any],
    benchmark_denylist: BenchmarkDenylist,
) -> dict[str, Any]:
    """Recompute corpus gates from rows instead of trusting readiness booleans."""

    errors: list[str] = []
    task_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    detection_split_counts: Counter[str] = Counter()
    detection_positive_rows = 0
    detection_negative_rows = 0
    detection = registry["detection_base"]
    identities = validate_source_identity_contract(registry)
    try:
        validate_composed_row_identities(rows, registry=registry, identities=identities)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"composition_source_identity_invalid:{exc}")
    try:
        event_leaks = validate_canonical_event_splits(rows)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"composition_canonical_event_invalid:{exc}")
        event_leaks = []
    if event_leaks:
        errors.append(f"composition_canonical_event_leakage:{len(event_leaks)}")
    benchmark_raw_hash_matches = sum(
        benchmark_matches(
            raw_sha256=str(row.get("image_sha256") or ""),
            denylist=benchmark_denylist,
        )["raw_sha256"]
        for row in rows
    )
    if benchmark_raw_hash_matches > int(registry["hard_gates"]["benchmark_hash_matches_max"]):
        errors.append(f"composition_benchmark_hash_matches:{benchmark_raw_hash_matches}")
    overlay_contracts = {
        str(contract["name"]): contract for contract in registry["overlay_sources"]
    }
    manifest_paths = {str(value) for value in detection["manifest_paths"].values()}
    for line_number, row in enumerate(rows, 1):
        points = row.get("anchor_points")
        point_list = points if isinstance(points, list) else []
        for task in ("segmentation", "point", "presence", "abstention"):
            if row.get(f"{task}_supervised") is True:
                task_counts[f"{task}_supervised_rows"] += 1
        if row.get("annotation_strength") == "negative":
            task_counts["explicit_negative_rows"] += 1

        overlay_refs = row.get("overlay_sources")
        if not isinstance(overlay_refs, list):
            errors.append(f"composition_overlay_references_invalid:{line_number}")
            overlay_refs = []
        row_families: set[str] = set()
        for reference in overlay_refs:
            if not isinstance(reference, dict):
                errors.append(f"composition_overlay_reference_invalid:{line_number}")
                continue
            name = str(reference.get("name") or "")
            contract = overlay_contracts.get(name)
            if contract is None:
                errors.append(f"composition_overlay_contract_unknown:{line_number}:{name}")
                continue
            expected = {
                "source_id": contract["source_id"],
                "source_family": contract["source_family"],
                "source_family_id": contract["source_family_id"],
                "source_revision": contract["source_revision"],
                "license": contract["license"],
                "manifest_sha256": contract["manifest_sha256"],
            }
            if any(reference.get(key) != value for key, value in expected.items()):
                errors.append(f"composition_overlay_contract_drift:{line_number}:{name}")
            row_families.add(str(reference.get("source_family_id") or ""))

        point_positive = row.get("point_supervised") is True and bool(point_list)
        if point_positive:
            task_counts["point_positive_rows"] += 1
            task_counts[f"point_positive_{row.get('split')}_rows"] += 1
            if len(row_families) != 1:
                errors.append(f"composition_point_family_ambiguous:{line_number}")
            else:
                family_counts.update(row_families)
            kinds = {
                str(point.get("kind") or "") for point in point_list if isinstance(point, dict)
            }
            for kind in ALLOWED_POINT_KINDS:
                if kind in kinds:
                    task_counts[f"{kind}_rows"] += 1
        if (
            row.get("segmentation_supervised") is True
            and row.get("annotation_strength") != "negative"
            and not overlay_refs
        ):
            errors.append(f"positive_segmentation_without_overlay:{line_number}")

        locator = row.get("image_locator")
        if not isinstance(locator, dict):
            errors.append(f"composition_image_locator_missing:{line_number}")
            continue
        locator_kind = locator.get("kind")
        if locator_kind == "hf_dataset_row":
            expected_locator = {
                "repository": detection["repository"],
                "revision": detection["revision"],
                "split": row.get("split"),
                "sample_id": row.get("sample_id"),
                "sha256": row.get("image_sha256"),
            }
            if (
                any(locator.get(key) != value for key, value in expected_locator.items())
                or row.get("source_manifest_path") not in manifest_paths
                or row.get("detection_validation_profile") != detection["validation_profile"]
                or row.get("detection_validation_run_id") != detection["validation_run_id"]
            ):
                errors.append(f"composition_detection_locator_drift:{line_number}")
            detection_split_counts[str(row.get("split"))] += 1
            if row.get("annotation_strength") == "negative":
                detection_negative_rows += 1
            else:
                detection_positive_rows += 1
        elif locator_kind == "s3_object":
            if len(overlay_refs) != 1:
                errors.append(f"composition_s3_row_without_single_overlay:{line_number}")
        else:
            errors.append(f"composition_image_locator_invalid:{line_number}")

    hard = registry["hard_gates"]
    detection_rows = detection_positive_rows + detection_negative_rows
    hard_expectations = {
        "detection_rows": (detection_rows, int(hard["detection_rows_exact"])),
        "detection_positive_rows": (
            detection_positive_rows,
            int(hard["detection_positive_rows_exact"]),
        ),
        "detection_explicit_negative_rows": (
            detection_negative_rows,
            int(hard["detection_explicit_negative_rows_exact"]),
        ),
    }
    for label, (actual, expected) in hard_expectations.items():
        if actual != expected:
            errors.append(f"composition_{label}_drift:{actual}!={expected}")
    expected_splits = {str(key): int(value) for key, value in detection["split_counts"].items()}
    actual_splits = {split: detection_split_counts[split] for split in sorted(ALL_SPLITS)}
    if actual_splits != dict(sorted(expected_splits.items())):
        errors.append("composition_detection_split_counts_drift")

    point_rows = task_counts["point_positive_rows"]
    family_shares = sorted(
        (count / point_rows for count in family_counts.values() if point_rows),
        reverse=True,
    )
    quality_values: dict[str, int | float] = {
        "pilot_point_positive_rows": point_rows,
        "professional_point_positive_rows": point_rows,
        "professional_fire_base_rows": task_counts["fire_base_rows"],
        "professional_smoke_column_base_rows": task_counts["smoke_column_base_rows"],
        "professional_point_validation_rows": task_counts["point_positive_validation_rows"],
        "professional_point_test_rows": task_counts["point_positive_test_rows"],
        "point_source_families": len(family_counts),
        "presence_supervised_rows": task_counts["presence_supervised_rows"],
        "explicit_negative_rows": task_counts["explicit_negative_rows"],
        "largest_point_source_share": family_shares[0] if family_shares else 0.0,
        "top_three_point_source_share": sum(family_shares[:3]),
    }
    quality = registry["quality_gates"]
    deficits: dict[str, dict[str, int | float]] = {}
    for label, actual in quality_values.items():
        minimum_key, maximum_key = f"{label}_min", f"{label}_max"
        if minimum_key in quality:
            minimum = int(quality[minimum_key])
            if actual < minimum:
                deficits[label] = {"actual": int(actual), "minimum": minimum}
        elif maximum_key in quality:
            maximum = float(quality[maximum_key])
            if actual > maximum:
                deficits[label] = {"actual": float(actual), "maximum": maximum}
        else:
            errors.append(f"composition_quality_gate_missing:{label}")
    pilot_ready = not errors and "pilot_point_positive_rows" not in deficits
    professional_ready = not errors and not deficits
    return {
        "errors": errors,
        "task_counts": dict(sorted(task_counts.items())),
        "point_source_family_counts": dict(sorted(family_counts.items())),
        "detection_rows": detection_rows,
        "detection_positive_rows": detection_positive_rows,
        "detection_explicit_negative_rows": detection_negative_rows,
        "detection_split_counts": actual_splits,
        "quality_gate_deficits": deficits,
        "pilot_corpus_ready": pilot_ready,
        "professional_corpus_ready": professional_ready,
        "canonical_event_leakage": event_leaks,
        "benchmark_raw_hash_matches": benchmark_raw_hash_matches,
    }


def _load_contract_json(path: Path | None, label: str, errors: list[str]) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        errors.append(f"{label}_missing")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"{label}_invalid:{type(exc).__name__}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label}_not_object")
        return None
    return value


def _validate_source_receipts(
    composition_report: dict[str, Any], registry: dict[str, Any], errors: list[str]
) -> None:
    overlay_receipts = composition_report.get("overlay_receipts")
    control_receipts = composition_report.get("control_receipts")
    detection_receipts = composition_report.get("detection_manifest_receipts")
    if not isinstance(overlay_receipts, list):
        errors.append("composition_overlay_receipts_invalid")
    else:
        actual = {str(item.get("name") or ""): item for item in overlay_receipts}
        expected = {str(item["name"]): item for item in registry["overlay_sources"]}
        if set(actual) != set(expected):
            errors.append("composition_overlay_receipt_set_drift")
        for name in set(actual) & set(expected):
            receipt, contract = actual[name], expected[name]
            if (
                receipt.get("manifest_sha256") != contract["manifest_sha256"]
                or receipt.get("report_sha256") != contract["report_sha256"]
                or receipt.get("source_gate_passed") is not True
                or not int(contract["strict_rows_min"])
                <= int(receipt.get("rows", -1))
                <= int(contract["strict_rows_max"])
            ):
                errors.append(f"composition_overlay_receipt_drift:{name}")
    if not isinstance(control_receipts, list):
        errors.append("composition_control_receipts_invalid")
    else:
        actual = {str(item.get("name") or ""): item for item in control_receipts}
        expected = {str(item["name"]): item for item in registry["control_sources"]}
        if set(actual) != set(expected):
            errors.append("composition_control_receipt_set_drift")
        for name in set(actual) & set(expected):
            receipt, contract = actual[name], expected[name]
            if (
                receipt.get("manifest_sha256") != contract["manifest_sha256"]
                or receipt.get("report_sha256") != contract["report_sha256"]
                or receipt.get("source_gate_passed") is not True
                or receipt.get("all_rows_accounted_without_append") is not True
                or not int(contract["strict_rows_min"])
                <= int(receipt.get("rows", -1))
                <= int(contract["strict_rows_max"])
            ):
                errors.append(f"composition_control_receipt_drift:{name}")
    detection = registry["detection_base"]
    if not isinstance(detection_receipts, list):
        errors.append("composition_detection_receipts_invalid")
    else:
        actual = {str(item.get("name") or ""): item for item in detection_receipts}
        expected = detection["manifest_paths"]
        if set(actual) != set(expected):
            errors.append("composition_detection_receipt_set_drift")
        for name in set(actual) & set(expected):
            receipt = actual[name]
            if (
                receipt.get("repository") != detection["repository"]
                or receipt.get("revision") != detection["revision"]
                or receipt.get("repository_path") != expected[name]
                or not _valid_sha256(receipt.get("sha256"))
                or int(receipt.get("bytes", 0)) <= 0
            ):
                errors.append(f"composition_detection_receipt_drift:{name}")


def _validate_composition_chain(
    *,
    hydrated_rows: list[dict[str, Any]],
    hydrated_manifest: Path | None,
    registry: dict[str, Any],
    registry_sha256: str,
    benchmark_denylist: BenchmarkDenylist,
    campaign_id: str,
    composition_manifest: Path | None,
    composition_report_path: Path | None,
    composition_integrity_receipt_path: Path | None,
    hydration_report_path: Path | None,
    hydration_integrity_receipt_path: Path | None,
    errors: list[str],
) -> dict[str, Any]:
    identities = validate_source_identity_contract(registry)
    accounting = _composition_accounting(hydrated_rows, registry, benchmark_denylist)
    errors.extend(accounting["errors"])
    source_rows: list[dict[str, Any]] = []
    if composition_manifest is None or not composition_manifest.is_file():
        errors.append("composition_manifest_missing")
    else:
        try:
            source_rows = _read_jsonl_path(composition_manifest)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            errors.append(f"composition_manifest_invalid:{type(exc).__name__}")
    composition_report = _load_contract_json(composition_report_path, "composition_report", errors)
    composition_integrity = _load_contract_json(
        composition_integrity_receipt_path,
        "composition_integrity_receipt",
        errors,
    )
    hydration_report = _load_contract_json(
        hydration_report_path, "composition_hydration_report", errors
    )
    hydration_integrity = _load_contract_json(
        hydration_integrity_receipt_path,
        "hydration_integrity_receipt",
        errors,
    )
    if source_rows:
        source_by_id = {str(row.get("sample_id") or ""): row for row in source_rows}
        hydrated_by_id = {str(row.get("sample_id") or ""): row for row in hydrated_rows}
        if (
            len(source_by_id) != len(source_rows)
            or len(hydrated_by_id) != len(hydrated_rows)
            or set(source_by_id) != set(hydrated_by_id)
        ):
            errors.append("composition_hydrated_identity_set_drift")
        else:
            for sample_id, source in source_by_id.items():
                hydrated = hydrated_by_id[sample_id]
                if any(hydrated.get(key) != value for key, value in source.items()):
                    errors.append(f"composition_hydrated_row_drift:{sample_id}")
                    break
    composition_manifest_sha = (
        _sha256(composition_manifest)
        if composition_manifest is not None and composition_manifest.is_file()
        else None
    )
    composition_report_sha = (
        _sha256(composition_report_path)
        if composition_report_path is not None and composition_report_path.is_file()
        else None
    )
    composition_integrity_sha = (
        _sha256(composition_integrity_receipt_path)
        if composition_integrity_receipt_path is not None
        and composition_integrity_receipt_path.is_file()
        else None
    )
    hydration_integrity_sha = (
        _sha256(hydration_integrity_receipt_path)
        if hydration_integrity_receipt_path is not None
        and hydration_integrity_receipt_path.is_file()
        else None
    )
    hydrated_manifest_sha = (
        _sha256(hydrated_manifest)
        if hydrated_manifest is not None and hydrated_manifest.is_file()
        else None
    )
    detection = registry["detection_base"]
    if composition_report is not None:
        expected = {
            "schema_version": 2,
            "campaign_id": campaign_id,
            "composition_registry_sha256": registry_sha256,
            "source_identity_contract_sha256": identities.contract_sha256,
            "benchmark_denylist_sha256": benchmark_denylist.sha256,
            "detection_repository": detection["repository"],
            "detection_revision": detection["revision"],
            "detection_rows": accounting["detection_rows"],
            "detection_positive_rows": accounting["detection_positive_rows"],
            "detection_explicit_negative_rows": accounting["detection_explicit_negative_rows"],
            "detection_split_counts": accounting["detection_split_counts"],
            "composition_rows": len(hydrated_rows),
            "task_counts": accounting["task_counts"],
            "point_source_family_counts": accounting["point_source_family_counts"],
            "hard_gates": registry["hard_gates"],
            "quality_gates": registry["quality_gates"],
            "hard_gate_errors": [],
            "integrity_gates_passed": True,
            "quality_gate_deficits": accounting["quality_gate_deficits"],
            "pilot_corpus_ready": accounting["pilot_corpus_ready"],
            "professional_corpus_ready": accounting["professional_corpus_ready"],
            "publication_allowed": False,
            "hf_replacement_allowed": False,
            "reviews_admitted": False,
            "manifest_sha256": composition_manifest_sha,
            "composition_integrity_receipt_sha256": composition_integrity_sha,
            "canonical_event_leakage": [],
            "benchmark_hash_matches": 0,
        }
        mismatches = [
            key for key, value in expected.items() if composition_report.get(key) != value
        ]
        if composition_report.get("split_group_leakage") != []:
            mismatches.append("split_group_leakage")
        if composition_report.get("bbox_derived_point_rows") != 0:
            mismatches.append("bbox_derived_point_rows")
        if composition_report.get("weak_or_teacher_generated_rows") != 0:
            mismatches.append("weak_or_teacher_generated_rows")
        if composition_report.get("independent_benchmark_rows") != 0:
            mismatches.append("independent_benchmark_rows")
        if mismatches:
            errors.append("composition_report_contract_failed:" + ",".join(sorted(set(mismatches))))
        _validate_source_receipts(composition_report, registry, errors)
    if composition_integrity is not None:
        report_receipts = composition_report or {}
        expected = {
            "schema_version": 2,
            "campaign_id": campaign_id,
            "composition_registry_sha256": registry_sha256,
            "source_identity_contract_sha256": identities.contract_sha256,
            "benchmark_denylist_sha256": benchmark_denylist.sha256,
            "manifest_sha256": composition_manifest_sha,
            "composition_rows": len(hydrated_rows),
            "detection_revision": detection["revision"],
            "integrity_gates_passed": True,
            "pilot_corpus_ready": accounting["pilot_corpus_ready"],
            "professional_corpus_ready": accounting["professional_corpus_ready"],
            "publication_allowed": False,
        }
        mismatches = [
            key for key, value in expected.items() if composition_integrity.get(key) != value
        ]
        detection_hashes = {
            str(item.get("name") or ""): item.get("sha256")
            for item in report_receipts.get("detection_manifest_receipts", [])
            if isinstance(item, dict)
        }
        overlay_hashes = {
            str(item.get("name") or ""): item.get("manifest_sha256")
            for item in report_receipts.get("overlay_receipts", [])
            if isinstance(item, dict)
        }
        control_hashes = {
            str(item.get("name") or ""): item.get("manifest_sha256")
            for item in report_receipts.get("control_receipts", [])
            if isinstance(item, dict)
        }
        if composition_integrity.get("detection_manifest_sha256") != detection_hashes:
            mismatches.append("detection_manifest_sha256")
        if composition_integrity.get("overlay_manifest_sha256") != overlay_hashes:
            mismatches.append("overlay_manifest_sha256")
        if composition_integrity.get("control_manifest_sha256") != control_hashes:
            mismatches.append("control_manifest_sha256")
        if mismatches:
            errors.append(
                "composition_integrity_receipt_contract_failed:" + ",".join(sorted(set(mismatches)))
            )
    if hydration_report is not None:
        near_duplicates = hydration_report.get("perceptual_near_duplicates")
        expected = {
            "schema_version": 2,
            "composition_manifest_sha256": composition_manifest_sha,
            "composition_report_sha256": composition_report_sha,
            "composition_integrity_receipt_sha256": composition_integrity_sha,
            "campaign_id": campaign_id,
            "composition_registry_sha256": registry_sha256,
            "source_identity_contract_sha256": identities.contract_sha256,
            "benchmark_denylist_sha256": benchmark_denylist.sha256,
            "rows": len(hydrated_rows),
            "verified_image_rows": len(hydrated_rows),
            "manifest_sha256": hydrated_manifest_sha,
            "hydration_integrity_passed": True,
            "quality_gate_deficits": accounting["quality_gate_deficits"],
            "pilot_corpus_ready": accounting["pilot_corpus_ready"],
            "professional_corpus_ready": accounting["professional_corpus_ready"],
            "ready_for_gpu_finite_loss_smoke": accounting["pilot_corpus_ready"],
            "ready_for_full_training": accounting["professional_corpus_ready"],
            "publication_allowed": False,
            "hard_gate_errors": [],
            "validation_errors": [],
            "decoded_pixel_duplicate_groups": 0,
            "split_group_leakage": [],
            "canonical_event_leakage": [],
            "benchmark_raw_hash_matches": 0,
            "benchmark_decoded_hash_matches": 0,
            "benchmark_phash_matches": 0,
            "hydration_integrity_receipt_sha256": hydration_integrity_sha,
        }
        mismatches = [key for key, value in expected.items() if hydration_report.get(key) != value]
        if (
            not isinstance(near_duplicates, dict)
            or near_duplicates.get("cross_split_pairs") != 0
            or near_duplicates.get("within_split_pairs") != 0
        ):
            mismatches.append("perceptual_near_duplicates")
        if mismatches:
            errors.append(
                "composition_hydration_report_gate_failed:" + ",".join(sorted(set(mismatches)))
            )
    if hydration_integrity is not None:
        expected = {
            "schema_version": 2,
            "manifest_sha256": hydrated_manifest_sha,
            "rows": len(hydrated_rows),
            "campaign_id": campaign_id,
            "composition_registry_sha256": registry_sha256,
            "source_identity_contract_sha256": identities.contract_sha256,
            "benchmark_denylist_sha256": benchmark_denylist.sha256,
            "composition_manifest_sha256": composition_manifest_sha,
            "composition_report_sha256": composition_report_sha,
            "composition_integrity_receipt_sha256": composition_integrity_sha,
            "hydration_integrity_passed": True,
            "pilot_corpus_ready": accounting["pilot_corpus_ready"],
            "professional_corpus_ready": accounting["professional_corpus_ready"],
            "quality_gate_deficits": accounting["quality_gate_deficits"],
            "training_ready": False,
            "publication_allowed": False,
        }
        mismatches = [
            key for key, value in expected.items() if hydration_integrity.get(key) != value
        ]
        if mismatches:
            errors.append(
                "hydration_integrity_receipt_contract_failed:" + ",".join(sorted(set(mismatches)))
            )
    return accounting


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_pointing_manifest(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = root / "manifest.jsonl"
    if not path.is_file():
        return [], [f"missing:{path}"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid_json:{line_number}:{exc.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"row_not_object:{line_number}")
            continue
        rows.append(row)
    return rows, errors


def _validate_smoke_report(
    path: Path | None,
    *,
    manifest: Path,
    training_contract: dict[str, Any],
    smoke_steps: int,
) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise ValueError("a successful GPU smoke report is required before full training")
    report = json.loads(path.read_text(encoding="utf-8"))
    hyperparameters = training_contract["hyperparameters"]
    contract_encoded = json.dumps(
        training_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_encoded).hexdigest()
    losses = report.get("loss_history")
    finite_losses = isinstance(losses, list) and bool(losses)
    required_loss_keys = {
        "loss",
        "segmentation_loss",
        "point_loss",
        "abstention_loss",
        "presence_loss",
    }
    if finite_losses:
        finite_losses = all(
            isinstance(step, dict)
            and set(step) == required_loss_keys
            and all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0
                for value in step.values()
            )
            for step in losses
        )
    expected = {
        "schema_version": 3,
        "model_schema_version": 4,
        "presence_labels": list(PRESENCE_LABELS),
        "passed": True,
        "all_gradients_finite": True,
        "model_id": training_contract["model_id"],
        "model_revision": training_contract["model_revision"],
        "manifest_sha256": _sha256(manifest),
        "training_contract": training_contract,
        "training_contract_sha256": contract_sha256,
        "device": "cuda",
        "image_size": hyperparameters["image_size"],
        "batch_size": hyperparameters["batch_size"],
        "smoke_steps": smoke_steps,
        "initialization": training_contract["initialization"],
        "initial_weights_sha256": training_contract["initial_weights_sha256"],
        "backbone_config_sha256": training_contract["backbone_config_sha256"],
        "optimizer_steps": math.ceil(smoke_steps / hyperparameters["gradient_accumulation_steps"]),
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    sample_ids = report.get("sample_ids")
    observed_roles = report.get("observed_roles")
    observed_sources = report.get("observed_sources")
    sampling = report.get("sampling")
    expected_sample_count = hyperparameters["batch_size"] * smoke_steps
    if not isinstance(sample_ids, list) or len(sample_ids) != expected_sample_count:
        mismatches.append("sample_ids")
    else:
        rows = _read_jsonl_path(manifest)
        rows_by_id = {str(row.get("sample_id") or ""): row for row in rows}
        sampled_rows = [rows_by_id.get(str(sample_id)) for sample_id in sample_ids]
        if any(row is None or row.get("split") != "train" for row in sampled_rows):
            mismatches.append("sample_ids_manifest_membership")
        else:
            role_counter: Counter[str] = Counter()
            source_counter: Counter[str] = Counter()
            for row in sampled_rows:
                assert row is not None
                if row.get("annotation_strength") in {"negative", "temporal_negative"}:
                    role = "negative"
                elif row.get("visual_abstention_reason") is not None:
                    role = "abstention"
                elif (
                    row.get("presence_supervised") is True and row.get("point_supervised") is False
                ):
                    role = "presence"
                else:
                    role = "positive"
                role_counter[role] += 1
                source_counter[str(row.get("source_id") or "unknown")] += 1
            if observed_roles != dict(sorted(role_counter.items())) or any(
                role_counter[role] <= 0
                for role in ("positive", "negative", "presence", "abstention")
            ):
                mismatches.append("observed_roles")
            if observed_sources != dict(sorted(source_counter.items())):
                mismatches.append("observed_sources")
    if (
        not isinstance(sampling, dict)
        or sampling.get("target_role_shares") != hyperparameters["role_targets"]
        or sampling.get("pyro_max_share") != hyperparameters["pyro_max_share"]
        or sampling.get("samples_per_epoch") != hyperparameters["samples_per_epoch"]
    ):
        mismatches.append("sampling")
    for key in ("gradient_tensors", "peak_vram_bytes", "trainable_parameters"):
        value = report.get(key)
        if not isinstance(value, int) or value <= 0:
            mismatches.append(key)
    if not isinstance(losses, list) or len(losses) != smoke_steps:
        mismatches.append("loss_history")
    gradient_heads = report.get("gradient_heads")
    expected_heads = {
        "segmentation_head",
        "point_head",
        "abstention_head",
        "presence_head",
    }
    if not isinstance(gradient_heads, dict) or set(gradient_heads) != expected_heads:
        mismatches.append("gradient_heads")
    else:
        for head, stats in gradient_heads.items():
            if (
                not isinstance(stats, dict)
                or int(stats.get("tensors", 0)) <= 0
                or int(stats.get("nonzero_tensors", 0)) <= 0
                or not isinstance(stats.get("max_abs"), (int, float))
                or not math.isfinite(float(stats["max_abs"]))
                or float(stats["max_abs"]) <= 0.0
            ):
                mismatches.append(f"gradient_heads.{head}")
    if mismatches or not finite_losses:
        raise ValueError(f"GPU smoke report contract failed: {sorted(mismatches)}")
    return report


def build_preflight_report(
    *,
    pointing_root: Path,
    multitask_manifest: Path | None,
    data_root: Path,
    composition_registry: Path | None,
    model_id: str,
    model_revision: str,
    benchmark_denylist_path: Path | None = DEFAULT_BENCHMARK_DENYLIST,
    initial_safetensors: Path | None = None,
    backbone_config: Path | None = None,
    composition_manifest: Path | None = None,
    composition_report: Path | None = None,
    composition_integrity_receipt: Path | None = None,
    hydration_report: Path | None = None,
    hydration_integrity_receipt: Path | None = None,
    require_professional_corpus: bool = True,
) -> dict[str, Any]:
    pointing_root = pointing_root.resolve()
    data_root = data_root.resolve()
    multitask_path = multitask_manifest.resolve() if multitask_manifest else None
    registry_sha256: str | None = None
    benchmark_denylist: BenchmarkDenylist | None = None
    source_identity_contract_sha256: str | None = None
    campaign_id: str | None = None
    registry: dict[str, Any] = {}
    if composition_registry is None or not composition_registry.is_file():
        registry_errors = ["composition_registry_missing"]
    else:
        registry_errors = []
        try:
            registry = json.loads(composition_registry.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            registry_errors.append(f"composition_registry_invalid:{type(exc).__name__}")
        else:
            registry_sha256 = _sha256(composition_registry)
            campaign_id = str(registry.get("campaign_id") or "")
            registry_errors.extend(_registry_contract_errors(registry))
            try:
                source_identity_contract_sha256 = validate_source_identity_contract(
                    registry
                ).contract_sha256
            except (KeyError, TypeError, ValueError):
                source_identity_contract_sha256 = None
            if not campaign_id:
                registry_errors.append("composition_registry_contract_invalid")
            if benchmark_denylist_path is None:
                registry_errors.append("benchmark_denylist_missing")
            else:
                try:
                    benchmark_denylist = load_benchmark_denylist(
                        benchmark_denylist_path, registry.get("benchmark_boundary", {})
                    )
                except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
                    registry_errors.append(f"benchmark_denylist_invalid:{type(exc).__name__}:{exc}")
    integrated_pointing = (
        multitask_path is not None
        and (pointing_root / "manifest.jsonl").resolve() == multitask_path
    )
    if (pointing_root / "manifest.jsonl").is_file() and not integrated_pointing:
        rows, errors = _read_pointing_manifest(pointing_root)
    else:
        rows, errors = [], []
    errors.extend(registry_errors)
    warnings: list[str] = []
    if not rows and not integrated_pointing:
        warnings.append("separate_pointing_corpus_absent_using_multitask_manifest")
    split_counts = Counter(str(row.get("split")) for row in rows)
    target_counts = Counter(
        str(target.get("semantic_anchor"))
        for row in rows
        for target in row.get("targets", [])
        if isinstance(target, dict)
    )
    empty_target_rows = sum(not row.get("targets") for row in rows)
    if rows and set(split_counts) != ALL_SPLITS:
        errors.append(f"missing_split:{sorted(ALL_SPLITS - set(split_counts))}")
    if rows and empty_target_rows:
        warnings.append(f"pointing_abstention_rows:{empty_target_rows}")
    if rows and not (pointing_root / "report.json").is_file():
        errors.append("pointing_report_missing")
    if rows and not (pointing_root / "checksums.sha256").is_file():
        errors.append("pointing_checksums_missing")

    mask_manifest_ready = False
    mask_quality_counts: Counter[str] = Counter()
    multitask_split_counts: Counter[str] = Counter()
    multitask_target_counts: Counter[str] = Counter()
    multitask_abstention_rows = 0
    sample_validation_status_counts: Counter[str] = Counter()
    invalid_sample_status_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    sample_weight_counts: Counter[str] = Counter()
    supervision_counts: Counter[str] = Counter()
    presence_target_counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = {}
    seen_sample_ids: set[str] = set()
    seen_image_sha256: set[str] = set()
    verified_artifacts = 0
    verified_decodable_artifacts = 0
    benchmark_raw_hash_matches = 0
    benchmark_decoded_hash_matches = 0
    benchmark_phash_matches = 0
    implicit_zero_masks = 0
    multitask_rows = 0
    composition_profile_rows = 0
    invalid_composition_profile_rows = 0
    invalid_registry_binding_rows = 0
    invalid_hydration_status_rows = 0
    multitask_records: list[dict[str, Any]] = []
    if multitask_manifest is None:
        errors.append("multitask_manifest_required_for_segmentation")
    else:
        path = multitask_path
        assert path is not None
        if not path.is_file():
            errors.append(f"missing:{path}")
        else:
            mask_manifest_ready = True
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"multitask_manifest_invalid_json:{line_number}:{exc.msg}")
                    continue
                if not isinstance(row, dict):
                    errors.append(f"multitask_manifest_row_not_object:{line_number}")
                    continue
                multitask_records.append(row)
                multitask_rows += 1
                serialized_row = json.dumps(row, sort_keys=True).casefold()
                if any(marker in serialized_row for marker in FORBIDDEN_BENCHMARK_MARKERS):
                    errors.append(f"independent_benchmark_reference_forbidden:{line_number}")
                split = str(row.get("split"))
                multitask_split_counts[split] += 1
                if split not in ALL_SPLITS:
                    errors.append(f"split_invalid:{line_number}:{split}")
                source_counts[str(row.get("source_id") or "unknown")] += 1
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    errors.append(f"sample_id_missing:{line_number}")
                elif sample_id in seen_sample_ids:
                    errors.append(f"sample_id_duplicate:{line_number}:{sample_id}")
                else:
                    seen_sample_ids.add(sample_id)
                group_value = row.get("split_group")
                if not isinstance(group_value, str) or not group_value.strip():
                    errors.append(f"split_group_missing:{line_number}")
                    group = f"missing:{line_number}"
                else:
                    group = group_value
                split_groups.setdefault(group, set()).add(split)

                flags: dict[str, bool] = {}
                for field in SUPERVISION_FIELDS:
                    value = row.get(field)
                    if not isinstance(value, bool):
                        errors.append(f"supervision_flag_invalid:{line_number}:{field}")
                        flags[field] = False
                    else:
                        flags[field] = value
                        if value:
                            supervision_counts[field] += 1
                if not any(flags.values()):
                    errors.append(f"no_supervised_task:{line_number}")

                points = row.get("anchor_points")
                if not isinstance(points, list):
                    errors.append(f"anchor_points_missing:{line_number}")
                    points = []
                if points and not flags["point_supervised"]:
                    errors.append(f"unsupervised_point_has_anchors:{line_number}")
                if (
                    flags["point_supervised"]
                    and not points
                    and str(row.get("annotation_strength")) not in {"negative", "temporal_negative"}
                ):
                    errors.append(f"point_target_missing:{line_number}")
                for point in points:
                    if not isinstance(point, dict):
                        errors.append(f"anchor_point_invalid:{line_number}")
                        continue
                    kind = str(point.get("kind"))
                    multitask_target_counts[kind] += 1
                    if kind not in ALLOWED_POINT_KINDS:
                        errors.append(f"anchor_point_kind_invalid:{line_number}:{kind}")
                    try:
                        x, y = float(point["x"]), float(point["y"])
                    except (KeyError, TypeError, ValueError):
                        errors.append(f"anchor_point_invalid:{line_number}")
                        continue
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        errors.append(f"anchor_point_out_of_bounds:{line_number}")
                point_derivation = str(row.get("point_derivation") or "").lower()
                forbidden_point_markers = ("bbox", "bounding_box", "box_center")
                if flags["point_supervised"] and (
                    row.get("bbox_to_point")
                    or any(marker in point_derivation for marker in forbidden_point_markers)
                ):
                    errors.append(f"bbox_derived_point_forbidden:{line_number}")

                abstention_reason = row.get("visual_abstention_reason")
                if flags["abstention_supervised"] and "visual_abstention_reason" not in row:
                    errors.append(f"visual_abstention_label_missing:{line_number}")
                elif abstention_reason is not None and (
                    not isinstance(abstention_reason, str) or not abstention_reason.strip()
                ):
                    errors.append(f"visual_abstention_label_invalid:{line_number}")
                elif not flags["abstention_supervised"] and abstention_reason is not None:
                    errors.append(f"unsupervised_abstention_has_label:{line_number}")
                elif flags["abstention_supervised"]:
                    if abstention_reason is not None and points:
                        errors.append(f"abstention_conflicts_with_point:{line_number}")
                    elif abstention_reason is None and not points:
                        errors.append(f"non_abstention_has_no_point_evidence:{line_number}")
                annotation_strength = str(row.get("annotation_strength") or "")
                if annotation_strength not in ALLOWED_ANNOTATION_STRENGTHS:
                    errors.append(
                        f"annotation_strength_not_admitted:{line_number}:{annotation_strength}"
                    )
                if annotation_strength in {
                    "negative",
                    "temporal_negative",
                }:
                    if flags["abstention_supervised"] and not str(abstention_reason or "").strip():
                        errors.append(f"negative_abstention_reason_missing:{line_number}")
                    role_counts["negative"] += 1
                    multitask_abstention_rows += int(
                        flags["abstention_supervised"]
                        and isinstance(abstention_reason, str)
                        and bool(abstention_reason.strip())
                    )
                elif (
                    flags["abstention_supervised"]
                    and isinstance(abstention_reason, str)
                    and bool(abstention_reason.strip())
                ):
                    multitask_abstention_rows += 1
                    role_counts["abstention"] += 1
                elif flags["presence_supervised"] and not flags["point_supervised"]:
                    role_counts["presence"] += 1
                else:
                    role_counts["positive"] += 1

                presence_targets = row.get("presence_targets")
                if flags["presence_supervised"]:
                    if not isinstance(presence_targets, dict) or set(presence_targets) != set(
                        PRESENCE_LABELS
                    ):
                        errors.append(f"presence_targets_invalid:{line_number}")
                    else:
                        for label in PRESENCE_LABELS:
                            value = presence_targets[label]
                            if not (
                                isinstance(value, bool)
                                or (isinstance(value, int) and value in (0, 1))
                            ):
                                errors.append(f"presence_target_not_boolean:{line_number}:{label}")
                            else:
                                presence_target_counts[f"{label}:{bool(value)}"] += 1
                    if not str(row.get("presence_provenance") or "").strip():
                        errors.append(f"presence_provenance_missing:{line_number}")
                elif presence_targets is not None:
                    errors.append(f"unsupervised_presence_has_targets:{line_number}")
                if isinstance(presence_targets, dict):
                    if annotation_strength == "negative":
                        if any(bool(value) for value in presence_targets.values()):
                            errors.append(f"negative_presence_conflict:{line_number}")
                        if row.get("mask_encoding") != "implicit_zero_from_explicit_negative":
                            errors.append(f"negative_zero_mask_contract_missing:{line_number}")
                    elif flags["segmentation_supervised"] and not any(
                        bool(value) for value in presence_targets.values()
                    ):
                        errors.append(f"positive_mask_presence_conflict:{line_number}")
                    for point in points:
                        if not isinstance(point, dict):
                            continue
                        expected_label = {
                            "fire_base": "flame_visible",
                            "smoke_column_base": "smoke_visible",
                        }.get(str(point.get("kind") or ""))
                        if expected_label and presence_targets.get(expected_label) is not True:
                            errors.append(f"point_presence_conflict:{line_number}:{expected_label}")

                try:
                    sample_weight = float(row.get("sample_weight", 1.0))
                except (TypeError, ValueError):
                    sample_weight = math.nan
                if not math.isfinite(sample_weight) or sample_weight <= 0.0:
                    errors.append(f"sample_weight_invalid:{line_number}")
                else:
                    sample_weight_counts[str(sample_weight)] += 1
                if flags["segmentation_supervised"]:
                    if not row.get("mask_quality"):
                        errors.append(f"mask_quality_missing:{line_number}")
                    else:
                        mask_quality_counts[str(row["mask_quality"])] += 1
                sample_status = str(row.get("sample_validation_status"))
                sample_validation_status_counts[sample_status] += 1
                if sample_status not in TRAINABLE_SAMPLE_STATUSES:
                    invalid_sample_status_counts[sample_status] += 1
                if row.get("validation_profile") == "fireviewer_multitask_composition_v1":
                    composition_profile_rows += 1
                    if sample_status != "strict_automated_validated":
                        errors.append(f"composition_row_not_strict:{line_number}")
                else:
                    invalid_composition_profile_rows += 1
                    errors.append(f"composition_validation_profile_invalid:{line_number}")
                if (
                    row.get("campaign_id") != campaign_id
                    or row.get("composition_registry_sha256") != registry_sha256
                ):
                    invalid_registry_binding_rows += 1
                invalid_hydration_status_rows += int(
                    row.get("hydration_status") != "sha256_and_semantic_validation_passed"
                )
                image_sha = str(row.get("image_sha256") or "").lower()
                if image_sha and image_sha in seen_image_sha256:
                    errors.append(f"image_sha256_duplicate:{line_number}:{image_sha}")
                elif image_sha:
                    seen_image_sha256.add(image_sha)
                if (
                    benchmark_denylist is not None
                    and benchmark_matches(
                        raw_sha256=image_sha,
                        denylist=benchmark_denylist,
                    )["raw_sha256"]
                ):
                    benchmark_raw_hash_matches += 1
                    errors.append(f"benchmark_raw_hash_forbidden:{line_number}")
                artifacts = [("image_relpath", "image_sha256")]
                if flags["segmentation_supervised"]:
                    if row.get("mask_relpath") or row.get("mask_sha256"):
                        artifacts.append(("mask_relpath", "mask_sha256"))
                    elif row.get("mask_encoding") == "implicit_zero_from_explicit_negative" and str(
                        row.get("annotation_strength")
                    ) in {"negative", "temporal_negative"}:
                        implicit_zero_masks += 1
                    else:
                        errors.append(f"mask_missing:{line_number}")
                elif row.get("mask_relpath") or row.get("mask_sha256"):
                    errors.append(f"unsupervised_segmentation_has_mask:{line_number}")
                if row.get("valid_mask_relpath") or row.get("valid_mask_sha256"):
                    artifacts.append(("valid_mask_relpath", "valid_mask_sha256"))
                artifact_dimensions: dict[str, tuple[int, int]] = {}
                for rel_key, sha_key in artifacts:
                    relpath, expected_sha = row.get(rel_key), row.get(sha_key)
                    if not relpath or not expected_sha:
                        errors.append(f"artifact_contract_missing:{line_number}:{rel_key}")
                        continue
                    artifact = (data_root / str(relpath)).resolve()
                    if artifact != data_root and data_root not in artifact.parents:
                        errors.append(f"artifact_path_escape:{line_number}:{rel_key}")
                    elif not artifact.is_file():
                        errors.append(f"artifact_missing:{line_number}:{rel_key}")
                    elif _sha256(artifact) != str(expected_sha).lower():
                        errors.append(f"artifact_checksum_mismatch:{line_number}:{rel_key}")
                    else:
                        verified_artifacts += 1
                        decoded_hash: str | None = None
                        image_phash: int | None = None
                        try:
                            with Image.open(artifact) as opened:
                                opened.load()
                                dimensions = opened.size
                                if rel_key == "image_relpath":
                                    decoded_hash = decoded_pixel_sha256(opened)
                                    image_phash = phash64_imagehash_v1(opened)
                            if dimensions[0] <= 0 or dimensions[1] <= 0:
                                raise ValueError("non-positive dimensions")
                        except (OSError, ValueError) as exc:
                            errors.append(
                                f"artifact_decode_failed:{line_number}:{rel_key}:"
                                f"{type(exc).__name__}"
                            )
                        else:
                            artifact_dimensions[rel_key] = dimensions
                            verified_decodable_artifacts += 1
                            if rel_key == "image_relpath":
                                expected_decoded = str(row.get("decoded_pixel_sha256") or "")
                                expected_phash = str(row.get("phash64_imagehash_v1") or "")
                                if decoded_hash != expected_decoded:
                                    errors.append(f"decoded_pixel_hash_drift:{line_number}")
                                if image_phash is None or f"{image_phash:016x}" != expected_phash:
                                    errors.append(f"phash_drift:{line_number}")
                                if benchmark_denylist is not None:
                                    matches = benchmark_matches(
                                        raw_sha256=image_sha,
                                        decoded_sha256=decoded_hash,
                                        phash64=image_phash,
                                        denylist=benchmark_denylist,
                                    )
                                    if matches["decoded_sha256"]:
                                        benchmark_decoded_hash_matches += 1
                                        errors.append(
                                            f"benchmark_decoded_hash_forbidden:{line_number}"
                                        )
                                    if matches["phash"]:
                                        benchmark_phash_matches += 1
                                        errors.append(f"benchmark_phash_forbidden:{line_number}")
                image_dimensions = artifact_dimensions.get("image_relpath")
                for mask_key in ("mask_relpath", "valid_mask_relpath"):
                    mask_dimensions = artifact_dimensions.get(mask_key)
                    if (
                        image_dimensions is not None
                        and mask_dimensions is not None
                        and mask_dimensions != image_dimensions
                    ):
                        errors.append(f"artifact_dimensions_mismatch:{line_number}:{mask_key}")

    if mask_manifest_ready and set(multitask_split_counts) != ALL_SPLITS:
        errors.append(f"multitask_missing_split:{sorted(ALL_SPLITS - set(multitask_split_counts))}")
    if invalid_sample_status_counts:
        errors.append(
            "sample_status_not_trainable:"
            + json.dumps(dict(sorted(invalid_sample_status_counts.items())), sort_keys=True)
        )
    professional_corpus_ready = False
    ready_for_gpu_finite_loss_smoke = False
    if invalid_hydration_status_rows:
        errors.append(f"composition_hydration_status_invalid:{invalid_hydration_status_rows}")
    if multitask_rows and composition_profile_rows != multitask_rows:
        errors.append(
            f"composition_profile_coverage_invalid:{composition_profile_rows}!={multitask_rows}"
        )
    if invalid_registry_binding_rows:
        errors.append(f"composition_registry_binding_invalid:{invalid_registry_binding_rows}")
    composition_accounting: dict[str, Any] = {
        "quality_gate_deficits": {},
        "pilot_corpus_ready": False,
        "professional_corpus_ready": False,
    }
    if (
        not registry_errors
        and registry_sha256
        and campaign_id
        and multitask_path
        and benchmark_denylist is not None
    ):
        composition_accounting = _validate_composition_chain(
            hydrated_rows=multitask_records,
            hydrated_manifest=multitask_path,
            registry=registry,
            registry_sha256=registry_sha256,
            benchmark_denylist=benchmark_denylist,
            campaign_id=campaign_id,
            composition_manifest=composition_manifest,
            composition_report_path=composition_report,
            composition_integrity_receipt_path=composition_integrity_receipt,
            hydration_report_path=hydration_report,
            hydration_integrity_receipt_path=hydration_integrity_receipt,
            errors=errors,
        )
        ready_for_gpu_finite_loss_smoke = composition_accounting["pilot_corpus_ready"]
        professional_corpus_ready = composition_accounting["professional_corpus_ready"]
    if require_professional_corpus and not professional_corpus_ready:
        errors.append("professional_corpus_not_ready")
    leaking_groups = sorted(group for group, owners in split_groups.items() if len(owners) != 1)
    if leaking_groups:
        errors.append(f"multitask_split_group_leakage:{leaking_groups}")

    if len(model_revision) != 40 or any(
        character not in "0123456789abcdef" for character in model_revision.casefold()
    ):
        errors.append("model_revision_required")
    if initial_safetensors is not None and not initial_safetensors.is_file():
        errors.append(f"initial_safetensors_missing:{initial_safetensors}")
    if initial_safetensors is not None and (
        backbone_config is None or not backbone_config.is_file()
    ):
        errors.append("backbone_config_required_with_initial_safetensors")

    return {
        "schema_version": 2,
        "model_family": "DINOv3 multi-task",
        "model_id": model_id,
        "model_revision": model_revision,
        "fine_tuning_mode": "full_model_all_parameters_trainable",
        "pointing_corpus": {
            "root": str(pointing_root),
            "manifest_sha256": (
                _sha256(pointing_root / "manifest.jsonl")
                if (pointing_root / "manifest.jsonl").is_file()
                else None
            ),
            "rows": len(rows),
            "split_counts": dict(sorted(split_counts.items())),
            "target_counts": dict(sorted(target_counts.items())),
            "empty_target_rows": empty_target_rows,
            "integrated_in_multitask_manifest": integrated_pointing,
        },
        "multitask_manifest": str(multitask_manifest.resolve()) if multitask_manifest else None,
        "multitask_manifest_sha256": _sha256(multitask_path)
        if multitask_path and multitask_path.is_file()
        else None,
        "composition_registry": (
            str(composition_registry.resolve()) if composition_registry else None
        ),
        "composition_registry_sha256": registry_sha256,
        "source_identity_contract_sha256": source_identity_contract_sha256,
        "benchmark_denylist": (
            str(benchmark_denylist_path.resolve()) if benchmark_denylist_path is not None else None
        ),
        "benchmark_denylist_sha256": (
            benchmark_denylist.sha256 if benchmark_denylist is not None else None
        ),
        "campaign_id": campaign_id,
        "composition_manifest": (
            str(composition_manifest.resolve()) if composition_manifest else None
        ),
        "composition_manifest_sha256": (
            _sha256(composition_manifest)
            if composition_manifest and composition_manifest.is_file()
            else None
        ),
        "composition_report": (str(composition_report.resolve()) if composition_report else None),
        "composition_report_sha256": (
            _sha256(composition_report)
            if composition_report and composition_report.is_file()
            else None
        ),
        "composition_integrity_receipt": (
            str(composition_integrity_receipt.resolve()) if composition_integrity_receipt else None
        ),
        "composition_integrity_receipt_sha256": (
            _sha256(composition_integrity_receipt)
            if composition_integrity_receipt and composition_integrity_receipt.is_file()
            else None
        ),
        "multitask_split_counts": dict(sorted(multitask_split_counts.items())),
        "multitask_target_counts": dict(sorted(multitask_target_counts.items())),
        "multitask_abstention_rows": multitask_abstention_rows,
        "multitask_role_counts": dict(sorted(role_counts.items())),
        "multitask_source_counts": dict(sorted(source_counts.items())),
        "supervision_counts": dict(sorted(supervision_counts.items())),
        "presence_target_counts": dict(sorted(presence_target_counts.items())),
        "implicit_zero_masks": implicit_zero_masks,
        "composition_profile_rows": composition_profile_rows,
        "invalid_composition_profile_rows": invalid_composition_profile_rows,
        "invalid_registry_binding_rows": invalid_registry_binding_rows,
        "invalid_hydration_status_rows": invalid_hydration_status_rows,
        "hydration_report": str(hydration_report.resolve()) if hydration_report else None,
        "hydration_report_sha256": (
            _sha256(hydration_report) if hydration_report and hydration_report.is_file() else None
        ),
        "hydration_integrity_receipt": (
            str(hydration_integrity_receipt.resolve()) if hydration_integrity_receipt else None
        ),
        "hydration_integrity_receipt_sha256": (
            _sha256(hydration_integrity_receipt)
            if hydration_integrity_receipt and hydration_integrity_receipt.is_file()
            else None
        ),
        "recomputed_quality_gate_deficits": composition_accounting["quality_gate_deficits"],
        "sample_weight_counts": dict(sorted(sample_weight_counts.items())),
        "verified_artifacts": verified_artifacts,
        "verified_decodable_artifacts": verified_decodable_artifacts,
        "professional_corpus_required": require_professional_corpus,
        "professional_corpus_ready": professional_corpus_ready,
        "ready_for_gpu_finite_loss_smoke": ready_for_gpu_finite_loss_smoke,
        "split_group_leakage": leaking_groups,
        "canonical_event_leakage": composition_accounting.get("canonical_event_leakage", []),
        "benchmark_raw_hash_matches": benchmark_raw_hash_matches,
        "benchmark_decoded_hash_matches": benchmark_decoded_hash_matches,
        "benchmark_phash_matches": benchmark_phash_matches,
        "training_errors": errors,
        "training_warnings": warnings,
        "training_ready": not errors,
        "promotion_ready": False,
        "promotion_errors": [
            "independent_benchmark_missing",
            "ground_truth_acceptance_gate_pending",
        ],
        "mask_manifest_detected": mask_manifest_ready,
        "mask_quality_counts": dict(sorted(mask_quality_counts.items())),
        "initialization": (
            "complete_v4_safetensors"
            if initial_safetensors is not None
            else "immutable_base_pretrained"
        ),
        "initial_safetensors": initial_safetensors.name if initial_safetensors else None,
        "initial_safetensors_sha256": (
            _sha256(initial_safetensors)
            if initial_safetensors and initial_safetensors.is_file()
            else None
        ),
        "backbone_config": backbone_config.name if backbone_config else None,
        "backbone_config_sha256": (
            _sha256(backbone_config) if backbone_config and backbone_config.is_file() else None
        ),
        "sample_validation_status_counts": dict(sorted(sample_validation_status_counts.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FireViewer DINOv3 multi-task training gate")
    parser.add_argument(
        "command",
        choices=("preflight", "plan", "smoke", "pilot-train", "train"),
    )
    parser.add_argument("--pointing-root", type=Path, default=DEFAULT_POINTING_ROOT)
    parser.add_argument("--multitask-manifest", type=Path, default=DEFAULT_MULTITASK_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--composition-registry", type=Path, default=DEFAULT_COMPOSITION_REGISTRY)
    parser.add_argument("--benchmark-denylist", type=Path, default=DEFAULT_BENCHMARK_DENYLIST)
    parser.add_argument("--composition-manifest", type=Path)
    parser.add_argument("--composition-report", type=Path)
    parser.add_argument("--composition-integrity-receipt", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--initial-safetensors", type=Path)
    parser.add_argument("--backbone-config", type=Path)
    parser.add_argument("--hydration-report", type=Path)
    parser.add_argument("--hydration-integrity-receipt", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument(
        "--reuse-passed-preflight",
        action="store_true",
        help=(
            "Reuse the immediately preceding passed preflight for smoke or pilot-train. "
            "Contract files are rehashed, but the already verified media are not decoded twice."
        ),
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--positive-share", type=float, default=0.35)
    parser.add_argument("--negative-share", type=float, default=0.25)
    parser.add_argument("--presence-share", type=float, default=0.25)
    parser.add_argument("--abstention-share", type=float, default=0.15)
    parser.add_argument("--pyro-max-share", type=float, default=0.33)
    parser.add_argument("--smoke-steps", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=8192)
    return parser


def _load_passed_smoke_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.command not in {"smoke", "pilot-train"}:
        raise ValueError("passed preflight reuse is restricted to smoke and pilot-train")
    if args.command == "pilot-train" and args.smoke_report is None:
        raise ValueError("pilot-train preflight reuse requires a smoke report")
    preflight_path = args.output / "preflight-report.json"
    if not preflight_path.is_file():
        raise ValueError("passed preflight report is required for reuse")
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("passed preflight report must be an object")
    if (
        args.command == "pilot-train"
        and args.smoke_report is not None
        and (
            not args.smoke_report.is_file()
            or args.smoke_report.stat().st_mtime_ns < preflight_path.stat().st_mtime_ns
        )
    ):
        raise ValueError("smoke report predates the reusable preflight")
    contracts = (
        ("multitask_manifest", "multitask_manifest_sha256", args.multitask_manifest),
        ("composition_registry", "composition_registry_sha256", args.composition_registry),
        ("benchmark_denylist", "benchmark_denylist_sha256", args.benchmark_denylist),
        ("composition_manifest", "composition_manifest_sha256", args.composition_manifest),
        ("composition_report", "composition_report_sha256", args.composition_report),
        (
            "composition_integrity_receipt",
            "composition_integrity_receipt_sha256",
            args.composition_integrity_receipt,
        ),
        ("hydration_report", "hydration_report_sha256", args.hydration_report),
        (
            "hydration_integrity_receipt",
            "hydration_integrity_receipt_sha256",
            args.hydration_integrity_receipt,
        ),
    )
    mismatches: list[str] = []
    for path_field, hash_field, path in contracts:
        if path is None or not path.is_file():
            mismatches.append(path_field)
            continue
        if report.get(path_field) != str(path.resolve()) or report.get(hash_field) != _sha256(path):
            mismatches.append(path_field)
    expected = {
        "schema_version": 2,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "training_ready": True,
        "ready_for_gpu_finite_loss_smoke": True,
        "professional_corpus_required": False,
    }
    mismatches.extend(key for key, value in expected.items() if report.get(key) != value)
    if mismatches:
        raise ValueError(
            "reusable preflight contract mismatch: " + ",".join(sorted(set(mismatches)))
        )
    return report


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.gradient_accumulation_steps <= 0
        or args.samples_per_epoch <= 0
    ):
        raise ValueError("epochs, batch-size and gradient-accumulation-steps must be positive")
    role_targets = {
        "positive": args.positive_share,
        "negative": args.negative_share,
        "presence": args.presence_share,
        "abstention": args.abstention_share,
    }
    if not math.isclose(sum(role_targets.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("positive, negative, presence and abstention shares must sum to 1.0")
    if not 0.30 <= args.pyro_max_share <= 0.35:
        raise ValueError("pyro-max-share must remain between 0.30 and 0.35")
    if args.reuse_passed_preflight:
        report = _load_passed_smoke_preflight(args)
    else:
        report = build_preflight_report(
            pointing_root=args.pointing_root,
            multitask_manifest=args.multitask_manifest,
            data_root=args.data_root,
            composition_registry=args.composition_registry,
            benchmark_denylist_path=args.benchmark_denylist,
            model_id=args.model_id,
            model_revision=args.model_revision,
            initial_safetensors=args.initial_safetensors,
            backbone_config=args.backbone_config,
            composition_manifest=args.composition_manifest,
            composition_report=args.composition_report,
            composition_integrity_receipt=args.composition_integrity_receipt,
            hydration_report=args.hydration_report,
            hydration_integrity_receipt=args.hydration_integrity_receipt,
            require_professional_corpus=args.command in {"preflight", "plan", "train"},
        )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "preflight-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "preflight":
        if not report["training_ready"]:
            raise SystemExit(2)
        return
    from fireviewer_model_lab.training.dinov3_adapter import (
        build_training_contract,
        training_contract_sha256,
    )

    training_provenance = {
        "composition_registry_sha256": report["composition_registry_sha256"],
        "source_identity_contract_sha256": report["source_identity_contract_sha256"],
        "benchmark_denylist_sha256": report["benchmark_denylist_sha256"],
        "composition_manifest_sha256": report["composition_manifest_sha256"],
        "composition_report_sha256": report["composition_report_sha256"],
        "composition_integrity_receipt_sha256": report["composition_integrity_receipt_sha256"],
        "hydration_report_sha256": report["hydration_report_sha256"],
        "hydration_integrity_receipt_sha256": report["hydration_integrity_receipt_sha256"],
    }
    training_contract = build_training_contract(
        manifest=args.multitask_manifest.resolve(),
        model_id=args.model_id,
        model_revision=args.model_revision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        image_size=args.image_size,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        balanced_sampling=args.balanced_sampling,
        role_targets=role_targets,
        pyro_share=args.pyro_max_share,
        samples_per_epoch=args.samples_per_epoch,
        provenance=training_provenance,
        initial_safetensors=args.initial_safetensors,
        backbone_config=args.backbone_config,
    )
    plan = {
        "schema_version": 3,
        "model_family": report["model_family"],
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "data_root": str(args.data_root.resolve()),
        "composition_registry": report["composition_registry"],
        "composition_registry_sha256": report["composition_registry_sha256"],
        "source_identity_contract_sha256": report["source_identity_contract_sha256"],
        "benchmark_denylist": report["benchmark_denylist"],
        "benchmark_denylist_sha256": report["benchmark_denylist_sha256"],
        "campaign_id": report["campaign_id"],
        "composition_manifest": report["composition_manifest"],
        "composition_manifest_sha256": report["composition_manifest_sha256"],
        "composition_report": report["composition_report"],
        "composition_report_sha256": report["composition_report_sha256"],
        "composition_integrity_receipt": report["composition_integrity_receipt"],
        "composition_integrity_receipt_sha256": report["composition_integrity_receipt_sha256"],
        "adapter": "fireviewer_model_lab.training.dinov3_adapter:DinoV3MultiTaskModel",
        "fine_tuning_mode": report["fine_tuning_mode"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "early_stopping_patience": args.early_stopping_patience,
            "balanced_sampling": args.balanced_sampling,
            "role_targets": role_targets,
            "pyro_max_share": args.pyro_max_share,
            "samples_per_epoch": args.samples_per_epoch,
        },
        "initialization": report["initialization"],
        "initial_safetensors": report["initial_safetensors"],
        "initial_safetensors_sha256": report["initial_safetensors_sha256"],
        "backbone_config": report["backbone_config"],
        "backbone_config_sha256": report["backbone_config_sha256"],
        "hydration_report": report["hydration_report"],
        "hydration_report_sha256": report["hydration_report_sha256"],
        "hydration_integrity_receipt": report["hydration_integrity_receipt"],
        "hydration_integrity_receipt_sha256": report["hydration_integrity_receipt_sha256"],
        "required_smoke_report": str(args.smoke_report.resolve()) if args.smoke_report else None,
        "resume_checkpoint": (
            str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None
        ),
        "training_contract": training_contract,
        "training_contract_sha256": training_contract_sha256(training_contract),
        "gates": [
            "explicit_task_supervision",
            "mask_integrity_when_supervised",
            "anchor_heatmaps_when_supervised",
            "presence_provenance",
            "visual_abstention_when_supervised",
            "one_image_per_sha256",
            "split_group_isolation",
            "canonical_event_isolation",
            "pinned_benchmark_hash_denylist",
            "bbox_to_point_forbidden",
        ],
        "training_ready": report["training_ready"],
        "corpus_tier": (
            "professional" if report["professional_corpus_ready"] else "pilot"
        ),
    }
    _write_json(args.output / "training-plan.json", plan)
    if args.command == "plan":
        return
    if not report["training_ready"]:
        raise RuntimeError("DINOv3 training gate failed; provide the canonical multi-task manifest")
    if args.command == "smoke":
        from fireviewer_model_lab.training.dinov3_adapter import finite_loss_probe

        smoke = finite_loss_probe(
            manifest=args.multitask_manifest.resolve(),
            data_root=args.data_root.resolve(),
            model_id=args.model_id,
            model_revision=args.model_revision,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            initial_safetensors=args.initial_safetensors,
            backbone_config=args.backbone_config,
            role_targets=role_targets,
            pyro_share=args.pyro_max_share,
            smoke_steps=args.smoke_steps,
            seed=args.seed,
            learning_rate=args.learning_rate,
            samples_per_epoch=args.samples_per_epoch,
            epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience,
            balanced_sampling=args.balanced_sampling,
            training_provenance=training_provenance,
        )
        _write_json(args.output / "smoke-report.json", smoke)
        print(json.dumps(smoke, ensure_ascii=False, indent=2, sort_keys=True))
        return
    from fireviewer_model_lab.training.dinov3_adapter import run_training

    smoke_receipt = _validate_smoke_report(
        args.smoke_report,
        manifest=args.multitask_manifest.resolve(),
        training_contract=training_contract,
        smoke_steps=args.smoke_steps,
    )
    _write_json(args.output / "accepted-smoke-receipt.json", smoke_receipt)

    try:
        result = run_training(
            manifest=args.multitask_manifest.resolve(),
            data_root=args.data_root.resolve(),
            output=args.output.resolve(),
            model_id=args.model_id,
            model_revision=args.model_revision,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            image_size=args.image_size,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience,
            initial_safetensors=args.initial_safetensors,
            backbone_config=args.backbone_config,
            balanced_sampling=args.balanced_sampling,
            role_targets=role_targets,
            pyro_share=args.pyro_max_share,
            samples_per_epoch=args.samples_per_epoch,
            training_provenance=training_provenance,
            resume_checkpoint=args.resume_checkpoint,
        )
    except OSError as exc:
        raise RuntimeError(
            "DINOv3 backbone is not locally available; accept the gated model "
            "terms and grant the HF token public-gated read access before train"
        ) from exc
    _write_json(args.output / "training-result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
