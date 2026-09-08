"""Fail-closed contract for the local FireViewer pointing V5 corpus."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPORT_SCHEMA = "fireviewer.pointing-dataset-v5-local-report.v1"
RELOAD_SCHEMA = "fireviewer.pointing-dataset-v5-reload-validation.v1"
EXPECTED_SPLITS = {"train", "validation", "test"}


class V5ContractError(RuntimeError):
    """Raised when the immutable V5 corpus no longer matches its receipt."""


@dataclass(frozen=True)
class V5DatasetContract:
    root: Path
    report: Mapping[str, Any]
    report_sha256: str
    selected_count: int
    split_counts: Mapping[str, int]
    source_counts: Mapping[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise V5ContractError(f"Cannot read valid JSON receipt {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise V5ContractError(f"JSON receipt is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise V5ContractError(f"Blank JSONL line in {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise V5ContractError(
                        f"JSONL row is not an object in {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise V5ContractError(f"Cannot read valid JSONL {path}: {error}") from error
    return rows


def _require_equal(report: Mapping[str, Any], key: str, expected: Any) -> None:
    actual = report.get(key)
    if actual != expected:
        raise V5ContractError(
            f"V5 report field {key!r} must be {expected!r}, got {actual!r}"
        )


def load_v5_contract(root: Path) -> V5DatasetContract:
    root = root.expanduser().resolve()
    report_path = root / "report.json"
    manifest_path = root / "selection_manifest.jsonl"
    reload_path = root / "reload_validation.json"
    for required in (report_path, manifest_path, reload_path, root / "data"):
        if not required.exists():
            raise V5ContractError(f"Required V5 corpus artifact is missing: {required}")

    report = _read_json(report_path)
    _require_equal(report, "schema", REPORT_SCHEMA)
    _require_equal(report, "status", "ready_for_training")
    _require_equal(report, "training_ready", True)
    _require_equal(report, "selected_count", 2909)
    _require_equal(report, "base_selected_count", 2810)
    _require_equal(report, "extension_selected_count", 99)
    _require_equal(report, "extension_available_count", 261)
    _require_equal(report, "extension_recurrence_excluded_count", 162)
    _require_equal(report, "extension_recurrence_cap", 12)
    _require_equal(report, "extension_maximum_recurrence_group_size", 12)
    _require_equal(report, "extension_scene_count", 9)
    _require_equal(
        report,
        "extension_visibility_counts",
        {"small_le_1pct": 15, "tiny_le_0p5pct": 47, "ultra_tiny_le_0p1pct": 37},
    )
    _require_equal(report, "aerial_retained_count", 0)
    _require_equal(report, "retained_foreground_person_or_selfie_count", 0)
    _require_equal(report, "exact_base_extension_overlap_count", 0)
    _require_equal(report, "cross_split_group_overlap_count", 0)
    _require_equal(report, "all_boxes_geometry_valid", True)
    _require_equal(report, "all_image_sha256_reverified", True)
    _require_equal(report, "materialization", {"copies": 0, "hardlinks": 2909})

    selected_count = int(report["selected_count"])
    split_counts = {str(k): int(v) for k, v in dict(report.get("split_counts", {})).items()}
    source_counts = {str(k): int(v) for k, v in dict(report.get("source_counts", {})).items()}
    if set(split_counts) != EXPECTED_SPLITS or sum(split_counts.values()) != selected_count:
        raise V5ContractError("V5 split accounting does not cover the corpus exactly")
    if sum(source_counts.values()) != selected_count:
        raise V5ContractError("V5 source accounting does not cover the corpus exactly")

    manifest_sha = _sha256(manifest_path)
    if manifest_sha != str(report.get("selection_manifest_sha256", "")):
        raise V5ContractError("V5 selection manifest SHA-256 differs from report.json")
    reload_receipt = _read_json(reload_path)
    if (
        reload_receipt.get("schema") != RELOAD_SCHEMA
        or reload_receipt.get("status") != "passed"
        or reload_receipt.get("all_images_decoded") is not True
        or reload_receipt.get("all_dimensions_match_metadata") is not True
        or reload_receipt.get("all_objects_reloadable") is not True
        or int(reload_receipt.get("decoded_total", -1)) != selected_count
        or {str(k): int(v) for k, v in dict(reload_receipt.get("decoded_counts", {})).items()}
        != split_counts
    ):
        raise V5ContractError("V5 reload validation receipt is incomplete or stale")

    return V5DatasetContract(
        root=root,
        report=report,
        report_sha256=_sha256(report_path),
        selected_count=selected_count,
        split_counts=split_counts,
        source_counts=source_counts,
    )


def validate_materialized_v5(contract: V5DatasetContract) -> Mapping[str, Any]:
    """Re-hash and account for every admitted row before GPU allocation."""

    manifest_path = contract.root / "selection_manifest.jsonl"
    manifest = _read_jsonl(manifest_path)
    if len(manifest) != contract.selected_count:
        raise V5ContractError(
            f"V5 manifest has {len(manifest)} rows, expected {contract.selected_count}"
        )

    manifest_by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    manifest_by_sha: dict[str, Mapping[str, Any]] = {}
    split_groups: dict[str, set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    image_bytes = 0
    hardlinks = 0
    for index, row in enumerate(manifest):
        split = str(row.get("split", ""))
        sha = str(row.get("sha256", ""))
        source = str(row.get("source_dataset", ""))
        group = str(row.get("split_group_id", ""))
        file_name = str(row.get("file_name", ""))
        if split not in EXPECTED_SPLITS:
            raise V5ContractError(f"Manifest row {index} has invalid split {split!r}")
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise V5ContractError(f"Manifest row {index} has invalid SHA-256")
        if sha in manifest_by_sha:
            raise V5ContractError(f"Duplicate admitted image SHA-256: {sha}")
        if not source or not group or not file_name.startswith("images/"):
            raise V5ContractError(f"Manifest row {index} lacks provenance fields")
        if row.get("aerial") is True:
            raise V5ContractError(f"Aerial row admitted despite V5 policy: {sha}")
        if (
            row.get("person_risk_requires_explicit_visual_clearance") is True
            and row.get("person_risk_reviewed_clear") is not True
        ):
            raise V5ContractError(f"Uncleared person-risk row admitted: {sha}")
        if source == "HPWREN-FIgLib":
            if (
                row.get("training_admitted") is not True
                or row.get("selection_eligible") is not True
                or row.get("selection_gate") != "visual_review_and_recurrence_cap_passed"
                or row.get("quality_gate")
                != "official_hpwren_bbox_and_fireviewer_visual_review_passed"
                or row.get("aerial") is not False
                or row.get("person_risk_reviewed_clear") is not True
            ):
                raise V5ContractError(f"HPWREN row lacks final visual-review authority: {sha}")

        manifest_by_sha[sha] = row
        manifest_by_split[split].append(row)
        split_groups[split].add(group)
        source_counts[source] += 1

    if dict(source_counts) != dict(contract.source_counts):
        raise V5ContractError("Materialized V5 source counts differ from report.json")
    if any(
        len(manifest_by_split[split]) != contract.split_counts[split]
        for split in EXPECTED_SPLITS
    ):
        raise V5ContractError("Materialized V5 split counts differ from report.json")
    for left in EXPECTED_SPLITS:
        for right in EXPECTED_SPLITS:
            if left < right and split_groups[left].intersection(split_groups[right]):
                raise V5ContractError(f"Split-group leakage detected between {left} and {right}")

    split_source_counts: dict[str, dict[str, int]] = {}
    for split in sorted(EXPECTED_SPLITS):
        metadata_path = contract.root / "data" / split / "metadata.jsonl"
        metadata = _read_jsonl(metadata_path)
        if len(metadata) != contract.split_counts[split]:
            raise V5ContractError(f"V5 {split} metadata cardinality mismatch")
        expected_by_sha = {str(row["sha256"]): row for row in manifest_by_split[split]}
        observed_sources: Counter[str] = Counter()
        for row in metadata:
            sha = str(row.get("sha256", ""))
            if sha not in expected_by_sha or row != expected_by_sha.pop(sha):
                raise V5ContractError(f"V5 {split} metadata differs from selection manifest: {sha}")
            image_path = contract.root / "data" / split / str(row["file_name"])
            if not image_path.is_file():
                raise V5ContractError(f"V5 image is missing: {image_path}")
            if _sha256(image_path) != sha:
                raise V5ContractError(f"V5 image SHA-256 mismatch: {image_path}")
            stat = image_path.stat()
            image_bytes += stat.st_size
            hardlinks += int(getattr(stat, "st_nlink", 1) >= 2)
            observed_sources[str(row["source_dataset"])] += 1
        if expected_by_sha:
            raise V5ContractError(f"V5 {split} metadata omitted admitted rows")
        split_source_counts[split] = dict(observed_sources)

    if hardlinks != contract.selected_count:
        raise V5ContractError(
            f"V5 storage policy expected {contract.selected_count} hardlinks, found {hardlinks}"
        )
    return {
        "status": "passed",
        "verified_image_count": contract.selected_count,
        "verified_image_bytes": image_bytes,
        "verified_hardlink_count": hardlinks,
        "split_source_counts": split_source_counts,
        "selection_manifest_sha256": _sha256(manifest_path),
    }
