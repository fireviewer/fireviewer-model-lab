"""Shared fail-closed training contracts for FireViewer challenger models.

The published RT-DETR and D-FINE trainers already establish the operational
discipline for detector runs: pinned inputs, split-leak prevention, preflight
reports, resumable checkpoints and immutable provenance.  This module applies
the same input contract to new model families without pretending that a local
unit test is a GPU or benchmark certification.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRAINING_SPLITS = frozenset({"train", "validation"})
ALL_SPLITS = frozenset({"train", "validation", "test"})
APPROVED_SAMPLE_STATUSES = frozenset({"source_provided", "double_validated"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest row is missing non-empty {key}")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class ChallengerRecord:
    """One immutable training sample shared by challenger trainers.

    Detection trainers use ``annotations``.  Segmentation and pointing trainers
    additionally require the mask fields and may declare a visual abstention.
    Relative paths are intentionally resolved only below the manifest directory.
    """

    sample_id: str
    source_id: str
    source_revision: str
    split: str
    split_group: str
    image_relpath: str
    image_sha256: str
    license_identifier: str
    sample_validation_status: str
    annotations: tuple[dict[str, Any], ...]
    mask_relpath: str | None = None
    mask_sha256: str | None = None
    anchor_points: tuple[dict[str, Any], ...] = ()
    visual_abstention_reason: str | None = None
    is_operational_incident: bool = False

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> ChallengerRecord:
        image_sha256 = str(row.get("image_sha256", row.get("sha256", ""))).lower()
        if not _is_sha256(image_sha256):
            raise ValueError("manifest row has invalid image SHA-256")
        mask_sha256 = row.get("mask_sha256")
        if mask_sha256 is not None:
            mask_sha256 = str(mask_sha256).lower()
            if not _is_sha256(mask_sha256):
                raise ValueError("manifest row has invalid mask SHA-256")
        annotations = row.get("annotations", [])
        if not isinstance(annotations, list) or not all(
            isinstance(item, dict) for item in annotations
        ):
            raise ValueError("manifest annotations must be a list of objects")
        anchors = row.get("anchor_points", [])
        if not isinstance(anchors, list) or not all(isinstance(item, dict) for item in anchors):
            raise ValueError("manifest anchor_points must be a list of objects")
        split = _require_string(row, "split")
        if split not in ALL_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            sample_id=_require_string(row, "sample_id"),
            source_id=_require_string(row, "source_id"),
            source_revision=_require_string(row, "source_revision"),
            split=split,
            split_group=_require_string(row, "split_group"),
            image_relpath=_require_string(row, "image_relpath"),
            image_sha256=image_sha256,
            license_identifier=_require_string(row, "license"),
            sample_validation_status=_require_string(row, "sample_validation_status"),
            annotations=tuple(annotations),
            mask_relpath=(str(row["mask_relpath"]) if row.get("mask_relpath") else None),
            mask_sha256=mask_sha256,
            anchor_points=tuple(anchors),
            visual_abstention_reason=(
                str(row["visual_abstention_reason"])
                if row.get("visual_abstention_reason")
                else None
            ),
            is_operational_incident=bool(row.get("is_operational_incident", False)),
        )


def load_records(manifests: Iterable[Path]) -> tuple[ChallengerRecord, ...]:
    records: list[ChallengerRecord] = []
    seen_sample_ids: set[str] = set()
    seen_images: set[str] = set()
    for manifest in manifests:
        path = manifest.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"training manifest is missing: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"manifest row is not an object at {path}:{line_number}")
            record = ChallengerRecord.from_json(raw)
            if record.sample_id in seen_sample_ids:
                raise ValueError(f"duplicate sample_id across manifests: {record.sample_id}")
            if record.image_sha256 in seen_images:
                raise ValueError(f"duplicate image SHA-256 across manifests: {record.image_sha256}")
            seen_sample_ids.add(record.sample_id)
            seen_images.add(record.image_sha256)
            records.append(record)
    if not records:
        raise ValueError("at least one non-empty training manifest is required")
    return tuple(records)


def preflight_report(
    records: Iterable[ChallengerRecord],
    *,
    model_family: str,
    requires_masks: bool,
    requires_anchors: bool,
) -> dict[str, Any]:
    values = tuple(records)
    split_counts: Counter[str] = Counter(record.split for record in values)
    source_counts: Counter[str] = Counter(record.source_id for record in values)
    groups: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    mask_count = 0
    anchor_count = 0
    abstention_count = 0
    for record in values:
        groups[record.split_group].add(record.split)
        if record.is_operational_incident:
            errors.append(f"operational_incident_forbidden:{record.sample_id}")
        if record.sample_validation_status not in APPROVED_SAMPLE_STATUSES:
            errors.append(f"sample_not_validated:{record.sample_id}")
        if requires_masks and (record.mask_relpath is None or record.mask_sha256 is None):
            errors.append(f"mask_missing:{record.sample_id}")
        if record.mask_relpath is not None:
            mask_count += 1
        if record.anchor_points:
            anchor_count += 1
        if record.visual_abstention_reason is not None:
            abstention_count += 1
        if (
            requires_anchors
            and not record.anchor_points
            and record.visual_abstention_reason is None
        ):
            errors.append(f"anchor_or_abstention_missing:{record.sample_id}")
    leaking_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    errors.extend(f"split_group_leakage:{group}" for group in leaking_groups)
    for split in TRAINING_SPLITS:
        if split_counts[split] == 0:
            errors.append(f"missing_split:{split}")
    # The test split is mandatory for promotion but deliberately not for an
    # exploratory train run; this keeps the report honest about its status.
    promotion_errors = list(errors)
    if split_counts["test"] == 0:
        promotion_errors.append("missing_held_out_test_split")
    if requires_anchors and abstention_count == 0:
        promotion_errors.append("visual_abstention_evaluation_missing")
    return {
        "schema_version": 1,
        "model_family": model_family,
        "record_count": len(values),
        "records_sha256": canonical_digest(
            [
                {
                    "sample_id": record.sample_id,
                    "image_sha256": record.image_sha256,
                    "mask_sha256": record.mask_sha256,
                    "split": record.split,
                    "split_group": record.split_group,
                    "source_revision": record.source_revision,
                }
                for record in sorted(values, key=lambda item: item.sample_id)
            ]
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "mask_records": mask_count,
        "anchor_records": anchor_count,
        "visual_abstention_records": abstention_count,
        "split_group_leakage": len(leaking_groups),
        "training_errors": errors,
        "promotion_errors": promotion_errors,
        "training_ready": not errors,
        "promotion_ready": not promotion_errors,
    }


def write_preflight_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "preflight-report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
