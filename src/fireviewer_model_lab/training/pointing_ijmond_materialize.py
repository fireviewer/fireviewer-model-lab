"""Materialize only the payloads admitted by a successful IJmond strict audit.

This is deliberately a second, fail-closed pass.  It binds the audit summary,
the two audit manifests, the immutable source archive, and all three exclusion
indexes before extracting any training payload.  Points are never inferred in
this step: source geometry is recomputed only to prove that the audited point
is reproducible.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

try:
    from fireviewer_model_lab.training.pointing_ijmond_audit import (
        ARCHIVE_FILENAME,
        MASK_VALUES,
        MAX_PHASH_DISTANCE,
        PROFESSIONAL_SOURCE_POINT_CAP,
        SOURCE_DOI,
        SOURCE_FAMILY,
        SOURCE_ID,
        SOURCE_LICENSE,
        SOURCE_LICENSE_URL,
        _hamming,
        _load_exclusion_index,
        _member_payload,
        _normalized_member_name,
        _perceptual_hash,
        _sha256_bytes,
        _sha256_file,
        _validated_members,
        _visible_source_point,
        download_archive,
    )
except ModuleNotFoundError:  # Standalone SageMaker code input.
    from pointing_ijmond_audit import (  # type: ignore[no-redef]
        ARCHIVE_FILENAME,
        MASK_VALUES,
        MAX_PHASH_DISTANCE,
        PROFESSIONAL_SOURCE_POINT_CAP,
        SOURCE_DOI,
        SOURCE_FAMILY,
        SOURCE_ID,
        SOURCE_LICENSE,
        SOURCE_LICENSE_URL,
        _hamming,
        _load_exclusion_index,
        _member_payload,
        _normalized_member_name,
        _perceptual_hash,
        _sha256_bytes,
        _sha256_file,
        _validated_members,
        _visible_source_point,
        download_archive,
    )

ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
EXPECTED_ANNOTATION_PROVENANCE = (
    "roboflow_sam_prompted_polygon_manually_refined_jointly_and_all_masks_"
    "manually_checked_edited_by_source_author"
)
EXPECTED_POINT_ORIGIN = "human_revised_mask_connected_visible_source_gate_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path.name}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path.name}:{line_number}")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _require_sha(value: str, *, name: str) -> str:
    normalized = value.casefold()
    if not HEX64.fullmatch(normalized):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return normalized


def _bind_file(path: Path, expected_sha256: str, *, name: str) -> str:
    expected = _require_sha(expected_sha256, name=name)
    if not path.is_file():
        raise ValueError(f"missing {name}: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")
    return actual


def _difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"


def _nearest_phash(signature: str, index: dict[str, Any]) -> tuple[int | None, str | None]:
    best_distance: int | None = None
    best_sample: str | None = None
    for candidate, sample_id in index["phash"]:
        distance = _hamming(signature, candidate)
        if best_distance is None or (distance, sample_id) < (best_distance, best_sample or ""):
            best_distance = distance
            best_sample = sample_id
    return best_distance, best_sample


def _validate_audit_contract(
    summary: dict[str, Any],
    strict_rows: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    *,
    archive_sha256: str,
    indexes: dict[str, dict[str, Any]],
    point_cap: int,
) -> None:
    if summary.get("source_gate_passed") is not True or summary.get("gate_errors") != []:
        raise ValueError("IJmond audit source gate did not pass cleanly")
    expected_identity = {
        "source_id": SOURCE_ID,
        "source_family": SOURCE_FAMILY,
        "source_revision": SOURCE_DOI,
        "declared_license": SOURCE_LICENSE,
        "license_url": SOURCE_LICENSE_URL,
    }
    for field, expected in expected_identity.items():
        if summary.get(field) != expected:
            raise ValueError(f"audit summary {field} does not match the pinned IJmond contract")
    source_receipt = summary.get("figshare_contract_receipt")
    if (
        not isinstance(source_receipt, dict)
        or source_receipt.get("source_contract_verified") is not True
    ):
        raise ValueError("audit summary lacks a successful Figshare source-contract receipt")
    archive_receipt = summary.get("archive_receipt")
    if not isinstance(archive_receipt, dict) or archive_receipt.get("sha256") != archive_sha256:
        raise ValueError("audit summary is not bound to the supplied source archive")
    if summary.get("reviews_admitted") is not False:
        raise ValueError("audit admitted review-derived rows")
    if summary.get("publication_allowed") is not False:
        raise ValueError("audit unexpectedly authorized publication")
    if summary.get("benchmark_hash_exclusion_only") is not True:
        raise ValueError("audit did not use the benchmark as a hash-only exclusion index")
    if summary.get("strict_payload_artifacts_materialized") != 0:
        raise ValueError("audit unexpectedly materialized payloads")
    if summary.get("strict_automated_validated_rows") != len(strict_rows):
        raise ValueError("strict audit row count disagrees with the bound manifest")
    if summary.get("smoke_column_base_points") != len(point_rows):
        raise ValueError("point audit row count disagrees with the bound point manifest")
    reported_cap = summary.get("professional_source_point_cap")
    if isinstance(reported_cap, bool) or not isinstance(reported_cap, int):
        raise ValueError("audit summary point cap is missing or invalid")
    if reported_cap > PROFESSIONAL_SOURCE_POINT_CAP or reported_cap > point_cap:
        raise ValueError("audited IJmond point cap exceeds the materialization ceiling")
    if len(point_rows) > point_cap:
        raise ValueError("audited IJmond point rows exceed the materialization ceiling")

    reported_indexes = summary.get("external_exclusion_indexes")
    if not isinstance(reported_indexes, dict) or set(reported_indexes) != set(indexes):
        raise ValueError("audit summary does not bind all three exclusion indexes")
    for name, index in indexes.items():
        reported = reported_indexes.get(name)
        if not isinstance(reported, dict):
            raise ValueError(f"audit summary exclusion binding is missing: {name}")
        if reported.get("receipt_sha256") != index["receipt_sha256"]:
            raise ValueError(f"audit summary exclusion receipt drifted: {name}")
        if reported.get("indexed_rows") != index["indexed_rows"]:
            raise ValueError(f"audit summary exclusion row count drifted: {name}")


def _validate_row_metadata(row: dict[str, Any], *, point_ids: set[str]) -> None:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("strict audit row lacks sample_id")
    expected = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_family": SOURCE_FAMILY,
        "source_revision": SOURCE_DOI,
        "license": SOURCE_LICENSE,
        "license_url": SOURCE_LICENSE_URL,
        "redistribution_allowed": True,
        "reviews_admitted": False,
        "strict_keep": True,
        "training_eligible": True,
        "segmentation_supervised": True,
        "presence_supervised": True,
        "abstention_supervised": True,
        "annotation_strength": "strong_human_revised",
        "annotation_provenance": EXPECTED_ANNOTATION_PROVENANCE,
        "mask_quality": "source_human_revised_all_masks_checked_and_edited",
        "mask_semantics": (
            "background_0_low_opacity_smoke_155_high_opacity_smoke_255"
        ),
        "presence_provenance": "source_human_revised_smoke_mask_only",
    }
    for field, wanted in expected.items():
        if row.get(field) != wanted:
            raise ValueError(f"{sample_id}: invalid strict field {field}")
    if row.get("exclusion_reasons") != []:
        raise ValueError(f"{sample_id}: strict row contains exclusion reasons")
    if row.get("publication_allowed") is not False:
        raise ValueError(f"{sample_id}: audit row unexpectedly authorizes publication")
    if row.get("benchmark_hash_exclusion_only") is not True:
        raise ValueError(f"{sample_id}: benchmark exclusion proof is missing")
    if not HEX64.fullmatch(str(row.get("source_archive_sha256") or "")):
        raise ValueError(f"{sample_id}: source archive provenance is invalid")
    split = row.get("split")
    if split not in ALLOWED_SPLITS or row.get("final_split") != split:
        raise ValueError(f"{sample_id}: invalid or inconsistent split")
    if not isinstance(row.get("split_group"), str) or not row["split_group"]:
        raise ValueError(f"{sample_id}: grouped split identity is missing")
    for field in ("objects", "boxes", "bbox", "bboxes"):
        if row.get(field):
            raise ValueError(f"{sample_id}: box-derived labels are forbidden")
    points = row.get("anchor_points")
    if not isinstance(points, list):
        raise ValueError(f"{sample_id}: anchor_points is not a list")
    is_point = sample_id in point_ids
    if row.get("point_supervised") is not is_point:
        raise ValueError(f"{sample_id}: point manifest and point_supervised disagree")
    if is_point:
        if len(points) != 1:
            raise ValueError(f"{sample_id}: admitted point row must contain exactly one point")
        point = points[0]
        if not isinstance(point, dict):
            raise ValueError(f"{sample_id}: admitted point is not an object")
        if point.get("kind") != "smoke_column_base" or point.get("origin") != EXPECTED_POINT_ORIGIN:
            raise ValueError(f"{sample_id}: point semantics or provenance are invalid")
        if row.get("point_derivation") != EXPECTED_POINT_ORIGIN:
            raise ValueError(f"{sample_id}: point derivation is not native-mask contracted")
        for axis in ("x", "y"):
            value = point.get(axis)
            if isinstance(value, bool):
                raise ValueError(f"{sample_id}: boolean point coordinate is forbidden")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{sample_id}: invalid point coordinate") from exc
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{sample_id}: point coordinate is outside image bounds")
    elif points:
        raise ValueError(f"{sample_id}: auxiliary row contains an uncontracted point")


def _validate_point_manifest(
    strict_rows: list[dict[str, Any]], point_rows: list[dict[str, Any]]
) -> set[str]:
    strict_by_id = {str(row.get("sample_id")): row for row in strict_rows}
    if len(strict_by_id) != len(strict_rows):
        raise ValueError("strict audit manifest contains duplicate sample_id values")
    point_ids: set[str] = set()
    for point_row in point_rows:
        sample_id = str(point_row.get("sample_id") or "")
        if not sample_id or sample_id in point_ids:
            raise ValueError("point audit manifest contains missing or duplicate sample_id values")
        strict_row = strict_by_id.get(sample_id)
        if strict_row is None or strict_row != point_row:
            raise ValueError(f"point audit row is not an exact strict-manifest subset: {sample_id}")
        point_ids.add(sample_id)
    declared = {
        sample_id
        for sample_id, row in strict_by_id.items()
        if row.get("point_supervised") is True
    }
    if declared != point_ids:
        raise ValueError("point audit manifest does not equal the strict point-supervised subset")
    return point_ids


def _verify_external_exclusions(
    row: dict[str, Any], indexes: dict[str, dict[str, Any]]
) -> None:
    sample_id = str(row["sample_id"])
    image_sha = str(row.get("source_image_sha256") or "").casefold()
    phash = str(row.get("phash") or "").casefold()
    if not HEX64.fullmatch(image_sha) or not re.fullmatch(r"[0-9a-f]{16}", phash):
        raise ValueError(f"{sample_id}: image SHA-256 or pHash is invalid")
    stored = row.get("external_exclusion_matches")
    if not isinstance(stored, dict) or set(stored) != set(indexes):
        raise ValueError(f"{sample_id}: stored exclusion checks are incomplete")
    for name, index in indexes.items():
        exact = index["sha"].get(image_sha)
        distance, nearest = _nearest_phash(phash, index)
        if exact is not None or (distance is not None and distance <= MAX_PHASH_DISTANCE):
            raise ValueError(f"{sample_id}: newly detected {name} corpus overlap")
        recorded = stored.get(name)
        if not isinstance(recorded, dict):
            raise ValueError(f"{sample_id}: stored {name} exclusion check is invalid")
        if recorded.get("exact_sha_sample_id") is not None:
            raise ValueError(f"{sample_id}: audit retained an exact {name} overlap")
        if recorded.get("nearest_phash_sample_id") != nearest:
            raise ValueError(f"{sample_id}: nearest {name} exclusion identity drifted")
        if recorded.get("nearest_phash_distance") != distance:
            raise ValueError(f"{sample_id}: nearest {name} exclusion distance drifted")


def _payload_row(
    row: dict[str, Any],
    *,
    image_relpath: str,
    mask_relpath: str,
    dhash: str,
) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "image_relpath": image_relpath,
            "mask_relpath": mask_relpath,
            "dhash": dhash,
            "media_license": SOURCE_LICENSE,
            "mask_license": SOURCE_LICENSE,
            "viewpoint": "ground_or_oblique",
            "variant": "clean",
            "sample_weight": 1.0,
            "negative": False,
        }
    )
    if result["point_supervised"]:
        result.update(
            {
                "sample_validation_status": "strict_automated_validated",
                "corpus_disposition": "eligible_genuinely_new_pool",
                "mask_to_point_conversion": EXPECTED_POINT_ORIGIN,
                "visual_abstention_reason": None,
            }
        )
    return result


def materialize_ijmond_archive(
    *,
    archive_path: Path,
    audit_summary_path: Path,
    audit_strict_manifest_path: Path,
    audit_point_manifest_path: Path,
    expected_archive_sha256: str,
    expected_audit_summary_sha256: str,
    expected_audit_strict_manifest_sha256: str,
    expected_audit_point_manifest_sha256: str,
    detection_index_root: Path,
    pointing_index_root: Path,
    benchmark_index_root: Path,
    expected_index_receipts: dict[str, dict[str, Any]],
    output_dir: Path,
    point_cap: int = PROFESSIONAL_SOURCE_POINT_CAP,
    output_s3_prefix: str | None = None,
) -> dict[str, Any]:
    """Extract payloads only after all immutable audit and exclusion gates pass."""
    if point_cap < 0 or point_cap > PROFESSIONAL_SOURCE_POINT_CAP:
        raise ValueError(f"point_cap must be between 0 and {PROFESSIONAL_SOURCE_POINT_CAP}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"materialization output directory is not empty: {output_dir}")

    archive_sha = _bind_file(archive_path, expected_archive_sha256, name="source archive")
    summary_sha = _bind_file(
        audit_summary_path, expected_audit_summary_sha256, name="audit summary"
    )
    strict_manifest_sha = _bind_file(
        audit_strict_manifest_path,
        expected_audit_strict_manifest_sha256,
        name="strict audit manifest",
    )
    point_manifest_sha = _bind_file(
        audit_point_manifest_path,
        expected_audit_point_manifest_sha256,
        name="point audit manifest",
    )
    if set(expected_index_receipts) != {"detection", "pointing", "benchmark"}:
        raise ValueError("exactly three pinned exclusion-index receipts are required")
    roots = {
        "detection": detection_index_root,
        "pointing": pointing_index_root,
        "benchmark": benchmark_index_root,
    }
    indexes = {
        name: _load_exclusion_index(
            roots[name],
            name,
            expected_receipt_sha256=str(expected_index_receipts[name]["receipt_sha256"]),
            expected_rows=int(expected_index_receipts[name]["rows"]),
        )
        for name in ("detection", "pointing", "benchmark")
    }
    index_errors = sorted(
        {error for index in indexes.values() for error in index.get("gate_errors", [])}
    )
    if index_errors:
        raise ValueError(f"exclusion indexes failed validation: {index_errors}")

    summary = _read_json(audit_summary_path)
    strict_rows = _read_jsonl(audit_strict_manifest_path)
    point_rows = _read_jsonl(audit_point_manifest_path)
    point_ids = _validate_point_manifest(strict_rows, point_rows)
    _validate_audit_contract(
        summary,
        strict_rows,
        point_rows,
        archive_sha256=archive_sha,
        indexes=indexes,
        point_cap=point_cap,
    )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in strict_rows:
        _validate_row_metadata(row, point_ids=point_ids)
        _verify_external_exclusions(row, indexes)
        group_splits[str(row["split_group"])].add(str(row["split"]))
    leaking = sorted(group for group, splits in group_splits.items() if len(splits) != 1)
    if leaking:
        raise ValueError(f"grouped split leakage detected: {leaking[:10]}")

    staging_dir = output_dir.with_name(f".{output_dir.name}.staging")
    if staging_dir.exists():
        raise ValueError(f"stale materialization staging directory exists: {staging_dir}")
    materialized: list[dict[str, Any]] = []
    seen_image_members: set[str] = set()
    seen_mask_members: set[str] = set()
    staging_dir.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = _validated_members(archive)
            for row in sorted(strict_rows, key=lambda item: str(item["sample_id"])):
                sample_id = str(row["sample_id"])
                if row.get("source_archive_sha256") != archive_sha:
                    raise ValueError(f"{sample_id}: row references a different source archive")
                image_name = _normalized_member_name(str(row.get("source_archive_member") or ""))
                mask_name = _normalized_member_name(str(row.get("mask_archive_member") or ""))
                image_member = members.get(image_name.casefold())
                mask_member = members.get(mask_name.casefold())
                if image_member is None or mask_member is None:
                    raise ValueError(f"{sample_id}: audited source member is missing from archive")
                if image_name in seen_image_members or mask_name in seen_mask_members:
                    raise ValueError(
                        f"{sample_id}: archive payload member is reused by multiple rows"
                    )
                seen_image_members.add(image_name)
                seen_mask_members.add(mask_name)
                image_payload = _member_payload(archive, image_member)
                mask_payload = _member_payload(archive, mask_member)
                image_sha = _sha256_bytes(image_payload)
                mask_sha = _sha256_bytes(mask_payload)
                if (
                    image_sha != row.get("source_image_sha256")
                    or image_sha != row.get("image_sha256")
                ):
                    raise ValueError(f"{sample_id}: source image payload SHA-256 drifted")
                if mask_sha != row.get("mask_sha256"):
                    raise ValueError(f"{sample_id}: source mask payload SHA-256 drifted")
                with Image.open(io.BytesIO(image_payload)) as opened:
                    opened.load()
                    if opened.mode != "RGB":
                        raise ValueError(f"{sample_id}: source image is not RGB")
                    image = opened.copy()
                with Image.open(io.BytesIO(mask_payload)) as opened_mask:
                    opened_mask.load()
                    raw_mask = np.asarray(opened_mask)
                    mask_size = opened_mask.size
                if raw_mask.ndim == 3 and all(
                    np.array_equal(raw_mask[..., 0], raw_mask[..., channel])
                    for channel in range(1, raw_mask.shape[2])
                ):
                    raw_mask = raw_mask[..., 0]
                if raw_mask.ndim != 2 or image.size != mask_size:
                    raise ValueError(f"{sample_id}: source image/mask geometry is invalid")
                if set(int(value) for value in np.unique(raw_mask)) - MASK_VALUES:
                    raise ValueError(f"{sample_id}: source mask values drifted")
                phash = _perceptual_hash(image)
                if phash != row.get("phash"):
                    raise ValueError(f"{sample_id}: source image pHash drifted")
                reproduced_point, _, _ = _visible_source_point(image, raw_mask > 0)
                if row["point_supervised"]:
                    if reproduced_point != row["anchor_points"][0]:
                        raise ValueError(f"{sample_id}: audited point is not geometry-reproducible")
                elif (
                    reproduced_point is not None
                    and row.get("point_cap_status") != "demoted_to_abstention"
                ):
                    raise ValueError(f"{sample_id}: an unrecorded reproducible point was omitted")

                suffix = PurePosixPath(image_name).suffix.casefold()
                mask_suffix = PurePosixPath(mask_name).suffix.casefold()
                split = str(row["split"])
                image_relpath = f"images/{split}/{image_sha}{suffix}"
                mask_relpath = f"masks/{split}/{image_sha}{mask_suffix}"
                image_path = staging_dir / Path(*PurePosixPath(image_relpath).parts)
                mask_path = staging_dir / Path(*PurePosixPath(mask_relpath).parts)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_payload)
                mask_path.write_bytes(mask_payload)
                materialized.append(
                    _payload_row(
                        row,
                        image_relpath=image_relpath,
                        mask_relpath=mask_relpath,
                        dhash=_difference_hash(image),
                    )
                )

        materialized.sort(key=lambda item: str(item["sample_id"]))
        strict_points = [row for row in materialized if row["point_supervised"]]
        materialized_path = staging_dir / "ijmond_materialized_manifest.jsonl"
        strict_path = staging_dir / "ijmond_strict_validated_manifest.jsonl"
        _write_jsonl(materialized_path, materialized)
        _write_jsonl(strict_path, strict_points)
        receipt = {
            "schema_version": 1,
            "source_id": SOURCE_ID,
            "source_family": SOURCE_FAMILY,
            "source_revision": SOURCE_DOI,
            "source_archive_sha256": archive_sha,
            "audit_summary_sha256": summary_sha,
            "audit_strict_manifest_sha256": strict_manifest_sha,
            "audit_point_manifest_sha256": point_manifest_sha,
            "exclusion_receipts": {
                name: {
                    "receipt_sha256": index["receipt_sha256"],
                    "index_sha256": index["receipt"].get("index_sha256"),
                    "indexed_rows": index["indexed_rows"],
                }
                for name, index in indexes.items()
            },
            "materialized_rows": len(materialized),
            "materialized_point_rows": len(strict_points),
            "materialized_auxiliary_rows": len(materialized) - len(strict_points),
            "materialized_payload_files": len(materialized) * 2,
            "split_counts": dict(
                sorted(Counter(str(row["split"]) for row in materialized).items())
            ),
            "point_split_counts": dict(
                sorted(Counter(str(row["split"]) for row in strict_points).items())
            ),
            "split_groups": len(group_splits),
            "split_group_leaks": 0,
            "professional_source_point_cap": point_cap,
            "materialized_manifest": materialized_path.name,
            "materialized_manifest_sha256": _sha256_file(materialized_path),
            "strict_point_manifest": strict_path.name,
            "strict_point_manifest_sha256": _sha256_file(strict_path),
            "annotation_provenance": EXPECTED_ANNOTATION_PROVENANCE,
            "point_origin": EXPECTED_POINT_ORIGIN,
            "declared_license": SOURCE_LICENSE,
            "license_url": SOURCE_LICENSE_URL,
            "redistribution_allowed": True,
            "reviews_admitted": False,
            "labels_generated_during_materialization": 0,
            "audited_native_geometry_derived_points": len(strict_points),
            "point_geometry_source": "source_human_revised_smoke_mask",
            "box_derived_labels": 0,
            "detection_corpus_used_for_training": False,
            "pointing_corpus_used_for_training": False,
            "independent_benchmark_used_for_training": False,
            "output_s3_prefix": output_s3_prefix,
            "materialization_gate_passed": True,
            "publication_allowed": False,
        }
        receipt_path = staging_dir / "ijmond_materialization_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            if any(output_dir.iterdir()):
                raise RuntimeError("output_dir must be empty before materialization commit")
            # SageMaker bind-mounts /opt/ml/processing/output, so the mount point
            # itself cannot be removed or atomically replaced. Commit each fully
            # staged top-level artifact into that empty mount instead.
            for child in staging_dir.iterdir():
                child.replace(output_dir / child.name)
            staging_dir.rmdir()
        else:
            staging_dir.replace(output_dir)
        return receipt
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    archive_source = parser.add_mutually_exclusive_group(required=True)
    archive_source.add_argument("--archive-path", type=Path)
    archive_source.add_argument("--download-pinned-archive", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--audit-strict-manifest", type=Path, required=True)
    parser.add_argument("--audit-point-manifest", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-audit-summary-sha256", required=True)
    parser.add_argument("--expected-audit-strict-manifest-sha256", required=True)
    parser.add_argument("--expected-audit-point-manifest-sha256", required=True)
    parser.add_argument("--detection-index-root", type=Path, required=True)
    parser.add_argument("--pointing-index-root", type=Path, required=True)
    parser.add_argument("--benchmark-index-root", type=Path, required=True)
    parser.add_argument("--detection-index-receipt-sha256", required=True)
    parser.add_argument("--pointing-index-receipt-sha256", required=True)
    parser.add_argument("--benchmark-index-receipt-sha256", required=True)
    parser.add_argument("--detection-index-rows", type=int, required=True)
    parser.add_argument("--pointing-index-rows", type=int, required=True)
    parser.add_argument("--benchmark-index-rows", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-cap", type=int, default=PROFESSIONAL_SOURCE_POINT_CAP)
    parser.add_argument("--output-s3-prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.download_pinned_archive:
        if args.work_dir is None:
            raise ValueError("--work-dir is required with --download-pinned-archive")
        args.work_dir.mkdir(parents=True, exist_ok=True)
        archive_path = args.work_dir / ARCHIVE_FILENAME
        download_archive(archive_path)
    else:
        archive_path = args.archive_path
        if archive_path is None:  # argparse enforces the mutually exclusive group.
            raise ValueError("an archive source is required")
    receipt = materialize_ijmond_archive(
        archive_path=archive_path,
        audit_summary_path=args.audit_summary,
        audit_strict_manifest_path=args.audit_strict_manifest,
        audit_point_manifest_path=args.audit_point_manifest,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_audit_summary_sha256=args.expected_audit_summary_sha256,
        expected_audit_strict_manifest_sha256=args.expected_audit_strict_manifest_sha256,
        expected_audit_point_manifest_sha256=args.expected_audit_point_manifest_sha256,
        detection_index_root=args.detection_index_root,
        pointing_index_root=args.pointing_index_root,
        benchmark_index_root=args.benchmark_index_root,
        expected_index_receipts={
            "detection": {
                "receipt_sha256": args.detection_index_receipt_sha256,
                "rows": args.detection_index_rows,
            },
            "pointing": {
                "receipt_sha256": args.pointing_index_receipt_sha256,
                "rows": args.pointing_index_rows,
            },
            "benchmark": {
                "receipt_sha256": args.benchmark_index_receipt_sha256,
                "rows": args.benchmark_index_rows,
            },
        },
        output_dir=args.output_dir,
        point_cap=args.point_cap,
        output_s3_prefix=args.output_s3_prefix,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
