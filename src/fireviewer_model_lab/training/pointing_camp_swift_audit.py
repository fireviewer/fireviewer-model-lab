"""Strictly audit Camp Swift EO/thermal-derived masks for the pointing corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

SOURCE_ID = "Camp Swift Fire Experiment 2014"
SOURCE_REVISION = "RDS-2018-0046+RDS-2018-0047"
HF_REPOSITORY = "fireviewer/dinov3-multitask-fireviewer-v3-dataset"
HF_REVISION = "06dad028c4e65fde36f36bb3a97c6fec766a270d"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _safe_local(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative.replace("\\", "/"))
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe Camp Swift path: {relative}")
    root = root.resolve()
    path = (root / Path(*posix.parts)).resolve()
    if root not in path.parents:
        raise ValueError(f"Camp Swift path escapes source root: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    if _sha256(destination) != expected_sha256:
        raise ValueError(f"Camp Swift strict payload copy mismatch: {destination.name}")


def _materialize_strict_payloads(
    rows: list[dict[str, Any]],
    *,
    source_root: Path,
    output_dir: Path,
    output_s3_prefix: str | None,
) -> int:
    copied_artifacts = 0
    for row in rows:
        artifacts = (
            (
                "image_relpath",
                "source_image_relpath",
                "image_sha256",
                "images",
                "image_s3_uri",
            ),
            ("mask_relpath", None, "mask_sha256", "masks", "mask_s3_uri"),
            (
                "valid_mask_relpath",
                None,
                "valid_mask_sha256",
                "valid-masks",
                "valid_mask_s3_uri",
            ),
        )
        for path_field, alias_field, sha_field, payload_kind, uri_field in artifacts:
            source = _safe_local(source_root, str(row[path_field]))
            digest = str(row[sha_field])
            suffix = source.suffix.casefold() or ".bin"
            relative = f"strict-payload/{payload_kind}/{digest}{suffix}"
            destination = output_dir / Path(*PurePosixPath(relative).parts)
            _copy_verified(source, destination, digest)
            row[path_field] = relative
            if alias_field is not None:
                row[alias_field] = relative
            if output_s3_prefix:
                row[uri_field] = f"{output_s3_prefix.rstrip('/')}/{relative}"
            copied_artifacts += 1
    return copied_artifacts


def _difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.tobytes())
    bits = 0
    for y_value in range(8):
        offset = y_value * 9
        for x_value in range(8):
            bits = (bits << 1) | int(pixels[offset + x_value] > pixels[offset + x_value + 1])
    return f"{bits:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _nearest(signature: str, others: list[tuple[str, str]]) -> tuple[int | None, str | None]:
    if not others:
        return None, None
    distance, sample_id = min((_hamming(signature, value), key) for value, key in others)
    return distance, sample_id


def _metrics(image: Image.Image) -> dict[str, float]:
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    stat = ImageStat.Stat(gray)
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
    return {
        "brightness": float(stat.mean[0] / 255.0),
        "contrast": float(stat.stddev[0] / 255.0),
        "edge_energy": float(edges.mean[0] / 255.0),
    }


def _base_point(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    bottom = int(ys.max())
    band_height = max(1, round(mask.shape[0] * 0.02))
    band_x = xs[ys >= max(0, bottom - band_height)]
    return (
        round(float(np.median(band_x)) / max(1, mask.shape[1] - 1), 8),
        round(float(bottom) / max(1, mask.shape[0] - 1), 8),
    )


def _load_baseline(root: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    manifests = list(root.rglob("strict_combined_manifest.jsonl"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"expected one strict combined manifest, found {len(manifests)}")
    sha_rows: dict[str, str] = {}
    dhash_rows: list[tuple[str, str]] = []
    for row in _read_jsonl(manifests[0]):
        sample_id = str(row["sample_id"])
        digest = str(row.get("image_sha256") or row.get("source_image_sha256") or "")
        signature = str(row.get("dhash") or "")
        if len(digest) == 64:
            sha_rows[digest] = sample_id
        if len(signature) == 16:
            dhash_rows.append((signature, sample_id))
    return sha_rows, dhash_rows


def _cross_split_near_exclusions(
    rows: list[dict[str, Any]], *, maximum_distance: int = 4
) -> tuple[list[dict[str, Any]], set[str]]:
    eligible = [row for row in rows if not row["exclusion_reasons"]]
    adjacency: dict[str, set[str]] = defaultdict(set)
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = _hamming(str(left["dhash"]), str(right["dhash"]))
            if distance > maximum_distance:
                continue
            left_id = str(left["sample_id"])
            right_id = str(right["sample_id"])
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)
            pairs.append(
                {
                    "left": left_id,
                    "left_split": str(left["split"]),
                    "right": right_id,
                    "right_split": str(right["split"]),
                    "dhash_distance": distance,
                }
            )
    split_priority = {"test": 0, "validation": 1, "train": 2}
    row_by_id = {str(row["sample_id"]): row for row in eligible}
    excluded: set[str] = set()
    visited: set[str] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        component: set[str] = set()
        stack = [start]
        while stack:
            sample_id = stack.pop()
            if sample_id in component:
                continue
            component.add(sample_id)
            stack.extend(adjacency[sample_id])
        visited.update(component)
        keep = min(
            component,
            key=lambda sample_id: (
                split_priority.get(str(row_by_id[sample_id]["split"]), 99),
                sample_id,
            ),
        )
        excluded.update(component - {keep})
    return pairs, excluded


def audit_camp_swift(
    *,
    source_root: Path,
    baseline_root: Path,
    output_dir: Path,
    output_s3_prefix: str | None = None,
) -> dict[str, Any]:
    manifest_path = source_root / "candidate_manifest.jsonl"
    source_rows = _read_jsonl(manifest_path)
    baseline_sha, baseline_dhash = _load_baseline(baseline_root)
    candidates: list[dict[str, Any]] = []
    decode_errors: list[str] = []
    for source in source_rows:
        sample_id = str(source.get("sample_id") or "")
        reasons: list[str] = []
        if source.get("source_id") != SOURCE_ID:
            reasons.append("unexpected_source_id")
        if source.get("source_revision") != SOURCE_REVISION:
            reasons.append("unexpected_source_revision")
        if source.get("source_repository") != HF_REPOSITORY:
            reasons.append("unexpected_source_repository")
        if source.get("source_repository_revision") != HF_REVISION:
            reasons.append("unexpected_source_repository_revision")
        if source.get("license") != "CC-BY-4.0" or source.get("redistribution_allowed") is not True:
            reasons.append("unknown_or_incompatible_rights")
        if source.get("sample_validation_status") != "sensor_derived":
            reasons.append("unexpected_source_validation_status")
        if source.get("annotation_strength") != "strong":
            reasons.append("non_strong_sensor_annotation")
        if source.get("mask_quality") != "sensor_derived_thermal_reprojection":
            reasons.append("unexpected_mask_quality")
        if source.get("mask_semantics") != "thermal_hot_fire_core":
            reasons.append("unexpected_mask_semantics")
        if int(source.get("pair_delta_ms") or 0) > 2000:
            reasons.append("eo_ir_pair_delta_exceeds_2000ms")
        try:
            image_path = _safe_local(source_root, str(source["image_relpath"]))
            mask_path = _safe_local(source_root, str(source["mask_relpath"]))
            valid_path = _safe_local(source_root, str(source["valid_mask_relpath"]))
            image_sha = _sha256(image_path)
            mask_sha = _sha256(mask_path)
            valid_sha = _sha256(valid_path)
            with Image.open(image_path) as opened:
                opened.load()
                image = opened.convert("RGB")
            with Image.open(mask_path) as opened_mask:
                opened_mask.load()
                mask = opened_mask.convert("L")
            with Image.open(valid_path) as opened_valid:
                opened_valid.load()
                valid = opened_valid.convert("L")
        except Exception as exc:
            decode_errors.append(f"{sample_id}:{type(exc).__name__}:{exc}")
            continue
        if image_sha != str(source.get("image_sha256") or ""):
            reasons.append("source_image_sha256_mismatch")
        if mask_sha != str(source.get("mask_sha256") or ""):
            reasons.append("mask_sha256_mismatch")
        if valid_sha != str(source.get("valid_mask_sha256") or ""):
            reasons.append("valid_mask_sha256_mismatch")
        if image.size != mask.size or image.size != valid.size:
            reasons.append("image_mask_dimension_mismatch")
        mask_array = np.asarray(mask, dtype=np.uint8)
        valid_array = np.asarray(valid, dtype=np.uint8)
        mask_values = sorted(int(value) for value in np.unique(mask_array))
        valid_values = sorted(int(value) for value in np.unique(valid_array))
        if not set(mask_values).issubset({0, 255}):
            reasons.append("unknown_mask_value")
        if not set(valid_values).issubset({0, 255}):
            reasons.append("unknown_valid_mask_value")
        fire_mask = (mask_array > 0) & (valid_array > 0)
        invalid_fire_pixels = int(np.count_nonzero((mask_array > 0) & (valid_array == 0)))
        if invalid_fire_pixels:
            reasons.append("fire_mask_outside_valid_reprojection")
        point = _base_point(fire_mask)
        if point is None:
            reasons.append("empty_thermal_fire_mask")
        dhash = _difference_hash(image)
        exact_match = baseline_sha.get(image_sha)
        baseline_distance, baseline_nearest = _nearest(dhash, baseline_dhash)
        if exact_match is not None:
            reasons.append("baseline_exact_sha_overlap")
        if baseline_distance is not None and baseline_distance <= 4:
            reasons.append("baseline_perceptual_overlap")
        anchor_points = (
            [
                {
                    "kind": "fire_base",
                    "x": point[0],
                    "y": point[1],
                    "origin": "sensor_mask_bottom_band_median",
                }
            ]
            if point is not None
            else []
        )
        candidates.append(
            {
                **source,
                "schema_version": 1,
                "source_family": "Camp Swift EO/IR thermal fire",
                "source_validation_status": "sensor_derived",
                "sample_validation_status": "pending_strict_automated_validation",
                "annotation_strength": "sensor_derived_strong",
                "annotation_provenance": "sensor_derived_thermal_reprojection",
                "anchor_points": anchor_points,
                "mask_to_point_conversion": "deterministic_thermal_mask_bottom_band_median",
                "point_derivation": "sensor_mask_bottom_band_median",
                "segmentation_supervised": True,
                "point_supervised": True,
                "presence_supervised": True,
                "abstention_supervised": True,
                "presence_targets": {
                    "flame_visible": True,
                    "smoke_visible": False,
                },
                "presence_provenance": "sensor_derived_thermal_hot_fire_core",
                "visual_abstention_reason": None,
                "mask_values": mask_values,
                "valid_mask_values": valid_values,
                "mask_nonzero_fraction": float(np.count_nonzero(fire_mask) / fire_mask.size),
                "image_sha256": image_sha,
                "mask_sha256": mask_sha,
                "valid_mask_sha256": valid_sha,
                "dhash": dhash,
                "baseline_exact_sha_match": exact_match,
                "baseline_nearest_dhash_distance": baseline_distance,
                "baseline_nearest_sample_id": baseline_nearest,
                "validation_profile": "fireviewer_pointing_strict_automated_v1",
                "reviews_admitted": False,
                "exclusion_reasons": sorted(set(reasons)),
                "strict_keep": False,
                "training_eligible": False,
                **_metrics(image),
            }
        )

    own_sha: dict[str, list[str]] = defaultdict(list)
    own_dhash: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        own_sha[str(row["image_sha256"])].append(str(row["sample_id"]))
        own_dhash[str(row["dhash"])].append(str(row["sample_id"]))
    exact_groups = {key: sorted(value) for key, value in own_sha.items() if len(value) > 1}
    dhash_groups = {key: sorted(value) for key, value in own_dhash.items() if len(value) > 1}
    cross_split_near_pairs, cross_split_near_excluded = _cross_split_near_exclusions(candidates)
    for row in candidates:
        reasons = list(row["exclusion_reasons"])
        digest = str(row["image_sha256"])
        signature = str(row["dhash"])
        if digest in exact_groups and str(row["sample_id"]) != exact_groups[digest][0]:
            reasons.append("within_source_exact_sha_duplicate")
        if signature in dhash_groups and str(row["sample_id"]) != dhash_groups[signature][0]:
            reasons.append("within_source_identical_dhash_neighbor")
        if str(row["sample_id"]) in cross_split_near_excluded:
            reasons.append("within_source_near_cross_split_duplicate")
        row["exclusion_reasons"] = sorted(set(reasons))
        row["strict_keep"] = not reasons
        row["training_eligible"] = not reasons
        row["sample_validation_status"] = (
            "strict_automated_validated" if not reasons else "excluded_unvalidated"
        )
        row["corpus_disposition"] = (
            "eligible_genuinely_new_pool" if not reasons else "excluded_unvalidated"
        )
    validated = [row for row in candidates if row["strict_keep"]]
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in validated:
        group_splits[str(row["split_group"])].add(str(row["split"]))
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"Camp Swift split-group leakage: {leaking_groups}")
    gate_errors: list[str] = []
    if decode_errors:
        gate_errors.append(f"decode_or_payload_errors:{len(decode_errors)}")
    if not validated:
        gate_errors.append("no_strict_automated_validated_rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_artifacts = _materialize_strict_payloads(
        validated,
        source_root=source_root,
        output_dir=output_dir,
        output_s3_prefix=output_s3_prefix,
    )
    _write_jsonl(output_dir / "camp_swift_automatic_dispositions.jsonl", candidates)
    _write_jsonl(output_dir / "camp_swift_strict_validated_manifest.jsonl", validated)
    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_repository": HF_REPOSITORY,
        "source_repository_revision": HF_REVISION,
        "rows_evaluated": len(candidates),
        "decode_or_payload_errors": decode_errors,
        "strict_automated_validated_rows": len(validated),
        "strict_payload_artifacts_materialized": payload_artifacts,
        "validated_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in validated).items())
        ),
        "fire_base_points": sum(bool(row["anchor_points"]) for row in validated),
        "exclusions_by_reason": dict(
            sorted(
                Counter(reason for row in candidates for reason in row["exclusion_reasons"]).items()
            )
        ),
        "baseline_images_compared": len(baseline_sha),
        "within_source_exact_sha_groups": len(exact_groups),
        "within_source_identical_dhash_groups": len(dhash_groups),
        "near_cross_split_pairs": cross_split_near_pairs,
        "near_cross_split_rows_excluded": len(cross_split_near_excluded),
        "split_group_leakage": leaking_groups,
        "gate_errors": gate_errors,
        "source_gate_passed": not gate_errors,
        "reviews_admitted": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "publication_allowed": False,
        "next_action": "merge_strict_sources_then_repeat_global_quality_gates",
    }
    (output_dir / "camp_swift_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-s3-prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_camp_swift(
        source_root=args.source_root,
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        output_s3_prefix=args.output_s3_prefix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
