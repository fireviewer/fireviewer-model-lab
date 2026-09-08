"""Audit FLAME2 RGB candidates against the isolated semantic-pointing corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

FRAME_PATTERN = re.compile(r"img_(?P<kind>rgb|ir|gt)_\((?P<frame>\d+)\)\.png$", re.I)
FRAME_ID_PATTERN = re.compile(r"\((?P<frame>\d+)\)\.png$", re.I)
MASK_CLASSES = {0: "background", 125: "smoke", 255: "fire"}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_base_point(mask: np.ndarray) -> tuple[float, float] | None:
    if mask.ndim != 2:
        raise ValueError("semantic mask must be two-dimensional")
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    bottom = int(ys.max())
    band_height = max(1, round(mask.shape[0] * 0.02))
    band_x = xs[ys >= max(0, bottom - band_height)]
    x = float(np.median(band_x)) / max(1, mask.shape[1] - 1)
    y = float(bottom) / max(1, mask.shape[0] - 1)
    return round(x, 8), round(y, 8)


def _frame_id(value: str) -> int:
    match = FRAME_ID_PATTERN.search(value.replace("\\", "/"))
    if match is None:
        raise ValueError(f"unrecognized FLAME2 path: {value}")
    return int(match.group("frame"))


def _listed_frames(path: Path) -> set[int]:
    return {
        _frame_id(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.tobytes())
    bits = 0
    for y_value in range(8):
        offset = y_value * 9
        for x_value in range(8):
            bits = (bits << 1) | int(pixels[offset + x_value] > pixels[offset + x_value + 1])
    return f"{bits:016x}"


def _metrics(image: Image.Image) -> dict[str, float]:
    gray = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    stat = ImageStat.Stat(gray)
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
    return {
        "brightness": float(stat.mean[0] / 255.0),
        "contrast": float(stat.stddev[0] / 255.0),
        "edge_energy": float(edges.mean[0] / 255.0),
    }


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _nearest(signature: str, others: list[tuple[str, str]]) -> tuple[int | None, str | None]:
    if not others:
        return None, None
    distance, sample_id = min((_hamming(signature, value), key) for value, key in others)
    return distance, sample_id


def _manifest_by_frame(rows: list[dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        relative_path = str(row["relative_path"])
        match = FRAME_PATTERN.search(relative_path.replace("\\", "/"))
        if match is None:
            continue
        frame = int(match.group("frame"))
        kind = match.group("kind").lower()
        if kind in grouped[frame]:
            raise ValueError(f"duplicate FLAME2 {kind} asset for frame {frame}")
        grouped[frame][kind] = row
    return grouped


def audit_flame2(
    *,
    source_root: Path,
    baseline_root: Path,
    output_dir: Path,
    source_s3_prefix: str,
) -> dict[str, Any]:
    manifest = _read_jsonl(source_root / "candidate_manifest.jsonl")
    frames = _manifest_by_frame(manifest)
    train_frames = _listed_frames(source_root / "source" / "lists" / "train_flm.txt")
    validation_frames = _listed_frames(source_root / "source" / "lists" / "val_flm.txt")
    test_frames = _listed_frames(source_root / "source" / "lists" / "test_flm.txt")
    overlap = (
        (train_frames & validation_frames)
        | (train_frames & test_frames)
        | (validation_frames & test_frames)
    )
    if overlap:
        raise ValueError(f"FLAME2 upstream split overlap: {len(overlap)} frames")

    baseline = _read_jsonl(baseline_root / "pixel_inventory.jsonl")
    failed_baseline = [row for row in baseline if row.get("status") != "ok"]
    if failed_baseline:
        raise ValueError(f"baseline pixel audit contains {len(failed_baseline)} failures")
    baseline_sha = {
        str(row["sha256"]): str(row["sample_id"]) for row in baseline if row.get("sha256")
    }
    baseline_dhash = [
        (str(row["dhash"]), str(row["sample_id"])) for row in baseline if row.get("dhash")
    ]

    candidates: list[dict[str, Any]] = []
    decode_errors: list[str] = []
    for frame, assets in sorted(frames.items()):
        missing = {"rgb", "ir", "gt"} - set(assets)
        if missing:
            decode_errors.append(f"frame:{frame}:missing_assets:{','.join(sorted(missing))}")
            continue
        rgb_row = assets["rgb"]
        mask_row = assets["gt"]
        rgb_path = source_root / "source" / str(rgb_row["relative_path"])
        mask_path = source_root / "source" / str(mask_row["relative_path"])
        try:
            with Image.open(rgb_path) as opened:
                opened.load()
                rgb = opened.convert("RGB")
            with Image.open(mask_path) as opened_mask:
                opened_mask.load()
                mask = opened_mask.convert("L")
        except Exception as exc:
            decode_errors.append(f"frame:{frame}:{type(exc).__name__}:{exc}")
            continue
        if rgb.size != mask.size:
            decode_errors.append(f"frame:{frame}:rgb_mask_dimension_mismatch")
            continue
        rgb_sha = _sha256(rgb_path)
        mask_sha = _sha256(mask_path)
        dhash = _difference_hash(rgb)
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_values = sorted(int(value) for value in np.unique(mask_array))
        invalid_mask_values = sorted(set(mask_values) - set(MASK_CLASSES))
        smoke_mask = mask_array == 125
        fire_mask = mask_array == 255
        smoke_pixels = int(np.count_nonzero(smoke_mask))
        fire_pixels = int(np.count_nonzero(fire_mask))
        mask_nonzero = smoke_pixels + fire_pixels
        anchor_points = []
        for kind, binary_mask in (
            ("smoke_column_base", smoke_mask),
            ("fire_base", fire_mask),
        ):
            point = _mask_base_point(binary_mask)
            if point is not None:
                anchor_points.append(
                    {
                        "kind": kind,
                        "x": point[0],
                        "y": point[1],
                        "origin": "source_mask_bottom_band_median",
                    }
                )
        if frame in train_frames:
            upstream_split = "train"
        elif frame in validation_frames:
            upstream_split = "validation"
        elif frame in test_frames:
            upstream_split = "test"
        else:
            upstream_split = "unlisted"
        exact_match = baseline_sha.get(str(rgb_row["sha256"]))
        baseline_distance, baseline_nearest = _nearest(dhash, baseline_dhash)
        candidate = {
            "schema_version": 1,
            "sample_id": f"flame2:{frame}",
            "frame_id": frame,
            "source_id": "robofirefusenet-flame2",
            "source_family": "FLAME2",
            "source_group": "flame2:source-sequence-unresolved",
            "split_group": "flame2:source-sequence-unresolved",
            "source_revision": str(rgb_row["source_revision"]),
            "upstream_split": upstream_split,
            "split": "train",
            "final_split": "train",
            "image_s3_uri": f"{source_s3_prefix.rstrip('/')}/source/{rgb_row['relative_path']}",
            "mask_s3_uri": f"{source_s3_prefix.rstrip('/')}/source/{mask_row['relative_path']}",
            "image_relpath": str(rgb_row["relative_path"]),
            "mask_relpath": str(mask_row["relative_path"]),
            "source_image_sha256": rgb_sha,
            "image_sha256": rgb_sha,
            "mask_sha256": mask_sha,
            "source_image_sha256_matches_manifest": rgb_sha == str(rgb_row["sha256"]),
            "mask_sha256_matches_manifest": mask_sha == str(mask_row["sha256"]),
            "width": rgb.width,
            "height": rgb.height,
            "dhash": dhash,
            "provided_mask_values": mask_values,
            "invalid_mask_values": invalid_mask_values,
            "provided_mask_nonzero_fraction": mask_nonzero / mask_array.size,
            "provided_mask_signal": mask_nonzero > 0,
            "smoke_pixels": smoke_pixels,
            "fire_pixels": fire_pixels,
            "provided_mask_role": "source_semantic_ground_truth",
            "mask_to_point_conversion": "deterministic_class_mask_bottom_band_median",
            "mask_class_mapping": MASK_CLASSES,
            "mask_quality": "source_provided_three_class",
            "anchor_points": anchor_points,
            "baseline_exact_sha_match": exact_match,
            "baseline_nearest_dhash_distance": baseline_distance,
            "baseline_nearest_sample_id": baseline_nearest,
            "exact_duplicate_rejected": exact_match is not None,
            "perceptual_similarity_excluded": (
                baseline_distance is not None and baseline_distance <= 4
            ),
            "point_annotation_status": "source_mask_derived_pending_strict_disposition",
            "sample_weight": 1.0,
            "visual_abstention_reason": None,
            "annotation_strength": "negative" if mask_nonzero == 0 else "source_provided",
            "annotation_provenance": (
                "source_provided_empty_semantic_mask"
                if mask_nonzero == 0
                else "source_provided_semantic_mask"
            ),
            "sample_validation_status": "pending_strict_automated_validation",
            "validation_profile": "fireviewer_pointing_strict_automated_v1",
            "corpus_disposition": "pending_strict_automated_validation",
            "strict_keep": False,
            "exclusion_reasons": [],
            "training_eligible": False,
            "media_license": "CC-BY-4.0",
            "mask_license": "MIT",
            "redistribution_allowed": True,
            "reviews_admitted": False,
            **_metrics(rgb),
        }
        candidates.append(candidate)

    own_sha: dict[str, list[str]] = defaultdict(list)
    own_dhash: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        own_sha[str(row["source_image_sha256"])].append(str(row["sample_id"]))
        own_dhash[str(row["dhash"])].append(str(row["sample_id"]))
    exact_sha_groups = {key: value for key, value in own_sha.items() if len(value) > 1}
    identical_dhash_groups = {key: value for key, value in own_dhash.items() if len(value) > 1}
    frame_by_sample = {str(row["sample_id"]): int(row["frame_id"]) for row in candidates}
    exact_sha_representatives = {
        digest: min(sample_ids, key=frame_by_sample.__getitem__)
        for digest, sample_ids in exact_sha_groups.items()
    }
    dhash_representatives = {
        signature: min(sample_ids, key=frame_by_sample.__getitem__)
        for signature, sample_ids in identical_dhash_groups.items()
    }

    for row in candidates:
        reasons = list(row["exclusion_reasons"])
        if not row["source_image_sha256_matches_manifest"]:
            reasons.append("source_image_sha256_mismatch")
        if not row["mask_sha256_matches_manifest"]:
            reasons.append("mask_sha256_mismatch")
        if row["invalid_mask_values"]:
            reasons.append("unknown_mask_class_value")
        if row["exact_duplicate_rejected"]:
            reasons.append("baseline_exact_sha_overlap")
        if row["perceptual_similarity_excluded"]:
            reasons.append("baseline_perceptual_overlap")
        if row["upstream_split"] == "unlisted":
            reasons.append("source_split_unresolved")
        digest = str(row["source_image_sha256"])
        if digest in exact_sha_representatives and (
            str(row["sample_id"]) != exact_sha_representatives[digest]
        ):
            reasons.append("within_source_exact_sha_duplicate")
        signature = str(row["dhash"])
        if signature in dhash_representatives and (
            str(row["sample_id"]) != dhash_representatives[signature]
        ):
            reasons.append("within_source_identical_dhash_neighbor")
        if row["provided_mask_signal"] and not row["anchor_points"]:
            reasons.append("mask_signal_without_anchor_point")
        row["exclusion_reasons"] = sorted(set(reasons))
        row["strict_keep"] = not reasons
        row["training_eligible"] = not reasons
        row["sample_validation_status"] = (
            "strict_automated_validated" if not reasons else "excluded_unvalidated"
        )
        row["point_annotation_status"] = (
            "source_mask_derived_strict_automated_validated"
            if not reasons
            else "excluded_unvalidated"
        )
        row["corpus_disposition"] = (
            "eligible_genuinely_new_pool" if not reasons else "excluded_unvalidated"
        )
    validated = [row for row in candidates if row["strict_keep"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "flame2_pixel_inventory.jsonl", candidates)
    _write_jsonl(output_dir / "flame2_automatic_dispositions.jsonl", candidates)
    _write_jsonl(output_dir / "flame2_strict_validated_manifest.jsonl", validated)
    report = {
        "schema_version": 1,
        "source_id": "robofirefusenet-flame2",
        "rgb_candidates": len(candidates),
        "infrared_candidates_excluded": sum(
            row.get("asset_kind") == "infrared" for row in manifest
        ),
        "mask_assets_used_as_source_ground_truth": sum(
            row.get("asset_kind") == "mask" for row in manifest
        ),
        "decode_or_pair_errors": decode_errors,
        "provided_mask_signal_counts": dict(
            sorted(Counter(str(row["provided_mask_signal"]).lower() for row in candidates).items())
        ),
        "mask_class_pixel_counts": {
            "background": sum(
                int(row["width"]) * int(row["height"])
                - int(row["smoke_pixels"])
                - int(row["fire_pixels"])
                for row in candidates
            ),
            "smoke": sum(int(row["smoke_pixels"]) for row in candidates),
            "fire": sum(int(row["fire_pixels"]) for row in candidates),
        },
        "point_target_rows": {
            "smoke_column_base": sum(
                any(point["kind"] == "smoke_column_base" for point in row["anchor_points"])
                for row in validated
            ),
            "fire_base": sum(
                any(point["kind"] == "fire_base" for point in row["anchor_points"])
                for row in validated
            ),
        },
        "negative_rows": sum(not row["provided_mask_signal"] for row in validated),
        "upstream_split_counts": dict(
            sorted(Counter(str(row["upstream_split"]) for row in candidates).items())
        ),
        "baseline_images_compared": len(baseline),
        "baseline_exact_sha_matches": sum(
            row["baseline_exact_sha_match"] is not None for row in candidates
        ),
        "baseline_perceptual_exclusions_distance_le_4": sum(
            row["perceptual_similarity_excluded"] for row in candidates
        ),
        "within_source_exact_sha_groups": len(exact_sha_groups),
        "within_source_identical_dhash_groups": len(identical_dhash_groups),
        "exclusions_by_reason": dict(
            sorted(
                Counter(reason for row in candidates for reason in row["exclusion_reasons"]).items()
            )
        ),
        "reviews_admitted": False,
        "excluded_unvalidated_rows": len(candidates) - len(validated),
        "point_ground_truth_rows": sum(bool(row["anchor_points"]) for row in validated),
        "training_eligible_rows": len(validated),
        "source_group_status": "unresolved_conservative_single_train_group",
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "publication_allowed": False,
        "next_action": "merge_with_additional_strict_sources_then_repeat_global_quality_gates",
    }
    (output_dir / "flame2_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-s3-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_flame2(
        source_root=args.source_root,
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        source_s3_prefix=args.source_s3_prefix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
