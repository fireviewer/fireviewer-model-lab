"""Stream and strictly audit the pinned Boreal segmentation bundle for pointing."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

HF_REPOSITORY = "fireviewer/firewarning-train-bundles-v1"
HF_REVISION = "3207a080f786e3349349759af34385e3d892b4e5"
BUNDLE_NAME = "wildfire-smoke-segmentation-v1.zip"
BUNDLE_SHA256 = "23134190da8ef71b157764453f3d5575a339fe469878934c70e4972db33eee0e"
BUNDLE_URL = f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/{HF_REVISION}/{BUNDLE_NAME}"
SOURCE_ROOT = "wildfire-smoke-segmentation-v1/sources/boreal-forest-fire-segmentation-v1/"
EXPECTED_SOURCE_ID = "boreal-forest-fire-segmentation-v1"
EXPECTED_LICENSE = "CC-BY-4.0"
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


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


def _mask_base_point(mask: np.ndarray) -> tuple[float, float] | None:
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


def _safe_member(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member reference: {value}")
    return path.as_posix()


def _load_baseline(root: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    names = {"pixel_inventory.jsonl", "flame2_strict_validated_manifest.jsonl"}
    manifests = [path for path in root.rglob("*.jsonl") if path.name in names]
    if not manifests:
        raise FileNotFoundError(f"no pointing baseline manifests below {root}")
    sha_rows: dict[str, str] = {}
    dhash_rows: list[tuple[str, str]] = []
    for path in sorted(manifests):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or "")
            digest = str(
                row.get("image_sha256") or row.get("sha256") or row.get("source_image_sha256") or ""
            )
            signature = str(row.get("dhash") or "")
            if len(digest) == 64:
                sha_rows[digest] = sample_id
            if len(signature) == 16:
                dhash_rows.append((signature, sample_id))
    return sha_rows, dhash_rows


def _archive_rows(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    manifest_name = SOURCE_ROOT + "manifest.jsonl"
    return [
        json.loads(line)
        for line in archive.read(manifest_name).decode("utf-8").splitlines()
        if line.strip()
    ]


def audit_boreal_archive(
    *,
    archive_path: Path,
    baseline_root: Path,
    output_dir: Path,
    output_s3_prefix: str,
) -> dict[str, Any]:
    baseline_sha, baseline_dhash = _load_baseline(baseline_root)
    candidates: list[dict[str, Any]] = []
    payload_members: dict[str, tuple[str, str]] = {}
    decode_errors: list[str] = []

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            _safe_member(member.filename)
        source_rows = _archive_rows(archive)
        for source in source_rows:
            sample_id = str(source.get("sample_id") or "")
            reasons: list[str] = []
            if source.get("source_id") != EXPECTED_SOURCE_ID:
                reasons.append("unexpected_source_id")
            if source.get("license") != EXPECTED_LICENSE:
                reasons.append("unknown_or_incompatible_rights")
            split = str(source.get("split") or "")
            if split not in ALLOWED_SPLITS:
                reasons.append("invalid_split")
            split_group = str(source.get("split_group") or "")
            if not split_group:
                reasons.append("missing_split_group")

            artifact = source.get("artifact") or {}
            artifact_member = SOURCE_ROOT + _safe_member(str(artifact.get("path") or ""))
            try:
                artifact_bytes = archive.read(artifact_member)
            except Exception as exc:
                decode_errors.append(f"{sample_id}:artifact:{type(exc).__name__}:{exc}")
                continue
            artifact_sha_ok = _sha256_bytes(artifact_bytes) == str(artifact.get("sha256") or "")
            if not artifact_sha_ok:
                reasons.append("artifact_sha256_mismatch")
            try:
                sample = json.loads(artifact_bytes)
                image_meta = sample["image"]
                mask_meta = sample["annotation"]
                image_member = SOURCE_ROOT + _safe_member(str(image_meta["path"]))
                mask_member = SOURCE_ROOT + _safe_member(str(mask_meta["path"]))
                image_bytes = archive.read(image_member)
                mask_bytes = archive.read(mask_member)
                with Image.open(io.BytesIO(image_bytes)) as opened:
                    opened.load()
                    image = opened.convert("RGB")
                with Image.open(io.BytesIO(mask_bytes)) as opened_mask:
                    opened_mask.load()
                    mask = opened_mask.convert("L")
            except Exception as exc:
                decode_errors.append(f"{sample_id}:payload:{type(exc).__name__}:{exc}")
                continue

            image_sha = _sha256_bytes(image_bytes)
            mask_sha = _sha256_bytes(mask_bytes)
            if image_sha != str(image_meta.get("sha256") or ""):
                reasons.append("source_image_sha256_mismatch")
            if mask_sha != str(mask_meta.get("sha256") or ""):
                reasons.append("mask_sha256_mismatch")
            if image.size != mask.size:
                reasons.append("image_mask_dimension_mismatch")
            mask_array = np.asarray(mask, dtype=np.uint8)
            mask_values = sorted(int(value) for value in np.unique(mask_array))
            if not set(mask_values).issubset({0, 255}):
                reasons.append("unknown_mask_class_value")
            point = _mask_base_point(mask_array > 0)
            if point is None:
                reasons.append("empty_smoke_mask")
            strength = str(sample.get("annotation_strength") or "")
            provenance = str(sample.get("annotation_provenance") or "")
            if strength != "strong" or provenance != "human_pixel_mask":
                reasons.append("weak_sam_mask_not_strict_ground_truth")

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
                        "kind": "smoke_column_base",
                        "x": point[0],
                        "y": point[1],
                        "origin": "source_mask_bottom_band_median",
                    }
                ]
                if point is not None
                else []
            )
            image_relpath = f"strict-payload/{image_meta['path']}"
            mask_relpath = f"strict-payload/{mask_meta['path']}"
            candidates.append(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "source_id": EXPECTED_SOURCE_ID,
                    "source_family": "Boreal Forest Fire smoke segmentation",
                    "source_revision": HF_REVISION,
                    "source_bundle_sha256": BUNDLE_SHA256,
                    "source_record_id": str(source.get("source_record_id") or ""),
                    "split": split,
                    "final_split": split,
                    "split_group": split_group,
                    "image_relpath": image_relpath,
                    "mask_relpath": mask_relpath,
                    "image_s3_uri": f"{output_s3_prefix.rstrip('/')}/{image_relpath}",
                    "mask_s3_uri": f"{output_s3_prefix.rstrip('/')}/{mask_relpath}",
                    "image_sha256": image_sha,
                    "source_image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "width": image.width,
                    "height": image.height,
                    "dhash": dhash,
                    "mask_values": mask_values,
                    "mask_quality": "source_provided_strong"
                    if strength == "strong"
                    else "sam_weak",
                    "annotation_strength": strength,
                    "annotation_provenance": provenance,
                    "mask_to_point_conversion": "deterministic_binary_mask_bottom_band_median",
                    "anchor_points": anchor_points,
                    "visual_abstention_reason": None if point is not None else "empty_smoke_mask",
                    "baseline_exact_sha_match": exact_match,
                    "baseline_nearest_dhash_distance": baseline_distance,
                    "baseline_nearest_sample_id": baseline_nearest,
                    "license": EXPECTED_LICENSE,
                    "redistribution_allowed": True,
                    "reviews_admitted": False,
                    "validation_profile": "fireviewer_pointing_strict_automated_v1",
                    "exclusion_reasons": sorted(set(reasons)),
                    "strict_keep": False,
                    "training_eligible": False,
                    "sample_validation_status": "pending_strict_automated_validation",
                    **_metrics(image),
                }
            )
            payload_members[sample_id] = (image_member, mask_member)

        sha_groups: dict[str, list[str]] = defaultdict(list)
        dhash_groups: dict[str, list[str]] = defaultdict(list)
        for row in candidates:
            sha_groups[str(row["image_sha256"])].append(str(row["sample_id"]))
            dhash_groups[str(row["dhash"])].append(str(row["sample_id"]))
        exact_duplicates = {
            key: sorted(value) for key, value in sha_groups.items() if len(value) > 1
        }
        identical_visuals = {
            key: sorted(value) for key, value in dhash_groups.items() if len(value) > 1
        }
        for row in candidates:
            reasons = list(row["exclusion_reasons"])
            digest = str(row["image_sha256"])
            signature = str(row["dhash"])
            if digest in exact_duplicates and str(row["sample_id"]) != exact_duplicates[digest][0]:
                reasons.append("within_source_exact_sha_duplicate")
            if (
                signature in identical_visuals
                and str(row["sample_id"]) != identical_visuals[signature][0]
            ):
                reasons.append("within_source_identical_dhash_neighbor")
            row["exclusion_reasons"] = sorted(set(reasons))
            row["strict_keep"] = not reasons
            row["training_eligible"] = not reasons
            row["sample_validation_status"] = (
                "strict_automated_validated" if not reasons else "excluded_unvalidated"
            )

        validated = [row for row in candidates if row["strict_keep"]]
        for row in validated:
            image_member, mask_member = payload_members[str(row["sample_id"])]
            for member, relative in (
                (image_member, str(row["image_relpath"])),
                (mask_member, str(row["mask_relpath"])),
            ):
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "boreal_automatic_dispositions.jsonl", candidates)
    _write_jsonl(output_dir / "boreal_strict_validated_manifest.jsonl", validated)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in validated:
        group_splits[str(row["split_group"])].add(str(row["split"]))
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"Boreal strict split-group leakage: {leaking_groups}")
    gate_errors: list[str] = []
    if decode_errors:
        gate_errors.append(f"decode_or_payload_errors:{len(decode_errors)}")
    if not validated:
        gate_errors.append("no_strict_automated_validated_rows")
    report = {
        "schema_version": 1,
        "source_id": EXPECTED_SOURCE_ID,
        "source_repository": HF_REPOSITORY,
        "source_revision": HF_REVISION,
        "source_bundle_sha256": BUNDLE_SHA256,
        "rows_evaluated": len(candidates),
        "decode_or_payload_errors": decode_errors,
        "annotation_strength_counts": dict(
            sorted(Counter(str(row["annotation_strength"]) for row in candidates).items())
        ),
        "strict_automated_validated_rows": len(validated),
        "validated_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in validated).items())
        ),
        "smoke_column_base_points": sum(bool(row["anchor_points"]) for row in validated),
        "exclusions_by_reason": dict(
            sorted(
                Counter(reason for row in candidates for reason in row["exclusion_reasons"]).items()
            )
        ),
        "baseline_images_compared": len(baseline_sha),
        "within_source_exact_sha_groups": len(exact_duplicates),
        "within_source_identical_dhash_groups": len(identical_visuals),
        "split_group_leakage": leaking_groups,
        "gate_errors": gate_errors,
        "source_gate_passed": not gate_errors,
        "reviews_admitted": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "publication_allowed": False,
        "next_action": "merge_strict_sources_then_repeat_global_quality_gates",
    }
    (output_dir / "boreal_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def download_bundle(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(  # noqa: S310 - immutable HTTPS source
        BUNDLE_URL,
        headers={"User-Agent": "FireViewer-Pointing-Corpus/2.0"},
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,  # noqa: S310
        destination.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    observed = _sha256_file(destination)
    if observed != BUNDLE_SHA256:
        raise ValueError(f"Boreal bundle SHA-256 mismatch: {observed}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-s3-prefix", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    archive_path = args.work_dir / BUNDLE_NAME
    download_bundle(archive_path)
    report = audit_boreal_archive(
        archive_path=archive_path,
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        output_s3_prefix=args.output_s3_prefix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
