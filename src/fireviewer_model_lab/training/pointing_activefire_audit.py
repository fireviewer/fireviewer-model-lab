"""Selectively acquire and strictly audit ActiveFire manual masks in SageMaker."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

SOURCE_ID = "activefire-landsat-manual"
SOURCE_REPOSITORY = "pereira-gha/activefire"
SOURCE_REVISION = "16ef99053bb472c5d25b6585a527e1d9c8bd34b7"
SOURCE_LICENSE = "CC-BY-4.0"
MANUAL_FILE_ID = "1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf"
MANUAL_ARCHIVE_SHA256 = "96b5a2b239748505cc72335b5747568efa500732c15d03d752765110e399d88e"
LANDSAT_FILE_ID = "1uZnc65_GRFdAoavGoUKkJfeVQOUI8lVg"
LANDSAT_FILE_COUNT = 9044
LANDSAT_UNCOMPRESSED_BYTES = 11_871_932_184
SPECTRAL_CONTRAST_MIN = 1.10
POINTING_SEMANTIC_EXCLUSION = "top_down_hotspot_mask_has_no_ground_base_semantics"
PATCH_PATTERN = re.compile(r"^(?P<scene>.+)_p\d+$", re.I)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _download_bytes(file_id: str) -> bytes:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    request = urllib.request.Request(  # noqa: S310 - pinned HTTPS provider and file id
        url,
        headers={"User-Agent": "FireViewer-Pointing-Corpus/2.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        return response.read()


def _validate_zip_members(archive: zipfile.ZipFile) -> list[str]:
    names: list[str] = []
    for member in archive.infolist():
        posix = PurePosixPath(member.filename.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"unsafe ZIP member: {member.filename}")
        names.append(posix.as_posix())
    return names


def normalize_manual_stem(path: str) -> str:
    stem = PurePosixPath(path).stem.lower().replace("_v1_", "_")
    return stem


def source_scene(stem: str) -> str:
    match = PATCH_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"unrecognized ActiveFire patch stem: {stem}")
    return match.group("scene").lower()


def assign_scene_splits(scenes: set[str]) -> dict[str, str]:
    ordered = sorted(scenes, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    if len(ordered) < 5:
        raise ValueError("ActiveFire requires at least five independent Landsat scenes")
    train_end = max(1, round(len(ordered) * 0.6))
    validation_end = max(train_end + 1, round(len(ordered) * 0.8))
    validation_end = min(validation_end, len(ordered) - 1)
    return {
        scene: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, scene in enumerate(ordered)
    }


def render_false_color_762(data: np.ndarray, scales: list[tuple[float, float]]) -> Image.Image:
    if data.ndim != 3 or data.shape[2] != 10:
        raise ValueError("ActiveFire image must contain ten Landsat bands")
    channels: list[np.ndarray] = []
    for band_index, (low, high) in zip((6, 5, 1), scales, strict=True):
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("invalid source-wide Landsat rendering scale")
        values = np.clip((data[..., band_index].astype(np.float32) - low) / (high - low), 0, 1)
        channels.append(np.rint(values * 255).astype(np.uint8))
    return Image.fromarray(np.stack(channels, axis=-1), mode="RGB")


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


def _metrics(image: Image.Image) -> dict[str, float]:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
    return {
        "brightness": float(stat.mean[0] / 255.0),
        "contrast": float(stat.stddev[0] / 255.0),
        "edge_energy": float(edges.mean[0] / 255.0),
    }


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


def _nearest(signature: str, others: list[tuple[str, str]]) -> tuple[int | None, str | None]:
    if not others:
        return None, None
    distance, sample_id = min((_hamming(signature, value), key) for value, key in others)
    return distance, sample_id


def _exclude_duplicates(rows: list[dict[str, Any]]) -> dict[str, int]:
    sha_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dhash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sha_groups[str(row["image_sha256"])].append(row)
        dhash_groups[str(row["dhash"])].append(row)
    exact_groups = [members for members in sha_groups.values() if len(members) > 1]
    identical_groups = [members for members in dhash_groups.values() if len(members) > 1]
    for groups, reason in (
        (exact_groups, "within_source_exact_sha_duplicate"),
        (identical_groups, "within_source_identical_dhash_neighbor"),
    ):
        for members in groups:
            keep = min(members, key=lambda row: str(row["sample_id"]))
            for row in members:
                if row is not keep:
                    row["exclusion_reasons"].append(reason)

    near_pairs = 0
    split_priority = {"test": 0, "validation": 1, "train": 2}
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            if _hamming(str(left["dhash"]), str(right["dhash"])) > 4:
                continue
            near_pairs += 1
            reject = max(
                (left, right),
                key=lambda row: (split_priority[str(row["split"])], str(row["sample_id"])),
            )
            reject["exclusion_reasons"].append("within_source_near_cross_split_duplicate")
    return {
        "exact_sha_groups": len(exact_groups),
        "identical_dhash_groups": len(identical_groups),
        "near_cross_split_pairs": near_pairs,
    }


def audit_activefire(
    *, baseline_root: Path, output_dir: Path, output_s3_uri: str, work_dir: Path
) -> dict[str, Any]:
    import tifffile
    from remotezip import RemoteZip

    work_dir.mkdir(parents=True, exist_ok=True)
    manual_payload = _download_bytes(MANUAL_FILE_ID)
    if _sha256_bytes(manual_payload) != MANUAL_ARCHIVE_SHA256:
        raise ValueError("ActiveFire manual annotation archive hash drift")
    with zipfile.ZipFile(io.BytesIO(manual_payload)) as manual_zip:
        manual_names = [
            name for name in _validate_zip_members(manual_zip) if name.lower().endswith(".tif")
        ]
        manual_masks = {name: manual_zip.read(name) for name in manual_names}
    if len(manual_masks) != 100:
        raise ValueError(f"ActiveFire manual-mask count drift: {len(manual_masks)}")

    landsat_url = (
        "https://drive.usercontent.google.com/download?"
        f"id={LANDSAT_FILE_ID}&export=download&confirm=t"
    )
    selected: list[dict[str, Any]] = []
    with RemoteZip(landsat_url) as remote:
        infos = [info for info in remote.infolist() if not info.is_dir()]
        if len(infos) != LANDSAT_FILE_COUNT:
            raise ValueError(f"ActiveFire Landsat member-count drift: {len(infos)}")
        if sum(info.file_size for info in infos) != LANDSAT_UNCOMPRESSED_BYTES:
            raise ValueError("ActiveFire Landsat uncompressed-byte count drift")
        by_stem = {PurePosixPath(info.filename).stem.lower(): info for info in infos}
        for manual_name, mask_payload in sorted(manual_masks.items()):
            stem = normalize_manual_stem(manual_name)
            info = by_stem.get(stem)
            if info is None:
                raise FileNotFoundError(f"missing ActiveFire Landsat pair for {manual_name}")
            image_payload = remote.read(info)
            data = tifffile.imread(io.BytesIO(image_payload))
            mask = tifffile.imread(io.BytesIO(mask_payload))
            if data.shape != (256, 256, 10) or mask.shape != (256, 256):
                raise ValueError(f"ActiveFire image-mask shape drift for {stem}")
            selected.append(
                {
                    "stem": stem,
                    "scene": source_scene(stem),
                    "remote_member": info.filename,
                    "remote_uncompressed_bytes": info.file_size,
                    "remote_compressed_bytes": info.compress_size,
                    "image_payload": image_payload,
                    "mask_payload": mask_payload,
                    "data": data,
                    "mask": mask,
                }
            )

    split_by_scene = assign_scene_splits({str(item["scene"]) for item in selected})
    sampled_channels: list[list[np.ndarray]] = [[], [], []]
    for item in selected:
        if split_by_scene[str(item["scene"])] != "train":
            continue
        data = np.asarray(item["data"])
        for target, band_index in zip(sampled_channels, (6, 5, 1), strict=True):
            values = data[::4, ::4, band_index].astype(np.float32).ravel()
            target.append(values[values > 0])
    scales: list[tuple[float, float]] = []
    for values in sampled_channels:
        combined = np.concatenate(values)
        scales.append((float(np.percentile(combined, 2)), float(np.percentile(combined, 99.5))))
    baseline_sha, baseline_dhash = _load_baseline(baseline_root)
    raw_dir = output_dir / "raw"
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    for directory in (raw_dir, image_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in selected:
        stem = str(item["stem"])
        data = np.asarray(item["data"])
        mask_array = np.asarray(item["mask"])
        mask_values = sorted(int(value) for value in np.unique(mask_array))
        # ActiveFire annotates top-down active-fire pixels.  Those pixels are useful
        # for segmentation, but no location in such a mask is a defensible semantic
        # fire base on the ground-view image plane required by this corpus.
        reasons: list[str] = [POINTING_SEMANTIC_EXCLUSION]
        if not set(mask_values).issubset({0, 1}):
            reasons.append("unknown_manual_mask_value")
        binary_mask = mask_array > 0
        mask_has_signal = bool(np.any(binary_mask))
        band7 = data[..., 6].astype(np.float64)
        if mask_has_signal:
            foreground = band7[binary_mask]
            background = band7[~binary_mask]
            spectral_contrast = float(foreground.mean() / max(1.0, np.median(background)))
            if spectral_contrast < SPECTRAL_CONTRAST_MIN:
                reasons.append("manual_fire_not_observable_in_swir2")
        else:
            spectral_contrast = None

        rendered = render_false_color_762(data, scales)
        raw_path = raw_dir / f"{stem}.tif"
        image_path = image_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}.png"
        raw_path.write_bytes(item["image_payload"])
        rendered.save(image_path, format="PNG", optimize=True)
        Image.fromarray((binary_mask.astype(np.uint8) * 255), mode="L").save(
            mask_path, format="PNG", optimize=True
        )
        image_sha = _sha256(image_path)
        dhash = _difference_hash(rendered)
        exact_match = baseline_sha.get(image_sha)
        baseline_distance, baseline_nearest = _nearest(dhash, baseline_dhash)
        if exact_match is not None:
            reasons.append("baseline_exact_sha_overlap")
        if baseline_distance is not None and baseline_distance <= 4:
            reasons.append("baseline_perceptual_overlap")
        anchor_points: list[dict[str, Any]] = []
        relative_raw = raw_path.relative_to(output_dir).as_posix()
        relative_image = image_path.relative_to(output_dir).as_posix()
        relative_mask = mask_path.relative_to(output_dir).as_posix()
        rows.append(
            {
                "schema_version": 1,
                "sample_id": f"activefire:{stem}",
                "source_id": SOURCE_ID,
                "source_family": "ActiveFire Landsat manual masks",
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "source_file_id": LANDSAT_FILE_ID,
                "source_member": item["remote_member"],
                "source_scene": item["scene"],
                "source_group": f"activefire:{item['scene']}",
                "split_group": f"activefire:{item['scene']}",
                "split": split_by_scene[str(item["scene"])],
                "image_relpath": relative_image,
                "mask_relpath": relative_mask,
                "source_image_relpath": relative_raw,
                "image_s3_uri": f"{output_s3_uri.rstrip('/')}/{relative_image}",
                "mask_s3_uri": f"{output_s3_uri.rstrip('/')}/{relative_mask}",
                "source_image_s3_uri": f"{output_s3_uri.rstrip('/')}/{relative_raw}",
                "source_image_sha256": _sha256_bytes(item["image_payload"]),
                "image_sha256": image_sha,
                "mask_sha256": _sha256(mask_path),
                "width": 256,
                "height": 256,
                "dhash": dhash,
                "provided_mask_values": mask_values,
                "provided_mask_nonzero_fraction": float(
                    np.count_nonzero(binary_mask) / mask_array.size
                ),
                "provided_mask_signal": mask_has_signal,
                "provided_mask_role": "source_manual_active_fire_mask",
                "mask_quality": "source_provided_manual_binary",
                "mask_semantics": "active_fire",
                "mask_to_point_conversion": "forbidden_top_down_mask_is_not_ground_base",
                "spectral_rendering": "landsat_bands_7_6_2_train_split_percentile_scale",
                "spectral_rendering_scales": scales,
                "spectral_rendering_scale_fit_split": "train",
                "fire_swir2_mean_over_background_median": spectral_contrast,
                "anchor_points": anchor_points,
                "annotation_strength": "auxiliary_segmentation_only",
                "visual_abstention_reason": POINTING_SEMANTIC_EXCLUSION,
                "sample_weight": 0.0,
                "variant": "clean",
                "baseline_exact_sha_match": exact_match,
                "baseline_nearest_dhash_distance": baseline_distance,
                "baseline_nearest_sample_id": baseline_nearest,
                "sample_validation_status": "pending_strict_automated_validation",
                "validation_profile": "fireviewer_pointing_strict_automated_v1",
                "exclusion_reasons": reasons,
                "strict_keep": False,
                "training_eligible": False,
                "media_license": SOURCE_LICENSE,
                "mask_license": SOURCE_LICENSE,
                "redistribution_allowed": True,
                "reviews_admitted": False,
                **_metrics(rendered),
            }
        )

    duplicate_report = _exclude_duplicates(rows)
    for row in rows:
        reasons = sorted(set(row["exclusion_reasons"]))
        row["exclusion_reasons"] = reasons
        row["strict_keep"] = not reasons
        row["training_eligible"] = not reasons
        row["sample_validation_status"] = (
            "strict_automated_validated" if not reasons else "excluded_unvalidated"
        )
        row["corpus_disposition"] = (
            "eligible_genuinely_new_pool" if not reasons else "excluded_unvalidated"
        )
    validated = [row for row in rows if row["strict_keep"]]
    rows.sort(key=lambda row: str(row["sample_id"]))
    validated.sort(key=lambda row: str(row["sample_id"]))
    _write_jsonl(output_dir / "activefire_automatic_dispositions.jsonl", rows)
    _write_jsonl(output_dir / "activefire_strict_validated_manifest.jsonl", validated)
    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "manual_archive_sha256": MANUAL_ARCHIVE_SHA256,
        "remote_landsat_members": LANDSAT_FILE_COUNT,
        "selected_manual_pairs": len(rows),
        "selected_remote_compressed_bytes": sum(
            int(item["remote_compressed_bytes"]) for item in selected
        ),
        "selected_remote_uncompressed_bytes": sum(
            int(item["remote_uncompressed_bytes"]) for item in selected
        ),
        "source_scenes": len(split_by_scene),
        "scene_split_counts": dict(sorted(Counter(split_by_scene.values()).items())),
        "strict_automated_validated_rows": len(validated),
        "validated_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in validated).items())
        ),
        "fire_base_points": sum(bool(row["anchor_points"]) for row in validated),
        "negative_rows": sum(not row["anchor_points"] for row in validated),
        "small_or_faint_positive_rows": sum(
            bool(row["anchor_points"]) and row["provided_mask_nonzero_fraction"] <= 0.02
            for row in validated
        ),
        "source_wide_rendering_scales": scales,
        "spectral_rendering_scale_fit_split": "train",
        "exclusions_by_reason": dict(
            sorted(Counter(reason for row in rows for reason in row["exclusion_reasons"]).items())
        ),
        "duplicate_report": duplicate_report,
        "baseline_images_compared": len(baseline_sha),
        "reviews_admitted": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "publication_allowed": False,
        "next_action": "merge_strict_sources_then_repeat_global_quality_gates",
    }
    (output_dir / "activefire_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_activefire(
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        output_s3_uri=args.output_s3_uri,
        work_dir=args.work_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
