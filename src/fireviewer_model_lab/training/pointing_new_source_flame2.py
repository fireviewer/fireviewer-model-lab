"""Acquire and inventory the pinned FLAME2 candidate without local corpus storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

SOURCE_REPOSITORY = "dimfot3/RoboFireFuseNet"
SOURCE_REVISION = "0d8ec502da0bafea7c388a989650aa53d1ecf278"
SOURCE_FILE_ID = "1mbooTgUZxXZ86_Lh56zI-_6WEyy6aZxN"
SOURCE_URL = (
    f"https://drive.usercontent.google.com/download?id={SOURCE_FILE_ID}&export=download&confirm=t"
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip_members(archive: zipfile.ZipFile) -> list[str]:
    names: list[str] = []
    for member in archive.infolist():
        posix = PurePosixPath(member.filename.replace("\\", "/"))
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"unsafe ZIP member: {member.filename}")
        names.append(posix.as_posix())
    return names


def _kind(path: Path) -> str:
    lowered = path.as_posix().lower()
    stem = path.stem.lower()
    if "mask" in lowered or "label" in lowered or "_gt_" in stem or stem.endswith("_gt"):
        return "mask"
    if "infrared" in lowered or "/ir/" in lowered or "_ir_" in stem or stem.endswith("_ir"):
        return "infrared"
    if "/rgb/" in lowered or "_rgb_" in stem or stem.endswith("_rgb"):
        return "rgb"
    return "unclassified_image"


def inventory_source(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    decode_errors: list[str] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(source_root).as_posix()
        kind = _kind(path.relative_to(source_root))
        width = 0
        height = 0
        image_format = ""
        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                image_format = str(image.format or "")
        except Exception as exc:
            decode_errors.append(f"{relative}:{type(exc).__name__}:{exc}")
        kinds[kind] += 1
        if kind == "rgb":
            point_annotation_status = "source_semantic_mask_pair_pending_strict_conversion"
            mask_to_point_conversion = "deterministic_class_mask_bottom_band_median"
            corpus_disposition = "pending_strict_automated_validation"
            exclusion_reasons: list[str] = []
        elif kind == "mask":
            point_annotation_status = "source_semantic_ground_truth_asset"
            mask_to_point_conversion = "deterministic_class_mask_bottom_band_median_input"
            corpus_disposition = "pending_strict_automated_validation"
            exclusion_reasons = []
        elif kind == "infrared":
            point_annotation_status = "not_applicable_non_rgb_asset"
            mask_to_point_conversion = "not_applicable"
            corpus_disposition = "excluded_outside_rgb_pointing_scope"
            exclusion_reasons = ["infrared_outside_rgb_pointing_scope"]
        else:
            point_annotation_status = "unclassified_asset"
            mask_to_point_conversion = "not_applicable"
            corpus_disposition = "excluded_unvalidated"
            exclusion_reasons = ["unclassified_asset"]
        rows.append(
            {
                "schema_version": 1,
                "source_id": "robofirefusenet-flame2",
                "source_repository": SOURCE_REPOSITORY,
                "source_revision": SOURCE_REVISION,
                "source_file_id": SOURCE_FILE_ID,
                "relative_path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "format": image_format,
                "asset_kind": kind,
                "media_license": "CC-BY-4.0",
                "mask_license": "MIT",
                "point_annotation_status": point_annotation_status,
                "mask_to_point_conversion": mask_to_point_conversion,
                "corpus_disposition": corpus_disposition,
                "exclusion_reasons": exclusion_reasons,
                "training_eligible": False,
            }
        )
    report = {
        "images": len(rows),
        "asset_kind_counts": dict(sorted(kinds.items())),
        "decode_errors": decode_errors,
        "decoded_images": len(rows) - len(decode_errors),
    }
    return rows, report


def acquire(*, output_dir: Path, work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = work_dir / "FLAME2.zip"
    request = urllib.request.Request(  # noqa: S310 - pinned HTTPS source URL
        SOURCE_URL,
        headers={"User-Agent": "FireViewer-Pointing-Corpus/2.0"},
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,  # noqa: S310
        archive_path.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    archive_sha = _sha256(archive_path)
    source_root = output_dir / "source"
    with zipfile.ZipFile(archive_path) as archive:
        members = validate_zip_members(archive)
        archive.extractall(source_root)
    rows, inventory = inventory_source(source_root)
    manifest = output_dir / "candidate_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    report = {
        "schema_version": 1,
        "source_id": "robofirefusenet-flame2",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_file_id": SOURCE_FILE_ID,
        "source_archive_sha256": archive_sha,
        "source_archive_bytes": archive_path.stat().st_size,
        "zip_members": len(members),
        "inventory": inventory,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "point_ground_truth_rows": 0,
        "training_eligible_rows": 0,
        "admission_status": "candidate_only_pending_strict_automatic_disposition",
        "reviews_admitted": False,
        "next_action": (
            "verify_source_mask_classes_hashes_and_splits_then_derive_points_"
            "deterministically_and_exclude_every_failed_row"
        ),
    }
    (output_dir / "acquisition_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = acquire(output_dir=args.output_dir, work_dir=args.work_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
