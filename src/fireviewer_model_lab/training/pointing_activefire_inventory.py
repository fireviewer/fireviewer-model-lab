"""Inventory ActiveFire manual labels in SageMaker before the large image transfer."""

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

SOURCE_REPOSITORY = "pereira-gha/activefire"
SOURCE_REVISION = "16ef99053bb472c5d25b6585a527e1d9c8bd34b7"
SOURCE_LICENSE = "CC-BY-4.0"
ARCHIVES = {
    "manual_annotations_patches": "1LdsX-rH5hy_82jfc1akO8p4n0_N8lRgf",
    "algorithm_masks_patches": "1RCURItVvqsT_oMxlhB5NYiRp8SJ9_xZ3",
}
IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff"})


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


def _download(file_id: str, destination: Path) -> None:
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    request = urllib.request.Request(  # noqa: S310 - pinned HTTPS provider and file id
        url,
        headers={"User-Agent": "FireViewer-Pointing-Corpus/2.0"},
    )
    with (
        urllib.request.urlopen(request, timeout=180) as response,  # noqa: S310
        destination.open("wb") as handle,
    ):
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _normalized_stem(member: str) -> str:
    stem = PurePosixPath(member).stem.lower()
    for token in ("manual_annotation_", "manual_annotations_", "annotation_", "mask_"):
        if stem.startswith(token):
            stem = stem[len(token) :]
    for token in ("_manual_annotation", "_manual", "_annotation", "_mask"):
        if stem.endswith(token):
            stem = stem[: -len(token)]
    return stem


def inventory_archive(archive_path: Path, extract_root: Path) -> dict[str, Any]:
    extract_root.mkdir(parents=True, exist_ok=True)
    decoded = 0
    decode_errors: list[str] = []
    modes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    value_sets: Counter[str] = Counter()
    image_members: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = validate_zip_members(archive)
        archive.extractall(extract_root)
    for member in members:
        path = extract_root / Path(*PurePosixPath(member).parts)
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image_members.append(member)
        try:
            with Image.open(path) as image:
                image.load()
                decoded += 1
                modes[image.mode] += 1
                sizes[f"{image.width}x{image.height}"] += 1
                if image.mode in {"1", "L", "P", "I", "I;16"}:
                    values = image.getdata()
                    unique = sorted({int(value) for value in values})
                    key = ",".join(str(value) for value in unique[:32])
                    if len(unique) > 32:
                        key += ",..."
                    value_sets[key] += 1
        except Exception as exc:
            decode_errors.append(f"{member}:{type(exc).__name__}:{exc}")
    return {
        "zip_members": len(members),
        "image_members": len(image_members),
        "decoded_images": decoded,
        "decode_errors": decode_errors,
        "modes": dict(sorted(modes.items())),
        "dimensions": dict(sorted(sizes.items())),
        "scalar_value_sets": dict(sorted(value_sets.items())),
        "normalized_stems": sorted({_normalized_stem(member) for member in image_members}),
        "member_examples": image_members[:20],
    }


def run_inventory(*, output_dir: Path, work_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    inventories: dict[str, dict[str, Any]] = {}
    archive_receipts: dict[str, dict[str, Any]] = {}
    for name, file_id in ARCHIVES.items():
        archive_path = work_dir / f"{name}.zip"
        _download(file_id, archive_path)
        archive_receipts[name] = {
            "google_drive_file_id": file_id,
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256(archive_path),
        }
        inventories[name] = inventory_archive(archive_path, work_dir / name)

    manual_stems = set(inventories["manual_annotations_patches"].pop("normalized_stems"))
    algorithm_stems = set(inventories["algorithm_masks_patches"].pop("normalized_stems"))
    report = {
        "schema_version": 1,
        "source_id": "activefire-landsat-manual",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
        "archive_receipts": archive_receipts,
        "inventories": inventories,
        "manual_annotation_stems": len(manual_stems),
        "algorithm_mask_stems": len(algorithm_stems),
        "normalized_stem_intersection": len(manual_stems & algorithm_stems),
        "manual_only_stems": sorted(manual_stems - algorithm_stems)[:50],
        "algorithm_only_stems": sorted(algorithm_stems - manual_stems)[:50],
        "large_landsat_archive_downloaded": False,
        "training_eligible_rows": 0,
        "reviews_admitted": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "next_action": "download_large_image_archive_in_cloud_only_if_manual_labels_are_valid",
    }
    (output_dir / "activefire_inventory_report.json").write_text(
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
    report = run_inventory(output_dir=args.output_dir, work_dir=args.work_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
