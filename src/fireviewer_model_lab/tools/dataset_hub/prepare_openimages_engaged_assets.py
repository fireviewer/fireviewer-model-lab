from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from download_supplemental_archives import ArchiveSpec, download_archive
from prepare_supplemental_sources import (
    SCHEMA_VERSION,
    _materialize_payload,
    _write_json,
    sha256_file,
    validate_normalized_source,
)

LANDING_PAGE = "https://storage.googleapis.com/openimages/web/download_v7.html"
IMAGE_OBJECT_BASE_URL = "https://open-images-dataset.s3.amazonaws.com"
TARGET_CLASSES = {
    "/m/012n7d": "ambulance",
    "/m/09ct_": "helicopter",
    "/m/0cmf2": "fixed-wing aircraft",
}
SPLIT_LIMITS = {
    "train": 1_500,
    "validation": 300,
    "test": 300,
}
OFFICIAL_FILES = (
    ArchiveSpec(
        "source/oidv7-class-descriptions-boxable.csv",
        "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
        12_064,
        "c5e7cb6b85d0539b2105db2d1974681f",
        "md5",
        "open-images-v7-engaged-assets",
    ),
    ArchiveSpec(
        "source/train-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        2_258_447_590,
        "3c3e70cfaba5757ea5c2604b19cac3b2",
        "md5",
        "open-images-v7-engaged-assets",
    ),
    ArchiveSpec(
        "source/validation-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        25_105_048,
        "c5e8200df129ea6867e913e8b21fcab9",
        "md5",
        "open-images-v7-engaged-assets",
    ),
    ArchiveSpec(
        "source/test-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        77_484_237,
        "1cc058a7003b4e73d47642276e9b123b",
        "md5",
        "open-images-v7-engaged-assets",
    ),
    ArchiveSpec(
        "source/train-images.csv",
        "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
        638_407_721,
        "a4ac0bcedb5c2df4d1b1230dc6b41b8f",
        "md5",
        "open-images-v7-engaged-assets",
    ),
    ArchiveSpec(
        "source/validation-images.csv",
        "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
        15_245_485,
        "643a54e43b0bab8acce8817b4a569780",
        "md5",
        "open-images-v7-engaged-assets",
    ),
    ArchiveSpec(
        "source/test-images.csv",
        "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
        45_227_339,
        "d832feb775b3cb78077bf8ae350adce5",
        "md5",
        "open-images-v7-engaged-assets",
    ),
)
ALLOWED_LICENSE_PREFIXES = (
    "https://creativecommons.org/licenses/by/",
    "http://creativecommons.org/licenses/by/",
    "https://creativecommons.org/licenses/by-sa/",
    "http://creativecommons.org/licenses/by-sa/",
    "https://creativecommons.org/publicdomain/zero/",
    "http://creativecommons.org/publicdomain/zero/",
)


def _rank(image_id: str) -> str:
    return hashlib.sha256(image_id.encode("ascii")).hexdigest()


def _valid_box(row: dict[str, str]) -> bool:
    if row["LabelName"] not in TARGET_CLASSES:
        return False
    if row["IsDepiction"] != "0" or row["IsInside"] != "0" or row["IsGroupOf"] != "0":
        return False
    coordinates = [float(row[key]) for key in ("XMin", "XMax", "YMin", "YMax")]
    xmin, xmax, ymin, ymax = coordinates
    return 0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1


def _read_candidate_boxes(path: Path) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if not _valid_box(row):
                continue
            candidates[row["ImageID"]].append(
                {
                    "class_id": row["LabelName"],
                    "class_name": TARGET_CLASSES[row["LabelName"]],
                    "bbox_normalized_xyxy": [
                        float(row["XMin"]),
                        float(row["YMin"]),
                        float(row["XMax"]),
                        float(row["YMax"]),
                    ],
                    "annotation_source": row["Source"],
                    "is_occluded": row["IsOccluded"] == "1",
                    "is_truncated": row["IsTruncated"] == "1",
                }
            )
    return candidates


def _candidate_ids_by_class(boxes: dict[str, list[dict[str, Any]]], limit: int) -> set[str]:
    by_class: dict[str, list[str]] = defaultdict(list)
    for image_id, image_annotations in boxes.items():
        for class_name in {str(annotation["class_name"]) for annotation in image_annotations}:
            by_class[class_name].append(image_id)
    selected: set[str] = set()
    for class_name in sorted(TARGET_CLASSES.values()):
        ranked = sorted(set(by_class[class_name]), key=_rank)
        selected.update(ranked[: limit * 2])
    return selected


def _read_licensed_metadata(path: Path, candidate_ids: set[str]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_id = row["ImageID"]
            if image_id not in candidate_ids:
                continue
            license_url = row["License"].strip()
            if not any(license_url.startswith(prefix) for prefix in ALLOWED_LICENSE_PREFIXES):
                continue
            metadata[image_id] = {
                "license": license_url,
                "original_url": row["OriginalURL"],
                "original_landing_url": row["OriginalLandingURL"],
                "author": row["Author"],
                "author_profile_url": row["AuthorProfileURL"],
                "title": row["Title"],
                "rotation": row["Rotation"],
            }
    return metadata


def _balanced_selection(
    boxes: dict[str, list[dict[str, Any]]],
    metadata: dict[str, dict[str, str]],
    limit: int,
) -> set[str]:
    selected: set[str] = set()
    for class_name in sorted(TARGET_CLASSES.values()):
        eligible = [
            image_id
            for image_id, annotations in boxes.items()
            if image_id in metadata
            and class_name in {annotation["class_name"] for annotation in annotations}
        ]
        selected.update(sorted(eligible, key=_rank)[:limit])
    return selected


def _download_image(split: str, image_id: str, output_root: Path) -> dict[str, Any]:
    destination = output_root / "images" / split / f"{image_id}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 2:
        with destination.open("rb") as stream:
            if stream.read(2) == b"\xff\xd8":
                return {
                    "image_id": image_id,
                    "path": destination,
                    "sha256": sha256_file(destination),
                    "size_bytes": destination.stat().st_size,
                    "status": "cache_hit",
                }
    partial = destination.with_suffix(".jpg.partial")
    # Match the official Open Images downloader: image objects are served by
    # the public CVDF S3 bucket with unsigned reads.  The similarly named
    # Google Storage URL returns 404 for these paths.
    url = f"{IMAGE_OBJECT_BASE_URL}/{split}/{image_id}.jpg"
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                digest = hashlib.sha256()
                size = 0
                with partial.open("wb") as output:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
            with partial.open("rb") as stream:
                if stream.read(2) != b"\xff\xd8":
                    raise ValueError(f"Open Images payload is not JPEG: {image_id}")
            os.replace(partial, destination)
            return {
                "image_id": image_id,
                "path": destination,
                "sha256": digest.hexdigest(),
                "size_bytes": size,
                "status": "downloaded",
            }
        except Exception as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"Open Images download failed: {split}/{image_id}") from last_error


def prepare_openimages(
    raw_root: Path, output_root: Path, *, force: bool, workers: int
) -> dict[str, Any]:
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    raw_root.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(OFFICIAL_FILES, start=1):
        result = download_archive(spec, raw_root)
        print(f"Open Images source {index}/{len(OFFICIAL_FILES)}: {result['status']}", flush=True)
    class_file = raw_root / "source" / "oidv7-class-descriptions-boxable.csv"
    observed_classes = {}
    with class_file.open(encoding="utf-8", newline="") as stream:
        for class_id, name in csv.reader(stream):
            if class_id in TARGET_CLASSES:
                observed_classes[class_id] = name
    if observed_classes != {
        "/m/012n7d": "Ambulance",
        "/m/09ct_": "Helicopter",
        "/m/0cmf2": "Fixed-wing aircraft",
    }:
        raise ValueError("Open Images target class mapping changed")

    split_data: dict[str, dict[str, Any]] = {}
    for split, limit in SPLIT_LIMITS.items():
        boxes = _read_candidate_boxes(raw_root / "source" / f"{split}-annotations-bbox.csv")
        candidate_ids = _candidate_ids_by_class(boxes, limit)
        metadata = _read_licensed_metadata(
            raw_root / "source" / f"{split}-images.csv", candidate_ids
        )
        selected = _balanced_selection(boxes, metadata, limit)
        if not selected:
            raise ValueError(f"Open Images {split} selected no licensed images")
        split_data[split] = {"boxes": boxes, "metadata": metadata, "selected": selected}
        print(f"Open Images {split}: {len(selected)} licensed images selected", flush=True)

    downloads: dict[tuple[str, str], dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(_download_image, split, image_id, raw_root): (split, image_id)
        for split, data in split_data.items()
        for image_id in data["selected"]
    }
    try:
        for completed, future in enumerate(as_completed(futures), start=1):
            key = futures[future]
            downloads[key] = future.result()
            if completed == 1 or completed % 250 == 0 or completed == len(futures):
                print(f"Open Images pixels: {completed}/{len(futures)} verified", flush=True)
    except BaseException:
        for pending in futures:
            pending.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    if output_root.exists():
        if not force:
            raise FileExistsError(output_root)
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    source_id = "open-images-v7-engaged-assets-v1"
    manifest_rows: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    class_image_counts: Counter[str] = Counter()
    digest_splits: dict[str, set[str]] = defaultdict(set)
    for split, data in split_data.items():
        for image_id in sorted(data["selected"]):
            download = downloads[(split, image_id)]
            digest = str(download["sha256"])
            destination_relative = Path("payload") / "media" / digest[:2] / f"{digest}.jpg"
            destination = output_root / destination_relative
            _materialize_payload(Path(download["path"]), destination)
            digest_splits[digest].add(split)
            image_ref = {
                "path": destination_relative.as_posix(),
                "sha256": digest,
                "size_bytes": int(download["size_bytes"]),
                "role": "media",
                "media_type": "image/jpeg",
            }
            annotations = data["boxes"][image_id]
            for class_name in {annotation["class_name"] for annotation in annotations}:
                class_image_counts[f"{split}:{class_name}"] += 1
            artifact = {
                "schema_version": SCHEMA_VERSION,
                "source_id": source_id,
                "sample_id": f"openimages:{split}:{image_id}",
                "task": "engaged_assets_object_detection",
                "split": split,
                "image": image_ref,
                "annotations": annotations,
                "image_license": data["metadata"][image_id],
            }
            artifact_relative = Path("samples") / split / f"{image_id}.json"
            artifact_path = output_root / artifact_relative
            _write_json(artifact_path, artifact)
            manifest_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": f"openimages:{split}:{image_id}",
                    "source_id": source_id,
                    "source_record_id": image_id,
                    "task": "engaged_assets_object_detection",
                    "split": split,
                    "split_group": f"openimages-image:{image_id}",
                    "license": data["metadata"][image_id]["license"],
                    "provenance": {
                        "landing_page": LANDING_PAGE,
                        "open_images_split": split,
                        "original_landing_url": data["metadata"][image_id]["original_landing_url"],
                        "author": data["metadata"][image_id]["author"],
                    },
                    "artifact": {
                        "path": artifact_relative.as_posix(),
                        "sha256": sha256_file(artifact_path),
                        "media_type": "application/json",
                    },
                    "referenced_payloads": [image_ref],
                }
            )
            split_counts[split] += 1
    leakage = {digest: splits for digest, splits in digest_splits.items() if len(splits) > 1}
    if leakage:
        raise ValueError(f"Open Images exact pixel leakage across splits: {len(leakage)}")
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    source_dir = output_root / "source"
    source_dir.mkdir()
    _materialize_payload(class_file, source_dir / class_file.name)
    source_manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "title": "Open Images V7 subset for engaged asset detection",
        "landing_page": LANDING_PAGE,
        "license": "per-image Creative Commons license; annotations CC-BY-4.0",
        "task": "engaged_assets_object_detection",
        "classes": observed_classes,
        "selection": {
            "per_class_limits": SPLIT_LIMITS,
            "depictions_inside_views_and_group_boxes_excluded": True,
            "noncommercial_and_no-derivatives_image_licenses_excluded": True,
        },
        "limitations": [
            "Open Images V7 has no boxable fire-engine class",
            "Aircraft and helicopters are not labeled by firefighting role",
            "This source is supplemental pretraining/evaluation, not proof of deployed means",
        ],
    }
    _write_json(output_root / "SOURCE_MANIFEST.json", source_manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": source_id,
        "samples": len(manifest_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "class_image_counts": dict(sorted(class_image_counts.items())),
        "exact_pixel_split_leakage": 0,
        "manifest_sha256": sha256_file(manifest),
    }
    _write_json(output_root / "VALIDATION_REPORT.json", report)
    report["normalized_validation"] = validate_normalized_source(output_root)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Open Images engaged-assets subset.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_openimages(
        args.raw_root.resolve(),
        args.output_root.resolve(),
        force=args.force,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
