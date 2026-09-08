#!/usr/bin/env python3
"""Create a storage-light, immutable COCO view of the reviewed V5 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V5_TRAINING = HERE.parent / "pointing_rtdetr_v5"
import sys

sys.path.insert(0, str(V5_TRAINING))
from v5_contract import load_v5_contract, validate_materialized_v5  # noqa: E402


SPLITS = {"train": "train", "validation": "valid", "test": "test"}
CATEGORIES = [
    {"id": 0, "name": "fire", "supercategory": "fire_smoke"},
    {"id": 1, "name": "smoke", "supercategory": "fire_smoke"},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def ensure_hardlink(source: Path, destination: Path) -> None:
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise RuntimeError(f"Existing COCO image is not a hardlink to source: {destination}")
        return
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"Hardlink verification failed: {destination}")


def convert_split(dataset_root: Path, output_root: Path, source_split: str, coco_split: str) -> dict[str, Any]:
    source_dir = dataset_root / "data" / source_split
    rows = load_jsonl(source_dir / "metadata.jsonl")
    destination = output_root / coco_split
    destination.mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    class_counts = {"fire": 0, "smoke": 0}
    negative_images = 0
    annotation_id = 1

    for coco_image_id, row in enumerate(rows, 1):
        width = int(row["width"])
        height = int(row["height"])
        source = source_dir / str(row["file_name"])
        if not source.is_file():
            raise FileNotFoundError(source)
        file_name = source.name
        ensure_hardlink(source, destination / file_name)

        images.append(
            {
                "id": coco_image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "fireviewer_sha256": row["sha256"],
                "fireviewer_source_dataset": row.get("source_dataset"),
                "fireviewer_source_group_id": row.get("source_group_id"),
                "fireviewer_scene_bin": row.get("scene_bin"),
            }
        )
        boxes = row["objects"]["bbox"]
        categories = row["objects"]["category"]
        if len(boxes) != len(categories):
            raise ValueError(f"bbox/category mismatch for {source}")
        if not boxes:
            negative_images += 1
        for raw_box, raw_category in zip(boxes, categories, strict=True):
            category = int(raw_category)
            if category not in (0, 1):
                raise ValueError(f"Unknown category {category} for {source}")
            x, y, box_width, box_height = (float(value) for value in raw_box)
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise ValueError(f"Invalid bbox {raw_box!r} for {source}")
            if x + box_width > width + 1e-3 or y + box_height > height + 1e-3:
                raise ValueError(f"Out-of-bounds bbox {raw_box!r} for {source}")
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": coco_image_id,
                    "category_id": category,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            class_counts["fire" if category == 0 else "smoke"] += 1

    annotation_path = destination / "_annotations.coco.json"
    atomic_json(
        annotation_path,
        {
            "info": {
                "description": "FireViewer pointing V5 reviewed corpus",
                "version": "5",
            },
            "licenses": [],
            "categories": CATEGORIES,
            "images": images,
            "annotations": annotations,
        },
    )
    return {
        "source_split": source_split,
        "coco_split": coco_split,
        "images": len(images),
        "annotations": len(annotations),
        "negative_images": negative_images,
        "class_counts": class_counts,
        "annotation_file": str(annotation_path.resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "hardlinks_verified": len(images),
        "additional_image_payload_bytes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if dataset_root == output_root or dataset_root in output_root.parents or output_root in dataset_root.parents:
        raise RuntimeError("COCO view and dataset roots must remain isolated")
    contract = load_v5_contract(dataset_root)
    materialized = validate_materialized_v5(contract)
    output_root.mkdir(parents=True, exist_ok=True)

    split_receipts = {
        source: convert_split(dataset_root, output_root, source, target)
        for source, target in SPLITS.items()
    }
    actual = {split: split_receipts[split]["images"] for split in SPLITS}
    if actual != contract.split_counts:
        raise RuntimeError(f"COCO split cardinality mismatch: {actual} != {contract.split_counts}")

    receipt = {
        "schema": "fireviewer.pointing-v5-coco-hardlink-view.v1",
        "status": "ready",
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "dataset_report_sha256": contract.report_sha256,
        "selection_manifest_sha256": contract.report["selection_manifest_sha256"],
        "materialized_validation": materialized,
        "categories": CATEGORIES,
        "splits": split_receipts,
        "storage_policy": {
            "image_materialization": "NTFS hardlinks",
            "additional_image_payload_bytes": 0,
            "retain_until_user_review": True,
            "automatic_cleanup": False,
        },
    }
    receipt_path = output_root / "coco_view_receipt.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
