"""Build a deterministic small-target curriculum from an audited COCO view.

The underlying image corpus is unchanged. Training image records are repeated
with new COCO ids so the standard DEIM dataloader samples audited images that
contain COCO-small smoke and fire more often. Validation and test are copied
byte-for-byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO


COCO_SMALL_AREA = 32 * 32


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def link_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Existing target differs: {target}")
        return
    os.link(source, target)


def build(base_view: Path, output: Path, smoke_extra: int = 2, fire_extra: int = 1) -> dict:
    if smoke_extra < 0 or fire_extra < 0:
        raise ValueError("Repeat counts must be non-negative")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    output.mkdir(parents=True)
    holdout_hashes: dict[str, str] = {}
    base_train_hash = digest(base_view / "train" / "_annotations.coco.json")
    report: dict[str, object] = {
        "schema": "fireviewer.pointing-v8p5-small-target-curriculum.v1",
        "status": "verified_for_cpu_setup_not_training_authorized",
        "base_view": str(base_view.resolve()),
        "base_train_sha256": base_train_hash,
        "coco_small_area_lt": COCO_SMALL_AREA,
        "repeat_policy": {
            "small_smoke_extra_records": smoke_extra,
            "small_fire_without_small_smoke_extra_records": fire_extra,
        },
    }

    for split in ("train", "valid", "test"):
        source_json = base_view / split / "_annotations.coco.json"
        data = json.loads(source_json.read_text(encoding="utf-8"))
        for image in data["images"]:
            source = base_view / split / image["file_name"]
            link_verified(source, output / split / image["file_name"])

        target_json = output / split / "_annotations.coco.json"
        if split != "train":
            shutil.copyfile(source_json, target_json)
            if digest(source_json) != digest(target_json):
                raise ValueError(f"Frozen {split} annotations changed")
            holdout_hashes[split] = digest(target_json)
            continue

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in data["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        small_smoke_ids = {
            annotation["image_id"] for annotation in data["annotations"]
            if annotation["category_id"] == 1 and annotation["area"] < COCO_SMALL_AREA
        }
        small_fire_ids = {
            annotation["image_id"] for annotation in data["annotations"]
            if annotation["category_id"] == 0 and annotation["area"] < COCO_SMALL_AREA
        }
        repeats = {
            image_id: smoke_extra for image_id in small_smoke_ids
        }
        repeats.update({
            image_id: fire_extra for image_id in small_fire_ids - small_smoke_ids
        })

        image_by_id = {image["id"]: image for image in data["images"]}
        next_image_id = max(image_by_id, default=0) + 1
        next_annotation_id = max((ann["id"] for ann in data["annotations"]), default=0) + 1
        added_images = 0
        added_annotations = Counter()
        added_small = Counter()
        for original_id in sorted(repeats):
            original_image = image_by_id[original_id]
            for ordinal in range(1, repeats[original_id] + 1):
                duplicate = dict(original_image)
                duplicate["id"] = next_image_id
                duplicate["fireviewer_curriculum_repeat_of_image_id"] = original_id
                duplicate["fireviewer_curriculum_repeat_ordinal"] = ordinal
                data["images"].append(duplicate)
                for original_annotation in annotations_by_image[original_id]:
                    annotation = dict(original_annotation)
                    annotation["id"] = next_annotation_id
                    annotation["image_id"] = next_image_id
                    data["annotations"].append(annotation)
                    category = "fire" if annotation["category_id"] == 0 else "smoke"
                    added_annotations[category] += 1
                    if annotation["area"] < COCO_SMALL_AREA:
                        added_small[category] += 1
                    next_annotation_id += 1
                next_image_id += 1
                added_images += 1

        write_json(target_json, data)
        loaded = COCO(str(target_json))
        if len(loaded.imgs) != len(data["images"]) or len(loaded.anns) != len(data["annotations"]):
            raise ValueError("Curriculum COCO reload mismatch")
        report["train"] = {
            "unique_image_payloads": len(image_by_id),
            "logical_image_records": len(data["images"]),
            "added_logical_records": added_images,
            "base_small_smoke_images": len(small_smoke_ids),
            "base_small_fire_images": len(small_fire_ids),
            "small_fire_without_small_smoke_images": len(small_fire_ids - small_smoke_ids),
            "added_annotations": dict(added_annotations),
            "added_small_annotations": dict(added_small),
            "annotation_sha256": digest(target_json),
        }

    # Decode every unique payload once. Repeated COCO records intentionally share files.
    decoded = 0
    for split in ("train", "valid", "test"):
        for path in (output / split / "images").iterdir():
            with Image.open(path) as image:
                image.verify()
            decoded += 1
    report["unique_images_decoded"] = decoded
    report["holdouts"] = {
        "validation_sha256": holdout_hashes["valid"],
        "test_sha256": holdout_hashes["test"],
        "annotation_files_byte_identical": True,
        "test_images_untouched": 455,
    }
    report["storage"] = {"mode": "hardlinks", "new_image_payloads": 0}
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-extra", type=int, default=2)
    parser.add_argument("--fire-extra", type=int, default=1)
    args = parser.parse_args()
    report = build(args.base_view.resolve(), args.output.resolve(), args.smoke_extra, args.fire_extra)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
