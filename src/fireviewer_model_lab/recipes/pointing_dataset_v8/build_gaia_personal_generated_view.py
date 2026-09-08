"""Build the V8P6 high-quality finishing view after the Gaia D-Fire stage.

The real V8P4 corpus and both frozen holdouts remain byte-identical. Reviewed
synthetic images are added to train only. Their COCO records are repeated in a
bounded curriculum so generated data remains below five percent of logical
training draws while small-smoke scenes receive useful exposure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO


SMALL_RELATIVE_AREA = 0.005
REGULAR_SYNTHETIC_DRAWS = 2
SMALL_SMOKE_SYNTHETIC_DRAWS = 8


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Emit identical bytes on Windows and Linux. Path.write_text() translates
    # newlines on Windows, which made the corpus receipt hash host-dependent.
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(payload)


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Existing target differs: {target}")
        return
    os.link(source, target)


def build(base_view: Path, synthetic_root: Path, authorization: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    auth = json.loads(authorization.read_text(encoding="utf-8"))
    if not auth.get("training_authorized") or not auth.get("reviewed_synthetic_mixing_authorized"):
        raise ValueError("Explicit reviewed-synthetic training authorization is required")

    synthetic_coco_path = synthetic_root / "coco/train/_annotations.coco.json"
    synthetic_coco = json.loads(synthetic_coco_path.read_text(encoding="utf-8"))
    reviewed_path = synthetic_root / "reviewed_synthetic_manifest.jsonl"
    reviewed = read_rows(reviewed_path)
    if len(reviewed) != 61 or len(synthetic_coco["images"]) != 61:
        raise ValueError("Expected exactly 61 reviewed synthetic images")
    reviewed_by_sha = {row["sha256"]: row for row in reviewed}
    if len(reviewed_by_sha) != 61:
        raise ValueError("Duplicate reviewed synthetic SHA-256")
    required_review = (
        "whole_scene_and_all_targets_reviewed",
        "annotations_exploitable",
        "corrected_overlay_verified",
        "safe_ground_view_reviewed",
        "no_selfie_or_dominant_person_reviewed",
    )
    if any(not all(row.get(key) is True for key in required_review) for row in reviewed):
        raise ValueError("A synthetic image lacks a required manual-review gate")

    synthetic_by_id = {image["id"]: image for image in synthetic_coco["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in synthetic_coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    synthetic_shas = {Path(image["file_name"]).stem for image in synthetic_coco["images"]}
    if synthetic_shas != set(reviewed_by_sha):
        raise ValueError("Reviewed manifest and synthetic COCO identities differ")

    output.mkdir(parents=True)
    split_counts: dict[str, dict] = {}
    holdout_hashes: dict[str, str] = {}
    logical_synthetic_draws = 0
    small_smoke_images = 0
    base_train_sha = None
    for split in ("train", "valid", "test"):
        base_json_path = base_view / split / "_annotations.coco.json"
        data = json.loads(base_json_path.read_text(encoding="utf-8"))
        if data["categories"] != synthetic_coco["categories"]:
            raise ValueError("Category mapping differs")
        for image in data["images"]:
            link(base_view / split / image["file_name"], output / split / image["file_name"])
        target_json = output / split / "_annotations.coco.json"
        if split != "train":
            shutil.copyfile(base_json_path, target_json)
            if digest(target_json) != digest(base_json_path):
                raise ValueError(f"Frozen {split} annotations changed")
            holdout_hashes[split] = digest(target_json)
            split_counts["validation" if split == "valid" else split] = {
                "images": len(data["images"]), "annotations": len(data["annotations"]),
                "annotation_sha256": digest(target_json),
            }
            continue

        base_train_sha = digest(base_json_path)
        base_names = {Path(image["file_name"]).stem for image in data["images"]}
        if base_names & synthetic_shas:
            raise ValueError("Exact synthetic/base identity overlap")
        next_image_id = max(image["id"] for image in data["images"]) + 1
        next_annotation_id = max(annotation["id"] for annotation in data["annotations"]) + 1
        for source_id in sorted(synthetic_by_id):
            source_image = synthetic_by_id[source_id]
            sha = Path(source_image["file_name"]).stem
            source_file = synthetic_root / "coco/train" / source_image["file_name"]
            if digest(source_file) != sha:
                raise ValueError(f"Synthetic payload SHA differs: {sha}")
            link(source_file, output / "train/images" / source_image["file_name"])
            relative_area = [
                annotation["area"] / (source_image["width"] * source_image["height"])
                for annotation in annotations_by_image[source_id]
                if annotation["category_id"] == 1
            ]
            small_smoke = any(area <= SMALL_RELATIVE_AREA for area in relative_area)
            draws = SMALL_SMOKE_SYNTHETIC_DRAWS if small_smoke else REGULAR_SYNTHETIC_DRAWS
            small_smoke_images += int(small_smoke)
            logical_synthetic_draws += draws
            for draw in range(draws):
                image = dict(source_image)
                image.update(
                    id=next_image_id,
                    file_name="images/" + source_image["file_name"],
                    synthetic=True,
                    fireviewer_sha256=sha,
                    fireviewer_synthetic_draw_ordinal=draw + 1,
                    fireviewer_synthetic_draws=draws,
                    fireviewer_review_status="manually_reviewed_all_targets",
                )
                data["images"].append(image)
                for source_annotation in annotations_by_image[source_id]:
                    annotation = dict(source_annotation)
                    annotation.update(id=next_annotation_id, image_id=next_image_id)
                    data["annotations"].append(annotation)
                    next_annotation_id += 1
                next_image_id += 1
        write_json(target_json, data)
        reloaded = COCO(str(target_json))
        if len(reloaded.imgs) != len(data["images"]) or len(reloaded.anns) != len(data["annotations"]):
            raise ValueError("Merged COCO reload mismatch")
        split_counts["train"] = {
            "images": len(data["images"]), "annotations": len(data["annotations"]),
            "annotation_sha256": digest(target_json),
        }

    for path in (output / "train/images").glob("*.png"):
        with Image.open(path) as image:
            image.verify()

    report = {
        "schema": "fireviewer.pointing-v8p6-gaia-personal-generated-finishing.v1",
        "status": "verified_for_cpu_setup",
        "stage_a": {
            "dataset": "Gaia Solutions on Demand D-Fire, pinned and technically filtered",
            "reuse_existing_checkpoint": True,
        },
        "stage_b": {
            "unique_real_images": 5100,
            "unique_reviewed_synthetic_images": 61,
            "unique_total_images": 5161,
            "logical_synthetic_draws": logical_synthetic_draws,
            "small_smoke_synthetic_images": small_smoke_images,
            "logical_train_records": split_counts["train"]["images"],
            "synthetic_logical_fraction": logical_synthetic_draws / split_counts["train"]["images"],
        },
        "split_counts": split_counts,
        "base_train_sha256": base_train_sha,
        "synthetic_coco_sha256": digest(synthetic_coco_path),
        "synthetic_review_manifest_sha256": digest(reviewed_path),
        "authorization_sha256": digest(authorization),
        "holdouts": {
            "validation_sha256": holdout_hashes["valid"],
            "test_sha256": holdout_hashes["test"],
            "byte_identical": True,
            "test_images_untouched": 455,
        },
        "new_image_payloads": 61,
        "test_evaluation_authorized": False,
    }
    write_json(output / "report.json", report)
    receipt = report | {
        "status": "verified_for_authorized_lightning_evaluation_run",
        "dataset_report_sha256": digest(output / "report.json"),
        "selection_manifest_sha256": digest(reviewed_path),
        "splits": split_counts,
        "real_images": 5100,
        "synthetic_images": 61,
        "local_training_authorization": auth,
        "coverage_qualified": False,
        "acknowledged_limitations": [
            "Stage A D-Fire annotations are source-provided, not exhaustively reviewed.",
            "Synthetic images are reviewed training examples and are not real incidents.",
            "The frozen test split is not evaluated during corpus selection.",
        ],
    }
    write_json(output / "coco_view_receipt.json", receipt)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-view", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(*(path.resolve() for path in (args.base_view, args.synthetic_root, args.authorization, args.output)))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
