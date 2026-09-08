"""Build a reviewed D-Fire complement on top of the frozen V8P2 view.

Only candidates covered by an explicit, packet-bound visual decision can be
added. Validation and test COCO files are copied byte-for-byte and every image
is hard-linked, so this command neither rewrites holdouts nor duplicates image
payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO


CATEGORIES = [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}]
SCHEMA = "fireviewer.pointing-v8p4-reviewed-dfire.v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def link_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f"Existing output differs: {target}")
        return
    os.link(source, target)


def verified_decisions(review_manifest: Path, decisions_path: Path) -> tuple[list[dict], list[dict]]:
    candidates = read_rows(review_manifest)
    decisions = read_rows(decisions_path)
    by_index = {row["review_index"]: row for row in candidates}
    if len(by_index) != len(candidates):
        raise ValueError("Duplicate review index in candidate manifest")
    if len(decisions) != len({row["review_index"] for row in decisions}):
        raise ValueError("Duplicate review decision")
    if {row["review_index"] for row in decisions} != set(by_index):
        raise ValueError("Every candidate requires exactly one explicit decision")

    accepted, rejected = [], []
    checked_pages: dict[str, str] = {}
    for decision in decisions:
        row = by_index[decision["review_index"]]
        for field in ("candidate_id", "sha256", "label_sha256"):
            if decision.get(field) != row.get(field):
                raise ValueError(f"Decision binding differs for review {decision['review_index']}: {field}")
        page = Path(decision["review_page"])
        actual_page_sha = checked_pages.setdefault(str(page.resolve()), digest(page))
        if actual_page_sha != decision.get("review_page_sha256"):
            raise ValueError(f"Review page changed for review {decision['review_index']}")
        if not decision.get("reason") or not decision.get("reviewed_at"):
            raise ValueError("A dated visual-review reason is mandatory")
        choice = decision.get("decision")
        if choice == "accept":
            if decision.get("full_frame_review_complete") is not True:
                raise ValueError("Accepted image lacks full-frame review")
            if decision.get("all_annotations_review_complete") is not True:
                raise ValueError("Accepted image lacks all-annotation review")
            source = Path(row["source_image"])
            if digest(source) != row["sha256"]:
                raise ValueError(f"Source image changed: {source}")
            with Image.open(source) as image:
                image.load()
                if image.size != (row["width"], row["height"]):
                    raise ValueError(f"Source dimensions changed: {source}")
            accepted.append(dict(row, review_status="accepted_full_frame_and_all_annotations",
                                 annotation_review_complete=True, v8_corpus_admitted=True,
                                 v8_training_admitted=True, review_decision=decision))
        elif choice == "reject":
            rejected.append({"review_index": row["review_index"], "candidate_id": row["candidate_id"],
                             "sha256": row["sha256"], "reason": decision["reason"],
                             "source_image_deleted": False, "review_decision": decision})
        else:
            raise ValueError(f"Invalid decision for review {decision['review_index']}")
    return accepted, rejected


def build(base_view: Path, review_manifest: Path, decisions_path: Path, output: Path,
          max_per_group: int = 2) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing corpus: {output}")
    accepted, rejected = verified_decisions(review_manifest, decisions_path)
    groups = Counter(row["source_group_id"] for row in accepted)
    if groups and max(groups.values()) > max_per_group:
        raise ValueError(f"Reviewed selection exceeds group cap {max_per_group}: {groups.most_common(1)[0]}")

    output.mkdir(parents=True)
    split_report = {}
    holdout_hashes = {}
    base_shas = set()
    for split in ("train", "valid", "test"):
        src_json = base_view / split / "_annotations.coco.json"
        coco = json.loads(src_json.read_text(encoding="utf-8"))
        if coco.get("categories") != CATEGORIES:
            raise ValueError("Unexpected base category mapping")
        for image in coco["images"]:
            source = base_view / split / image["file_name"]
            link_verified(source, output / split / image["file_name"])
            if image.get("fireviewer_sha256"):
                base_shas.add(image["fireviewer_sha256"])

        dest_json = output / split / "_annotations.coco.json"
        if split in {"valid", "test"}:
            shutil.copyfile(src_json, dest_json)
            if digest(src_json) != digest(dest_json):
                raise ValueError(f"Frozen {split} annotations changed")
            holdout_hashes[split] = digest(dest_json)
        else:
            next_image_id = max((item["id"] for item in coco["images"]), default=-1) + 1
            next_annotation_id = max((item["id"] for item in coco["annotations"]), default=-1) + 1
            for row in accepted:
                if row["sha256"] in base_shas:
                    raise ValueError("Reviewed addition already exists in the base")
                filename = f"images/{row['sha256']}.jpg"
                link_verified(Path(row["source_image"]), output / split / filename)
                coco["images"].append({
                    "id": next_image_id, "file_name": filename,
                    "width": row["width"], "height": row["height"],
                    "fireviewer_sha256": row["sha256"],
                    "fireviewer_source_group_id": row["source_group_id"],
                    "annotation_review_status": "accepted_full_frame_and_all_annotations",
                })
                for box, category, area in zip(row["objects"]["bbox"], row["objects"]["category"],
                                               row["objects"]["area"]):
                    coco["annotations"].append({
                        "id": next_annotation_id, "image_id": next_image_id,
                        "category_id": category, "bbox": box, "area": area, "iscrowd": 0,
                    })
                    next_annotation_id += 1
                next_image_id += 1
            write_json(dest_json, coco)

        loaded = COCO(str(dest_json))
        if len(loaded.imgs) != len(coco["images"]) or len(loaded.anns) != len(coco["annotations"]):
            raise ValueError(f"COCO reload failed for {split}")
        for image in coco["images"]:
            path = output / split / image["file_name"]
            with Image.open(path) as decoded:
                decoded.verify()
        split_report[split] = {"images": len(coco["images"]), "annotations": len(coco["annotations"]),
                               "annotation_sha256": digest(dest_json), "decoded_images": len(coco["images"])}

    accepted_boxes = Counter()
    small = Counter()
    for row in accepted:
        pixels = row["width"] * row["height"]
        for box, category in zip(row["objects"]["bbox"], row["objects"]["category"]):
            name = CATEGORIES[category]["name"]
            accepted_boxes[name] += 1
            if box[2] * box[3] / pixels <= 0.005:
                small[name] += 1
    report = {
        "schema": SCHEMA,
        "status": "verified_for_cpu_setup_not_training_authorized",
        "base_view": str(base_view.resolve()),
        "inputs": {
            "base_train_sha256": digest(base_view / "train" / "_annotations.coco.json"),
            "base_valid_sha256": holdout_hashes["valid"],
            "base_test_sha256": holdout_hashes["test"],
            "review_manifest_sha256": digest(review_manifest),
            "decisions_sha256": digest(decisions_path),
        },
        "review": {"reviewed": len(accepted) + len(rejected), "accepted": len(accepted),
                   "rejected": len(rejected), "group_cap": max_per_group,
                   "maximum_accepted_per_group": max(groups.values(), default=0)},
        "accepted_annotations": dict(accepted_boxes),
        "accepted_small_annotations": dict(small),
        "splits": split_report,
        "total_images": sum(item["images"] for item in split_report.values()),
        "holdouts": {"validation_images_unchanged": split_report["valid"]["images"],
                     "test_images_unchanged": split_report["test"]["images"],
                     "annotation_files_byte_identical": True},
        "storage": {"mode": "hardlinks", "image_payload_copied": False},
    }
    write_rows(output / "accepted_additions.jsonl", accepted)
    write_rows(output / "rejected_additions.jsonl", rejected)
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-view", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-group", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build(args.base_view.resolve(), args.review_manifest.resolve(),
                           args.decisions.resolve(), args.output.resolve(), args.max_per_group), indent=2))


if __name__ == "__main__":
    main()
