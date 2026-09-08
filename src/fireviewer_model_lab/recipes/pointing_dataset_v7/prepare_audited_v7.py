"""Create a non-destructive, byte-identical V7 benchmark view after split quarantine."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, make_registry, read_rows, verify_coco


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    for flag in ("dataset", "coco", "audit", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_rows(args.dataset / "selection_manifest.jsonl")
    audit = json.loads((args.audit / "report.json").read_text(encoding="utf-8"))
    if audit["manifest_sha256"] != digest(args.dataset / "selection_manifest.jsonl"):
        raise ValueError("Audit is not bound to the source corpus")
    quarantine = {row["sha256"] for row in read_rows(args.audit / "quarantine.jsonl")}
    kept = [row for row in rows if row["sha256"] not in quarantine]
    if any(row["split"] == "train" and row["sha256"] in quarantine for row in rows):
        raise ValueError("Cannot rewrite the actual V7 training exposure")
    args.output.mkdir(parents=True)
    manifest = args.output / "selection_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in kept), encoding="utf-8")
    report = {"schema": "fireviewer.pointing-v7-audited-benchmark-view.v1", "status": "ready_for_benchmark",
              "source_dataset": str(args.dataset.resolve()), "source_coco": str(args.coco.resolve()),
              "source_manifest_sha256": audit["manifest_sha256"], "split_audit_sha256": digest(args.audit / "report.json"),
              "counts": dict(Counter(r["split"] for r in kept)), "quarantined_images": len(quarantine),
              "selection_rule": "Declared source-group conflicts and mirrored pHash <=4, expanded to whole evaluation groups, before inference.",
              "limitations": ["Group identifiers are not all physical events.", "Conservative quarantine is not confirmation that all similar frames are duplicates."],
              "automatic_cleanup": False, "additional_image_payload_bytes": 0}
    write_json(args.output / "report.json", report)
    coco_root = args.output / "coco"
    splits = {}
    hashes = {r["sha256"] for r in kept}
    for split, folder in (("train", "train"), ("validation", "valid"), ("test", "test")):
        original = json.loads((args.coco / folder / "_annotations.coco.json").read_text(encoding="utf-8"))
        images = [r for r in original["images"] if r["fireviewer_sha256"] in hashes]
        ids = {r["id"] for r in images}
        annotations = [r for r in original["annotations"] if r["image_id"] in ids]
        for row in images:
            src = args.coco / folder / row["file_name"]
            dst = coco_root / folder / row["file_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)  # fail, never silently copy the image payload
            if not os.path.samefile(src, dst):
                raise ValueError("Hardlink identity mismatch")
        annotation_path = coco_root / folder / "_annotations.coco.json"
        write_json(annotation_path, original | {"images": images, "annotations": annotations})
        classes = Counter(r["category_id"] for r in annotations)
        splits[split] = {"images": len(images), "annotations": len(annotations),
                         "negative_images": len(images) - len({r["image_id"] for r in annotations}),
                         "class_counts": {"fire": classes[0], "smoke": classes[1]},
                         "annotation_sha256": digest(annotation_path), "hardlinks_verified": len(images)}
    receipt = {"schema": "fireviewer.pointing-v7-audited-coco-hardlink-view.v1", "status": "ready",
               "dataset_root": str(args.output.resolve()), "dataset_report_sha256": digest(args.output / "report.json"),
               "selection_manifest_sha256": digest(manifest), "splits": splits,
               "storage_policy": {"automatic_cleanup": False, "additional_image_payload_bytes": 0}}
    write_json(coco_root / "coco_view_receipt.json", receipt)
    registry = make_registry(kept)
    registry["verification"] = verify_coco(coco_root.resolve(), registry)
    if registry["verification"]["status"] != "passed":
        raise ValueError("Source group conflicts remain after quarantine")
    write_json(args.output / "split_registry.json", registry)
    # The complete exposure history still locks retired validation/test rows out of V8 train.
    write_json(args.output / "historical_split_registry.json", make_registry(rows))
    print(json.dumps(report | {"verified": registry["verification"], "splits": splits}, indent=2), flush=True)


if __name__ == "__main__":
    main()
