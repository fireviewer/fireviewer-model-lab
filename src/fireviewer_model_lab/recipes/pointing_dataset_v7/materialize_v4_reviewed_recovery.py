#!/usr/bin/env python3
"""Materialize reviewed V4 rows omitted from the current V7 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from training.pointing_dataset_v7.build_v7_ready import file_sha, read_jsonl, write_jsonl


POOL_SPECS = {
    "r2": ("pointing-dataset-v4-review-plan-r2-20260824", "review_pool.jsonl"),
    "fire_supplement": ("pointing-dataset-v4-fire-supplement-20260824", "fire_supplement_pool.jsonl"),
    "dfire_extension": ("pointing-dataset-v4-dfire-fire-extension-20260824", "review_pool.jsonl"),
    "dfire_reusegroup": ("pointing-dataset-v4-dfire-fire-reusegroup-extension-20260824", "review_pool.jsonl"),
    "cqu_secondary": ("pointing-dataset-v4-cqu-fire-extension-20260824", "review_pool.jsonl"),
}


def payloads(rows: list[dict]) -> dict[str, bytes]:
    requests: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    parquet_files: dict[str, pq.ParquetFile] = {}
    for row in rows:
        if row["image_storage"] != "parquet_embedded":
            continue
        locator = row["image_locator"]
        path = str(Path(locator["parquet"]).resolve())
        parquet = parquet_files.setdefault(path, pq.ParquetFile(path))
        absolute = int(locator["row"])
        offset = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if absolute < offset + count:
                requests[(path, group)].append((absolute - offset, row["sha256"]))
                break
            offset += count
        else:
            raise RuntimeError(f"parquet row out of range: {locator}")
    output = {}
    for (path, group), items in requests.items():
        table = parquet_files[path].read_row_group(group, columns=["image"])
        for local_row, sha in items:
            value = table["image"][local_row].as_py()
            data = value["bytes"] if isinstance(value, dict) else value
            if hashlib.sha256(data).hexdigest() != sha:
                raise RuntimeError(f"embedded image SHA mismatch: {sha}")
            output[sha] = data
    return output


def source_path(root: Path, lot: str, row: dict) -> Path:
    locator = row["image_locator"]
    if row["image_storage"] == "local_file":
        return Path(locator)
    if row["image_storage"] == "download_cache_file":
        return root / POOL_SPECS[lot][0] / locator
    raise RuntimeError(f"not a file-backed row: {lot} {row['pool_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    known = {row["sha256"] for row in read_jsonl(args.current / "selection_manifest.jsonl")}
    decisions = {
        (row["lot"], row["pool_id"]): row
        for row in read_jsonl(args.artifact_root / "pointing-dataset-v4-final-zero-human-merged-20260824" / "decision_ledger.jsonl")
        if row["decision"] == "accepted" and row.get("full_resolution_visual_confirmation") is True
    }
    selected = []
    for lot, (directory, filename) in POOL_SPECS.items():
        for row in read_jsonl(args.artifact_root / directory / filename):
            if (lot, row["pool_id"]) in decisions and row["sha256"] not in known:
                selected.append((lot, row, decisions[(lot, row["pool_id"])]))
                known.add(row["sha256"])
    parquet_payloads = payloads([row for _, row, _ in selected])
    manifest = []
    materialization = Counter()
    for lot, raw, decision in selected:
        destination = args.output / "images" / f"{raw['sha256']}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if raw["image_storage"] == "parquet_embedded":
            destination.write_bytes(parquet_payloads[raw["sha256"]])
            materialization["parquet_extract"] += 1
        else:
            source = source_path(args.artifact_root, lot, raw)
            if not source.is_file():
                raise RuntimeError(f"missing reviewed source image: {source}")
            try:
                os.link(source, destination)
                materialization["hardlink"] += 1
            except OSError:
                destination.write_bytes(source.read_bytes())
                materialization["copy_fallback"] += 1
        if file_sha(destination) != raw["sha256"]:
            raise RuntimeError(f"materialized SHA mismatch: {raw['sha256']}")
        with Image.open(destination) as image:
            if (image.width, image.height) != (raw["width"], raw["height"]):
                raise RuntimeError(f"dimension mismatch: {raw['pool_id']}")
        boxes = [[float(value) for value in ann["bbox_xywh"]] for ann in raw["annotations"]]
        categories = [int(ann["class_id"]) for ann in raw["annotations"]]
        manifest.append({
            "schema": "fireviewer.pointing-training-sample.v1",
            "sha256": raw["sha256"], "width": raw["width"], "height": raw["height"],
            "objects": {"bbox": boxes, "category": categories, "area": [box[2] * box[3] for box in boxes]},
            "caption": f"Verified annotations: fire={categories.count(0)}, smoke={categories.count(1)}.",
            "source_dataset": raw["source_dataset"], "source_revision": raw.get("source_revision", "v4-pinned"),
            "source_record_id": raw.get("source_record_id", raw["pool_id"]), "source_split": raw.get("source_split", raw.get("original_split", "train")),
            "source_group_id": raw["source_group_id"], "split_group_id": raw.get("split_group_id", raw["recurrence_group_id"]),
            "license": raw["license"], "quality_gate": "v4_full_resolution_visual_review_passed_recovered_for_v7",
            "annotation_exploitable": True, "negative_verified": not boxes, "training_admitted": True,
            "scene_bin": "hard_negative" if not boxes else raw["scene_bin_v4"], "aerial": False,
            "centered": raw.get("centered_v4", False), "poor_framing": raw.get("poor_framing_v4", False),
            "person_risk_reviewed_clear": decision.get("zero_human_foreground_confirmed", True),
            "v7_origin": "v4_reviewed_ledger_recovery", "visual_review_decision_id": decision["review_id"],
            "recovery_lot": lot, "recovery_pool_id": raw["pool_id"],
            "artifact_image": f"images/{raw['sha256']}.jpg",
        })
    write_jsonl(args.output / "selection_manifest.jsonl", sorted(manifest, key=lambda row: row["sha256"]))
    report = {
        "schema": "fireviewer.pointing-v7-v4-reviewed-recovery.v1", "status": "verified",
        "selected_count": len(manifest), "lots": dict(Counter(row["recovery_lot"] for row in manifest)),
        "scenes": dict(Counter(row["scene_bin"] for row in manifest)), "storage": dict(Counter(raw["image_storage"] for _, raw, _ in selected)),
        "materialization": dict(materialization), "selection_manifest_sha256": file_sha(args.output / "selection_manifest.jsonl"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
