#!/usr/bin/env python3
"""Build the expanded, reviewed FireViewer pointing V7 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from training.pointing_dataset_v7.split_registry import assert_frozen_holdouts_retained, load_registry, make_registry

from training.pointing_dataset_v7.build_v7_ready import (
    assign_groupwise_splits,
    cap_recurrence,
    coco_for_split,
    enforce_distribution_caps,
    file_sha,
    link,
    read_jsonl,
    stable_number,
    validate,
    write_jsonl,
)


def base_rows(artifact: Path) -> tuple[list[dict], dict[str, Path]]:
    rows = read_jsonl(artifact / "selection_manifest.jsonl")
    sources = {row["sha256"]: artifact / "data" / row["split"] / row["file_name"] for row in rows}
    return rows, sources


def normalize_existing(row: dict, origin: str) -> dict:
    output = dict(row)
    boxes = [[float(value) for value in box] for box in row["objects"]["bbox"]]
    categories = [int(value) for value in row["objects"]["category"]]
    output["objects"] = {"bbox": boxes, "category": categories, "area": [box[2] * box[3] for box in boxes]}
    output["caption"] = f"Verified annotations: fire={categories.count(0)}, smoke={categories.count(1)}."
    output["v7_origin"] = origin
    output["training_admitted"] = True
    return output


def normalize_candidate(row: dict, origin: str, dataset: str, revision: str, quality: str) -> dict:
    boxes, categories = [], []
    for annotation in row["annotations"]:
        x, y, width, height = [float(value) for value in annotation.get("bbox_xywh", annotation.get("bbox"))]
        # YOLO-to-XYWH conversion can put an edge at -1 px through rounding.
        # Clip to the decoded image bounds without expanding the annotation.
        x2 = min(float(row["width"]), x + width)
        y2 = min(float(row["height"]), y + height)
        x = max(0.0, x)
        y = max(0.0, y)
        box = [x, y, x2 - x, y2 - y]
        if box[2] <= 0 or box[3] <= 0:
            raise RuntimeError(f"candidate box collapsed after clipping: {row['candidate_id']}")
        boxes.append(box)
        if "class_name" in annotation:
            categories.append(1 if "smoke" in annotation["class_name"].lower() else 0)
        else:
            categories.append(int(annotation["category"]))
    return {
        "schema": "fireviewer.pointing-training-sample.v1",
        "sha256": row["sha256"],
        "width": int(row["width"]),
        "height": int(row["height"]),
        "objects": {"bbox": boxes, "category": categories, "area": [box[2] * box[3] for box in boxes]},
        "caption": f"Verified annotations: fire={categories.count(0)}, smoke={categories.count(1)}.",
        "source_dataset": dataset,
        "source_revision": revision,
        "source_record_id": row.get("source_record_id", row.get("source_filename", row["candidate_id"])),
        "source_split": row.get("source_split", row.get("split", "train")),
        "source_group_id": row.get("source_group_id", row["split_group"]),
        "split_group_id": row.get("split_group_id", row["split_group"]),
        "license": row["license"],
        "quality_gate": quality,
        "annotation_exploitable": True,
        "negative_verified": not boxes,
        "training_admitted": True,
        "scene_bin": row["gap_bucket"],
        "aerial": False,
        "centered": bool(row.get("off_center") is False),
        "poor_framing": bool(row.get("off_center")),
        "person_risk_reviewed_clear": True,
        "v7_origin": origin,
        "visual_review_decision_id": row["candidate_id"],
    }


def accepted_by_id(artifact: Path) -> set[str]:
    return {row["candidate_id"] for row in read_jsonl(artifact / "review_decisions.jsonl") if row["decision"] == "accept"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--hpwren", type=Path, required=True)
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--dfire", type=Path, required=True)
    parser.add_argument("--v4-recovery", type=Path)
    parser.add_argument("--local-hard-negatives", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    rows, sources = base_rows(args.base)
    registry_path = args.base / "split_registry.json"
    registry = make_registry(rows, load_registry(registry_path) if registry_path.exists() else None)
    known = {row["sha256"] for row in rows}
    additions = Counter()

    for split in ("train", "validation", "test"):
        for raw in read_jsonl(args.v4 / "data" / split / "metadata.jsonl"):
            if raw["sha256"] in known:
                continue
            row = normalize_existing(raw, "v4_reviewed_recovered_extension")
            rows.append(row)
            sources[row["sha256"]] = args.v4 / "data" / split / raw["file_name"]
            known.add(row["sha256"])
            additions["v4_reviewed_recovered"] += 1

    if args.v4_recovery:
        for raw in read_jsonl(args.v4_recovery / "selection_manifest.jsonl"):
            if raw["sha256"] in known:
                continue
            row = normalize_existing(raw, "v4_reviewed_ledger_recovery")
            rows.append(row)
            sources[row["sha256"]] = args.v4_recovery / raw["artifact_image"]
            known.add(row["sha256"])
            additions["v4_reviewed_ledger_recovery"] += 1

    for raw in read_jsonl(args.hpwren / "selection_manifest.jsonl"):
        if raw["sha256"] in known:
            continue
        row = normalize_existing(raw, "hpwren_reviewed_extension")
        rows.append(row)
        sources[row["sha256"]] = args.hpwren / "data" / raw["split"] / raw["file_name"]
        known.add(row["sha256"])
        additions["hpwren_reviewed"] += 1

    for artifact, dataset, revision, origin, quality in (
        (args.nemo, "NEMO-AlertWildfire", "a2e5576fec00ea0c0532cb4c81040ba388e54097", "nemo_reviewed_extension", "nemo_context_and_target_zoom_visual_review_passed"),
        (args.dfire, "dfire", "local-fire-smoke-ground-corpus-pinned", "dfire_reviewed_ground_extension", "dfire_full_frame_ground_subset_visual_review_passed"),
    ):
        accepted = accepted_by_id(artifact)
        for raw in read_jsonl(artifact / "candidate_manifest.jsonl"):
            if raw["candidate_id"] not in accepted or raw["sha256"] in known:
                continue
            row = normalize_candidate(raw, origin, dataset, revision, quality)
            rows.append(row)
            sources[row["sha256"]] = Path(raw["source_image"])
            known.add(row["sha256"])
            additions[origin] += 1

    if args.local_hard_negatives:
        accepted = accepted_by_id(args.local_hard_negatives)
        for raw in read_jsonl(args.local_hard_negatives / "candidate_manifest.jsonl"):
            if raw["candidate_id"] not in accepted or raw["sha256"] in known:
                continue
            if raw.get("annotations"):
                raise RuntimeError(f"hard negative has annotations: {raw['candidate_id']}")
            source_id = str(raw.get("source_id", "local-reviewed-negative"))
            dataset = "FASDD-v9" if source_id == "fasdd_v9" else "Pyro-SDIS"
            row = normalize_candidate(
                raw,
                "v7_reviewed_contextual_hard_negative_extension",
                dataset,
                "fire-smoke-ground-elite-rfdetr-small-v1-local-manifest",
                "manual_context_sheet_negative_review_passed",
            )
            rows.append(row)
            sources[row["sha256"]] = Path(raw["source_image"])
            known.add(row["sha256"])
            additions["v7_reviewed_contextual_hard_negative_extension"] += 1

    if len(known) != len(rows):
        raise RuntimeError("exact SHA duplication before capping")
    rows, recurrence = cap_recurrence(rows, maximum=12)
    rows, distribution_caps = enforce_distribution_caps(rows, source_cap=0.35, negative_cap=0.10)
    assert_frozen_holdouts_retained(rows, registry)
    assignment = assign_groupwise_splits(rows, registry)
    splits = {name: [] for name in ("train", "validation", "test")}
    materialization = Counter()
    for row in sorted(rows, key=lambda item: item["sha256"]):
        split = assignment[row["split_group_id"]]
        row["split"] = split
        row["file_name"] = f"images/{row['sha256']}.jpg"
        row["image_id"] = stable_number(row["sha256"]) % (2**53 - 1)
        materialization[link(sources[row["sha256"]], args.output / "data" / split / row["file_name"])] += 1
        splits[split].append(row)
    for split, members in splits.items():
        write_jsonl(args.output / "data" / split / "metadata.jsonl", members)
        coco = args.output / "annotations" / f"instances_{split}.json"
        coco.parent.mkdir(parents=True, exist_ok=True)
        coco.write_text(json.dumps(coco_for_split(members, split), sort_keys=True), encoding="utf-8")
    write_jsonl(args.output / "selection_manifest.jsonl", sorted(rows, key=lambda item: item["sha256"]))
    (args.output / "split_registry.json").write_text(json.dumps(make_registry(rows, registry), indent=2, sort_keys=True), encoding="utf-8")

    validation = validate(args.output, splits)
    (args.output / "reload_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    origin_counts = Counter(row["v7_origin"] for row in rows)
    source_counts = Counter(row["source_dataset"] for row in rows)
    scene_counts = Counter(row.get("scene_bin", "unknown") for row in rows)
    target_ratios = [area / (row["width"] * row["height"]) for row in rows for area in row["objects"]["area"]]
    provenance = {
        "schema": "fireviewer.pointing-v7-expanded-provenance.v1",
        "sources": [{"source_dataset": key[0], "source_revision": key[1], "license": key[2], "count": count}
                    for key, count in sorted(Counter((row["source_dataset"], row.get("source_revision", "unknown"), row.get("license", "unknown")) for row in rows).items())],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    split_audit = {split: {"images": len(members), "negative_images": sum(not row["objects"]["bbox"] for row in members), "sources": dict(Counter(row["source_dataset"] for row in members)), "scenes": dict(Counter(row.get("scene_bin", "unknown") for row in members))} for split, members in splits.items()}
    (args.output / "split_audit.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "schema": "fireviewer.pointing-v7-expanded-ready-local.v1",
        "status": "ready_for_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_count": len(rows),
        "counts": {key: len(value) for key, value in splits.items()},
        "candidate_additions_before_caps": dict(additions),
        "retained_by_origin": dict(origin_counts),
        "source_counts": dict(source_counts),
        "scene_counts": dict(scene_counts),
        "annotation_counts": validation["annotation_counts"],
        "negative_counts": validation["negative_images"],
        "negative_ratio": sum(validation["negative_images"].values()) / len(rows),
        "small_target_annotations": {"le_1pct": sum(value <= .01 for value in target_ratios), "le_0p5pct": sum(value <= .005 for value in target_ratios), "le_0p1pct": sum(value <= .001 for value in target_ratios)},
        "recurrence": recurrence,
        "distribution_caps": distribution_caps,
        "materialization": dict(materialization),
        "split_group_overlap": validation["split_group_overlap"],
        "test_status": "internal_groupwise_holdout_with_negatives; independent external test still required",
        "storage_policy": "hardlinks preferred; all source caches retained pending user review",
    }
    tracked = [args.output / name for name in ("selection_manifest.jsonl", "reload_validation.json", "provenance.json", "split_audit.json")]
    report["artifact_hashes"] = {path.name: file_sha(path) for path in tracked}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    card = f"""# FireViewer Pointing V7 Expanded\n\nStatus: `ready_for_training` (local artifact only).\n\n- Images: {len(rows)}\n- Split: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}\n- Exact SHA leakage: 0; split-group leakage: 0\n- Negatives: {sum(validation['negative_images'].values())} ({report['negative_ratio']:.2%})\n- This is an internal groupwise holdout, not an independent external test.\n- Source caches and prior artifacts are retained until user review.\n"""
    (args.output / "CORPUS_CARD.md").write_text(card, encoding="utf-8")
    inventory = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file() and item.name not in {"artifact_inventory.jsonl", "artifact_receipt.json"}):
        inventory.append({"path": path.relative_to(args.output).as_posix(), "bytes": path.stat().st_size, "sha256": path.stem if path.suffix.lower() == ".jpg" and len(path.stem) == 64 else file_sha(path)})
    write_jsonl(args.output / "artifact_inventory.jsonl", inventory)
    receipt = {"schema": "fireviewer.pointing-v7-expanded-artifact-receipt.v1", "status": "verified", "file_count": len(inventory), "logical_bytes": sum(row["bytes"] for row in inventory), "inventory_sha256": file_sha(args.output / "artifact_inventory.jsonl"), "report_sha256": file_sha(args.output / "report.json"), "selection_manifest_sha256": file_sha(args.output / "selection_manifest.jsonl")}
    (args.output / "artifact_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
