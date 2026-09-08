#!/usr/bin/env python3
"""Build the reviewed V7 pointing corpus from V6 plus strictly reviewed additions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from training.pointing_dataset_v7.split_registry import (
    assign_locked, assert_frozen_holdouts_retained, make_registry,
)


LOCAL_ACCEPT = {
    *range(285, 301),
    302, 305, 307, 308, 309, 311, 312, 314, 315, 317, 318, 319, 320, 321, 322,
    323, 328, 333, 340, 345, 352, 358, 363, 370, 376, 382, 390, 397, 405, 413,
    424, 434, 445, 454, 464, 475, 486, 498, 510, 530, 557,
    573, 574, 575, 576, 580, 583, 587, 591, 592, 594, 597, 598, 600, 602, 605,
    610, 613, 616, 620, 623, 627, 628, 629, 631, 634, 638, 643, 646, 648, 649,
    651, 654, 656, 658, 661, 663, 667, 671, 672, 675,
}
SAINET_ACCEPT = {130, 137, 159}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_number(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def canonical_kind(row: dict[str, Any]) -> str:
    categories = set(row["objects"]["category"])
    if not categories:
        return "negative"
    if categories == {1}:
        return "smoke_only"
    if categories == {0}:
        return "fire_only"
    return "fire_smoke"


def assign_groupwise_splits(rows: list[dict[str, Any]], registry: dict | None = None) -> dict[str, str]:
    return assign_locked(rows, registry)


def cap_recurrence(rows: list[dict[str, Any]], maximum: int = 12) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["split_group_id"]].append(row)
    kept: list[dict[str, Any]] = []
    removed = 0
    for group_id, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda row: stable_number(f"v7-recurrence:{group_id}:{row['sha256']}"))
        kept.extend(ordered[:maximum])
        removed += max(0, len(ordered) - maximum)
    return kept, {"maximum_per_group": maximum, "removed": removed, "group_count": len(groups)}


def enforce_distribution_caps(rows: list[dict[str, Any]], source_cap: float = 0.35, negative_cap: float = 0.10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept = list(rows)
    removed = Counter()
    while True:
        source_counts = Counter(row["source_dataset"] for row in kept)
        source, count = source_counts.most_common(1)[0]
        if count / len(kept) <= source_cap:
            break
        remove_count = math.ceil((count - source_cap * len(kept)) / (1.0 - source_cap))
        candidates = [
            row for row in kept
            if row["source_dataset"] == source and row["objects"]["bbox"] and row.get("scene_bin") in {"fire_medium", "fire_contextual", "fire_small"}
        ]
        candidates.sort(key=lambda row: (row.get("scene_bin") != "fire_medium", stable_number(f"source-cap:{row['sha256']}")))
        if len(candidates) < remove_count:
            raise RuntimeError(f"cannot enforce source cap for {source}")
        rejected = {row["sha256"] for row in candidates[:remove_count]}
        kept = [row for row in kept if row["sha256"] not in rejected]
        removed[f"source_cap:{source}"] += remove_count
    while sum(not row["objects"]["bbox"] for row in kept) / len(kept) > negative_cap:
        candidates = [row for row in kept if not row["objects"]["bbox"] and row["v7_origin"] == "v4_reviewed_negative_reuse"]
        if not candidates:
            raise RuntimeError("cannot enforce negative cap")
        candidate = min(candidates, key=lambda row: stable_number(f"negative-cap:{row['sha256']}"))
        kept.remove(candidate)
        removed["negative_cap"] += 1
    return kept, {"source_cap": source_cap, "negative_cap": negative_cap, "removed": dict(removed)}


def source_for_base(v6: Path, old_split: str, file_name: str) -> Path:
    return v6 / "data" / old_split / file_name


def normalize_base(v6: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(v6 / "data" / split / "metadata.jsonl"):
            row = dict(row)
            sha = row["sha256"]
            sources[sha] = source_for_base(v6, split, row["file_name"])
            row["v6_split"] = split
            row["v7_origin"] = "v6_reviewed_base"
            rows.append(row)
    return rows, sources


def normalize_local(row: dict[str, Any]) -> dict[str, Any]:
    boxes, categories, areas = [], [], []
    for annotation in row["annotations"]:
        box = [float(value) for value in annotation["bbox_xywh"]]
        category = 1 if "smoke" in annotation["class_name"].lower() else 0
        boxes.append(box)
        categories.append(category)
        areas.append(box[2] * box[3])
    negative = not boxes
    return {
        "schema": "fireviewer.pointing-training-sample.v1",
        "sha256": row["sha256"],
        "width": row["width"],
        "height": row["height"],
        "objects": {"bbox": boxes, "category": categories, "area": areas},
        "caption": f"Verified annotations: fire={categories.count(0)}, smoke={categories.count(1)}.",
        "source_dataset": {"fasdd_v9": "fasdd", "pyro_sdis_a1e553e": "pyro-sdis"}.get(row["source_id"], row["source_id"]),
        "source_revision": "fire-smoke-ground-elite-rfdetr-small-v1",
        "source_record_id": row["source_record_id"],
        "source_split": row["split"],
        "source_group_id": row["split_group"],
        "split_group_id": row["split_group"],
        "license": row["license"],
        "quality_gate": "v7_full_sheet_and_target_zoom_visual_review_passed",
        "annotation_exploitable": True,
        "negative_verified": negative,
        "training_admitted": True,
        "scene_bin": row["gap_bucket"],
        "aerial": False,
        "centered": False,
        "poor_framing": False,
        "person_risk_reviewed_clear": True,
        "v7_origin": "v7_local_reviewed_extension",
        "visual_review_decision_id": row["candidate_id"],
    }


def normalize_sainet(row: dict[str, Any]) -> dict[str, Any]:
    boxes = [[float(value) for value in box] for box in row["objects"]["bbox"]]
    categories = [int(value) for value in row["objects"]["category"]]
    return {
        "schema": "fireviewer.pointing-training-sample.v1",
        "sha256": row["sha256"],
        "width": row["width"],
        "height": row["height"],
        "objects": {"bbox": boxes, "category": categories, "area": [box[2] * box[3] for box in boxes]},
        "caption": f"Verified annotations: fire={categories.count(0)}, smoke={categories.count(1)}.",
        "source_dataset": "SAINetset_v8.0",
        "source_revision": "ff34e4058939ec6ec5d2024bdaf3e58199494372",
        "source_record_id": row["source_record_id"],
        "source_split": "train",
        "source_group_id": row["split_group"],
        "split_group_id": row["split_group"],
        "license": row["license"],
        "quality_gate": "v7_full_sheet_and_target_zoom_visual_review_passed",
        "annotation_exploitable": True,
        "negative_verified": False,
        "training_admitted": True,
        "scene_bin": row["gap_bucket"],
        "aerial": False,
        "centered": False,
        "poor_framing": False,
        "person_risk_reviewed_clear": True,
        "v7_origin": "v7_hf_reviewed_extension",
        "visual_review_decision_id": row["candidate_id"],
    }


def load_v4_reviewed_negatives(artifact_root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    ledger_path = artifact_root / "pointing-dataset-v4-final-zero-human-merged-20260824" / "decision_ledger.jsonl"
    accepted = {
        (row["lot"], row["pool_id"])
        for row in read_jsonl(ledger_path)
        if row["decision"] == "accepted" and row.get("full_resolution_visual_confirmation") is True
    }
    pool_specs = {
        "r2": artifact_root / "pointing-dataset-v4-review-plan-r2-20260824" / "review_pool.jsonl",
        "fire_supplement": artifact_root / "pointing-dataset-v4-fire-supplement-20260824" / "fire_supplement_pool.jsonl",
    }
    selected: list[dict[str, Any]] = []
    for lot, path in pool_specs.items():
        for row in read_jsonl(path):
            if (lot, row["pool_id"]) in accepted and row.get("annotation_count", len(row.get("annotations", []))) == 0:
                if row["image_storage"] != "parquet_embedded":
                    raise RuntimeError(f"unexpected V4 negative storage: {row['pool_id']}")
                selected.append(row)
    # Bounded reuse keeps the clean-negative ratio inside 8-10% after
    # recurrence capping. Deterministic SHA order makes it reproducible.
    selected = sorted(selected, key=lambda row: row["sha256"])[:96]

    import pyarrow.parquet as pq

    requests: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    parquet_files: dict[str, Any] = {}
    for row in selected:
        locator = row["image_locator"]
        path = str(Path(locator["parquet"]).resolve())
        parquet = parquet_files.setdefault(path, pq.ParquetFile(path))
        absolute_row = int(locator["row"])
        offset = 0
        for group_index in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group_index).num_rows
            if absolute_row < offset + count:
                requests[(path, group_index)].append((absolute_row - offset, row["sha256"]))
                break
            offset += count
        else:
            raise RuntimeError(f"V4 parquet row out of range: {locator}")
    payloads: dict[str, bytes] = {}
    for (path, group_index), items in requests.items():
        table = parquet_files[path].read_row_group(group_index, columns=["image"])
        for local_row, sha in items:
            payload = table["image"][local_row].as_py()["bytes"]
            if hashlib.sha256(payload).hexdigest() != sha:
                raise RuntimeError(f"V4 embedded image SHA mismatch: {sha}")
            payloads[sha] = payload

    normalized = []
    for row in selected:
        normalized.append({
            "schema": "fireviewer.pointing-training-sample.v1",
            "sha256": row["sha256"],
            "width": row["width"],
            "height": row["height"],
            "objects": {"bbox": [], "category": [], "area": []},
            "caption": "Visually verified contextual hard negative: no visible fire or smoke.",
            "source_dataset": row["source_dataset"],
            "source_revision": row.get("source_revision", row.get("source_version", "v4-pinned-source")),
            "source_record_id": row.get("source_record_id", row["pool_id"]),
            "source_split": row.get("original_split", "test"),
            "source_group_id": row["recurrence_group_id"],
            "split_group_id": row["recurrence_group_id"],
            "license": row["license"],
            "quality_gate": "v4_full_resolution_exhaustive_visual_review_passed_reused_in_v7",
            "annotation_exploitable": True,
            "negative_verified": True,
            "training_admitted": True,
            "scene_bin": "hard_negative",
            "aerial": False,
            "centered": False,
            "poor_framing": False,
            "person_risk_reviewed_clear": True,
            "v7_origin": "v4_reviewed_negative_reuse",
            "visual_review_decision_id": row["pool_id"],
        })
    return normalized, payloads


def decision_rows(rows: list[dict[str, Any]], accepted: set[int], prefix: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        number = int(row["candidate_id"].rsplit("-", 1)[-1])
        accept = number in accepted
        if accept:
            reason = "manual_full_sheet_and_zoom_review_passed"
        elif row.get("automatic_exclusion_reasons"):
            reason = ";".join(row["automatic_exclusion_reasons"])
        elif prefix == "local" and number <= 284:
            reason = "out_of_domain_negative_or_person_product_indoor_contamination"
        elif prefix == "sainet" and any(token in row.get("source_image_path", "").lower() for token in ("generated", "sintetico", "seedream")):
            reason = "synthetic_or_generated_image_excluded"
        else:
            reason = "visual_recurrence_risk_or_task_geometry_context_not_exploitable"
        output.append({"candidate_id": row["candidate_id"], "decision": "accept" if accept else "reject", "reason": reason})
    return output


def link(source: Path | bytes, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(source, bytes):
        destination.write_bytes(source)
        return "parquet_extract"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy_fallback"


def coco_for_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    images, annotations = [], []
    annotation_id = 1
    for image_id, row in enumerate(rows, 1):
        images.append({"id": image_id, "file_name": f"../data/{split}/{row['file_name']}", "width": row["width"], "height": row["height"]})
        for box, category, area in zip(row["objects"]["bbox"], row["objects"]["category"], row["objects"]["area"]):
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": category + 1, "bbox": box, "area": area, "iscrowd": 0})
            annotation_id += 1
    return {"info": {"description": "FireViewer pointing V7", "version": "7"}, "images": images, "annotations": annotations, "categories": [{"id": 1, "name": "fire"}, {"id": 2, "name": "smoke"}]}


def validate(output: Path, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    groups: dict[str, set[str]] = {}
    negatives: dict[str, int] = {}
    annotations = Counter()
    for split, rows in splits.items():
        groups[split] = {row["split_group_id"] for row in rows}
        negatives[split] = sum(not row["objects"]["bbox"] for row in rows)
        for row in rows:
            path = output / "data" / split / row["file_name"]
            observed = file_sha(path)
            if observed != row["sha256"]:
                raise RuntimeError(f"SHA mismatch: {path}")
            hashes[row["sha256"]] = split
            with Image.open(path) as image:
                if image.width != row["width"] or image.height != row["height"]:
                    raise RuntimeError(f"dimension mismatch: {path}")
            for box, category in zip(row["objects"]["bbox"], row["objects"]["category"]):
                x, y, width, height = box
                if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > row["width"] + 1e-3 or y + height > row["height"] + 1e-3:
                    raise RuntimeError(f"invalid bbox: {row['sha256']} {box}")
                annotations["fire" if category == 0 else "smoke"] += 1
    if len(hashes) != sum(map(len, splits.values())):
        raise RuntimeError("exact duplicate SHA across V7")
    overlaps = {f"{a}_{b}": len(groups[a] & groups[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(overlaps.values()):
        raise RuntimeError(f"group leakage: {overlaps}")
    if any(value == 0 for value in negatives.values()):
        raise RuntimeError(f"negative coverage missing: {negatives}")
    return {"status": "passed", "image_count": len(hashes), "split_group_overlap": overlaps, "negative_images": negatives, "annotation_counts": dict(annotations)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--local-candidates", type=Path, required=True)
    parser.add_argument("--sainet-candidates", type=Path, required=True)
    parser.add_argument("--v4-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    base_rows, sources = normalize_base(args.v6)
    registry = make_registry([row | {"split": row["v6_split"]} for row in base_rows])
    local_all = read_jsonl(args.local_candidates / "candidate_manifest.jsonl")
    sainet_all = read_jsonl(args.sainet_candidates / "candidate_manifest.jsonl")
    local_selected = [row for row in local_all if int(row["candidate_id"].rsplit("-", 1)[-1]) in LOCAL_ACCEPT]
    sainet_selected = [row for row in sainet_all if int(row["candidate_id"].rsplit("-", 1)[-1]) in SAINET_ACCEPT]
    v4_negatives, v4_payloads = load_v4_reviewed_negatives(args.v4_artifact_root)
    rows = list(base_rows)
    for raw in local_selected:
        normalized = normalize_local(raw)
        rows.append(normalized)
        sources[normalized["sha256"]] = Path(raw["source_image"])
    for raw in sainet_selected:
        normalized = normalize_sainet(raw)
        rows.append(normalized)
        sources[normalized["sha256"]] = args.sainet_candidates / raw["candidate_image"]
    for normalized in v4_negatives:
        rows.append(normalized)
        sources[normalized["sha256"]] = v4_payloads[normalized["sha256"]]

    deduped = {row["sha256"]: row for row in rows}
    if len(deduped) != len(rows):
        raise RuntimeError("selected additions contain exact V6 duplicates")
    rows, recurrence = cap_recurrence(list(deduped.values()))
    rows, distribution_caps = enforce_distribution_caps(rows)
    assert_frozen_holdouts_retained(rows, registry)
    assignment = assign_groupwise_splits(rows, registry)
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    materialization = Counter()
    for row in sorted(rows, key=lambda item: item["sha256"]):
        split = assignment[row["split_group_id"]]
        row["split"] = split
        row["file_name"] = f"images/{row['sha256']}.jpg"
        row["image_id"] = stable_number(row["sha256"]) % (2**53 - 1)
        materialization[link(sources[row["sha256"]], args.output / "data" / split / row["file_name"])] += 1
        splits[split].append(row)
    for split, split_rows in splits.items():
        write_jsonl(args.output / "data" / split / "metadata.jsonl", split_rows)
        coco_path = args.output / "annotations" / f"instances_{split}.json"
        coco_path.parent.mkdir(parents=True, exist_ok=True)
        coco_path.write_text(json.dumps(coco_for_split(split_rows, split), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    local_decisions = decision_rows(local_all, LOCAL_ACCEPT, "local")
    sainet_decisions = decision_rows(sainet_all, SAINET_ACCEPT, "sainet")
    write_jsonl(args.output / "review" / "local_candidate_decisions.jsonl", local_decisions)
    write_jsonl(args.output / "review" / "sainet_candidate_decisions.jsonl", sainet_decisions)
    write_jsonl(args.output / "selection_manifest.jsonl", sorted(rows, key=lambda item: item["sha256"]))
    (args.output / "split_registry.json").write_text(json.dumps(make_registry(rows, registry), indent=2, sort_keys=True), encoding="utf-8")

    legacy_rows = read_jsonl(args.v6 / "data" / "test" / "metadata.jsonl")
    write_jsonl(args.output / "benchmark_panels" / "legacy_v6_test" / "metadata.jsonl", legacy_rows)
    legacy_links = Counter()
    for row in legacy_rows:
        legacy_links[link(args.v6 / "data" / "test" / row["file_name"], args.output / "benchmark_panels" / "legacy_v6_test" / row["file_name"])] += 1

    validation = validate(args.output, splits)
    (args.output / "reload_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")
    target_ratios = []
    for row in rows:
        pixels = row["width"] * row["height"]
        target_ratios.extend(area / pixels for area in row["objects"]["area"])
    quality_metrics = {
        "aerial_images": sum(bool(row.get("aerial")) for row in rows),
        "poor_framing_images": sum(bool(row.get("poor_framing")) for row in rows),
        "poor_framing_ratio": sum(bool(row.get("poor_framing")) for row in rows) / len(rows),
        "target_annotations_le_1pct": sum(ratio <= 0.01 for ratio in target_ratios),
        "target_annotations_le_0p5pct": sum(ratio <= 0.005 for ratio in target_ratios),
        "target_annotations_le_0p1pct": sum(ratio <= 0.001 for ratio in target_ratios),
        "maximum_source_ratio": max(Counter(row["source_dataset"] for row in rows).values()) / len(rows),
    }
    provenance = {
        "schema": "fireviewer.pointing-v7-provenance.v1",
        "sources": [
            {"source_dataset": key[0], "source_revision": key[1], "license": key[2], "count": count}
            for key, count in sorted(Counter((row["source_dataset"], row.get("source_revision", "unknown"), row.get("license", "unknown")) for row in rows).items())
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    split_audit = {
        split: {
            "images": len(split_rows),
            "negative_images": sum(not row["objects"]["bbox"] for row in split_rows),
            "sources": dict(Counter(row["source_dataset"] for row in split_rows)),
            "scenes": dict(Counter(row.get("scene_bin", "unknown") for row in split_rows)),
        }
        for split, split_rows in splits.items()
    }
    (args.output / "split_audit.json").write_text(json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8")
    report = {
        "schema": "fireviewer.pointing-v7-ready-local.v1",
        "status": "ready_for_training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base": str(args.v6.resolve()),
        "counts": {split: len(values) for split, values in splits.items()},
        "selected_count": len(rows),
        "extension": {
            "candidate_accepts": {"local_reviewed": len(local_selected), "sainet_reviewed": len(sainet_selected), "v4_reviewed_negative_reuse": len(v4_negatives)},
            "retained_by_origin": dict(Counter(row["v7_origin"] for row in rows)),
        },
        "negative_counts": validation["negative_images"],
        "negative_ratio": sum(validation["negative_images"].values()) / len(rows),
        "annotation_counts": validation["annotation_counts"],
        "source_counts": dict(Counter(row["source_dataset"] for row in rows)),
        "scene_counts": dict(Counter(row.get("scene_bin", "unknown") for row in rows)),
        "quality_metrics": quality_metrics,
        "recurrence": recurrence,
        "distribution_caps": distribution_caps,
        "materialization": dict(materialization),
        "legacy_regression_panel": {"count": len(legacy_rows), "materialization": dict(legacy_links), "role": "paired V5/V6 regression only; not an external independent test"},
        "test_status": "internal_groupwise_holdout_with_negatives; external_independent_test_not_available",
        "storage_policy": "hardlinks used where supported; source caches retained pending user review",
        "review_contract": {"synthetic_generated_excluded": True, "out_of_domain_negatives_excluded": True, "selfie_person_foreground_unsafe_aerial_excluded_from_new_extension": True, "recurrence_manual_selection_plus_deterministic_cap": True},
    }
    tracked = [args.output / "selection_manifest.jsonl", args.output / "reload_validation.json", args.output / "provenance.json", args.output / "split_audit.json", args.output / "review" / "local_candidate_decisions.jsonl", args.output / "review" / "sainet_candidate_decisions.jsonl"]
    report["artifact_hashes"] = {path.relative_to(args.output).as_posix(): file_sha(path) for path in tracked}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    card = f"""# FireViewer Pointing V7\n\nStatus: `ready_for_training` (local artifact only).\n\n- Images: {len(rows)} (`train={len(splits['train'])}`, `validation={len(splits['validation'])}`, `test={len(splits['test'])}`)\n- Clean contextual negatives: {sum(validation['negative_images'].values())} ({report['negative_ratio']:.2%})\n- Group leakage: 0; exact SHA leakage: 0\n- Maximum recurrence group: {recurrence['maximum_per_group']} images\n- COCO categories: `fire`, `smoke`\n- Test status: internal groupwise holdout. It is not an independent external test.\n- The frozen V6 test panel is retained only for historical paired regression; it must not be used to claim V7 generalization.\n- Images, source caches, and prior artifacts are retained until manual review.\n"""
    (args.output / "CORPUS_CARD.md").write_text(card, encoding="utf-8", newline="\n")
    inventory_rows = []
    for path in sorted(item for item in args.output.rglob("*") if item.is_file() and item.name not in {"artifact_inventory.jsonl", "artifact_receipt.json"}):
        relative = path.relative_to(args.output).as_posix()
        observed_sha = path.stem if path.suffix.lower() == ".jpg" and len(path.stem) == 64 else file_sha(path)
        inventory_rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed_sha})
    write_jsonl(args.output / "artifact_inventory.jsonl", inventory_rows)
    receipt = {
        "schema": "fireviewer.pointing-v7-artifact-receipt.v1",
        "status": "verified",
        "file_count": len(inventory_rows),
        "logical_bytes": sum(row["bytes"] for row in inventory_rows),
        "inventory_sha256": file_sha(args.output / "artifact_inventory.jsonl"),
        "report_sha256": file_sha(args.output / "report.json"),
        "selection_manifest_sha256": file_sha(args.output / "selection_manifest.jsonl"),
    }
    (args.output / "artifact_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
