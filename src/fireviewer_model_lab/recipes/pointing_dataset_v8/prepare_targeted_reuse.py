"""Find genuinely additional reviewed small-target cases in retained local pools."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from training.pointing_dataset_v7.split_registry import canonical_source_group as event_key, digest, read_rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("artifact-root", "dataset", "fingerprints", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = read_rows(args.dataset / "selection_manifest.jsonl")
    known = {r["sha256"] for r in base}
    reserved = {event_key(r["source_group_id"]) for r in base if r["split"] != "train"}
    group_counts = Counter(event_key(r["source_group_id"]) for r in base)
    index = [(int(r["phash"], 16), int(r["phash_flipped"], 16)) for r in read_rows(args.fingerprints)]
    if {r["sha256"] for r in read_rows(args.fingerprints)} != known:
        raise ValueError("Fingerprint index does not cover the exact V7 corpus")
    pool = {}
    decisions = defaultdict(list)
    hp = args.artifact_root / "pointing-dataset-v5-hpwren-distant-smoke-ready-20260824"
    hp_accepted = {r["sha256"] for r in read_rows(hp / "review_decisions.jsonl") if r["decision"] == "accepted"}
    for row in read_rows(hp / "selection_manifest.jsonl"):
        if row["sha256"] in hp_accepted:
            pool[row["sha256"]] = row | {"source_image": str((hp / "data" / row["split"] / row["file_name"]).resolve()),
                "prior_review_ledger": str((hp / "review_decisions.jsonl").resolve())}
    for path in sorted(args.artifact_root.glob("pointing-v7-nemo-*/review_decisions.jsonl")):
        manifest = {r["candidate_id"]: r for r in read_rows(path.parent / "candidate_manifest.jsonl")}
        for decision in read_rows(path):
            raw = manifest[decision["candidate_id"]]
            decisions[raw["sha256"]].append(decision["decision"])
            if decision["decision"] != "accept":
                continue
            boxes = [r.get("bbox", r.get("bbox_xywh")) for r in raw["annotations"]]
            pool[raw["sha256"]] = raw | {"source_dataset": "NEMO", "source_group_id": raw["split_group"],
                "objects": {"bbox": boxes, "category": [1] * len(boxes), "area": [b[2] * b[3] for b in boxes]},
                "prior_review_ledger": str(path.resolve())}
    exclusions = Counter()
    selected = []
    sorted_pool = sorted(pool.values(), key=lambda r: (min(b[2] * b[3] for b in r["objects"]["bbox"]) / (r["width"] * r["height"]), r["sha256"]))
    for row in sorted_pool:
        sha = row["sha256"]
        group = event_key(row["source_group_id"])
        ratios = [box[2] * box[3] / (row["width"] * row["height"]) for box in row["objects"]["bbox"]]
        reason = ("already_in_v7" if sha in known else "conflicting_old_reviews" if "reject" in decisions[sha]
                  else "frozen_evaluation_event" if group in reserved else "not_small_target_priority" if min(ratios) > .005
                  else "event_already_has_four_views" if group_counts[group] >= 4 else None)
        if reason:
            exclusions[reason] += 1
            continue
        path = Path(row["source_image"])
        if digest(path) != sha:
            raise ValueError(f"Retained candidate bytes changed: {sha}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            a, af = int(str(imagehash.phash(rgb)), 16), int(str(imagehash.phash(ImageOps.mirror(rgb))), 16)
        if any(min((a ^ b).bit_count(), (af ^ b).bit_count(), (a ^ bf).bit_count()) <= 4 for b, bf in index):
            exclusions["near_duplicate_v7_or_selected_le4"] += 1
            continue
        selected.append(row | {"candidate_id": f"v8-reuse-{len(selected)+1:05d}",
                              "status": "previously_reviewed_candidate_pending_v8_admission", "training_admitted": False,
                              "canonical_event": group, "target_area_ratio_min": min(ratios),
                              "prior_review_ledger_sha256": digest(Path(row["prior_review_ledger"]))})
        index.append((a, af))
        group_counts[group] += 1
    args.output.mkdir(parents=True)
    (args.output / "candidate_manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in selected), encoding="utf-8")
    smoke100 = read_rows(args.artifact_root / "pointing-v7-smoke100-candidates-20260825-r1" / "candidate_manifest.jsonl")
    report = {"schema": "fireviewer.pointing-v8-targeted-reuse.v1", "status": "candidate_pool_only_not_training_ready",
              "reviewed_pool_unique_images": len(pool), "selected_candidates": len(selected),
              "source_counts": dict(Counter(r["source_dataset"] for r in selected)), "exclusions": dict(exclusions),
              "smoke100_existing_candidates": len(smoke100),
              "smoke100_targets_le_1pct": sum(r["target_area_ratio_min"] <= .01 for r in smoke100),
              "new_image_payload_bytes": 0, "training_started": False,
              "unresolved": ["V8 visual/admission review", "Additional complementary source acquisition still required for a substantial expansion"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
