"""Read-only split audit: bytes, original source groups and mirrored pHash candidates."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from training.pointing_dataset_v7.split_registry import canonical_source_group, digest, read_rows


def fingerprints(item):
    row, root = item
    path = root / "data" / row["split"] / row["file_name"]
    if digest(path) != row["sha256"]:
        raise ValueError(f"Image digest mismatch: {path}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (row["width"], row["height"]):
            raise ValueError(f"Image dimensions mismatch: {path}")
        return {"sha256": row["sha256"], "split": row["split"],
                "phash": str(imagehash.phash(rgb)), "phash_flipped": str(imagehash.phash(ImageOps.mirror(rgb)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_rows(args.dataset / "selection_manifest.jsonl")
    groups = defaultdict(list)
    for row in rows:
        groups[canonical_source_group(row["source_group_id"])].append(row)
    conflicts = [{"source_group": group, "members": [{"sha256": r["sha256"], "split": r["split"],
                  "split_group_id": r["split_group_id"]} for r in members]}
                 for group, members in groups.items() if len({r["split"] for r in members}) > 1]
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = list(executor.map(fingerprints, [(r, args.dataset) for r in rows]))
    print(f"Verified and fingerprinted {len(hashes)} images", flush=True)
    pairs = []
    values = [(h, int(h["phash"], 16), int(h["phash_flipped"], 16)) for h in hashes]
    for index, (left, a, af) in enumerate(values):
        for right, b, bf in values[index + 1:]:
            if left["split"] == right["split"]:
                continue
            distance = min((a ^ b).bit_count(), (a ^ bf).bit_count(), (af ^ b).bit_count())
            if distance <= 4:
                pairs.append({"left": left["sha256"], "left_split": left["split"], "right": right["sha256"],
                              "right_split": right["split"], "phash_distance": distance})
    # Quarantine evaluation members, never undo previous training exposure or move test into training.
    quarantine = defaultdict(set)
    for conflict in conflicts:
        members = conflict["members"]
        if any(r["split"] == "train" for r in members):
            for r in members:
                if r["split"] != "train":
                    quarantine[r["sha256"]].add("declared_source_group_shared_with_train")
        else:
            for r in members:
                if r["split"] == "test":
                    quarantine[r["sha256"]].add("declared_source_group_shared_with_validation")
    for pair in pairs:
        for side, other in (("left", "right"), ("right", "left")):
            split, other_split = pair[f"{side}_split"], pair[f"{other}_split"]
            if split != "train" and (other_split == "train" or split == "test"):
                quarantine[pair[side]].add(f"phash_candidate_shared_with_{other_split}")
    # A suspect frame excludes its complete evaluation group, not just that frame.
    changed = True
    while changed:
        changed = False
        blocked_groups = {r["split_group_id"] for r in rows if r["sha256"] in quarantine}
        blocked_sources = {canonical_source_group(r["source_group_id"]) for r in rows if r["sha256"] in quarantine}
        for row in rows:
            if row["split"] != "train" and row["sha256"] not in quarantine and (
                row["split_group_id"] in blocked_groups or canonical_source_group(row["source_group_id"]) in blocked_sources
            ):
                quarantine[row["sha256"]].add("same_evaluation_group_as_quarantined_image")
                changed = True
    args.output.mkdir(parents=True)
    for name, data in (("fingerprints.jsonl", hashes), ("source_group_conflicts.jsonl", conflicts),
                       ("near_duplicate_candidates.jsonl", pairs),
                       ("quarantine.jsonl", [{"sha256": sha, "reasons": sorted(reasons)} for sha, reasons in sorted(quarantine.items())])):
        (args.output / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in data), encoding="utf-8")
    report = {"schema": "fireviewer.pointing-split-forensic-audit.v1", "status": "completed",
              "manifest_sha256": digest(args.dataset / "selection_manifest.jsonl"), "images_verified": len(hashes),
              "source_group_conflicts": len(conflicts), "cross_split_phash_pairs_le4": len(pairs),
              "quarantined_evaluation_images": dict(Counter(r["split"] for r in rows if r["sha256"] in quarantine)),
              "remaining_counts": dict(Counter(r["split"] for r in rows if r["sha256"] not in quarantine)),
              "qualification": "Conservative exclusion of unresolved similarities, not a claim that every pHash candidate is a duplicate.",
              "original_images_modified_or_deleted": False, "test_model_predictions_used_for_selection": False}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
