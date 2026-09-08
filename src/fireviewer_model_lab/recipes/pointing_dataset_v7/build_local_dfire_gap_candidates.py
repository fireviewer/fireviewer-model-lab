#!/usr/bin/env python3
"""Reuse verified local D-Fire caches to create a strict V7 review pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from training.pointing_dataset_v7.build_dfire_gap_candidates import (
    MIRROR_REVISION,
    OFFICIAL_REVISION,
    base_hashes,
    bucket,
    deduplicate,
    draw_sheets,
    parse_labels,
    read_jsonl,
    write_jsonl,
)


def load_sources(root: Path) -> list[dict]:
    specs = [
        (root / "pointing-dataset-v4-dfire-fire-extension-20260824", "source_rows.jsonl"),
        (root / "pointing-dataset-v4-dfire-fire-reusegroup-extension-20260824", "source_rows.jsonl"),
        (root / "pointing-dataset-v4-fire-supplement-20260824", "external_source_all_rows.jsonl"),
    ]
    by_sha = {}
    for artifact, name in specs:
        for row in read_jsonl(artifact / name):
            if "filename" not in row or not row["filename"].lower().startswith("web"):
                continue
            source_image = artifact / row["image_relpath"]
            if not source_image.is_file():
                continue
            by_sha[row["image_sha256"]] = {**row, "source_image": str(source_image.resolve())}
    return list(by_sha.values())


def deterministic(row: dict) -> int:
    return int(hashlib.sha256(f"local-dfire:{row['filename']}".encode()).hexdigest(), 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    (args.output / "candidate_images").mkdir(parents=True)

    base_rows = read_jsonl(args.base / "selection_manifest.jsonl")
    exact = {row["sha256"] for row in base_rows}
    prepared = []
    for row in load_sources(args.v4_root):
        if row["image_sha256"] in exact:
            continue
        annotations = parse_labels(row["label_text"], int(row["width"]), int(row["height"]))
        gap = bucket(annotations)
        if gap is None:
            continue
        if min(min(annotation["bbox"][2], annotation["bbox"][3]) * 704 / max(row["width"], row["height"]) for annotation in annotations) < 4:
            continue
        with Image.open(row["source_image"]) as raw:
            rgb = raw.convert("RGB")
            phash = str(imagehash.phash(rgb, hash_size=8))
            flipped = str(imagehash.phash(ImageOps.mirror(rgb), hash_size=8))
        prepared.append({
            **row,
            "sha256": row["image_sha256"],
            "annotations": annotations,
            "gap_bucket": gap,
            "phash64": phash,
            "phash64_flipped": flipped,
            "candidate_image": f"{row['image_sha256']}.jpg",
            "bytes": Path(row["source_image"]).stat().st_size,
        })

    _, known_phashes = base_hashes(args.base)
    deduped, exclusions = deduplicate(prepared, exact, known_phashes)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in deduped:
        number = int(Path(row["filename"]).stem[3:])
        group = f"dfire:web:block-{number // 100:04d}"
        row["split_group"] = group
        grouped[group].append(row)
    candidates = []
    for group, rows in sorted(grouped.items()):
        candidates.extend(sorted(rows, key=lambda row: (row["gap_bucket"] not in {"mixed_small", "fire_small"}, deterministic(row)))[:12])
    candidates.sort(key=lambda row: (row["gap_bucket"], deterministic(row)))
    for index, row in enumerate(candidates, 1):
        row["candidate_id"] = f"v7-dfire-local-{index:05d}"
        row["status"] = "needs_visual_review"
        row["training_admitted"] = False
        row["source_dataset"] = "dfire"
        row["source_revision"] = OFFICIAL_REVISION
        row["mirror_revision"] = MIRROR_REVISION
        row["license"] = "CC0-1.0"
        destination = args.output / "candidate_images" / row["candidate_image"]
        os.link(row["source_image"], destination)
    write_jsonl(args.output / "candidate_manifest.jsonl", candidates)
    write_jsonl(args.output / "review_decisions.template.jsonl", [{"candidate_id": row["candidate_id"], "decision": "", "reason": ""} for row in candidates])
    draw_sheets(candidates, args.output)
    report = {
        "schema": "fireviewer.pointing-v7-local-dfire-gap-candidates.v1",
        "status": "awaiting_exhaustive_visual_review",
        "local_unique_source_images": len(load_sources(args.v4_root)),
        "technically_prepared": len(prepared),
        "candidate_count": len(candidates),
        "candidate_buckets": dict(Counter(row["gap_bucket"] for row in candidates)),
        "group_count": len(grouped),
        "maximum_per_source_block": 12,
        "automatic_exclusions": dict(exclusions),
        "candidate_bytes_logical": sum(row["bytes"] for row in candidates),
        "materialization": "hardlink_only",
        "source": {"official_revision": OFFICIAL_REVISION, "mirror_revision": MIRROR_REVISION, "license": "CC0-1.0"},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
