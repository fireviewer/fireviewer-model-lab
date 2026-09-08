#!/usr/bin/env python3
"""Build a bounded, event-grouped NEMO review pool for FireViewer V7.

The source archive contains correlated frames from wildfire videos.  This
builder rejects collages and invalid geometry, then keeps at most two frames
per normalized source event.  Nothing is admitted to training automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps

from training.pointing_dataset_v7.build_dfire_gap_candidates import (
    base_hashes,
    deduplicate,
    draw_sheets,
    write_jsonl,
)


OFFICIAL_REPOSITORY = "SayBender/Nemo"
OFFICIAL_REVISION = "a2e5576fec00ea0c0532cb4c81040ba388e54097"
SOURCE_ARCHIVE_SHA256 = "2e3e96ea0a82d5d5075be59baa9d6b37e7f4d7e92f400bb33a4250194513d19f"
LICENSE = "Apache-2.0"


def normalized_event(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"_(?:FR|SS)-?\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[-_]\d+$", "", stem)
    return re.sub(r"_+", "_", stem).strip("_").lower()


def deterministic(row: dict[str, Any]) -> int:
    return int(hashlib.sha256(f"nemo-v7:{row['source_split']}:{row['source_filename']}".encode()).hexdigest(), 16)


def gap_bucket(annotations: list[dict[str, Any]]) -> str:
    maximum = max(annotation["area_ratio"] for annotation in annotations)
    if maximum <= 0.001:
        return "smoke_tiny"
    if maximum <= 0.01:
        return "smoke_distant"
    if maximum <= 0.05:
        return "smoke_small"
    return "smoke_contextual"


def load_rows(source: Path) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    exclusions: Counter = Counter()
    for split in ("train", "val"):
        ann_root = source / split / "ann"
        image_root = source / split / "img"
        for annotation_path in sorted(ann_root.glob("*.json")):
            filename = annotation_path.name[:-5]
            if "collage" in filename.lower():
                exclusions["collage"] += 1
                continue
            image_path = image_root / filename
            if not image_path.is_file():
                exclusions["missing_image"] += 1
                continue
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            width = int(payload["size"]["width"])
            height = int(payload["size"]["height"])
            annotations = []
            valid = True
            for obj in payload.get("objects", []):
                if obj.get("geometryType") != "rectangle" or "smoke" not in obj.get("classTitle", "").lower():
                    valid = False
                    break
                (x1, y1), (x2, y2) = obj["points"]["exterior"]
                x1, x2 = sorted((max(0.0, float(x1)), min(float(width), float(x2))))
                y1, y2 = sorted((max(0.0, float(y1)), min(float(height), float(y2))))
                box_width, box_height = x2 - x1, y2 - y1
                if box_width <= 0 or box_height <= 0:
                    valid = False
                    break
                annotations.append({
                    "bbox": [x1, y1, box_width, box_height],
                    "category": 1,
                    "class_name": "smoke",
                    "source_density_class": obj["classTitle"],
                    "area_ratio": box_width * box_height / (width * height),
                })
            if not valid or not annotations:
                exclusions["empty_or_invalid_annotation"] += 1
                continue
            if max(a["area_ratio"] for a in annotations) > 0.40:
                exclusions["target_too_large"] += 1
                continue
            if min(min(a["bbox"][2], a["bbox"][3]) * 704 / max(width, height) for a in annotations) < 4:
                exclusions["target_below_four_pixels_at_704"] += 1
                continue
            with Image.open(image_path) as raw:
                rgb = raw.convert("RGB")
                if rgb.size != (width, height):
                    exclusions["dimension_mismatch"] += 1
                    continue
                sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
                phash = str(imagehash.phash(rgb, hash_size=8))
                flipped = str(imagehash.phash(ImageOps.mirror(rgb), hash_size=8))
            union_x1 = min(a["bbox"][0] for a in annotations)
            union_y1 = min(a["bbox"][1] for a in annotations)
            union_x2 = max(a["bbox"][0] + a["bbox"][2] for a in annotations)
            union_y2 = max(a["bbox"][1] + a["bbox"][3] for a in annotations)
            center_x = ((union_x1 + union_x2) / 2) / width
            center_y = ((union_y1 + union_y2) / 2) / height
            rows.append({
                "source_split": split,
                "filename": filename,
                "source_filename": filename,
                "source_image": str(image_path.resolve()),
                "source_annotation": str(annotation_path.resolve()),
                "source_event": normalized_event(filename),
                "width": width,
                "height": height,
                "annotations": annotations,
                "gap_bucket": gap_bucket(annotations),
                "target_max_area_ratio": max(a["area_ratio"] for a in annotations),
                "target_center_normalized": [center_x, center_y],
                "off_center": abs(center_x - 0.5) >= 0.2 or abs(center_y - 0.5) >= 0.2,
                "sha256": sha256,
                "phash64": phash,
                "phash64_flipped": flipped,
                "bytes": image_path.stat().st_size,
            })
    return rows, exclusions


def choose_per_event(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_event"]].append(row)
    chosen = []
    for event, event_rows in sorted(grouped.items()):
        # Prefer the smallest valid plume, then a geometrically different frame.
        ordered = sorted(event_rows, key=lambda row: (row["target_max_area_ratio"], deterministic(row)))
        selected = [ordered[0]]
        while len(selected) < cap and len(selected) < len(ordered):
            def distance(candidate: dict[str, Any]) -> tuple[float, int]:
                log_area = abs(__import__("math").log10(candidate["target_max_area_ratio"]) - __import__("math").log10(selected[0]["target_max_area_ratio"]))
                center = sum(abs(a - b) for a, b in zip(candidate["target_center_normalized"], selected[0]["target_center_normalized"]))
                return (log_area + center, -deterministic(candidate))
            selected.append(max((row for row in ordered if row not in selected), key=distance))
        for row in selected:
            row["split_group"] = f"nemo:event:{event}"
            chosen.append(row)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-event", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    (args.output / "candidate_images").mkdir(parents=True)

    prepared, source_exclusions = load_rows(args.source)
    event_selected = choose_per_event(prepared, args.max_per_event)
    exact, known_phashes = base_hashes(args.base)
    candidates, duplicate_exclusions = deduplicate(event_selected, exact, known_phashes)
    candidates.sort(key=lambda row: (row["gap_bucket"], row["source_event"], deterministic(row)))
    for index, row in enumerate(candidates, 1):
        row["candidate_id"] = f"v7-nemo-{index:05d}"
        row["candidate_image"] = f"{row['sha256']}.jpg"
        row["status"] = "needs_exhaustive_visual_review"
        row["training_admitted"] = False
        row["source_dataset"] = "nemo"
        row["source_repository"] = OFFICIAL_REPOSITORY
        row["source_revision"] = OFFICIAL_REVISION
        row["source_archive_sha256"] = SOURCE_ARCHIVE_SHA256
        row["license"] = LICENSE
        os.link(row["source_image"], args.output / "candidate_images" / row["candidate_image"])

    write_jsonl(args.output / "candidate_manifest.jsonl", candidates)
    write_jsonl(args.output / "review_decisions.template.jsonl", [
        {"candidate_id": row["candidate_id"], "decision": "", "reason": ""} for row in candidates
    ])
    draw_sheets(candidates, args.output)
    report = {
        "schema": "fireviewer.pointing-v7-nemo-gap-candidates.v1",
        "status": "awaiting_exhaustive_visual_review",
        "source_rows": 2934,
        "technically_prepared": len(prepared),
        "normalized_source_events": len({row["source_event"] for row in prepared}),
        "maximum_per_source_event": args.max_per_event,
        "selected_before_cross_corpus_dedup": len(event_selected),
        "candidate_count": len(candidates),
        "candidate_buckets": dict(Counter(row["gap_bucket"] for row in candidates)),
        "source_exclusions": dict(source_exclusions),
        "duplicate_exclusions": dict(duplicate_exclusions),
        "candidate_bytes_logical": sum(row["bytes"] for row in candidates),
        "materialization": "hardlink_only_to_retained_source_cache",
        "source": {
            "official_repository": OFFICIAL_REPOSITORY,
            "official_revision": OFFICIAL_REVISION,
            "archive_sha256": SOURCE_ARCHIVE_SHA256,
            "license": LICENSE,
        },
        "admission_rule": "no automatic admission; visual decision required for every candidate",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
