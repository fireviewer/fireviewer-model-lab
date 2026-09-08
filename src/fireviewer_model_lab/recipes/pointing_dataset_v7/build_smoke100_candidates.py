#!/usr/bin/env python3
"""Prepare a bounded, review-only Smoke100 extension from one pinned HF zip."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import imagehash
from PIL import Image, ImageDraw, ImageOps

from training.pointing_dataset_v7.build_dfire_gap_candidates import base_hashes, hamming, write_jsonl

REVISION = "ccfc48a7e02b349c04c506937c014b85945130ee"


def key(row: dict) -> int:
    return int(hashlib.sha256(f"smoke100:{row['source_record_id']}".encode()).hexdigest(), 16)


def group_for(filename: str) -> str:
    match = re.match(r"([A-Za-z]+)_(\d+)", filename)
    if not match:
        return f"smoke100:file:{filename.lower()}"
    return f"smoke100:{match.group(1).lower()}:block-{int(match.group(2)) // 25:04d}"


def draw_sheets(rows: list[dict], output: Path, per_sheet: int = 20) -> None:
    sheets = output / "review_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    tw, th = 320, 250
    for page_index in range(math.ceil(len(rows) / per_sheet)):
        page = rows[page_index * per_sheet : (page_index + 1) * per_sheet]
        sheet = Image.new("RGB", (tw * 5, th * 4), "white")
        for index, row in enumerate(page):
            with Image.open(output / "candidate_images" / row["candidate_image"]) as raw:
                image = ImageOps.contain(raw.convert("RGB"), (tw, th - 25))
            draw = ImageDraw.Draw(image)
            sx, sy = image.width / row["width"], image.height / row["height"]
            for x, y, w, h in row["objects"]["bbox"]:
                draw.rectangle((x * sx, y * sy, (x + w) * sx, (y + h) * sy), outline="#1683ff", width=3)
            x0, y0 = (index % 5) * tw, (index // 5) * th
            sheet.paste(image, (x0, y0))
            ImageDraw.Draw(sheet).text((x0 + 4, y0 + th - 22), f"{row['candidate_id']} smoke_varied", fill="black")
        sheet.save(sheets / f"review-{page_index + 1:04d}.jpg", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    (args.output / "candidate_images").mkdir(parents=True)
    base_exact, base_phashes = base_hashes(args.base)
    rows = []
    with ZipFile(args.zip) as archive:
        coco = json.loads(archive.read("_annotations.coco.json"))
        annotations = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
        for image_row in coco["images"]:
            image_annotations = annotations[image_row["id"]]
            if not image_annotations:
                continue
            boxes = [[float(value) for value in annotation["bbox"]] for annotation in image_annotations]
            ratios = [box[2] * box[3] / (image_row["width"] * image_row["height"]) for box in boxes]
            if max(ratios) > 0.40:
                continue
            payload = archive.read(image_row["file_name"])
            sha = hashlib.sha256(payload).hexdigest()
            if sha in base_exact:
                continue
            with Image.open(io.BytesIO(payload)) as raw:
                rgb = raw.convert("RGB")
                phash = str(imagehash.phash(rgb, hash_size=8))
                flipped = str(imagehash.phash(ImageOps.mirror(rgb), hash_size=8))
            if any(hamming(probe, known) <= 3 for probe in (phash, flipped) for known in base_phashes):
                continue
            rows.append({
                "source_record_id": image_row["file_name"], "sha256": sha, "width": image_row["width"], "height": image_row["height"],
                "objects": {"bbox": boxes, "category": [1] * len(boxes), "area": [box[2] * box[3] for box in boxes]},
                "target_area_ratio_min": min(ratios), "target_area_ratio_max": max(ratios), "phash64": phash, "phash64_flipped": flipped,
                "payload": payload, "split_group": group_for(image_row["file_name"]), "gap_bucket": "smoke_varied",
            })
    accepted = []
    known = []
    for row in sorted(rows, key=key):
        if any(hamming(probe, other) <= 3 for probe in (row["phash64"], row["phash64_flipped"]) for other in known):
            continue
        accepted.append(row)
        known.extend((row["phash64"], row["phash64_flipped"]))
    grouped = defaultdict(list)
    for row in accepted:
        grouped[row["split_group"]].append(row)
    capped = []
    for group, group_rows in sorted(grouped.items()):
        capped.extend(sorted(group_rows, key=key)[:6])
    capped.sort(key=key)
    with ZipFile(args.zip) as archive:
        for index, row in enumerate(capped, 1):
            row["candidate_id"] = f"v7-smoke100-{index:05d}"
            row["candidate_image"] = f"{row['sha256']}.jpg"
            (args.output / "candidate_images" / row["candidate_image"]).write_bytes(row.pop("payload"))
            row.update({"status": "needs_visual_review", "training_admitted": False, "source_dataset": "Smoke100", "source_revision": REVISION, "license": "CC-BY-4.0"})
    write_jsonl(args.output / "candidate_manifest.jsonl", capped)
    write_jsonl(args.output / "review_decisions.template.jsonl", [{"candidate_id": row["candidate_id"], "decision": "", "reason": ""} for row in capped])
    draw_sheets(capped, args.output)
    report = {
        "schema": "fireviewer.pointing-v7-smoke100-candidates.v1", "status": "awaiting_exhaustive_visual_review",
        "zip_sha256": hashlib.sha256(args.zip.read_bytes()).hexdigest(), "source_revision": REVISION, "license": "CC-BY-4.0",
        "geometry_eligible_before_dedup": len(rows), "candidate_count": len(capped), "group_count": len(grouped),
        "maximum_per_numeric_block": 6, "candidate_bytes": sum((args.output / "candidate_images" / row["candidate_image"]).stat().st_size for row in capped),
        "storage_policy": "only pinned test zip plus bounded candidates retained pending review",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
