#!/usr/bin/env python3
"""Build a bounded, review-only D-Fire extension without downloading the full dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import imagehash
import requests
from PIL import Image, ImageDraw, ImageOps

DATASET = "badsaarow/d-fire"
MIRROR_REVISION = "27f81e6f7d32fe3b29d366da9573b6150a3d148f"
OFFICIAL_REVISION = "4bf9c31b18fadcd44d5f0b6d66f82bc56fa5e328"
ROWS_URL = "https://datasets-server.huggingface.co/rows"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fetch_page(split: str, offset: int, length: int = 100) -> list[dict[str, Any]]:
    for attempt in range(8):
        response = requests.get(ROWS_URL, params={"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": length}, timeout=60)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()["rows"]
        time.sleep(min(30, 2 ** attempt))
    response.raise_for_status()
    raise RuntimeError("unreachable")


def fetch_inventory() -> list[dict[str, Any]]:
    jobs = [("train", offset) for offset in range(0, 17221, 100)] + [("test", offset) for offset in range(0, 4306, 100)]
    pages: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_map = {executor.submit(fetch_page, split, offset): (split, offset) for split, offset in jobs}
        for future in as_completed(future_map):
            pages[future_map[future]] = future.result()
    rows = []
    for split, offset in jobs:
        for wrapped in pages[(split, offset)]:
            row = wrapped["row"]
            rows.append({"source_split": split, "row_idx": wrapped["row_idx"], **row})
    return rows


def parse_labels(text: str, width: int, height: int) -> list[dict[str, Any]]:
    annotations = []
    for line in text.splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, bw, bh = map(float, line.split())
        # Official D-Fire mapping is smoke=0, fire=1. Canonical V7 is fire=0, smoke=1.
        canonical = 1 if int(class_id) == 0 else 0
        x, y, w, h = (cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height
        annotations.append({"bbox": [x, y, w, h], "category": canonical, "area_ratio": bw * bh})
    return annotations


def bucket(annotations: list[dict[str, Any]]) -> str | None:
    if not annotations:
        return None
    categories = {annotation["category"] for annotation in annotations}
    maximum = max(annotation["area_ratio"] for annotation in annotations)
    minimum = min(annotation["area_ratio"] for annotation in annotations)
    if maximum > 0.40:
        return None
    if categories == {1}:
        return "smoke_small" if maximum <= 0.03 else "smoke_varied"
    if categories == {0}:
        return "fire_small" if maximum <= 0.03 else "fire_contextual"
    if minimum <= 0.03:
        return "mixed_small"
    return "mixed_contextual"


def deterministic_key(row: dict[str, Any]) -> int:
    return int(hashlib.sha256(f"dfire-v7:{row['source_split']}:{row['filename']}".encode()).hexdigest(), 16)


def select_metadata(rows: list[dict[str, Any]], known_records: set[str]) -> list[dict[str, Any]]:
    quotas = {"smoke_small": 500, "smoke_varied": 350, "fire_small": 350, "fire_contextual": 250, "mixed_small": 350, "mixed_contextual": 200}
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    for row in rows:
        if row["filename"] in known_records or not row["filename"].lower().startswith("web"):
            continue
        width, height = int(row["image"]["width"]), int(row["image"]["height"])
        annotations = parse_labels(row["label"], width, height)
        gap = bucket(annotations)
        if not gap:
            continue
        if min(min(annotation["bbox"][2], annotation["bbox"][3]) * 704 / max(width, height) for annotation in annotations) < 4:
            continue
        enriched = {**row, "width": width, "height": height, "annotations": annotations, "gap_bucket": gap}
        buckets[gap].append(enriched)
    selected = []
    for gap, quota in quotas.items():
        selected.extend(sorted(buckets[gap], key=deterministic_key)[:quota])
    return selected


def download(row: dict[str, Any], destination: Path) -> dict[str, Any]:
    response = requests.get(row["image"]["src"], timeout=90)
    response.raise_for_status()
    payload = response.content
    sha = hashlib.sha256(payload).hexdigest()
    path = destination / f"{sha}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    with Image.open(path) as raw:
        rgb = raw.convert("RGB")
        if rgb.width != row["width"] or rgb.height != row["height"]:
            raise RuntimeError(f"dimension mismatch {row['filename']}")
        phash = str(imagehash.phash(rgb, hash_size=8))
        flipped = str(imagehash.phash(ImageOps.mirror(rgb), hash_size=8))
    return {**row, "sha256": sha, "candidate_image": path.name, "bytes": len(payload), "phash64": phash, "phash64_flipped": flipped}


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def base_hashes(base: Path) -> tuple[set[str], list[str]]:
    rows = read_jsonl(base / "selection_manifest.jsonl")
    exact = {row["sha256"] for row in rows}
    hashes = []
    for row in rows:
        image_path = base / "data" / row["split"] / row["file_name"]
        with Image.open(image_path) as raw:
            rgb = raw.convert("RGB")
            hashes.extend((str(imagehash.phash(rgb, hash_size=8)), str(imagehash.phash(ImageOps.mirror(rgb), hash_size=8))))
    return exact, hashes


def deduplicate(rows: list[dict[str, Any]], exact: set[str], known_phashes: list[str]) -> tuple[list[dict[str, Any]], Counter]:
    accepted, reasons = [], Counter()
    accepted_hashes: list[str] = []
    for row in sorted(rows, key=deterministic_key):
        if row["sha256"] in exact:
            reasons["exact_v7"] += 1
            continue
        probes = (row["phash64"], row["phash64_flipped"])
        if any(hamming(probe, known) <= 3 for probe in probes for known in known_phashes):
            reasons["phash_v7_le3"] += 1
            continue
        if any(hamming(probe, known) <= 3 for probe in probes for known in accepted_hashes):
            reasons["phash_candidate_le3"] += 1
            continue
        accepted.append(row)
        accepted_hashes.extend(probes)
    return accepted, reasons


def draw_sheets(rows: list[dict[str, Any]], root: Path, per_sheet: int = 20) -> None:
    sheet_root = root / "review_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    tile_w, tile_h = 320, 250
    for page_index in range(math.ceil(len(rows) / per_sheet)):
        page = rows[page_index * per_sheet : (page_index + 1) * per_sheet]
        sheet = Image.new("RGB", (tile_w * 5, tile_h * 4), "white")
        for index, row in enumerate(page):
            with Image.open(root / "candidate_images" / row["candidate_image"]) as raw:
                image = ImageOps.contain(raw.convert("RGB"), (tile_w, tile_h - 25))
            draw = ImageDraw.Draw(image)
            sx, sy = image.width / row["width"], image.height / row["height"]
            for annotation in row["annotations"]:
                x, y, w, h = annotation["bbox"]
                color = "#1683ff" if annotation["category"] == 1 else "#ff7a00"
                draw.rectangle((x * sx, y * sy, (x + w) * sx, (y + h) * sy), outline=color, width=3)
            x0, y0 = (index % 5) * tile_w, (index // 5) * tile_h
            sheet.paste(image, (x0, y0))
            ImageDraw.Draw(sheet).text((x0 + 4, y0 + tile_h - 22), f"{row['candidate_id']} {row['gap_bucket']}", fill="black")
        sheet.save(sheet_root / f"review-{page_index + 1:04d}.jpg", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    known_records = {row.get("source_record_id", "") for row in read_jsonl(args.base / "selection_manifest.jsonl")}
    for path in args.v4_root.rglob("*pool.jsonl"):
        for row in read_jsonl(path):
            known_records.add(Path(str(row.get("source_record_id", ""))).name)
    inventory = fetch_inventory()
    write_jsonl(args.output / "source_inventory.jsonl", inventory)
    selected = select_metadata(inventory, known_records)
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download, row, args.output / "candidate_images") for row in selected]
        downloaded = [future.result() for future in as_completed(futures)]
    exact, hashes = base_hashes(args.base)
    candidates, exclusions = deduplicate(downloaded, exact, hashes)
    candidates.sort(key=lambda row: (row["gap_bucket"], deterministic_key(row)))
    for index, row in enumerate(candidates, 1):
        row["candidate_id"] = f"v7-dfire-{index:05d}"
        row["status"] = "needs_visual_review"
        row["training_admitted"] = False
        row["source_dataset"] = "dfire"
        row["source_revision"] = OFFICIAL_REVISION
        row["mirror_revision"] = MIRROR_REVISION
        row["license"] = "CC0-1.0"
        row["split_group"] = f"dfire:web:{Path(row['filename']).stem.lower()}"
        row.pop("image", None)
        row.pop("label", None)
    write_jsonl(args.output / "candidate_manifest.jsonl", candidates)
    write_jsonl(args.output / "review_decisions.template.jsonl", [{"candidate_id": row["candidate_id"], "decision": "", "reason": ""} for row in candidates])
    draw_sheets(candidates, args.output)
    report = {
        "schema": "fireviewer.pointing-v7-dfire-gap-candidates.v1",
        "status": "awaiting_exhaustive_visual_review",
        "source_rows": len(inventory),
        "metadata_selected": len(selected),
        "candidate_count": len(candidates),
        "candidate_buckets": dict(Counter(row["gap_bucket"] for row in candidates)),
        "automatic_exclusions": dict(exclusions),
        "candidate_bytes": sum(row["bytes"] for row in candidates),
        "source": {"official_repository": "gaia-solutions-on-demand/DFireDataset", "official_revision": OFFICIAL_REVISION, "mirror": DATASET, "mirror_revision": MIRROR_REVISION, "license": "CC0-1.0"},
        "storage_policy": "only bounded candidates downloaded; full 3.1 GB dataset not downloaded; retain until user review",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
