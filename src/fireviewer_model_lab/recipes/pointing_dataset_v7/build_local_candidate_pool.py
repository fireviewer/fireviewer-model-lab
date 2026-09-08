"""Build a bounded, review-only V7 candidate pool from the local ground-elite cache.

No candidate emitted by this script is training-admitted.  It deliberately
excludes every source record already seen by the V4 human-review campaign and
every exact/near duplicate of V6.  Admission happens in a separate finalizer
and requires an explicit visual decision for each candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


SPLITS = ("train", "valid", "test")
SOURCE_MANIFESTS = ("fasdd-cv", "pyro-sdis")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def index_pixels(coco_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for split in SPLITS:
        for path in sorted((coco_root / split).glob("*.jpg")):
            digest = sha256_path(path)
            previous = output.setdefault(digest, path)
            if previous != path:
                raise RuntimeError(f"duplicate pixel SHA in elite cache: {digest}")
    return output


def reviewed_source_records(pool_paths: list[Path]) -> set[str]:
    records: set[str] = set()
    for path in pool_paths:
        records.update(str(row["source_record_id"]) for row in read_jsonl(path))
    return records


def annotation_stats(row: dict[str, Any]) -> dict[str, Any]:
    width, height = int(row["width"]), int(row["height"])
    by_class: dict[str, list[float]] = {"fire": [], "smoke": []}
    scaled_min_side: list[float] = []
    for annotation in row.get("annotations", []):
        x, y, w, h = (float(value) for value in annotation["bbox_xywh"])
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width + 1 or y + h > height + 1:
            raise ValueError("invalid annotation geometry")
        name = str(annotation.get("class_name", "")).casefold()
        key = "smoke" if "smoke" in name else "fire" if "flame" in name or "fire" in name else ""
        if not key:
            raise ValueError(f"unsupported class: {name}")
        by_class[key].append((w * h) / (width * height))
        scale = 704.0 / min(width, height)
        scaled_min_side.append(min(w, h) * scale)
    return {
        "fire_count": len(by_class["fire"]),
        "smoke_count": len(by_class["smoke"]),
        "max_fire_area_ratio": max(by_class["fire"], default=0.0),
        "max_smoke_area_ratio": max(by_class["smoke"], default=0.0),
        "min_target_side_at_704": min(scaled_min_side, default=0.0),
    }


def bucket_for(stats: dict[str, Any]) -> str | None:
    if stats["fire_count"] == 0 and stats["smoke_count"] == 0:
        return "hard_negative"
    if stats["smoke_count"] and stats["max_smoke_area_ratio"] <= 0.01:
        return "small_distant_smoke"
    if stats["fire_count"] and stats["max_fire_area_ratio"] <= 0.03:
        return "small_fire_mixed"
    if stats["smoke_count"] and stats["max_smoke_area_ratio"] <= 0.10:
        return "varied_smoke"
    return None


def deterministic_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    stats = row["v7_stats"]
    bucket = row["gap_bucket"]
    if bucket == "small_distant_smoke":
        metric = stats["max_smoke_area_ratio"]
    elif bucket == "small_fire_mixed":
        metric = stats["max_fire_area_ratio"]
    else:
        metric = 0.0
    return (metric, hashlib.sha256(str(row["sha256"]).encode()).hexdigest())


def make_sheets(rows: list[dict[str, Any]], output: Path, size: int = 25) -> list[dict[str, Any]]:
    sheets = output / "review_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    cell_w, cell_h, columns = 300, 235, 5
    font = ImageFont.load_default()
    for start in range(0, len(rows), size):
        subset = rows[start : start + size]
        canvas = Image.new("RGB", (columns * cell_w, 5 * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(subset):
            x0, y0 = (index % columns) * cell_w, (index // columns) * cell_h
            with Image.open(row["source_image"]) as source:
                image = source.convert("RGB")
                image.thumbnail((cell_w - 8, cell_h - 44), Image.Resampling.LANCZOS)
                paste_x, paste_y = x0 + (cell_w - image.width) // 2, y0 + 22
                canvas.paste(image, (paste_x, paste_y))
                scale_x = image.width / float(row["width"])
                scale_y = image.height / float(row["height"])
                for annotation in row.get("annotations", []):
                    bx, by, bw, bh = (float(value) for value in annotation["bbox_xywh"])
                    name = str(annotation.get("class_name", "")).casefold()
                    color = "#1687ff" if "smoke" in name else "#ff6a00"
                    draw.rectangle(
                        (
                            paste_x + bx * scale_x,
                            paste_y + by * scale_y,
                            paste_x + (bx + bw) * scale_x,
                            paste_y + (by + bh) * scale_y,
                        ),
                        outline=color,
                        width=2,
                    )
            label = f"{row['candidate_id']} {row['gap_bucket']} {row['source_id']}"
            draw.text((x0 + 3, y0 + 3), label[:48], fill="black", font=font)
            manifest.append({
                "candidate_id": row["candidate_id"],
                "sheet": f"review_sheets/review-{start // size + 1:04d}.jpg",
                "tile": index + 1,
            })
        sheet_path = sheets / f"review-{start // size + 1:04d}.jpg"
        canvas.save(sheet_path, quality=90, optimize=True)
    return manifest


def build(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    v6_rows = read_jsonl(args.v6_manifest)
    blocked_sha = {str(row["sha256"]) for row in v6_rows}
    base_phashes = [str(row["phash64"]) for row in v6_rows if row.get("phash64")]
    reviewed_records = reviewed_source_records(args.review_pool)
    pixel_index = index_pixels(args.elite_root / "_rfdetr_coco")

    rows: list[dict[str, Any]] = []
    for source in SOURCE_MANIFESTS:
        rows.extend(read_jsonl(args.elite_root / "manifests" / source / "manifest.jsonl"))

    exclusions: Counter[str] = Counter()
    accepted_candidates: list[dict[str, Any]] = []
    candidate_phashes: list[str] = []
    used_groups: set[str] = set()
    quotas = {
        "hard_negative": args.hard_negatives,
        "small_distant_smoke": args.small_smoke,
        "varied_smoke": args.varied_smoke,
        "small_fire_mixed": args.small_fire,
    }

    eligible: list[dict[str, Any]] = []
    for row in rows:
        sha = str(row["sha256"])
        if sha in blocked_sha:
            exclusions["exact_v6_overlap"] += 1
            continue
        if str(row["source_record_id"]) in reviewed_records:
            exclusions["record_seen_by_v4_review"] += 1
            continue
        if row.get("near_duplicate_of"):
            exclusions["source_near_duplicate"] += 1
            continue
        image = pixel_index.get(sha)
        if image is None:
            exclusions["missing_sha_matched_pixel"] += 1
            continue
        if int(row["width"]) < 320 or int(row["height"]) < 240:
            exclusions["low_resolution"] += 1
            continue
        try:
            stats = annotation_stats(row)
        except ValueError:
            exclusions["invalid_annotation"] += 1
            continue
        bucket = bucket_for(stats)
        if bucket is None:
            exclusions["outside_gap_buckets"] += 1
            continue
        if bucket != "hard_negative" and stats["min_target_side_at_704"] < 4.0:
            exclusions["subpixel_at_704"] += 1
            continue
        phash = str(row.get("phash", ""))
        if phash and any(hamming(phash, other) <= args.phash_threshold for other in base_phashes):
            exclusions["phash_near_v6"] += 1
            continue
        copied = dict(row)
        copied.update({
            "gap_bucket": bucket,
            "source_image": str(image.resolve()),
            "status": "needs_visual_review",
            "training_admitted": False,
            "v7_stats": stats,
        })
        eligible.append(copied)

    for bucket, quota in quotas.items():
        bucket_rows = sorted((row for row in eligible if row["gap_bucket"] == bucket), key=deterministic_rank)
        for row in bucket_rows:
            group = str(row["split_group"])
            if group in used_groups:
                exclusions["candidate_group_cap_one"] += 1
                continue
            phash = str(row.get("phash", ""))
            if phash and any(hamming(phash, other) <= args.phash_threshold for other in candidate_phashes):
                exclusions["candidate_phash_near_duplicate"] += 1
                continue
            used_groups.add(group)
            if phash:
                candidate_phashes.append(phash)
            accepted_candidates.append(row)
            if sum(item["gap_bucket"] == bucket for item in accepted_candidates) >= quota:
                break

    accepted_candidates.sort(key=lambda row: (row["gap_bucket"], row["source_id"], row["sha256"]))
    for index, row in enumerate(accepted_candidates, start=1):
        row["candidate_id"] = f"v7-local-{index:05d}"

    write_jsonl(output / "candidate_manifest.jsonl", accepted_candidates)
    decisions = ({"candidate_id": row["candidate_id"], "decision": "", "reason": ""} for row in accepted_candidates)
    write_jsonl(output / "visual_decisions.template.jsonl", decisions)
    sheet_manifest = make_sheets(accepted_candidates, output)
    write_jsonl(output / "review_sheets_manifest.jsonl", sheet_manifest)

    counts = Counter((row["gap_bucket"], row["source_id"]) for row in accepted_candidates)
    report = {
        "schema": "fireviewer.pointing-v7-local-candidates.v1",
        "status": "awaiting_exhaustive_visual_review",
        "training_admissions": 0,
        "candidate_count": len(accepted_candidates),
        "candidate_counts": [
            {"bucket": bucket, "source": source, "count": count}
            for (bucket, source), count in sorted(counts.items())
        ],
        "exclusions": dict(sorted(exclusions.items())),
        "policy": {
            "one_candidate_per_split_group": True,
            "phash_hamming_threshold": args.phash_threshold,
            "minimum_target_side_at_704": 4.0,
            "every_admission_requires_explicit_visual_review": True,
            "previously_reviewed_records_are_not_reopened": True,
        },
        "inputs": {
            "v6_manifest": str(args.v6_manifest.resolve()),
            "v6_manifest_sha256": sha256_path(args.v6_manifest),
            "elite_root": str(args.elite_root.resolve()),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elite-root", type=Path, required=True)
    parser.add_argument("--v6-manifest", type=Path, required=True)
    parser.add_argument("--review-pool", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=int, default=300)
    parser.add_argument("--small-smoke", type=int, default=450)
    parser.add_argument("--varied-smoke", type=int, default=250)
    parser.add_argument("--small-fire", type=int, default=250)
    parser.add_argument("--phash-threshold", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
