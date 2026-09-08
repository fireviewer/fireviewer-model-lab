"""Acquire a bounded WUI classification pool for explicit detection annotation.

Upstream presence/distance/time-of-day labels are selection hints only, never
visual decisions or detection ground truth. Original bytes are retained once.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from training.pointing_dataset_v7.split_registry import read_rows
from training.pointing_dataset_v8.acquire_pyrosdis import session


class ByteBudget:
    def __init__(self, limit, used=0):
        self.limit, self.used = limit, used
        self.lock = threading.Lock()

    def add(self, count):
        with self.lock:
            if self.used + count > self.limit:
                raise ValueError("WUI acquisition byte budget reached")
            self.used += count


def select(rows, negative_per_group=30):
    groups = defaultdict(list)
    for row in rows:
        parts = row["source_path"].split("/")
        if row["mime"] not in {"image/jpeg", "image/png"} or len(parts) < 6 or "Perto" in parts:
            continue
        positive, lighting, distance = parts[2] == "1", parts[3], parts[4]
        kind = parts[5] if positive else "negative"
        groups[(positive, lighting, distance, kind)].append(row)
    result = []
    for key, items in sorted(groups.items()):
        positive, lighting, distance, kind = key
        # Broad night/evening and small-flame coverage; bounded negatives.
        limit = 50 if positive and kind == "Fire" else 22 if positive else negative_per_group
        items.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        result.extend(items[:limit])
    return result


def acquire(row, output, budget):
    path = output / "receipts" / (row["id"] + ".json")
    if path.exists():
        cached = json.loads(path.read_text())
        if hashlib.sha256(Path(cached["source_image"]).read_bytes()).hexdigest() != cached["sha256"]:
            raise ValueError("WUI cached image differs from bound receipt")
        return cached
    with session().get("https://drive.google.com/uc", params={"export": "download", "id": row["id"]},
                       timeout=40, stream=True) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", 0)) > 8_000_000:
            raise ValueError("WUI original exceeds individual 8 MB budget")
        payload = bytearray()
        for chunk in response.iter_content(65536):
            budget.add(len(chunk))
            payload.extend(chunk)
            if len(payload) > 8_000_000:
                raise ValueError("WUI original exceeds individual 8 MB streamed budget")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        phash = str(imagehash.phash(image.convert("RGB")))
        flipped = str(imagehash.phash(ImageOps.mirror(image.convert("RGB"))))
    sha = hashlib.sha256(payload).hexdigest()
    image_path = output / "images" / (sha + "." + ("png" if row["mime"] == "image/png" else "jpg"))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image_path.write_bytes(payload)
    candidate = row | {"candidate_id": "WUI-" + row["id"], "source_record_id": row["source_path"],
        "sha256": sha, "source_image": str(image_path.resolve()), "image_bytes": len(payload),
        "phash": phash, "phash_flipped": flipped, "width": width, "height": height,
        "source_family": "WUI-Fire-Detection", "split": "train", "synthetic": False,
        "review_status": "pending_visual_annotation", "v8_corpus_admitted": False,
        "v8_training_admitted": False, "objects": {"bbox": [], "category": [], "area": []}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-mb", type=int, default=1000)
    parser.add_argument("--negative-per-group", type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6 or not 1 <= args.budget_mb <= 1500 or not 1 <= args.negative_per_group <= 50:
        raise ValueError("WUI acquisition limits exceeded")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = select(read_rows(args.catalogue), args.negative_per_group)
    budget = ByteBudget(args.budget_mb*1_000_000,
                        sum(p.stat().st_size for p in (args.output / "images").glob("*")))
    (args.output / "acquisition_plan.jsonl").write_text("".join(json.dumps(r)+"\n" for r in plan), encoding="utf-8")
    candidates, errors = [], []
    def worker(row):
        try:
            return acquire(row, args.output, budget), None
        except Exception as exc:
            return None, {"id": row["id"], "error_type": type(exc).__name__,
                          "message": str(exc).split(" for url:")[0][:250]}
    print(json.dumps({"planned_images": len(plan), "max_download_mb": args.budget_mb}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, (candidate, error) in enumerate(pool.map(worker, plan), 1):
            if candidate:
                candidates.append(candidate)
            else:
                errors.append(error)
            if count % 25 == 0:
                print(json.dumps({"processed": count, "candidates": len(candidates), "errors": len(errors),
                                  "download_mb": round(budget.used/1e6, 1)}), flush=True)
    for name, rows in (("candidate_manifest.jsonl", candidates), ("acquisition_errors.jsonl", errors)):
        (args.output / name).write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    report = {"planned": len(plan), "downloaded_candidates": len(candidates), "errors": len(errors),
              "bytes_budget_used": budget.used, "admitted": 0,
              "error_types": dict(Counter(r["error_type"] for r in errors)),
              "annotation_state": "classification_only_not_detection_boxes"}
    (args.output / "acquisition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
