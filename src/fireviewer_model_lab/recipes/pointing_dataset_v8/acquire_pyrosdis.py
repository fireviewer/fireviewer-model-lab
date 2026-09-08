"""Bounded, pinned Pyro-SDIS acquisition. Download candidates, never admit them.

The Dataset Viewer is used for metadata and selected images only. No full
snapshot/parquet/image archive is downloaded. Signed asset URLs are transient.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import get_token

from training.pointing_dataset_v7.split_registry import read_rows
from training.pointing_dataset_v8.audit_coverage import geometries
from training.pointing_dataset_v8.source_identity import frozen_camera_views, pyro_camera_view

REPO = "pyronear/pyro-sdis"
REVISION = "a1e553ec4d806f71fc6db744cc22bc3469487382"
API = "https://datasets-server.huggingface.co"
_API_LOCK = threading.Lock()
_API_NEXT = 0.0


def session():
    result = requests.Session()
    # A throttled host must pause acquisition, not hold a worker in a long
    # Retry-After sleep or multiply requests against a resource budget.
    result.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.5,
                 status_forcelist=[500, 502, 503, 504], respect_retry_after_header=False)))
    return result


def api_rows(params, revision=REVISION):
    global _API_NEXT
    # Bound the shared request rate, including concurrent workers. Cached pages
    # do not spend requests. Authentication is never logged or persisted here.
    with _API_LOCK:
        delay = max(0, _API_NEXT - time.monotonic())
        if delay:
            time.sleep(delay)
        _API_NEXT = time.monotonic() + 1.25
    client = session()
    token = get_token()
    if token:
        client.headers["Authorization"] = "Bearer " + token
    response = client.get(API + "/rows", params=params, timeout=45)
    response.raise_for_status()
    if response.headers.get("x-revision") != revision:
        raise ValueError("Dataset Viewer revision differs from pinned source")
    return response.json()


def parse_annotations(text, width, height):
    boxes = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"0", "1"}:
            raise ValueError("Unrecognized upstream single-smoke-class annotation")
        cx, cy, w, h = map(float, fields[1:])
        if not all(math.isfinite(v) for v in (cx, cy, w, h)) or min(w, h) <= 0:
            raise ValueError("Invalid normalized source bbox")
        left, top, right, bottom = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        if min(left, top) < -0.000002 or max(right, bottom) > 1.000002:
            raise ValueError("Source bbox outside full image")
        # Tolerate only the upstream six-decimal rounding at the frame boundary.
        left, top, right, bottom = max(0, left), max(0, top), min(1, right), min(1, bottom)
        boxes.append([left*width, top*height, (right-left)*width, (bottom-top)*height])
    return {"bbox": boxes, "category": [1]*len(boxes), "area": [b[2]*b[3] for b in boxes]}


def capture_key(row):
    name = row.get("image_name", row.get("source_record_id", ""))
    match = re.fullmatch(r"(.+?)_(.+?)_(\d{4}-\d{2}-\d{2})T.+\.(?:jpg|png|jpeg)", name)
    return "Pyro-SDIS:" + ":".join(match.groups()) if match else None


def fetch_page(offset, output):
    path = output / "metadata" / f"train-{offset:05d}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["revision"] != REVISION or cached["offset"] != offset:
            raise ValueError("Cached metadata revision differs")
        return cached
    data = api_rows({"dataset": REPO, "config": "default", "split": "train", "offset": offset, "length": 100})
    rows = []
    for wrapper in data["rows"]:
        row = wrapper["row"]
        image = row["image"]
        # No signed URLs are persisted. Reacquire the selected page when needed.
        rows.append({k: row[k] for k in ("annotations", "image_name", "partner", "camera", "date")} |
                    {"row_idx": wrapper["row_idx"], "width": image["width"], "height": image["height"]})
    cached = {"revision": REVISION, "offset": offset, "num_rows_total": data["num_rows_total"], "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cached), encoding="utf-8")
    return cached


def select(rows, baseline, limit):
    known = {r.get("source_record_id") for r in baseline}
    frozen = {capture_key(r) for r in baseline if r.get("split") in {"validation", "test"}}
    frozen.discard(None)
    heldout_cameras = frozen_camera_views(baseline)
    groups = defaultdict(list)
    excluded = Counter()
    for row in rows:
        group = capture_key(row)
        if not group or group in frozen or row["image_name"] in known or pyro_camera_view(row) in heldout_cameras:
            excluded["known_or_frozen_capture"] += 1
            continue
        try:
            obj = parse_annotations(row["annotations"], row["width"], row["height"])
        except ValueError:
            excluded["invalid_source_geometry"] += 1
            continue
        if not obj["bbox"]:
            excluded["not_a_positive_smoke_annotation"] += 1
            continue
        smallest = min(obj["area"]) / (row["width"]*row["height"])
        if smallest > .02:
            excluded["outside_distant_priority"] += 1
            continue
        groups[group].append(row | {"source_group_id": group, "objects": obj, "smallest_area_ratio": smallest})
    for items in groups.values():
        items.sort(key=lambda r: (r["smallest_area_ratio"] > .005, hashlib.sha256(r["image_name"].encode()).hexdigest()))
    # Round robin across camera-days first: never thousands of a single sequence.
    picked = []
    keys = sorted(groups, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    for round_index in range(6):
        for key in keys:
            if round_index < len(groups[key]):
                picked.append(groups[key][round_index])
                if len(picked) >= limit:
                    return picked, dict(excluded)
    return picked, dict(excluded)


def acquire_page(items, output):
    offset = items[0]["row_idx"] // 100 * 100
    missing = [r for r in items if not (output / "receipts" / f"{r['row_idx']:05d}.json").exists()]
    urls = {}
    if missing:
        data = api_rows({"dataset": REPO, "config": "default", "split": "train", "offset": offset, "length": 100})
        for wrapper in data["rows"]:
            urls[wrapper["row_idx"]] = wrapper["row"]["image"]["src"]
    result = []
    for row in items:
        receipt = output / "receipts" / f"{row['row_idx']:05d}.json"
        if receipt.exists():
            candidate = json.loads(receipt.read_text())
            path = Path(candidate["source_image"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != candidate["sha256"]:
                raise ValueError("Cached candidate bytes changed")
            result.append(candidate)
            continue
        with session().get(urls[row["row_idx"]], timeout=30, stream=True) as response:
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_content(65536):
                payload.extend(chunk)
                if len(payload) > 2_000_000:
                    raise ValueError("Selected image exceeds 2 MB limit")
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.size != (row["width"], row["height"]):
                raise ValueError("Dataset Viewer image dimensions changed")
            phash = str(imagehash.phash(image.convert("RGB")))
            flipped = str(imagehash.phash(ImageOps.mirror(image.convert("RGB"))))
        sha = hashlib.sha256(payload).hexdigest()
        path = output / "images" / f"{sha}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        candidate = row | {"candidate_id": f"PS-{row['row_idx']:05d}", "sha256": sha, "source_image": str(path.resolve()),
            "phash": phash, "phash_flipped": flipped, "image_bytes": len(payload), "source_dataset": "Pyro-SDIS",
            "source_family": "Pyro-SDIS", "source_revision": REVISION, "source_record_id": row["image_name"],
            "source_split": "train", "split": "train", "split_group_id": row["source_group_id"],
            "license": "Apache-2.0", "license_evidence": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
            "upstream_class_mapping": "single smoke class; source integer 0/1 both map to FireViewer smoke=1",
            "synthetic": False, "review_status": "pending_visual_review", "v8_corpus_admitted": False,
            "v8_training_admitted": False, "annotation_state": "source_boxes_not_yet_reviewed"}
        geometries(candidate)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(candidate), encoding="utf-8")
        result.append(candidate)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--cached-metadata-only", action="store_true",
                        help="Select from already indexed pages, without claiming full-source coverage")
    args = parser.parse_args()
    if not 1 <= args.limit <= 1600 or not 1 <= args.workers <= 6:
        raise ValueError("Acquisition budget exceeded")
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    if args.cached_metadata_only:
        for path in sorted((args.output / "metadata").glob("train-*.json")):
            cached = json.loads(path.read_text())
            if cached["revision"] != REVISION:
                raise ValueError("Cached source revision differs")
            all_rows.extend(cached["rows"])
    else:
        first = fetch_page(0, args.output)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, page in enumerate(pool.map(lambda offset: fetch_page(offset, args.output), range(0, first["num_rows_total"], 100))):
                all_rows.extend(page["rows"])
                if i % 25 == 0:
                    print(json.dumps({"metadata_rows": len(all_rows)}), flush=True)
    selected, excluded = select(all_rows, read_rows(args.baseline), args.limit)
    (args.output / "acquisition_plan.jsonl").write_text("".join(json.dumps(r)+"\n" for r in selected), encoding="utf-8")
    report = {"revision": REVISION, "metadata_rows": len(all_rows), "selected": len(selected),
              "full_source_metadata_indexed": not args.cached_metadata_only,
              "selected_camera_days": len({r["source_group_id"] for r in selected}), "exclusions": excluded,
              "note": "Camera-day grouping is not verified physical incident identity; visual admission is separate."}
    (args.output / "acquisition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if args.metadata_only:
        return
    batches = defaultdict(list)
    for row in selected:
        batches[row["row_idx"] // 100].append(row)
    candidates = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(lambda items: acquire_page(items, args.output), batches.values()):
            candidates.extend(batch)
            print(json.dumps({"downloaded_candidates": len(candidates), "admitted": 0}), flush=True)
    candidates.sort(key=lambda r: r["candidate_id"])
    (args.output / "candidate_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in candidates), encoding="utf-8")


if __name__ == "__main__":
    main()
