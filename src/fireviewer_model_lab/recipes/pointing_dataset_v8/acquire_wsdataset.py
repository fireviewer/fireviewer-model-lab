"""Acquire a bounded, video-stratified WSDataset pool; never grant visual admission."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import imagehash
import numpy as np
import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.audit_coverage import geometries

BASE = "https://www.kaggle.com/api/v1/datasets/download/gloryvu/wildfire-smoke-detection"
PREFIX = "wildfire_smoke_dataset/wildfire_smoke_dataset"


def capture_key(name, video):
    """Conservatively join clips from the same named camera/capture.

    A camera/capture is a grouping constraint, not proof of a physical incident.
    """
    stem = PurePosixPath(name).stem
    if "frame_camera_" in stem:
        return stem.removeprefix("frame_").rsplit("_", 1)[0]
    if "camera_" in stem:
        return stem[stem.index("camera_"):]
    if "rtsp___" in stem:
        # The same stream recorded at different times is not an independent camera.
        return "stream_" + hashlib.sha256(stem.split("rtsp___", 1)[1].encode()).hexdigest()[:16]
    return "video_" + video


def plan(files, positives_per_video=8, negatives_per_video=4):
    groups = defaultdict(list)
    detection_images = {r["name"]: r for r in files if "/train/images/" in r["name"]}
    frozen_captures = set()
    for item in files:
        parts = PurePosixPath(item["name"]).parts
        if len(parts) == 7 and parts[2:4] == ("classification", "test"):
            frozen_captures.add(capture_key(parts[-1], parts[5]))
    for item in files:
        parts = PurePosixPath(item["name"]).parts
        if len(parts) != 7 or parts[2:4] != ("classification", "train") or parts[4] not in {"smoke", "nonsmoke"}:
            continue
        positive = parts[4] == "smoke"
        primary = f"{PREFIX}/train/images/{parts[-1]}"
        if positive and primary not in detection_images:
            continue
        capture = capture_key(parts[-1], parts[5])
        if capture in frozen_captures:
            continue
        groups[(positive, capture)].append(item | {"positive": positive, "video": parts[5], "capture": capture,
                                                   "primary_image": primary if positive else None,
                                                   "primary_bytes": detection_images[primary]["bytes"] if positive else 0})
    selected = []
    for (positive, video), items in sorted(groups.items()):
        items.sort(key=lambda r: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", r["name"])])
        count = min(len(items), positives_per_video if positive else negatives_per_video)
        for i in range(count):
            selected.append(items[(2 * i + 1) * len(items) // (2 * count)])
    return selected


def get_file(name: str, destination: Path | None, limit: int) -> bytes:
    if destination is not None and destination.exists():
        data = destination.read_bytes()
        if len(data) > limit:
            raise ValueError("cached file exceeds byte budget")
        return data
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=.3, status_forcelist=[429, 500, 502, 503, 504])))
    with session.get(BASE + "/" + quote(name, safe=""), params={"datasetVersionNumber": 7}, stream=True, timeout=25) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", "0")) > limit:
            raise ValueError("remote file exceeds byte budget")
        body = bytearray()
        for chunk in response.iter_content(65536):
            body.extend(chunk)
            if len(body) > limit:
                raise ValueError("streamed file exceeds byte budget")
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
    return bytes(body)


def check_image_pair(original: bytes, detection: bytes) -> dict:
    """Verify full-frame correspondence despite upstream 640-square stretching.

    Normalized YOLO boxes then map back to the unwarped original. This technical
    check does not replace the later visual check of every annotation.
    """
    with Image.open(io.BytesIO(original)) as a, Image.open(io.BytesIO(detection)) as b:
        a.load()
        b.load()
        a, b = a.convert("RGB"), b.convert("RGB")
        distance = int(imagehash.phash(a) - imagehash.phash(b))
        aa = np.asarray(a.resize((256, 256)), dtype=np.float32)
        bb = np.asarray(b.resize((256, 256)), dtype=np.float32)
        error = float(np.abs(aa - bb).mean() / 255)
        evidence = {"original_size": list(a.size), "detection_size": list(b.size),
                    "normalized_pixel_mae": error, "phash_distance": distance,
                    "coordinate_mapping": "normalized_full_frame_xywh_to_original_resolution"}
    if distance > 4 or error > .025:
        raise ValueError("classification/detection full-frame correspondence is not established")
    return evidence


def acquire(item: dict, output: Path):
    key = hashlib.sha256(item["name"].encode()).hexdigest()
    source_image = output / "images" / (key + ".jpg")
    data = get_file(item["name"], source_image, 3_000_000)
    if len(data) != item["bytes"]:
        raise ValueError("versioned image size differs from metadata")
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        rgb = image.convert("RGB")
        phash = str(imagehash.phash(rgb))
        flipped = str(imagehash.phash(ImageOps.mirror(rgb)))
    boxes = []
    label_path = None
    primary_hash = None
    pairing = None
    if item["positive"]:
        # Keep the original resolution. The resized primary is checked in memory,
        # avoiding a second stored copy and not relying on filename equality.
        pairing_path = output / "pairing" / (key + ".json")
        cached_pairing = json.loads(pairing_path.read_text()) if pairing_path.exists() else None
        image_sha = hashlib.sha256(data).hexdigest()
        if cached_pairing and cached_pairing["original_sha256"] == image_sha and cached_pairing["primary_image"] == item["primary_image"]:
            primary_hash = cached_pairing["primary_sha256"]
            pairing = cached_pairing["evidence"]
        else:
            primary = get_file(item["primary_image"], None, 3_000_000)
            if len(primary) != item["primary_bytes"]:
                raise ValueError("versioned detection image size differs from metadata")
            primary_hash = hashlib.sha256(primary).hexdigest()
            pairing = check_image_pair(data, primary)
            pairing_path.parent.mkdir(parents=True, exist_ok=True)
            pairing_path.write_text(json.dumps({"original_sha256": image_sha, "primary_image": item["primary_image"],
                                                "primary_sha256": primary_hash, "evidence": pairing}), encoding="utf-8")
        label_name = item["primary_image"].replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        label_path = output / "labels" / (key + ".txt")
        label = get_file(label_name, label_path, 32_768).decode("utf-8-sig")
        for line in label.splitlines():
            if not line.strip():
                continue
            parts = list(map(float, line.split()))
            if len(parts) != 5 or parts[0] != 0:
                raise ValueError("unknown WSDataset annotation format/category")
            _, x, y, w, h = parts
            boxes.append([(x - w / 2) * width, (y - h / 2) * height, w * width, h * height])
        if not boxes:
            raise ValueError("positive classification image has no detection annotations")
    row = {"sha256": hashlib.sha256(data).hexdigest(), "source_image": str(source_image.resolve()),
           "source_dataset": "WSDataset", "source_family": "WSDataset", "source_revision": "kaggle-version-7",
           "source_record_id": item["name"], "source_group_id": "wsdataset:" + item["capture"],
           "source_split": "train", "split": "train", "split_group_id": "wsdataset:" + item["capture"],
           "source_video_id": item["video"], "source_capture_key": item["capture"],
           "source_group_evidence": "official_video_directory_and_conservative_capture_identity", "physical_event_verified": False,
           "source_annotation": str(label_path.resolve()) if label_path else None,
           "annotation_sha256": digest(label_path) if label_path else None, "primary_pair_image_sha256": primary_hash,
           "image_pair_evidence": pairing,
           "license": "MIT", "license_evidence": "https://www.kaggle.com/datasets/gloryvu/wildfire-smoke-detection",
           "width": width, "height": height, "objects": {"bbox": boxes, "category": [1] * len(boxes), "area": [b[2] * b[3] for b in boxes]},
        "phash": phash, "phash_flipped": flipped, "synthetic": False,
           "upstream_negative_label": not item["positive"], "negative_verified": False,
           "review_status": "needs_visual_review", "v8_training_admitted": False,
           "target_area_ratio_min": min((b[2] * b[3] / (width * height) for b in boxes), default=None)}
    geometries(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--external-samples", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers not in range(1, 7):
        raise ValueError("Use at most six network workers")
    files = read_rows(args.source_root / "file_index.jsonl")
    # A partial full-index is sufficient only after all classification folders and
    # detection training images have been observed (listing already entered labels).
    if not any("/train/labels/" in row["name"] for row in files):
        raise ValueError("Incomplete classification/detection image index")
    planned = plan(files)
    if len(planned) > 800 or sum(r["bytes"] + r["primary_bytes"] for r in planned) > 650_000_000:
        raise ValueError("Bounded acquisition budget exceeded")
    args.source_root.mkdir(parents=True, exist_ok=True)
    classes = get_file(f"{PREFIX}/train/labels/classes.txt", args.source_root / "classes.txt", 1024).decode().strip()
    if classes != "smoke":
        raise ValueError("Class map differs from the audited source")
    (args.source_root / "acquisition_plan.json").write_text(json.dumps(planned, indent=2), encoding="utf-8")
    print(json.dumps({"planned": len(planned), "positive": sum(r["positive"] for r in planned), "video_groups": len({r["video"] for r in planned}), "unique_payload_bytes": sum(r["bytes"] for r in planned)}), flush=True)
    base_fingerprints = read_rows(args.fingerprints)
    hashes = {r["sha256"] for r in base_fingerprints}
    index = [(int(r["phash"], 16), int(r["phash_flipped"], 16)) for r in base_fingerprints]
    # Exclude perceptual variants as well as exact external benchmark images.
    for row in read_rows(args.external_samples):
        sha = row.get("image_sha256", row.get("sha256"))
        if sha:
            hashes.add(sha)
        external_image = args.external_samples.parent / "images" / row["image_relpath"]
        if digest(external_image) != sha:
            raise ValueError("External holdout image integrity mismatch")
        with Image.open(external_image) as external:
            index.append((int(str(imagehash.phash(external)), 16), int(str(imagehash.phash(ImageOps.mirror(external))), 16)))
    errors, rejected, accepted = [], [], []

    def worker(item):
        try:
            return acquire(item, args.source_root), None
        except Exception as exc:
            return None, {"source_record_id": item["name"], "error_type": type(exc).__name__, "error": str(exc).split(" for url:")[0]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, (row, error) in enumerate(pool.map(worker, planned), 1):
            if error:
                errors.append(error)
            else:
                a, af = int(row["phash"], 16), int(row["phash_flipped"], 16)
                reason = "exact_overlap" if row["sha256"] in hashes else "near_duplicate_le4" if any(min((a ^ b).bit_count(), (af ^ b).bit_count(), (a ^ bf).bit_count()) <= 4 for b, bf in index) else None
                if reason:
                    rejected.append({"sha256": row["sha256"], "source_record_id": row["source_record_id"], "reason": reason})
                else:
                    row["candidate_id"] = f"v8-ws-{len(accepted) + 1:05d}"
                    accepted.append(row)
                    hashes.add(row["sha256"])
                    index.append((a, af))
            if number % 50 == 0:
                print(json.dumps({"processed": number, "candidates": len(accepted), "duplicates": len(rejected), "acquisition_errors": len(errors)}), flush=True)
    for filename, rows in (("candidate_manifest.jsonl", accepted), ("duplicate_exclusions.jsonl", rejected), ("acquisition_errors.jsonl", errors)):
        (args.source_root / filename).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    report = {"schema": "fireviewer.wsdataset-v8-acquisition.v1", "status": "candidate_pool_not_training_ready", "planned": len(planned),
              "candidates": len(accepted), "positive_candidates": sum(bool(r["objects"]["bbox"]) for r in accepted),
              "small_smoke_candidates_le_half_percent": sum(r["target_area_ratio_min"] is not None and r["target_area_ratio_min"] <= .005 for r in accepted),
              "negative_candidates": sum(not r["objects"]["bbox"] for r in accepted), "declared_video_groups": len({r["source_group_id"] for r in accepted}),
              "duplicate_exclusions": dict(Counter(r["reason"] for r in rejected)), "acquisition_errors": len(errors),
              "admitted_images": 0, "full_archive_downloaded": False,
              "limitations": ["Per-image visual and annotation review still required.", "Video IDs are not automatically independent physical incidents.", "Near-duplicate screening is conservative, not proof of source independence.", "A complementary source with small flames is still required."]}
    (args.source_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
