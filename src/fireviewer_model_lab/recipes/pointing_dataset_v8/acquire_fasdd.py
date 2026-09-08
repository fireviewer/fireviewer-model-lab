"""Select real, ground-photo FASDD positives from the existing pinned HF corpus.

Stream the small metadata manifest, not the 46 GB image corpus. Keep only the
eligible metadata locally; image downloads are bounded, resumable and never an
admission. Existing rejections, upstream holdouts and historical split locks are
excluded before requesting a single image. Every image still needs visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import threading
import time
from datetime import datetime, timezone
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, unquote

import imagehash
from PIL import Image, ImageOps

from training.pointing_dataset_v7.split_registry import canonical_source_group, digest, read_rows
from training.pointing_dataset_v8.acquire_pyrosdis import api_rows, session
from training.pointing_dataset_v8.acquire_wui import ByteBudget
from training.pointing_dataset_v8.acquire_dfire import rate_limit_delay
from training.pointing_dataset_v8.audit_coverage import geometries

REPO = "fireviewer/fire-smoke-detection-corpus-v1"
REVISION = "85ad763e6275537386f7eefdae5e3a18a55f1c71"
MANIFEST = "manifests/fasdd/manifest.jsonl"
MANIFEST_BYTES = 167_944_737
MANIFEST_SHA256 = "e8c8536de79bcb6c5f62c7f35a788e41a1684cf6b4641c363f79980a19561339"
LICENSE = "CC-BY-SA-4.0"
NORMALIZER_VERSION = 2
CLASS_NAMES = {"fire": 0, "flame": 0, "fire_visible": 0, "flame_visible": 0, "smoke": 1, "smoke_visible": 1}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def source_candidate(row, row_index):
    """Source labels are selection hints, never a semantic/ground-view review."""
    if (row.get("source_id") != "fasdd_v9" or row.get("split") != "train"
            or not row.get("source_record_id", "").casefold().startswith("cv:train:")
            or row.get("license") != LICENSE or row.get("near_duplicate_of")):
        return None
    boxes, labels = [], []
    for annotation in row.get("annotations", []):
        label = annotation.get("class_name", "").casefold()
        if label not in CLASS_NAMES:
            raise ValueError(f"Unknown FASDD class: {label}")
        boxes.append(annotation["bbox_xywh"])
        labels.append(CLASS_NAMES[label])
    if not boxes or row["width"] < 480 or row["height"] < 320:
        return None
    out = {key: row[key] for key in ("source_record_id", "sample_id", "sha256", "width", "height", "license", "phash")}
    out.update({"row_index": row_index, "source_group_id": row["split_group"],
                "source_split": "train", "upstream_original_split": "CV:train",
                "objects": {"bbox": boxes, "category": labels, "area": [b[2] * b[3] for b in boxes]}})
    geometries(out)
    ratios = [box[2] * box[3] / (row["width"] * row["height"]) for box in boxes]
    # Do not spend storage on extreme closeups or subpixel source boxes.
    if max(ratios) > .30 or min(min(b[2:]) for b in boxes) < 3:
        return None
    small = {label for label, ratio in zip(labels, ratios) if ratio <= .005}
    if not small and not (0 in labels and max(r for r, label in zip(ratios, labels) if label == 0) <= .04):
        return None
    out.update({"selection_hint_small_fire": 0 in small, "selection_hint_small_smoke": 1 in small,
                "annotation_state": "source_boxes_not_yet_reviewed"})
    return out


def metadata(output):
    path, receipt = output / "eligible_metadata.jsonl", output / "metadata_receipt.json"
    if path.exists():
        authority = json.loads(receipt.read_text())
        if (authority["source_manifest_sha256"] != MANIFEST_SHA256 or authority["revision"] != REVISION
                or authority["eligible_metadata_sha256"] != digest(path)):
            raise ValueError("Cached FASDD metadata identity changed")
        if authority.get("normalizer_version") == NORMALIZER_VERSION:
            return read_rows(path)
    rows, counts, checksum, pending, transferred = [], Counter(), hashlib.sha256(), b"", 0
    train_index = 0

    def consume(line):
        nonlocal train_index
        if not line.strip():
            return
        row = json.loads(line)
        counts["manifest_rows"] += 1
        index = train_index
        if row["split"] == "train":
            train_index += 1
        try:
            candidate = source_candidate(row, index)
        except (ValueError, KeyError, TypeError):
            counts["invalid_source_geometry_or_category"] += 1
            return
        if candidate is not None:
            rows.append(candidate)

    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{MANIFEST}"
    with session().get(url, stream=True, timeout=45) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Full metadata stream required for its published SHA-256")
        for chunk in response.iter_content(1 << 20):
            transferred += len(chunk)
            if transferred > MANIFEST_BYTES:
                raise ValueError("Metadata transfer exceeds the pinned manifest size")
            checksum.update(chunk)
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                consume(line)
        consume(pending)
    if transferred != MANIFEST_BYTES or checksum.hexdigest() != MANIFEST_SHA256:
        raise ValueError("FASDD manifest bytes/hash differ from the pinned Hub object")
    write_rows(path, rows)
    receipt.write_text(json.dumps({"repository": REPO, "revision": REVISION,
        "source_manifest": MANIFEST, "source_manifest_sha256": checksum.hexdigest(),
        "source_manifest_streamed_bytes": transferred, "full_manifest_stored": False,
        "eligible_metadata_sha256": digest(path), "eligible_rows": len(rows), "normalizer_version": NORMALIZER_VERSION,
        "source_train_rows": train_index, **counts}, indent=2), encoding="utf-8")
    print(json.dumps({"metadata_verified": True, "eligible_rows": len(rows), **counts}), flush=True)
    return rows


def prior_identities(root, base, history):
    """Do not reintroduce rejected V4/V7 records with a fresh download URL."""
    known, records = {r["sha256"] for r in base}, {r.get("source_record_id") for r in base}
    known.update(r["source_original_sha256"] for r in base if r.get("source_original_sha256"))
    patterns = ("pointing-dataset-v4*/review_pool.jsonl", "pointing-dataset-v4*/fire_supplement_pool.jsonl",
                "pointing-v7*/candidate_manifest.jsonl", "pointing-v8-review-*/review_manifest.jsonl")
    for pattern in patterns:
        for path in root.glob(pattern):
            for row in read_rows(path):
                known.add(row.get("sha256", row.get("image_sha256")))
                records.add(row.get("source_record_id"))
    known.update(history.get("sha256", {}))
    frozen = {canonical_source_group(group) for group, splits in history.get("source_groups", {}).items()
              if any(split != "train" for split in splits)}
    return known, records, frozen


def select(rows, known, records, frozen, limit):
    groups, excluded = defaultdict(list), Counter()
    for row in rows:
        fire = [b[2] * b[3] / (row["width"] * row["height"])
                for b, label in zip(row["objects"]["bbox"], row["objects"]["category"]) if label == 0]
        # A giant fire front with one tiny secondary annotation is not the
        # intended missing case. This is a queue filter, never visual clearance.
        if not fire or len(fire) > 4 or max(fire) > .005:
            excluded["not_an_entirely_small_fire_scene_hint"] += 1
            continue
        reason = ("already_known_image_or_review" if row["sha256"] in known or row["source_record_id"] in records
                  else "historical_holdout_group" if canonical_source_group(row["source_group_id"]) in frozen else None)
        if reason:
            excluded[reason] += 1
            continue
        groups[row["source_group_id"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: (not r["selection_hint_small_fire"], not r["selection_hint_small_smoke"], r["sha256"]))
    planned = [group[n] for n in range(2) for _, group in sorted(groups.items()) if len(group) > n]
    planned.sort(key=lambda r: (not r["selection_hint_small_fire"], not r["selection_hint_small_smoke"], r["sha256"]))
    return planned[:limit], dict(excluded)


class Viewer:
    def __init__(self):
        self.pages, self.lock = {}, threading.Lock()

    def image_url(self, row):
        offset = row["row_index"] // 100 * 100
        with self.lock:
            if offset not in self.pages:
                data = api_rows({"dataset": REPO, "config": "default", "split": "train", "offset": offset, "length": 100}, revision=REVISION)
                self.pages[offset] = {item["row_idx"]: item["row"] for item in data["rows"]}
            item = self.pages[offset].get(row["row_index"])
        if (not item or item["sha256"] != row["sha256"] or item["source_record_id"] != row["source_record_id"]
                or item["source_name"] != "fasdd" or item["license"] != LICENSE):
            raise ValueError("FASDD Viewer row does not match the immutable source manifest")
        # Dataset Viewer saves PIL images again (JPEG by default); these assets
        # are not byte-identical to the original. Bind the exact row/revision,
        # same dimensions and annotation payload before using that representation.
        annotations = json.loads(item["annotations_json"])
        objects = {"bbox": [a["bbox_xywh"] for a in annotations],
                   "category": [CLASS_NAMES[a["class_name"].casefold()] for a in annotations]}
        if (objects["bbox"] != row["objects"]["bbox"] or objects["category"] != row["objects"]["category"]
                or (item["image"]["width"], item["image"]["height"]) != (row["width"], row["height"])):
            raise ValueError("Viewer annotations or dimensions differ from the source record")
        url = item["image"]["src"]
        if f"/{REPO}/--/{REVISION}/--/default/train/{row['row_index']}/image/" not in unquote(urlsplit(url).path):
            raise ValueError("Viewer asset URL is not bound to the expected source row/revision")
        return url


def acquire(row, output, budget, viewer):
    receipt = output / "receipts" / (row["sha256"] + ".json")
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if (result["source_revision"] != REVISION or result["source_original_sha256"] != row["sha256"]
                or digest(Path(result["source_image"])) != result["sha256"]):
            raise ValueError("Retained FASDD image changed")
        return result
    with session().get(viewer.image_url(row), timeout=40, stream=True) as response:
        response.raise_for_status()
        payload = bytearray()
        for chunk in response.iter_content(65536):
            budget.add(len(chunk))
            payload.extend(chunk)
            if len(payload) > 8_000_000:
                raise ValueError("FASDD individual image exceeds 8 MB")
    asset_sha = hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        if source.size != (row["width"], row["height"]):
            raise ValueError("Source resolution and annotation coordinates differ")
        rgb = source.convert("RGB")
        phash = str(imagehash.phash(rgb))
        flipped = str(imagehash.phash(ImageOps.mirror(rgb)))
    source_distance = (int(phash, 16) ^ int(row["phash"], 16)).bit_count()
    if source_distance > 4:
        raise ValueError("Viewer representation is not perceptually consistent with the original source fingerprint")
    path = output / "images" / (asset_sha + ".jpg")
    result = row | {"candidate_id": "FASDD-" + row["sha256"][:16], "source_image": str(path.resolve()),
        "sha256": asset_sha, "source_original_sha256": row["sha256"], "source_original_phash": row["phash"],
        "source_phash_distance": source_distance, "source_image_representation": "hf_viewer_jpeg_reencoded_same_dimensions",
        "representation_evidence": "https://github.com/huggingface/dataset-viewer/blob/main/libs/libcommon/src/libcommon/viewer_utils/asset.py#create_image_file",
        "image_bytes": len(payload), "phash": phash, "phash_flipped": flipped,
        "source_repository": REPO, "source_revision": REVISION, "source_dataset": "FASDD-v9", "source_family": "FASDD",
        "license_evidence": f"https://huggingface.co/datasets/{REPO}/blob/{REVISION}/README.md",
        "split": "train", "split_group_id": row["source_group_id"], "synthetic": False,
        "source_group_evidence": "Conservative upstream block; physical scene identity still requires visual review.",
        "review_status": "pending_visual_review", "v8_corpus_admitted": False, "v8_training_admitted": False}
    geometries(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) != asset_sha:
        raise ValueError("Refusing to overwrite a changed source image")
    if not path.exists():
        path.write_bytes(payload)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000 or not 1 <= args.workers <= 4:
        raise ValueError("Bounded FASDD acquisition limits exceeded")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = metadata(args.output)
    known, records, frozen = prior_identities(args.root, read_rows(args.base), json.loads(args.history.read_text()))
    planned, excluded = select(rows, known, records, frozen, args.limit)
    write_rows(args.output / "acquisition_plan.jsonl", planned)
    report = {"eligible_metadata": len(rows), "planned": len(planned), "exclusions": excluded,
              "small_fire_hints": sum(r["selection_hint_small_fire"] for r in planned),
              "small_smoke_hints": sum(r["selection_hint_small_smoke"] for r in planned), "admitted": 0}
    print(json.dumps(report), flush=True)
    (args.output / "acquisition_plan_report.json").write_text(json.dumps(report | {
        "plan_sha256": digest(args.output / "acquisition_plan.jsonl"),
        "selection_policy": "every_fire_box_at_most_0.5pct_and_at_most_four_fire_instances; hints_only"}, indent=2), encoding="utf-8")
    if args.metadata_only:
        return
    status_path = args.output / "acquisition_report.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        resume_at = previous.get("not_before_utc")
        fallback = status_path.stat().st_mtime + 600 if any("429" in r.get("message", "") for r in previous.get("errors_this_pass", [])) else 0
        if (datetime.fromisoformat(resume_at).timestamp() if resume_at else fallback) > time.time():
            raise ValueError("FASDD server cooldown is active; metadata-only review remains available")
    retained = [json.loads(p.read_text()) for p in sorted((args.output / "receipts").glob("*.json"))]
    for row in retained:
        if row["source_revision"] != REVISION or digest(Path(row["source_image"])) != row["sha256"]:
            raise ValueError("Retained FASDD receipt/image mismatch")
    budget = ByteBudget(1_000_000_000, sum(p.stat().st_size for p in (args.output / "images").glob("*")))
    viewer, stopped, resume_after = Viewer(), threading.Event(), [0.0]
    candidates, errors = {r["sha256"]: r for r in retained}, []

    def worker(row):
        if stopped.is_set():
            return None, None
        try:
            return acquire(row, args.output, budget, viewer), None
        except Exception as error:
            # Stop on a server throttle, budget, or provenance problem; no retry loop.
            stopped.set()
            response = getattr(error, "response", None)
            if response is not None and response.status_code == 429:
                resume_after[0] = max(resume_after[0], time.time() + rate_limit_delay(response))
            return None, {"source_record_id": row["source_record_id"], "error_type": type(error).__name__, "message": str(error).split("https://")[0]}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, (row, error) in enumerate(pool.map(worker, sorted(planned, key=lambda r: r["row_index"])), 1):
            if row:
                candidates[row["sha256"]] = row
            if error:
                errors.append(error)
            if number % 24 == 0:
                print(json.dumps({"queue_entries_processed": number, "retained_candidates": len(candidates), "errors": len(errors), "image_bytes_transferred_including_cache": budget.used}), flush=True)
    write_rows(args.output / "candidate_manifest.jsonl", list(candidates.values()))
    # Preserve earlier errors on resumed passes, just as source images are preserved.
    error_path = args.output / "acquisition_errors.jsonl"
    write_rows(error_path, (read_rows(error_path) if error_path.exists() else []) + errors)
    report.update({"status": "paused_source_error" if errors else "candidates_acquired_not_admitted",
        "retained_candidates": len(candidates), "errors_this_pass": errors,
        "image_bytes_retained": sum(p.stat().st_size for p in (args.output / "images").glob("*")),
        "image_bytes_transferred_including_cache": budget.used,
        "source_revision": REVISION, "image_archive_downloaded": False})
    if resume_after[0]:
        report["not_before_utc"] = datetime.fromtimestamp(resume_after[0], timezone.utc).isoformat()
    (args.output / "acquisition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
