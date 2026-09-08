"""Reuse pinned D-Fire metadata and acquire only fresh small-fire candidates."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import requests
from PIL import Image, ImageOps

from training.pointing_dataset_v7.build_dfire_gap_candidates import DATASET, MIRROR_REVISION, OFFICIAL_REVISION, parse_labels
from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.acquire_pyrosdis import api_rows, session
from training.pointing_dataset_v8.acquire_wui import ByteBudget
from training.pointing_dataset_v8.audit_coverage import geometries

LICENSE_EVIDENCE = f"https://github.com/gaia-solutions-on-demand/DFireDataset/blob/{OFFICIAL_REVISION}/README.md"


def select(root, prefixes=("WEB",), target="small_fire"):
    if target not in {"small_fire", "small_smoke"}:
        raise ValueError("Unknown small-target selection profile")
    target_class = 0 if target == "small_fire" else 1
    authority = root / "pointing-dataset-v4-dfire-fire-extension-20260824"
    info = json.loads((authority / "source_authority.json").read_text())
    if digest(authority / "source_metadata.jsonl") != info["source_metadata_sha256"]:
        raise ValueError("Retained pinned D-Fire metadata changed")
    known = set()
    pools = (list(root.glob("pointing-dataset-v4*/review_pool.jsonl"))
             + list(root.glob("pointing-dataset-v4*/fire_supplement_pool.jsonl"))
             + list(root.glob("pointing-dataset-v4*/candidate_manifest.jsonl"))
             + list(root.glob("pointing-v7*/*manifest.jsonl"))
             + list(root.glob("fireviewer-pointing-v7-*/selection_manifest.jsonl"))
             + list(root.glob("pointing-v8-source-*/candidate_manifest.jsonl"))
             + list(root.glob("pointing-v8-review-*/review_manifest.jsonl")))
    for path in pools:
        for row in read_rows(path):
            for key in ("source_record_id", "filename", "source_ref"):
                known.update(s.lower() for s in re.findall(r"(?:WEB|AoF|PublicDataset)\d+\.jpg", str(row.get(key, "")), re.I))
    history = json.loads((root / "fireviewer-pointing-v7-audited-20260827/historical_split_registry.json").read_text())
    frozen = {key.lower() for key, splits in history["source_groups"].items() if any(s != "train" for s in splits)}
    result = []
    allowed = {prefix.lower() for prefix in prefixes}
    if not allowed or not allowed <= {"web", "aof", "publicdataset"}:
        raise ValueError("Unknown pinned D-Fire filename family")
    for row in read_rows(authority / "source_metadata.jsonl"):
        name = row["filename"]
        match = re.fullmatch(r"(WEB|AoF|PublicDataset)(\d+)\.jpg", name, re.I)
        if not match or match[1].lower() not in allowed or name.lower() in known:
            continue
        prefix, number = match[1].lower(), int(match[2])
        group = f"dfire:{prefix}:block-{row['row_index']//100:04d}"
        aliases = {group, f"dfire:{prefix}:block-{number//100:04d}", f"dfire:{prefix}:{number//50}"}
        if aliases & frozen:
            continue
        if hashlib.sha256(row["label_text"].encode()).hexdigest() != row["label_sha256"]:
            raise ValueError("Pinned D-Fire label text changed")
        labels = parse_labels(row["label_text"], 1, 1)
        if not labels or max(a["area_ratio"] for a in labels) > .25:
            continue
        if not any(a["category"] == target_class and a["area_ratio"] <= .005 for a in labels):
            continue
        result.append(row | {"source_group_id": group,
                            "source_group_aliases": sorted(aliases),
                            "grouping_basis": "conservative_filename_blocks_not_verified_incident_identity"})
    return result


class ViewerRows:
    """Read one revision-bound metadata page, not a failing Hub URL per image."""

    def __init__(self):
        self.pages = {}
        self.lock = threading.Lock()

    def image_url(self, row):
        offset = row["row_index"] // 100 * 100
        with self.lock:
            if offset not in self.pages:
                page = api_rows({"dataset": DATASET, "config": "default", "split": "train",
                                 "offset": offset, "length": 100}, revision=MIRROR_REVISION)
                self.pages[offset] = {item["row_idx"]: item["row"] for item in page["rows"]}
            value = self.pages[offset].get(row["row_index"])
        if not value or value.get("filename") != row["filename"] or value.get("label") != row["label_text"]:
            raise ValueError("D-Fire Viewer row/filename/annotations mismatch")
        return value["image"]["src"]  # Signed URLs remain in memory only.


def retained_candidates(output):
    """A resumed acquisition must preserve all prior, still-valid receipts."""
    result = {}
    for receipt in sorted((output / "receipts").glob("*.json")):
        row = json.loads(receipt.read_text(encoding="utf-8"))
        if (row.get("source_revision") != MIRROR_REVISION
                or digest(Path(row["source_image"])) != row["sha256"]):
            raise ValueError("Retained D-Fire receipt/image identity changed")
        if row["filename"] in result and result[row["filename"]] != row:
            raise ValueError("Conflicting retained D-Fire receipts")
        result[row["filename"]] = row
    manifest = output / "candidate_manifest.jsonl"
    for row in read_rows(manifest) if manifest.exists() else []:
        if row["filename"] not in result or result[row["filename"]]["sha256"] != row["sha256"]:
            raise ValueError("D-Fire manifest has no matching retained receipt")
    return result


def rate_limit_delay(response):
    value = response.headers.get("Retry-After", "")
    try:
        return max(1, int(value))
    except ValueError:
        try:
            return max(1, int(parsedate_to_datetime(value).timestamp() - time.time()))
        except (ValueError, TypeError, OverflowError):
            return 600


def acquire(row, output, budget, viewer=None):
    receipt = output / "receipts" / (Path(row["filename"]).stem + ".json")
    if receipt.exists():
        cached = json.loads(receipt.read_text())
        if digest(Path(cached["source_image"])) != cached["sha256"]:
            raise ValueError("Cached D-Fire image changed")
        if cached["license_evidence"] != LICENSE_EVIDENCE:
            cached["license_evidence"] = LICENSE_EVIDENCE
            receipt.write_text(json.dumps(cached), encoding="utf-8")
        return cached
    url = (viewer or ViewerRows()).image_url(row)
    response = session().get(url, timeout=35, stream=True)
    with response:
        response.raise_for_status()
        payload = bytearray()
        for chunk in response.iter_content(65536):
            budget.add(len(chunk))
            payload.extend(chunk)
            if len(payload) > 3_000_000:
                raise ValueError("D-Fire individual image byte cap reached")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        phash = str(imagehash.phash(image.convert("RGB")))
        flipped = str(imagehash.phash(ImageOps.mirror(image.convert("RGB"))))
    objects = parse_labels(row["label_text"], width, height)
    sha = hashlib.sha256(payload).hexdigest()
    path = output / "images" / (sha + ".jpg")
    candidate = row | {"candidate_id": "DF-" + Path(row["filename"]).stem, "sha256": sha,
        "source_image": str(path.resolve()), "image_bytes": len(payload), "width": width, "height": height,
        "phash": phash, "phash_flipped": flipped,
        "objects": {"bbox": [a["bbox"] for a in objects], "category": [a["category"] for a in objects],
                    "area": [a["bbox"][2]*a["bbox"][3] for a in objects]},
        "source_dataset": "D-Fire", "source_family": "D-Fire", "source_record_id": row["filename"],
        "source_revision": MIRROR_REVISION, "source_split": "train", "split": "train",
        "split_group_id": row["source_group_id"], "synthetic": False, "license": "CC0-1.0",
        "license_evidence": LICENSE_EVIDENCE,
        "annotation_state": "source_boxes_not_yet_reviewed", "review_status": "pending_visual_review",
        "v8_corpus_admitted": False, "v8_training_admitted": False}
    geometries(candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(payload)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(candidate), encoding="utf-8")
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefixes", nargs="+", choices=["WEB", "AoF", "PublicDataset"], default=["WEB"])
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--target", choices=["small_fire", "small_smoke"], default="small_fire",
                        help="Select source-box hints only; every image still requires visual review")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.limit <= 400 or not 1 <= args.workers <= 4:
        raise ValueError("Bounded small-fire acquisition limits exceeded")
    retained = retained_candidates(args.output)
    eligible = [r for r in select(args.root, args.prefixes, args.target) if r["filename"] not in retained]
    # Spread the review queue across source blocks instead of taking a run of
    # adjacent frames. These are selection hints, never certified scene groups.
    by_group = {}
    for row in sorted(eligible, key=lambda r: hashlib.sha256(r['filename'].encode()).hexdigest()):
        by_group.setdefault(row['source_group_id'], []).append(row)
    plan = [items[rank] for rank in range(8) for _, items in sorted(by_group.items()) if rank < len(items)][:args.limit]
    (args.output / "acquisition_plan.jsonl").write_text("".join(json.dumps(r)+"\n" for r in plan), encoding="utf-8")
    print(json.dumps({"eligible_target_metadata": len(eligible), "planned_target_candidates": len(plan),
                      "target_profile": args.target,
                      "filename_families": args.prefixes, "archive_downloaded": False, "admitted": 0}), flush=True)
    if args.metadata_only:
        return
    status_path = args.output / "acquisition_status.json"
    if status_path.exists():
        prior = json.loads(status_path.read_text())
        if prior.get("not_before_utc") and datetime.fromisoformat(prior["not_before_utc"]).timestamp() > time.time():
            raise ValueError("D-Fire server cooldown is still active; no network request sent")
    budget = ByteBudget(250_000_000, sum(p.stat().st_size for p in (args.output/"images").glob("*")))
    candidates, errors, paused = dict(retained), [], threading.Event()
    viewer, resume_after = ViewerRows(), [0.0]
    def worker(row):
        if paused.is_set():
            return None, {"record": row["filename"], "error_type": "SkippedAfterResourceLimit", "message": "No new request sent after a resource limit"}
        try:
            return acquire(row, args.output, budget, viewer), None
        except Exception as exc:
            limited = isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 429
            if limited:
                resume_after[0] = max(resume_after[0], time.time() + rate_limit_delay(exc.response))
            if limited or "budget reached" in str(exc):
                paused.set()
            return None, {"record": row["filename"], "error_type": type(exc).__name__,
                          "message": str(exc).split(" for url:")[0][:250]}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, (candidate, error) in enumerate(pool.map(worker, plan), 1):
            if candidate:
                candidates[candidate["filename"]] = candidate
            else:
                errors.append(error)
            if number % 25 == 0:
                print(json.dumps({"processed": number, "candidates": len(candidates), "download_mb": round(budget.used/1e6, 1)}), flush=True)
    for name, rows in (("candidate_manifest.jsonl", [candidates[k] for k in sorted(candidates)]), ("acquisition_errors.jsonl", errors)):
        (args.output/name).write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    status = {"candidates": len(candidates), "previous_candidates_preserved": len(retained),
              "target_profile": args.target,
              "new_candidates": len(candidates) - len(retained), "errors": len(errors), "admitted": 0,
              "status": "paused_resource_limit_not_complete" if paused.is_set() else "bounded_acquisition_complete",
              "image_bytes_downloaded": budget.used, "image_byte_budget": budget.limit,
              "automatic_retry_scheduled": False, "training_started": False}
    if resume_after[0]:
        status["not_before_utc"] = datetime.fromtimestamp(resume_after[0], timezone.utc).isoformat()
    (args.output / "acquisition_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
