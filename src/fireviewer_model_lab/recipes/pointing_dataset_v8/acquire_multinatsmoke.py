"""Selective, resumable AusSmoke/FireSpot acquisition; never grants admission.

Reuse the V4 pilot's pinned authority and strict HTTP Range checks. Only the
central-directory ranges are cached; selected image/mask pairs have individual
receipts. Neither the 43.7 GB ZIP nor derived TestSmall/TestLarge views are fetched.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import imagehash
import numpy as np
import requests
from PIL import ImageOps
from remotezip import PartialBuffer, RemoteZip

from training.pointing_dataset_v4 import pilot_firespot_remotezip as pilot
from training.pointing_dataset_v7.split_registry import digest, read_rows

# The pinned release spells the Australian collection "AuSmoke" in the ZIP.
FAMILIES = {"aussmoke": "AusSmoke", "ausmoke": "AusSmoke", "firespot": "FireSpot"}
MAX_RANGE_BYTES = 40 * 1024 * 1024
MAX_MEMBER_BYTES = 12 * 1024 * 1024
USER_AGENT = "fireviewer-pointing-v8-selective-multinatsmoke/1"


class RangeBudget:
    def __init__(self, limit, already_read=0):
        self.limit, self.reserved = limit, already_read
        self.lock = threading.Lock()

    def reserve(self, size):
        with self.lock:
            if self.reserved + size > self.limit:
                raise ValueError("Cumulative HTTP byte budget reached; existing files retained")
            self.reserved += size


def write_json(path, value):
    pilot.atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def write_rows(path, rows):
    pilot.atomic_write(path, b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in rows))


class CachedRangeFetcher(pilot.RecordingRemoteFetcher):
    """Bound transfers before requesting data; cache only ZIP index ranges."""

    def __init__(self, *args, cache_root, max_total_bytes, budget=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_root = Path(cache_root)
        self.max_total_bytes = max_total_bytes
        self.budget = budget or RangeBudget(max_total_bytes, sum(r["bytes_read"] for r in self._request_log))

    def fetch(self, data_range, stream=False):
        start, end = data_range
        if start < 0 and end is None:
            # The immutable API authority is checked before opening this ZIP.
            start, end = max(0, pilot.HF_ARCHIVE_SIZE + start), pilot.HF_ARCHIVE_SIZE - 1
        if end is None or not 0 <= start <= end < pilot.HF_ARCHIVE_SIZE:
            raise ValueError("Unbounded or out-of-archive range refused")
        length = end - start + 1
        if length > MAX_RANGE_BYTES:
            raise ValueError("Range exceeds 40 MiB; full-archive download refused")
        cached = self.cache_root / f"{start}-{end}.bin"
        receipt = cached.with_suffix(".json")
        if not stream and cached.exists():
            meta = json.loads(receipt.read_text(encoding="utf-8"))
            if (meta.get("revision") != pilot.HF_REVISION or cached.stat().st_size != length
                    or digest(cached) != meta.get("sha256")):
                raise ValueError("Cached archive index changed")
            payload = cached.read_bytes()
        else:
            self.budget.reserve(length)
            part = super().fetch((start, end), stream=False)
            try:
                payload = part.read()
            finally:
                part.close()
            if len(payload) != length:
                raise ValueError("Truncated archive range")
            if not stream:
                pilot.atomic_write(cached, payload)
                write_json(receipt, {"revision": pilot.HF_REVISION, "bytes": length,
                                     "sha256": pilot.sha256_bytes(payload)})
        return PartialBuffer(io.BytesIO(payload), start, length, False)


def member_role(name):
    path = PurePosixPath(name)
    parts = [p.casefold() for p in path.parts]
    if "__macosx" in parts or path.name.startswith("._"):
        return None
    families = [FAMILIES[p] for p in parts if p in FAMILIES]
    if len(families) != 1:
        return None
    splits = [p for p in parts if p in {"train", "test"}]
    if len(splits) != 1:
        return None
    parent = set(parts[:-1])
    if parent & pilot.IMAGE_DIR_NAMES and path.suffix.casefold() in pilot.IMAGE_EXTENSIONS:
        return families[0], splits[0], "image"
    if parent & pilot.MASK_DIR_NAMES and path.suffix.casefold() in pilot.MASK_EXTENSIONS:
        return families[0], splits[0], "mask"
    return None


def build_pairs(infos):
    by_key = defaultdict(dict)
    for info in infos:
        role = member_role(info.filename)
        if not role:
            continue
        family, split, kind = role
        key = family, split, PurePosixPath(info.filename).stem.casefold()
        if kind in by_key[key]:
            raise ValueError(f"Duplicate {kind} for {key}")
        by_key[key][kind] = info
    pairs, incomplete = [], []
    for (family, split, stem), members in sorted(by_key.items()):
        if set(members) != {"image", "mask"}:
            incomplete.append({"family": family, "split": split, "stem": stem})
            continue
        pairs.append({"family": family, "source_split": split, "stem": stem,
                      "image_member": members["image"].filename,
                      "mask_member": members["mask"].filename,
                      "image_bytes": members["image"].file_size,
                      "mask_bytes": members["mask"].file_size})
    return pairs, incomplete


def select_pairs(pairs, per_group=8):
    """Spread over declared sequences, excluding known aerial/artificial cues.

    Filename groups are acquisition strata only, not verified independent scenes.
    A visual reviewer must merge correlated views before V8 assembly.
    """
    if not 1 <= per_group <= 8:
        raise ValueError("At most eight candidates per declared sequence")
    groups, excluded = defaultdict(list), Counter()
    for row in pairs:
        if row["source_split"] != "train":
            excluded["upstream_test_untouched"] += 1
            continue
        stem = row["stem"]
        if any(cue in stem for cue in ("dji", "smokemachine", "test_run_smoke", "tower")):
            excluded["aerial_obstructed_or_artificial_provenance_cue"] += 1
            continue
        if row["family"] == "AusSmoke":
            group = re.sub(r"_\d+-\+\d+$", "", stem)
            if group == stem:
                excluded["unparsed_sequence"] += 1
                continue
        else:
            if not re.fullmatch(r"\d{2}-\d{2}-\d{2}-\d+-1", stem):
                excluded["unparsed_sequence"] += 1
                continue
            group = "-".join(stem.split("-")[:2])
        groups[row["family"]+":"+group].append(row)
    picked = []
    # Include early stages and later appearance; no detector scores guide selection.
    quantiles = (.02, .08, .18, .30, .45, .60, .75, .90)[:per_group]
    for group, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: r["stem"])
        indices = sorted({round(q*(len(ordered)-1)) for q in quantiles})
        for index in indices:
            picked.append(ordered[index] | {"source_group_id": group,
                "source_group_evidence": "Conservative filename sequence/spot prefix; scene/camera correlation still requires visual review."})
    return picked, {"candidate_pairs": len(picked), "declared_sequence_groups": len(groups),
                    "families": dict(Counter(r["family"] for r in picked)), "exclusions": dict(excluded),
                    "predicted_original_image_mask_bytes": sum(r["image_bytes"]+r["mask_bytes"] for r in picked),
                    "independent_scenes_not_yet_verified": True}


def source_documents(archive, infos, output):
    documents = []
    for info in infos:
        parts = [p.casefold() for p in PurePosixPath(info.filename).parts]
        if (not any(p in FAMILIES for p in parts) or info.file_size > 1024 * 1024
                or PurePosixPath(info.filename).name.casefold() not in
                {"info.txt", "license", "license.txt", "readme.md", "readme.txt"}):
            continue
        name = pilot.sha256_bytes(info.filename.encode())[:12] + "-" + PurePosixPath(info.filename).name
        local = output / "sources" / name
        if not local.exists():
            pilot.atomic_write(local, archive.read(info))
        documents.append({"archive_member": info.filename, "local": str(local.resolve()),
                          "sha256": digest(local), "bytes": local.stat().st_size})
    return documents


def mask_envelope(inspection):
    """A conservative proposal, not a claim that every plume/visible flame is labelled."""
    mask = inspection["mask"]
    if mask["unique_gray_value_count"] > 2 or any(v not in {0, 1, 255} for v in mask["unique_gray_values"]):
        raise ValueError("Mask is not binary; manual interpretation required")
    bbox = mask["foreground_bbox_xyxy_nonzero"]
    if bbox is None:
        raise ValueError("Empty smoke mask; no positive acquisition")
    x, y, right, bottom = bbox
    box = [x, y, right-x, bottom-y]
    return {"bbox": [box], "category": [1], "area": [box[2]*box[3]]}


def source_mask_proposal(mask_image, inspection, family):
    raw = inspection["mask"]
    if raw["unique_gray_value_count"] <= 2 and all(v in {0, 1, 255} for v in raw["unique_gray_values"]):
        return mask_envelope(inspection), {"method": "source_binary_nonzero", "original_mask_unchanged": True}
    pixels = np.asarray(mask_image.convert("L"))
    values = np.unique(pixels)
    # FireSpot's released masks contain near-black/near-white compression values.
    # Do not interpret arbitrary intermediate transparency as a binary annotation.
    if family != "FireSpot" or not np.all((values <= 16) | (values >= 239)):
        raise ValueError("Mask has unresolved intermediate levels; manual interpretation required")
    foreground = pixels >= 128
    ys, xs = np.nonzero(foreground)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if len(xs) else None
    normalized = {"mask": {"unique_gray_value_count": 2, "unique_gray_values": [0, 255],
                            "foreground_bbox_xyxy_nonzero": bbox}}
    return mask_envelope(normalized), {"method": "firespot_near_binary_compression_threshold_128",
        "accepted_source_gray_bands": [[0, 16], [239, 255]], "original_mask_unchanged": True,
        "foreground_pixels": int(foreground.sum()), "foreground_fraction": float(foreground.mean()),
        "foreground_bbox_xyxy": bbox}


def acquire_pair(archive, row, output):
    if row["source_split"] != "train":
        raise ValueError("Source test examples are not acquired for this train extension")
    if not row.get("source_group_id") or not row.get("source_group_evidence"):
        raise ValueError("Explicit conservative acquisition grouping is required")
    cid = ("AUS-" if row["family"] == "AusSmoke" else "FS-") + pilot.sha256_bytes(row["image_member"].encode())[:14]
    receipt = output / "receipts" / f"{cid}.json"
    if receipt.exists():
        cached = json.loads(receipt.read_text(encoding="utf-8"))
        for field, sha_field in (("source_image", "sha256"), ("source_mask", "mask_sha256")):
            if digest(Path(cached[field])) != cached[sha_field]:
                raise ValueError("Cached candidate image/mask changed")
        if cached["source_revision"] != pilot.HF_REVISION or cached["source_record_id"] != row["image_member"]:
            raise ValueError("Cached source identity mismatch")
        return cached
    members = [archive.getinfo(row[k]) for k in ("image_member", "mask_member")]
    if any(i.file_size > MAX_MEMBER_BYTES or i.compress_size > MAX_MEMBER_BYTES for i in members):
        raise ValueError("Selected member exceeds 12 MiB")
    raw_receipt = output / "raw-pair-receipts" / f"{cid}.json"
    if raw_receipt.exists():
        raw = json.loads(raw_receipt.read_text(encoding="utf-8"))
        if raw["image_member"] != row["image_member"] or raw["revision"] != pilot.HF_REVISION:
            raise ValueError("Raw cached pair identity differs")
        image_payload, mask_payload = (Path(raw[k]).read_bytes() for k in ("source_image", "source_mask"))
        if (pilot.sha256_bytes(image_payload), pilot.sha256_bytes(mask_payload)) != (raw["sha256"], raw["mask_sha256"]):
            raise ValueError("Raw cached pair changed")
    else:
        image_payload, mask_payload = (archive.read(i) for i in members)
    image, mask_image, inspection = pilot.inspect_pair(image_payload, mask_payload)
    sha, mask_sha = pilot.sha256_bytes(image_payload), pilot.sha256_bytes(mask_payload)
    image_path = output / "images" / (sha + PurePosixPath(row["image_member"]).suffix.lower())
    mask_path = output / "masks" / (mask_sha + PurePosixPath(row["mask_member"]).suffix.lower())
    for path, payload in ((image_path, image_payload), (mask_path, mask_payload)):
        if path.exists() and digest(path) != pilot.sha256_bytes(payload):
            raise ValueError("Existing content-addressed file changed")
        if not path.exists():
            pilot.atomic_write(path, payload)
    # Preserve original bytes even if annotation interpretation fails below.
    write_json(raw_receipt, {"revision": pilot.HF_REVISION, "image_member": row["image_member"],
                            "source_image": str(image_path.resolve()), "source_mask": str(mask_path.resolve()),
                            "sha256": sha, "mask_sha256": mask_sha})
    objects, processing = source_mask_proposal(mask_image, inspection, row["family"])
    width, height = image.size
    license_name = "CC-BY-4.0" if row["family"] == "FireSpot" else None
    candidate = row | {"candidate_id": cid, "sha256": sha, "source_image": str(image_path.resolve()),
        "source_mask": str(mask_path.resolve()), "mask_sha256": mask_sha, "image_bytes": len(image_payload),
        "mask_bytes": len(mask_payload), "width": width, "height": height,
        "phash": str(imagehash.phash(image.convert("RGB"))),
        "phash_flipped": str(imagehash.phash(ImageOps.mirror(image.convert("RGB")))),
        "source_dataset": row["family"], "source_family": row["family"], "source_revision": pilot.HF_REVISION,
        "source_repository": pilot.HF_REPO, "source_record_id": row["image_member"], "split": "train",
        "split_group_id": row["source_group_id"], "license": license_name,
        "license_evidence": (f"https://github.com/henryzhao0615/MultiNatSmoke/blob/{pilot.MULTINATSMOKE_GIT_REVISION}/README.md" if license_name else None),
        "license_provenance": "declared_by_redistributor_not_independent_legal_clearance" if license_name else "unresolved",
        "public_redistribution_approved": False, "synthetic": False, "objects": objects,
        "upstream_mask_inspection": inspection["mask"],
        "proposal_mask_processing": processing,
        "mask_pixel_fraction_is_not_bbox_area_fraction": True,
        "annotation_state": "source_mask_envelope_not_reviewed",
        "annotation_warning": "Smoke mask envelope only; inspect whole scene, separate distinct plumes and annotate visible flames before admission.",
        "review_status": "pending_visual_review", "v8_corpus_admitted": False, "v8_training_admitted": False,
        "archive_members": [pilot._zip_member_receipt(i, p) for i, p in zip(members, (image_payload, mask_payload))]}
    write_json(receipt, candidate)
    return candidate


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "http-range-ledger.jsonl"
    range_log = read_rows(ledger) if ledger.exists() else []
    before = sum(r["bytes_read"] for r in range_log)
    budget = RangeBudget(args.max_http_mib*1024*1024, before)
    client = requests.Session()
    client.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    try:
        # Always revalidate the small immutable tree; no credential or signed URL is persisted.
        payload, meta = pilot._request_bytes(client, pilot.HF_TREE_URL)
        pilot.validate_hf_metadata(payload)
        pilot.atomic_write(output / "sources" / "hf-tree.json", payload)
        write_json(output / "sources" / "hf-tree-receipt.json", meta)

        def factory(*a, **kw):
            return CachedRangeFetcher(*a, **kw, request_log=range_log, cache_root=output / "zip-index-cache",
                                      max_total_bytes=args.max_http_mib*1024*1024, budget=budget)

        with RemoteZip(pilot.HF_ARCHIVE_URL, session=client, fetcher=factory, support_suffix_range=False,
                       headers=dict(client.headers), timeout=(10, 45)) as archive:
            infos = archive.infolist()
            pairs, incomplete = build_pairs(infos)
            write_rows(output / "pair_index.jsonl", pairs)
            documents = source_documents(archive, infos, output)
            report = {"status": "indexed_only_no_admission", "hf_repo": pilot.HF_REPO,
                "revision": pilot.HF_REVISION, "archive_bytes": pilot.HF_ARCHIVE_SIZE,
                "archive_entries": len(infos), "pairs": dict(Counter(r["family"]+":"+r["source_split"] for r in pairs)),
                "incomplete_pairs": incomplete,
                "root_folders": dict(Counter("/".join(PurePosixPath(i.filename).parts[:3]) for i in infos)),
                "source_documents": documents, "automatic_admissions": 0}
            write_json(output / "index_report.json", report)
            print(json.dumps({k: report[k] for k in ("status", "archive_entries", "pairs")}), flush=True)
            if args.make_selection:
                selection, selection_report = select_pairs(pairs)
                args.selection = output / "acquisition_selection.jsonl"
                write_rows(args.selection, selection)
                write_json(output / "selection_report.json", selection_report)
                print(json.dumps(selection_report), flush=True)
            if args.selection:
                selection = read_rows(args.selection)
                if len(selection) > 900:
                    raise ValueError("Selection exceeds bounded 900-pair acquisition")
                lookup = {r["image_member"]: r for r in pairs}
                if len(selection) != len({r["image_member"] for r in selection}):
                    raise ValueError("Duplicate requested pair")
                failures, requested = [], []
                for chosen in selection:
                    source = lookup[chosen["image_member"]]
                    if any(chosen.get(k) != source[k] for k in ("family", "source_split", "stem", "mask_member")):
                        raise ValueError("Selection identity differs from pinned ZIP index")
                    requested.append(source | {k: chosen[k] for k in ("source_group_id", "source_group_evidence")})
                local, opened = threading.local(), []

                def worker(chosen):
                    if args.workers == 1:
                        selected_archive = archive
                    else:
                        if not hasattr(local, "archive"):
                            session = requests.Session()
                            local.archive = RemoteZip(pilot.HF_ARCHIVE_URL, session=session, fetcher=factory,
                                support_suffix_range=False, headers=dict(client.headers), timeout=(10, 45))
                            opened.append((local.archive, session))
                        selected_archive = local.archive
                    try:
                        acquire_pair(selected_archive, chosen, output)
                        return None
                    except ValueError as error:
                        return {"image_member": chosen["image_member"], "error": str(error)}

                pool = ThreadPoolExecutor(max_workers=args.workers)
                try:
                    consecutive_failures = 0
                    for index, failure in enumerate(pool.map(worker, requested), 1):
                        consecutive_failures = consecutive_failures + 1 if failure else 0
                        if failure:
                            failures.append(failure)
                            if len(failures) <= 3:
                                print(json.dumps(failure), flush=True)
                        if consecutive_failures >= 3:
                            raise ValueError("Three consecutive acquisition refusals; inspect errors before further transfer")
                        if index % 16 == 0 or index == len(selection):
                            write_rows(ledger, range_log)
                            print(json.dumps({"pairs_processed": index, "selected": len(selection), "failures": len(failures),
                                "http_mib_this_invocation": round((sum(r["bytes_read"] for r in range_log)-before)/1048576, 2),
                                "admitted": 0}), flush=True)
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
                    for selected_archive, session in opened:
                        selected_archive.close()
                        session.close()
                    write_rows(output / "acquisition_failures.jsonl", failures)
                candidates = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "receipts").glob("*.json"))]
                write_rows(output / "candidate_manifest.jsonl", candidates)
                write_rows(output / "acquisition_failures.jsonl", failures)
                write_json(output / "acquisition_report.json", {"status": "pending_visual_review", "candidates": len(candidates),
                    "families": dict(Counter(r["source_family"] for r in candidates)), "requested": len(selection), "failures": len(failures),
                    "original_image_mask_bytes": sum(r["image_bytes"]+r["mask_bytes"] for r in candidates),
                    "public_redistribution_approved": False, "automatic_admissions": 0})
    finally:
        write_rows(ledger, range_log)
        client.close()
    print(json.dumps({"range_bytes_cumulative": sum(r["bytes_read"] for r in range_log),
                      "range_bytes_this_invocation": sum(r["bytes_read"] for r in range_log)-before,
                      "full_archive_downloaded": False}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--make-selection", action="store_true")
    parser.add_argument("--max-http-mib", type=int, default=700)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 40 <= args.max_http_mib <= 1500:
        raise ValueError("HTTP budget must remain between 40 and 1500 MiB")
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four acquisition workers")
    run(args)


if __name__ == "__main__":
    main()
