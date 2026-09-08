"""Acquire and strictly validate Pyro-SDIS explicit negatives in SageMaker."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat

HF_REPOSITORY = "pyronear/pyro-sdis"
HF_REVISION = "a1e553ec4d806f71fc6db744cc22bc3469487382"
HF_CONFIG = "default"
HF_SPLITS = ("train", "val")
SOURCE_ID = "pyro-sdis-explicit-negatives"
SOURCE_LICENSE = "Apache-2.0"
EXPECTED_EMPTY_ROWS = 5_499
EXPECTED_SOURCE_ROWS = 33_636
SOURCE_EMPTY_PREDICATE = "annotations == ''"
ROWS_PAGE_SIZE = 100
NEAR_DUPLICATE_DISTANCE = 4
DEFAULT_MAX_VALIDATED = 600
FINAL_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

JsonFetcher = Callable[[str], dict[str, Any]]
ByteFetcher = Callable[[str], bytes]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _difference_hash(image: Image.Image) -> str:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(resized.tobytes())
    bits = 0
    for y_value in range(8):
        offset = y_value * 9
        for x_value in range(8):
            bits = (bits << 1) | int(pixels[offset + x_value] > pixels[offset + x_value + 1])
    return f"{bits:016x}"


def _metrics(image: Image.Image) -> dict[str, float]:
    gray = image.convert("L").resize((256, 144), Image.Resampling.BILINEAR)
    stat = ImageStat.Stat(gray)
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
    return {
        "brightness": float(stat.mean[0] / 255.0),
        "contrast": float(stat.stddev[0] / 255.0),
        "edge_energy": float(edges.mean[0] / 255.0),
    }


def _read_response_bytes(response: Any) -> bytes:
    try:
        return response.read()
    finally:
        response.close()


def _http_bytes(url: str, *, attempts: int = 5) -> bytes:
    delay = 2.0
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(  # noqa: S310 - pinned HTTPS services only
            url,
            headers={"User-Agent": "FireViewer-Pointing-Corpus/2.0"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=120)  # noqa: S310
            return _read_response_bytes(response)
        except urllib.error.HTTPError as error:
            last_error = error
            retryable = error.code in {425, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 == attempts:
                raise
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 == attempts:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 30.0)
    raise RuntimeError("HTTP download exhausted retries") from last_error


def _http_json(url: str, *, attempts: int = 20) -> dict[str, Any]:
    delay = 3.0
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            payload = json.loads(_http_bytes(url, attempts=1))
            if not isinstance(payload, dict):
                raise ValueError("HTTP JSON response is not an object")
            error_text = str(payload.get("error") or "")
            if "index is loading" not in error_text.lower():
                return payload
            last_error = RuntimeError(error_text)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {425, 429, 500, 502, 503, 504}:
                raise
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
        if attempt + 1 == attempts:
            break
        time.sleep(delay)
        delay = min(delay * 1.5, 30.0)
    raise RuntimeError("Dataset Viewer did not become ready") from last_error


def _query_url(endpoint: str, **params: str | int) -> str:
    query = urllib.parse.urlencode(params)
    return f"https://datasets-server.huggingface.co/{endpoint}?{query}"


def validate_pinned_source(fetch_json: JsonFetcher = _http_json) -> dict[str, Any]:
    revision_url = f"https://huggingface.co/api/datasets/{HF_REPOSITORY}/revision/{HF_REVISION}"
    main_url = f"https://huggingface.co/api/datasets/{HF_REPOSITORY}"
    revision = fetch_json(revision_url)
    current = fetch_json(main_url)
    if revision.get("sha") != HF_REVISION:
        raise ValueError("Pyro-SDIS immutable revision did not resolve exactly")
    if current.get("sha") != HF_REVISION:
        raise ValueError(
            "Dataset Viewer only serves the current revision and Pyro-SDIS HEAD has drifted"
        )
    card_license = str((revision.get("cardData") or {}).get("license") or "").lower()
    if card_license not in {"apache-2.0", "apache 2.0"}:
        raise ValueError(f"Pyro-SDIS license drift: {card_license!r}")
    return {
        "repository": HF_REPOSITORY,
        "revision": HF_REVISION,
        "license": SOURCE_LICENSE,
        "last_modified": revision.get("lastModified"),
    }


def _asset_revision(url: str) -> str | None:
    parts = urllib.parse.unquote(urllib.parse.urlparse(url).path).split("/")
    try:
        marker = parts.index("--")
    except ValueError:
        return None
    revision_index = marker + 1
    if revision_index >= len(parts):
        return None
    return parts[revision_index]


def validate_viewer_row(entry: dict[str, Any], source_split: str) -> dict[str, Any]:
    if source_split not in HF_SPLITS:
        raise ValueError(f"unexpected Pyro-SDIS source split: {source_split}")
    if entry.get("truncated_cells"):
        raise ValueError("Dataset Viewer returned a truncated Pyro-SDIS row")
    row_index = entry.get("row_idx")
    row = entry.get("row")
    if not isinstance(row_index, int) or row_index < 0 or not isinstance(row, dict):
        raise ValueError("invalid Dataset Viewer row envelope")
    if row.get("annotations") != "":
        raise ValueError("Pyro-SDIS filter returned a non-empty annotation")

    image = row.get("image")
    if not isinstance(image, dict):
        raise ValueError("Pyro-SDIS row has no Dataset Viewer image object")
    image_url = str(image.get("src") or "")
    parsed_url = urllib.parse.urlparse(image_url)
    if parsed_url.scheme != "https" or _asset_revision(image_url) != HF_REVISION:
        raise ValueError("Pyro-SDIS cached image is not bound to the pinned revision")
    width = image.get("width")
    height = image.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Pyro-SDIS row has invalid image dimensions")

    image_name = str(row.get("image_name") or "")
    partner = str(row.get("partner") or "")
    camera = str(row.get("camera") or "")
    captured_at = str(row.get("date") or "")
    if not image_name.lower().endswith((".jpg", ".jpeg")) or Path(image_name).name != image_name:
        raise ValueError("Pyro-SDIS row has an unsafe or unexpected image name")
    if not partner or not camera:
        raise ValueError("Pyro-SDIS row has an incomplete camera identity")
    try:
        datetime.strptime(captured_at, "%Y-%m-%dT%H-%M-%S")
    except ValueError as error:
        raise ValueError("Pyro-SDIS row has an invalid capture timestamp") from error

    stable_source = {
        "source_split": source_split,
        "source_row_index": row_index,
        "image_name": image_name,
        "partner": partner,
        "camera": camera,
        "captured_at": captured_at,
        "annotations": "",
    }
    sample_id = f"pyro-sdis:{source_split}:{row_index:06d}"
    split_group = f"pyro-sdis-camera:{partner}:{camera}"
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_id": SOURCE_ID,
        "source_family": "Pyro-SDIS camera explicit negatives",
        "source_repository": HF_REPOSITORY,
        "source_revision": HF_REVISION,
        "source_split": source_split,
        "source_row_index": row_index,
        "source_row_sha256": _sha256_bytes(
            json.dumps(stable_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "source_image_name": image_name,
        "source_partner": partner,
        "source_camera": camera,
        "source_capture_timestamp": captured_at,
        "source_group": split_group,
        "split_group": split_group,
        "declared_width": width,
        "declared_height": height,
        "source_annotations": "",
        "source_annotations_exactly_empty": True,
        "annotation_strength": "negative",
        "annotation_provenance": "source_explicit_empty_yolo_annotation",
        "anchor_points": [],
        "visual_abstention_reason": None,
        "negative_evidence": "source_annotations_exactly_empty",
        "negative": True,
        "mask_quality": "source_explicit_empty_annotation_rasterized",
        "mask_semantics": "no_fire_or_smoke_annotation",
        "mask_nonzero_fraction": 0.0,
        "sample_weight": 1.0,
        "variant": "clean",
        "media_license": SOURCE_LICENSE,
        "mask_license": SOURCE_LICENSE,
        "license": SOURCE_LICENSE,
        "license_evidence_uri": (
            f"https://huggingface.co/datasets/{HF_REPOSITORY}/blob/{HF_REVISION}/README.md"
        ),
        "redistribution_allowed": True,
        "reviews_admitted": False,
        "validation_profile": "fireviewer_pointing_strict_automated_v1",
        "strict_keep": False,
        "training_eligible": False,
        "sample_validation_status": "pending_strict_automated_validation",
        "validation_passed": False,
        "exclusion_reasons": [],
        "_image_url": image_url,
    }


def fetch_exact_empty_rows(fetch_json: JsonFetcher = _http_json) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    source_rows_seen = 0
    for source_split in HF_SPLITS:
        offset = 0
        expected_split_total: int | None = None
        while expected_split_total is None or offset < expected_split_total:
            url = _query_url(
                "rows",
                dataset=HF_REPOSITORY,
                config=HF_CONFIG,
                split=source_split,
                offset=offset,
                length=ROWS_PAGE_SIZE,
            )
            payload = fetch_json(url)
            if payload.get("partial") is not False:
                raise ValueError("Dataset Viewer returned a partial Pyro-SDIS row index")
            total = payload.get("num_rows_total")
            rows = payload.get("rows")
            if not isinstance(total, int) or total < 0 or not isinstance(rows, list):
                raise ValueError("Dataset Viewer returned an invalid rows response")
            if expected_split_total is None:
                expected_split_total = total
            elif total != expected_split_total:
                raise ValueError("Dataset Viewer Pyro-SDIS row count changed while paging")
            if not rows and offset < total:
                raise ValueError("Dataset Viewer Pyro-SDIS pagination stopped early")
            for entry in rows:
                row = entry.get("row") if isinstance(entry, dict) else None
                if not isinstance(row, dict) or not isinstance(row.get("annotations"), str):
                    raise ValueError("Dataset Viewer returned a malformed annotation cell")
                if row["annotations"] == "":
                    candidates.append(validate_viewer_row(entry, source_split))
            source_rows_seen += len(rows)
            offset += len(rows)

    identities = [(row["source_split"], row["source_row_index"]) for row in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("Dataset Viewer repeated a filtered Pyro-SDIS row")
    if source_rows_seen != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Pyro-SDIS source row-count drift: {source_rows_seen} != {EXPECTED_SOURCE_ROWS}"
        )
    if len(candidates) != EXPECTED_EMPTY_ROWS:
        raise ValueError(
            f"Pyro-SDIS exact-empty count drift: {len(candidates)} != {EXPECTED_EMPTY_ROWS}"
        )
    return candidates


def assign_group_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    group_counts = Counter(str(row["split_group"]) for row in rows)
    if len(group_counts) < 3:
        raise ValueError("Pyro-SDIS needs at least three independent camera groups")
    total = sum(group_counts.values())
    targets = {name: total * ratio for name, ratio in FINAL_SPLIT_RATIOS.items()}
    assigned_counts = dict.fromkeys(FINAL_SPLIT_RATIOS, 0)
    result: dict[str, str] = {}
    groups = sorted(
        group_counts,
        key=lambda group: (-group_counts[group], _stable_digest(group), group),
    )
    for index, group in enumerate(groups):
        group_count = group_counts[group]
        empty_splits = [name for name, count in assigned_counts.items() if count == 0]
        remaining_groups = len(groups) - index
        allowed = empty_splits if remaining_groups == len(empty_splits) else list(targets)

        scores: dict[str, tuple[float, float, str]] = {}
        for split in allowed:
            projected = dict(assigned_counts)
            projected[split] += group_count
            normalized_error = sum(
                ((projected[name] - targets[name]) / max(1.0, targets[name])) ** 2
                for name in targets
            )
            deficit = targets[split] - assigned_counts[split]
            scores[split] = (normalized_error, -deficit, split)
        destination = min(allowed, key=scores.__getitem__)
        result[group] = destination
        assigned_counts[destination] += group_count
    if set(result.values()) != set(FINAL_SPLIT_RATIOS):
        raise ValueError("Pyro-SDIS camera split assignment is incomplete")
    return result


def _inspect_payload(row: dict[str, Any], payload: bytes, staging_root: Path) -> dict[str, Any]:
    candidate = dict(row)
    reasons = list(candidate["exclusion_reasons"])
    try:
        from io import BytesIO

        with Image.open(BytesIO(payload)) as opened:
            opened.load()
            source_format = str(opened.format or "").upper()
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as error:
        reasons.append(f"image_decode_error:{type(error).__name__}")
        candidate["exclusion_reasons"] = reasons
        return candidate

    if source_format != "JPEG":
        reasons.append("unexpected_source_image_format")
    declared_size = (int(candidate["declared_width"]), int(candidate["declared_height"]))
    if normalized.size != declared_size:
        reasons.append("decoded_dimension_mismatch")
    pixel_header = f"RGB:{normalized.width}x{normalized.height}:".encode()
    source_sha = _sha256_bytes(payload)
    staged_name = f"{candidate['source_split']}-{candidate['source_row_index']:06d}.jpg"
    staged_path = staging_root / staged_name
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_bytes(payload)
    candidate.update(
        {
            "source_image_sha256": source_sha,
            "image_sha256": source_sha,
            "decoded_pixel_sha256": _sha256_bytes(
                pixel_header + normalized.tobytes()
            ),
            "width": normalized.width,
            "height": normalized.height,
            "dhash": _difference_hash(normalized),
            "source_image_format": source_format,
            "image_decode_valid": not any(
                reason in {"unexpected_source_image_format", "decoded_dimension_mismatch"}
                for reason in reasons
            ),
            "_staged_path": str(staged_path),
            "exclusion_reasons": reasons,
            **_metrics(normalized),
        }
    )
    return candidate


def _download_and_inspect(
    row: dict[str, Any], staging_root: Path, fetch_bytes: ByteFetcher
) -> dict[str, Any]:
    try:
        payload = fetch_bytes(str(row["_image_url"]))
    except Exception as error:
        candidate = dict(row)
        candidate["exclusion_reasons"] = [
            *candidate["exclusion_reasons"],
            f"image_download_error:{type(error).__name__}",
        ]
        return candidate
    try:
        return _inspect_payload(row, payload, staging_root)
    except Exception as error:
        candidate = dict(row)
        candidate["exclusion_reasons"] = [
            *candidate["exclusion_reasons"],
            f"image_processing_error:{type(error).__name__}",
        ]
        return candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _load_baseline(root: Path) -> tuple[dict[str, str], list[tuple[str, str]], list[str]]:
    combined = sorted(root.rglob("strict_combined_manifest.jsonl"))
    manifests = combined or sorted(root.rglob("*_strict_validated_manifest.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"no strict pointing baseline manifest below {root}")
    sha_rows: dict[str, str] = {}
    dhash_rows: list[tuple[str, str]] = []
    for manifest in manifests:
        for row in _read_jsonl(manifest):
            sample_id = str(row.get("sample_id") or "")
            digest = str(row.get("image_sha256") or row.get("source_image_sha256") or "")
            signature = str(row.get("dhash") or "")
            if len(digest) == 64:
                sha_rows[digest] = sample_id
            if len(signature) == 16:
                dhash_rows.append((signature, sample_id))
    if not sha_rows or not dhash_rows:
        raise ValueError("strict pointing baseline has no usable image hashes")
    return sha_rows, dhash_rows, [path.name for path in manifests]


def _nearest(signature: str, rows: list[tuple[str, str]]) -> tuple[int | None, str | None]:
    if not rows:
        return None, None
    distance, sample_id = min((_hamming(signature, value), key) for value, key in rows)
    return distance, sample_id


def apply_strict_deduplication(
    rows: list[dict[str, Any]],
    *,
    baseline_sha: dict[str, str],
    baseline_dhash: list[tuple[str, str]],
) -> dict[str, int]:
    accepted_sha: dict[str, str] = {}
    accepted_dhash: list[tuple[str, str]] = []
    ordered = sorted(
        rows,
        key=lambda item: (_stable_digest(str(item["sample_id"])), item["sample_id"]),
    )
    for row in ordered:
        reasons = list(row["exclusion_reasons"])
        digest = str(row.get("image_sha256") or "")
        signature = str(row.get("dhash") or "")
        if reasons or len(digest) != 64 or len(signature) != 16:
            row["exclusion_reasons"] = sorted(set(reasons or ["missing_validated_image_hashes"]))
            continue
        exact_baseline = baseline_sha.get(digest)
        baseline_distance, baseline_sample = _nearest(signature, baseline_dhash)
        row["baseline_exact_sha_match"] = exact_baseline
        row["baseline_nearest_dhash_distance"] = baseline_distance
        row["baseline_nearest_sample_id"] = baseline_sample
        if exact_baseline is not None:
            reasons.append("baseline_exact_sha_overlap")
        if baseline_distance is not None and baseline_distance <= NEAR_DUPLICATE_DISTANCE:
            reasons.append("baseline_perceptual_overlap")

        exact_internal = accepted_sha.get(digest)
        internal_distance, internal_sample = _nearest(signature, accepted_dhash)
        row["within_source_exact_sha_match"] = exact_internal
        row["within_source_nearest_dhash_distance"] = internal_distance
        row["within_source_nearest_sample_id"] = internal_sample
        if exact_internal is not None:
            reasons.append("within_source_exact_sha_duplicate")
        if internal_distance is not None and internal_distance <= NEAR_DUPLICATE_DISTANCE:
            reasons.append("within_source_perceptual_duplicate")
        row["exclusion_reasons"] = sorted(set(reasons))
        if not reasons:
            accepted_sha[digest] = str(row["sample_id"])
            accepted_dhash.append((signature, str(row["sample_id"])))
    return {
        "baseline_exact_overlaps": sum(
            "baseline_exact_sha_overlap" in row["exclusion_reasons"] for row in rows
        ),
        "baseline_perceptual_overlaps": sum(
            "baseline_perceptual_overlap" in row["exclusion_reasons"] for row in rows
        ),
        "within_source_exact_duplicates": sum(
            "within_source_exact_sha_duplicate" in row["exclusion_reasons"] for row in rows
        ),
        "within_source_perceptual_duplicates": sum(
            "within_source_perceptual_duplicate" in row["exclusion_reasons"] for row in rows
        ),
    }


def _balanced_take(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    queues: dict[str, deque[dict[str, Any]]] = {}
    groups = sorted(
        {str(row["split_group"]) for row in rows},
        key=lambda value: (_stable_digest(value), value),
    )
    for group in groups:
        group_rows = [row for row in rows if row["split_group"] == group]
        queues[group] = deque(
            sorted(
                group_rows,
                key=lambda row: (_stable_digest(str(row["sample_id"])), row["sample_id"]),
            )
        )
    selected: list[dict[str, Any]] = []
    while queues and len(selected) < count:
        for group in list(queues):
            selected.append(queues[group].popleft())
            if not queues[group]:
                del queues[group]
            if len(selected) == count:
                break
    return selected


def select_balanced_strict_rows(
    rows: list[dict[str, Any]], *, max_validated: int
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if not row["exclusion_reasons"]]
    if max_validated <= 0 or len(eligible) <= max_validated:
        return eligible
    requested = {
        "train": round(max_validated * FINAL_SPLIT_RATIOS["train"]),
        "validation": round(max_validated * FINAL_SPLIT_RATIOS["validation"]),
    }
    requested["test"] = max_validated - requested["train"] - requested["validation"]
    selected: list[dict[str, Any]] = []
    for split in FINAL_SPLIT_RATIOS:
        split_rows = [row for row in eligible if row["split"] == split]
        selected.extend(_balanced_take(split_rows, min(requested[split], len(split_rows))))
    if len(selected) < max_validated:
        selected_ids = {str(row["sample_id"]) for row in selected}
        remainder = [row for row in eligible if str(row["sample_id"]) not in selected_ids]
        selected.extend(_balanced_take(remainder, max_validated - len(selected)))
    selected_ids = {str(row["sample_id"]) for row in selected}
    for row in eligible:
        if str(row["sample_id"]) not in selected_ids:
            row["exclusion_reasons"].append("strict_source_balance_quota")
    return selected


def _serializable_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_serializable_row(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def _assert_isolated_uri(value: str) -> None:
    lowered = value.lower()
    forbidden = ("fire-smoke-detection-corpus-v1", "benchdata", "benchmark", "fireviewer_bench")
    if any(marker in lowered for marker in forbidden) or "pointing" not in lowered:
        raise ValueError("Pyro-SDIS negatives escaped the isolated pointing campaign")


def audit_pyro_negatives(
    *,
    baseline_root: Path,
    output_dir: Path,
    output_s3_uri: str,
    work_dir: Path,
    workers: int,
    max_validated: int,
    fetch_json: JsonFetcher = _http_json,
    fetch_bytes: ByteFetcher = _http_bytes,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_validated < 0:
        raise ValueError("max_validated cannot be negative")
    _assert_isolated_uri(output_s3_uri)
    source_receipt = validate_pinned_source(fetch_json)
    rows = fetch_exact_empty_rows(fetch_json)
    split_by_group = assign_group_splits(rows)
    for row in rows:
        row["split"] = split_by_group[str(row["split_group"])]
        row["final_split"] = row["split"]

    staging_root = work_dir / "pyro-sdis-explicit-negatives"
    staging_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected: list[dict[str, Any]] = []
        for index, row in enumerate(
            executor.map(
                lambda row: _download_and_inspect(row, staging_root, fetch_bytes),
                rows,
            ),
            start=1,
        ):
            inspected.append(row)
            if index % 500 == 0 or index == len(rows):
                print(json.dumps({"stage": "image_validation", "completed": index}))
        rows = inspected

    payload_failures = [
        reason
        for row in rows
        for reason in row["exclusion_reasons"]
        if reason.startswith(
            ("image_download_error:", "image_decode_error:", "image_processing_error:")
        )
    ]
    if payload_failures:
        raise RuntimeError(
            f"Pyro-SDIS payload validation incomplete: {len(payload_failures)} failures"
        )

    baseline_sha, baseline_dhash, baseline_manifests = _load_baseline(baseline_root)
    duplicate_report = apply_strict_deduplication(
        rows,
        baseline_sha=baseline_sha,
        baseline_dhash=baseline_dhash,
    )
    selected = select_balanced_strict_rows(rows, max_validated=max_validated)
    if max_validated > 0 and len(selected) != max_validated:
        raise RuntimeError(
            f"Pyro-SDIS strict selection incomplete: {len(selected)} != {max_validated}"
        )
    if max_validated > 0:
        expected_splits = {
            "train": round(max_validated * FINAL_SPLIT_RATIOS["train"]),
            "validation": round(max_validated * FINAL_SPLIT_RATIOS["validation"]),
        }
        expected_splits["test"] = (
            max_validated - expected_splits["train"] - expected_splits["validation"]
        )
        actual_splits = Counter(str(row["split"]) for row in selected)
        if dict(actual_splits) != expected_splits:
            raise RuntimeError(
                f"Pyro-SDIS strict split quota drift: {dict(actual_splits)} != {expected_splits}"
            )
    selected_ids = {str(row["sample_id"]) for row in selected}
    for row in rows:
        reasons = sorted(set(row["exclusion_reasons"]))
        row["exclusion_reasons"] = reasons
        row["validation_passed"] = not any(
            reason != "strict_source_balance_quota" for reason in reasons
        )
        is_selected = str(row["sample_id"]) in selected_ids and not reasons
        row["strict_keep"] = is_selected
        row["training_eligible"] = is_selected
        if is_selected:
            row["sample_validation_status"] = "strict_automated_validated"
            row["corpus_disposition"] = "eligible_genuinely_new_negative"
            image_name = f"{row['source_split']}-{row['source_row_index']:06d}.jpg"
            relative = f"images/{row['split']}/{image_name}"
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(row["_staged_path"]), destination)
            row["image_relpath"] = relative
            row["source_image_relpath"] = relative
            row["image_s3_uri"] = f"{output_s3_uri.rstrip('/')}/{relative}"
            row["source_image_s3_uri"] = row["image_s3_uri"]
            mask_name = f"{row['source_split']}-{row['source_row_index']:06d}.png"
            mask_relative = f"masks/{row['split']}/{mask_name}"
            mask_destination = output_dir / mask_relative
            mask_destination.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (int(row["width"]), int(row["height"])), 0).save(
                mask_destination,
                format="PNG",
                optimize=True,
            )
            row["mask_relpath"] = mask_relative
            row["mask_s3_uri"] = f"{output_s3_uri.rstrip('/')}/{mask_relative}"
            row["mask_sha256"] = _sha256_file(mask_destination)
        elif reasons == ["strict_source_balance_quota"]:
            row["sample_validation_status"] = "excluded_strict_source_balance_quota"
            row["corpus_disposition"] = "validated_negative_not_selected"
        else:
            row["sample_validation_status"] = "excluded_unvalidated"
            row["corpus_disposition"] = "excluded_unvalidated"

    selected = sorted(
        (row for row in rows if row["strict_keep"]), key=lambda row: str(row["sample_id"])
    )
    rows.sort(key=lambda row: str(row["sample_id"]))
    _write_jsonl(output_dir / "pyro_sdis_negative_automatic_dispositions.jsonl", rows)
    _write_jsonl(output_dir / "pyro_sdis_negative_strict_validated_manifest.jsonl", selected)

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        group_splits[str(row["split_group"])].add(str(row["split"]))
    leaking_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if leaking_groups:
        raise ValueError(f"Pyro-SDIS strict camera-group leakage: {leaking_groups}")
    selected_hashes = [str(row["image_sha256"]) for row in selected]
    selected_dhashes = [str(row["dhash"]) for row in selected]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise ValueError("Pyro-SDIS strict manifest still contains exact duplicates")
    for left_index, left in enumerate(selected_dhashes):
        later = selected_dhashes[left_index + 1 :]
        if any(_hamming(left, right) <= NEAR_DUPLICATE_DISTANCE for right in later):
            raise ValueError("Pyro-SDIS strict manifest still contains perceptual duplicates")

    gate_errors: list[str] = []
    if payload_failures:
        gate_errors.append(f"decode_or_payload_errors:{len(payload_failures)}")
    if len(selected) != max_validated:
        gate_errors.append(f"strict_selection_rows:{len(selected)}!={max_validated}")

    report = {
        "schema_version": 1,
        "source_id": SOURCE_ID,
        "source_repository": HF_REPOSITORY,
        "source_revision": HF_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_receipt": source_receipt,
        "dataset_viewer_endpoint": "rows",
        "source_empty_predicate": SOURCE_EMPTY_PREDICATE,
        "dataset_viewer_full_enumeration_complete": True,
        "expected_source_rows": EXPECTED_SOURCE_ROWS,
        "expected_exact_empty_rows": EXPECTED_EMPTY_ROWS,
        "rows_evaluated": len(rows),
        "source_split_counts": dict(
            sorted(Counter(str(row["source_split"]) for row in rows).items())
        ),
        "camera_groups": len(split_by_group),
        "camera_group_split_counts": dict(
            sorted(Counter(split_by_group.values()).items())
        ),
        "decoded_images": sum(bool(row.get("image_decode_valid")) for row in rows),
        "strict_pool_before_balance_quota": sum(
            not any(reason != "strict_source_balance_quota" for reason in row["exclusion_reasons"])
            for row in rows
        ),
        "max_validated": max_validated,
        "strict_automated_validated_rows": len(selected),
        "explicit_negative_rows": len(selected),
        "validated_split_counts": dict(
            sorted(Counter(str(row["split"]) for row in selected).items())
        ),
        "validated_camera_groups": len({str(row["split_group"]) for row in selected}),
        "split_group_leakage": leaking_groups,
        "exclusions_by_reason": dict(
            sorted(Counter(reason for row in rows for reason in row["exclusion_reasons"]).items())
        ),
        "duplicate_report": duplicate_report,
        "baseline_manifests": baseline_manifests,
        "baseline_images_compared": len(baseline_sha),
        "gate_errors": gate_errors,
        "source_gate_passed": not gate_errors,
        "reviews_admitted": False,
        "detection_corpus_used": False,
        "independent_benchmark_used": False,
        "developer_workstation_corpus_materialized": False,
        "sagemaker_ephemeral_candidates_used": True,
        "publication_allowed": False,
        "next_action": "merge_strict_sources_then_repeat_global_quality_gates",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pyro_sdis_negative_audit_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-validated", type=int, default=DEFAULT_MAX_VALIDATED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_pyro_negatives(
        baseline_root=args.baseline_root,
        output_dir=args.output_dir,
        output_s3_uri=args.output_s3_uri,
        work_dir=args.work_dir,
        workers=args.workers,
        max_validated=args.max_validated,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
