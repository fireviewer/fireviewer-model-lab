"""Build hash-only exclusion indexes for the isolated IJmond pointing audit.

The builder verifies the pinned composition chain before emitting anything.
Detection manifests are fetched from the exact Hugging Face revision.  Mounted
pointing images are rehashed and decoded only in memory.  Benchmark media is
never mounted: only the immutable external guard is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tempfile
import threading
import time
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit

import boto3
import numpy as np
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import scipy.fftpack
from PIL import Image

BENCHMARK_GUARD_FILENAME = "external_guard_samples.jsonl"
BENCHMARK_GUARD_SHA256 = "685c8a345d8df8a9534676b492496c79a8d96483b06b233e9517fa299cdbbccc"
BENCHMARK_CORPUS_ID = "dfire-pointing-external-v1"
BENCHMARK_ROWS = 200
PUBLICATION_MANIFEST_FILENAME = "publication-manifest.json"
PUBLICATION_MANIFEST_SHA256 = "155fd807ed14934fa5d22a7c5c081cde912f4bd79aa0764af784753916cfc155"
PUBLICATION_STAGING_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
PUBLICATION_ROWS = 102_257
PUBLICATION_SHARDS = 32
PUBLICATION_FORMAT = "parquet_with_embedded_images"
PUBLICATION_RELEASE_KIND = "strict_clean_replacement"
S3_REGION = "eu-west-2"
EXPECTED_POINTING_NAMES = frozenset({"boreal", "camp-swift", "kit"})
EXPECTED_DETECTION_MANIFESTS = 4
VALID_SPLITS = frozenset({"train", "validation", "test"})
IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff"})
MAX_MANIFEST_BYTES = 256 * 1024**2
MAX_IMAGE_BYTES = 64 * 1024**2
HTTP_MAX_ATTEMPTS = 6
HTTP_RETRY_DELAY_MAX_SECONDS = 60.0
HTTP_REQUEST_INTERVAL_SECONDS = 0.05
PARQUET_PROJECTED_COLUMNS = ("sample_id", "sha256", "phash")
UTC_COMPAT = timezone.utc  # noqa: UP017 - SageMaker image currently uses Python 3.10

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_PHASH = re.compile(r"^[0-9a-f]{16}$")
_HTTP_REQUEST_LOCK = threading.Lock()
_HTTP_LAST_REQUEST_STARTED = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path.name}")
    return value


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path.name}:{line_number}")
            yield line_number, value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _valid_sha(value: Any) -> bool:
    return bool(_SHA256.fullmatch(str(value or "").casefold()))


def _valid_phash(value: Any) -> bool:
    return bool(_PHASH.fullmatch(str(value or "").casefold()))


def _valid_revision(value: Any) -> bool:
    return bool(_IMMUTABLE_REVISION.fullmatch(str(value or "").casefold()))


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return path.as_posix()


def _safe_local(root: Path, relative: str) -> Path:
    posix = PurePosixPath(_safe_relative(relative))
    resolved_root = root.resolve()
    path = (resolved_root / Path(*posix.parts)).resolve()
    if path == resolved_root or resolved_root not in path.parents:
        raise ValueError(f"path escapes mounted source root: {relative}")
    return path


def _find_unique(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename)) if root.is_dir() else []
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} below {root}, found {len(matches)}"
        )
    return matches[0]


def _hf_url(repository: str, revision: str, relative: str) -> str:
    path = _safe_relative(relative)
    if not repository or "/" not in repository or not _valid_revision(revision):
        raise ValueError("invalid pinned Hugging Face repository or revision")
    url = (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/"
        f"{quote(path, safe='/')}"
    )
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
        raise ValueError("Hugging Face URL escaped its pinned origin")
    return url


def _retry_delay(headers: Any, attempt: int) -> float:
    fallback = min(float(2**attempt), HTTP_RETRY_DELAY_MAX_SECONDS)
    value = str(headers.get("Retry-After") or "").strip() if headers is not None else ""
    if not value:
        return fallback
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC_COMPAT)
            seconds = (retry_at - datetime.now(UTC_COMPAT)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return fallback
    return min(max(seconds, fallback, 0.0), HTTP_RETRY_DELAY_MAX_SECONDS)


def _pace_http_request() -> None:
    global _HTTP_LAST_REQUEST_STARTED
    with _HTTP_REQUEST_LOCK:
        elapsed = time.monotonic() - _HTTP_LAST_REQUEST_STARTED
        if elapsed < HTTP_REQUEST_INTERVAL_SECONDS:
            time.sleep(HTTP_REQUEST_INTERVAL_SECONDS - elapsed)
        _HTTP_LAST_REQUEST_STARTED = time.monotonic()


def _http_open_with_retry(url: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(HTTP_MAX_ATTEMPTS):
        request = urllib.request.Request(  # noqa: S310 - pinned HTTPS origin is validated
            url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "FireViewer-Pointing-Exclusion-Index/1.0",
            },
        )
        _pace_http_request()
        try:
            return urllib.request.urlopen(request, timeout=180)  # noqa: S310
        except HTTPError as exc:
            last_error = exc
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            if not retryable or attempt + 1 == HTTP_MAX_ATTEMPTS:
                raise
            delay = _retry_delay(exc.headers, attempt)
            exc.close()
        except URLError as exc:
            last_error = exc
            if attempt + 1 == HTTP_MAX_ATTEMPTS:
                raise
            delay = min(float(2**attempt), HTTP_RETRY_DELAY_MAX_SECONDS)
        time.sleep(delay)
    raise RuntimeError("bounded HTTP retry loop exhausted") from last_error


def _http_download_to_path(url: str, destination: Path, maximum_bytes: int) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with (
            _http_open_with_retry(url) as response,
            partial.open("wb") as output,
        ):
            if urlsplit(response.geturl()).scheme != "https":
                raise ValueError("download redirected away from HTTPS")
            while chunk := response.read(4 * 1024 * 1024):
                downloaded += len(chunk)
                if downloaded > maximum_bytes:
                    raise ValueError(f"download exceeds bounded size: {url}")
                digest.update(chunk)
                output.write(chunk)
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return {"bytes": downloaded, "sha256": digest.hexdigest(), "url": url}


def _phash64(image: Image.Image) -> str:
    """Return the 64-bit pHash encoded exactly like ImageHash 4.3.2."""
    resized = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized)
    coefficients = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    low_frequency = coefficients[:8, :8]
    median = np.median(low_frequency)
    bits = low_frequency > median
    bit_string = "".join("1" if value else "0" for value in bits.flatten())
    return f"{int(bit_string, 2):016x}"


def _image_phash(payload: bytes, *, expected_sha256: str) -> str:
    observed = _sha256_bytes(payload)
    if observed != expected_sha256:
        raise ValueError(f"image payload SHA-256 mismatch: {observed} != {expected_sha256}")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            value = _phash64(opened.convert("RGB")).casefold()
    except Exception as exc:
        raise ValueError(f"image payload decode failed: {type(exc).__name__}:{exc}") from exc
    if not _valid_phash(value):
        raise ValueError("computed image pHash is not 16 hexadecimal characters")
    return value


def _manifest_phash(row: dict[str, Any]) -> str | None:
    for field in ("phash", "phash64", "perceptual_hash"):
        value = str(row.get(field) or "").casefold()
        if _valid_phash(value):
            return value
    return None


def _validate_publication_manifest(
    *,
    publication_manifest_path: Path,
    registry: dict[str, Any],
    composition_contract: dict[str, Any],
    expected_sha256: str,
    expected_rows: int,
    expected_shards: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    if not _valid_sha(expected_sha256):
        raise ValueError("expected publication-manifest SHA-256 is invalid")
    if publication_manifest_path.name != PUBLICATION_MANIFEST_FILENAME:
        raise ValueError("publication metadata input has an unexpected filename")
    observed_sha = _sha256(publication_manifest_path)
    if observed_sha != expected_sha256:
        raise ValueError("publication-manifest SHA-256 mismatch")
    publication = _load_json(publication_manifest_path)
    detection = registry["detection_base"]
    validation_run_id = str(detection.get("validation_run_id") or "")
    if (
        publication.get("schema_version") != 2
        or publication.get("format") != PUBLICATION_FORMAT
        or publication.get("release_kind") != PUBLICATION_RELEASE_KIND
        or publication.get("archives_included") is not False
        or publication.get("repo_id") != detection["repository"]
        or publication.get("validation_profile") != detection["validation_profile"]
        or publication.get("validation_run_id") != validation_run_id
        or not validation_run_id
        or not _valid_revision(detection.get("revision"))
        or int(detection.get("rows", -1)) != expected_rows
    ):
        raise ValueError("publication-manifest repository/revision/profile/run contract failed")

    metadata = publication.get("metadata")
    if not isinstance(metadata, list):
        raise ValueError("publication-manifest metadata receipts are missing")
    metadata_by_path: dict[str, dict[str, Any]] = {}
    for item in metadata:
        if not isinstance(item, dict):
            raise ValueError("publication-manifest metadata receipt is not an object")
        path = _safe_relative(str(item.get("path") or ""))
        if path in metadata_by_path:
            raise ValueError(f"duplicate publication metadata path: {path}")
        metadata_by_path[path] = item
    for name, relative in detection["manifest_paths"].items():
        receipt = composition_contract["detection_receipts"][name]
        item = metadata_by_path.get(str(relative))
        if (
            item is None
            or item.get("sha256") != receipt["sha256"]
            or int(item.get("bytes", -1)) != int(receipt["bytes"])
        ):
            raise ValueError(f"publication metadata manifest receipt drifted: {name}")

    shards = publication.get("shards")
    if not isinstance(shards, list) or len(shards) != expected_shards:
        raise ValueError(
            f"publication shard-count mismatch: "
            f"{len(shards) if isinstance(shards, list) else -1} != {expected_shards}"
        )
    repository_name = str(detection["repository"]).split("/", 1)[-1]
    expected_prefix = (
        f"{repository_name}/staging/strict-clean/runs/{validation_run_id}/"
    )
    seen_paths: set[str] = set()
    seen_keys: set[str] = set()
    seen_versions: set[str] = set()
    seen_hashes: set[str] = set()
    split_rows: dict[str, int] = {split: 0 for split in VALID_SPLITS}
    split_indices: dict[str, set[int]] = {split: set() for split in VALID_SPLITS}
    validated_shards: list[dict[str, Any]] = []
    for shard_number, item in enumerate(shards):
        if not isinstance(item, dict):
            raise ValueError(f"publication shard is not an object: {shard_number}")
        split = str(item.get("split") or "")
        try:
            index = int(item.get("index", -1))
            rows = int(item.get("rows", -1))
            size = int(item.get("parquet_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"publication shard counters are invalid: {shard_number}") from exc
        path = _safe_relative(str(item.get("path") or ""))
        key = _safe_relative(str(item.get("staged_s3_key") or ""))
        version_id = str(item.get("staged_s3_version_id") or "")
        digest = str(item.get("sha256") or "").casefold()
        expected_path = f"data/{split}/{split}-{index:05d}.parquet"
        if (
            split not in VALID_SPLITS
            or index < 0
            or rows <= 0
            or size <= 0
            or path != expected_path
            or key != f"{expected_prefix}{path}"
            or not version_id
            or not _valid_sha(digest)
            or path in seen_paths
            or key in seen_keys
            or version_id in seen_versions
            or digest in seen_hashes
        ):
            raise ValueError(f"publication shard pin is invalid or duplicate: {shard_number}")
        seen_paths.add(path)
        seen_keys.add(key)
        seen_versions.add(version_id)
        seen_hashes.add(digest)
        split_rows[split] += rows
        split_indices[split].add(index)
        validated_shards.append(
            {
                "path": path,
                "key": key,
                "version_id": version_id,
                "bytes": size,
                "rows": rows,
                "sha256": digest,
                "split": split,
                "index": index,
            }
        )
    if sum(split_rows.values()) != expected_rows:
        raise ValueError("publication shard row total drifted")

    split_summary = publication.get("splits")
    registry_splits = detection.get("split_counts")
    if not isinstance(split_summary, dict) or not isinstance(registry_splits, dict):
        raise ValueError("publication or registry split summary is missing")
    for split in sorted(VALID_SPLITS):
        expected_indices = set(range(len(split_indices[split])))
        summary = split_summary.get(split)
        if (
            split_indices[split] != expected_indices
            or not isinstance(summary, dict)
            or int(summary.get("rows", -1)) != split_rows[split]
            or int(summary.get("shards", -1)) != len(split_indices[split])
            or int(registry_splits.get(split, -1)) != split_rows[split]
        ):
            raise ValueError(f"publication split contract drifted: {split}")
    return publication, sorted(validated_shards, key=lambda item: item["path"]), observed_sha


def _head_current_shard(s3_client: Any, shard: dict[str, Any]) -> dict[str, Any]:
    response = s3_client.head_object(
        Bucket=PUBLICATION_STAGING_BUCKET,
        Key=shard["key"],
    )
    version_id = str(response.get("VersionId") or "")
    size = int(response.get("ContentLength", -1))
    if response.get("DeleteMarker") or (
        version_id != shard["version_id"] or size != shard["bytes"]
    ):
        raise ValueError(f"current S3 object does not match publication pin: {shard['path']}")
    return {"version_id": version_id, "bytes": size}


def _load_publication_metadata(
    *,
    publication_manifest_path: Path,
    registry: dict[str, Any],
    composition_contract: dict[str, Any],
    s3_client: Any,
    parquet_filesystem: Any,
    expected_sha256: str,
    expected_rows: int,
    expected_shards: int,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    _, shards, publication_sha = _validate_publication_manifest(
        publication_manifest_path=publication_manifest_path,
        registry=registry,
        composition_contract=composition_contract,
        expected_sha256=expected_sha256,
        expected_rows=expected_rows,
        expected_shards=expected_shards,
    )
    metadata_by_sample: dict[str, dict[str, str]] = {}
    seen_sha: set[str] = set()
    missing_phash_rows = 0
    recovered_phash_rows = 0
    fallback_image_row_groups = 0
    shard_receipts: list[dict[str, Any]] = []
    for shard in shards:
        before = _head_current_shard(s3_client, shard)
        source = parquet_filesystem.open_input_file(
            f"{PUBLICATION_STAGING_BUCKET}/{shard['key']}"
        )
        shard_rows = 0
        try:
            parquet = pq.ParquetFile(source)
            schema_names = set(parquet.schema_arrow.names)
            missing_columns = set(PARQUET_PROJECTED_COLUMNS) - schema_names
            if missing_columns:
                raise ValueError(
                    f"Parquet metadata columns missing for {shard['path']}: "
                    f"{sorted(missing_columns)}"
                )
            if int(parquet.metadata.num_rows) != shard["rows"]:
                raise ValueError(f"Parquet footer row count drifted: {shard['path']}")
            for row_group_index in range(parquet.metadata.num_row_groups):
                projected = parquet.read_row_group(
                    row_group_index,
                    columns=list(PARQUET_PROJECTED_COLUMNS),
                    use_threads=False,
                ).to_pydict()
                missing_indexes = [
                    index
                    for index, value in enumerate(projected["phash"])
                    if not _valid_phash(str(value or "").casefold())
                ]
                fallback_images: list[Any] | None = None
                if missing_indexes:
                    if "image" not in schema_names:
                        raise ValueError(
                            f"Parquet image fallback column missing for {shard['path']}"
                        )
                    fallback_images = parquet.read_row_group(
                        row_group_index,
                        columns=["image"],
                        use_threads=False,
                    ).to_pydict()["image"]
                    fallback_image_row_groups += 1
                for row_index, (sample_value, sha_value, phash_value) in enumerate(
                    zip(
                        projected["sample_id"],
                        projected["sha256"],
                        projected["phash"],
                        strict=True,
                    )
                ):
                    sample_id = str(sample_value or "")
                    digest = str(sha_value or "").casefold()
                    phash = str(phash_value or "").casefold()
                    if (
                        not sample_id
                        or sample_id in metadata_by_sample
                        or not _valid_sha(digest)
                        or digest in seen_sha
                    ):
                        raise ValueError(
                            f"invalid authoritative Parquet metadata row: "
                            f"{shard['path']}:{sample_id}"
                        )
                    if not _valid_phash(phash):
                        missing_phash_rows += 1
                        value = fallback_images[row_index] if fallback_images is not None else None
                        payload = (
                            value.get("bytes")
                            if isinstance(value, dict)
                            else value
                            if isinstance(value, bytes)
                            else None
                        )
                        if not isinstance(payload, bytes):
                            raise ValueError(
                                f"Parquet fallback image payload missing: "
                                f"{shard['path']}:{sample_id}"
                            )
                        phash = _image_phash(payload, expected_sha256=digest)
                        recovered_phash_rows += 1
                    metadata_by_sample[sample_id] = {
                        "sha256": digest,
                        "phash": phash,
                    }
                    seen_sha.add(digest)
                    shard_rows += 1
        finally:
            source.close()
        after = _head_current_shard(s3_client, shard)
        if before != after or shard_rows != shard["rows"]:
            raise ValueError(f"S3 version or projected row count changed: {shard['path']}")
        shard_receipts.append(
            {
                **shard,
                "head_before": before,
                "head_after": after,
                "projected_columns": list(PARQUET_PROJECTED_COLUMNS),
                "projected_rows": shard_rows,
                "image_column_read": False,
            }
        )
    if len(metadata_by_sample) != expected_rows or len(seen_sha) != expected_rows:
        raise ValueError("authoritative Parquet metadata total is incomplete")
    return metadata_by_sample, {
        "publication_manifest_filename": PUBLICATION_MANIFEST_FILENAME,
        "publication_manifest_sha256": publication_sha,
        "staging_bucket": PUBLICATION_STAGING_BUCKET,
        "s3_region": S3_REGION,
        "shards": len(shards),
        "rows": len(metadata_by_sample),
        "missing_phash_rows": missing_phash_rows,
        "recovered_phash_rows": recovered_phash_rows,
        "fallback_image_row_groups": fallback_image_row_groups,
        "projected_columns": list(PARQUET_PROJECTED_COLUMNS),
        "image_column_read": recovered_phash_rows > 0,
        "shard_receipts": shard_receipts,
    }


def _validate_composition(
    *,
    registry_path: Path,
    composition_root: Path,
    expected_report_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not _valid_sha(expected_report_sha256):
        raise ValueError("expected composition_report SHA-256 is invalid")
    registry = _load_json(registry_path)
    registry_sha = _sha256(registry_path)
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported composition registry schema")
    if set(registry.get("detection_base", {}).get("manifest_paths", {})) == set():
        raise ValueError("composition registry has no detection manifests")
    if len(registry["detection_base"]["manifest_paths"]) != EXPECTED_DETECTION_MANIFESTS:
        raise ValueError("composition registry must pin exactly four detection manifests")
    overlay_sources = registry.get("overlay_sources", [])
    overlay_names = {str(item.get("name") or "") for item in overlay_sources}
    if len(overlay_sources) != len(EXPECTED_POINTING_NAMES) or (
        overlay_names != EXPECTED_POINTING_NAMES
    ):
        raise ValueError("composition registry pointing source set drifted")

    report_path = _find_unique(composition_root, "composition_report.json")
    report_sha = _sha256(report_path)
    if report_sha != expected_report_sha256:
        raise ValueError("pinned composition_report SHA-256 mismatch")
    report = _load_json(report_path)
    if (
        report.get("schema_version") != 1
        or report.get("campaign_id") != registry.get("campaign_id")
        or report.get("composition_registry_sha256") != registry_sha
        or report.get("integrity_gates_passed") is not True
        or report.get("publication_allowed") is not False
        or report.get("reviews_admitted") is not False
    ):
        raise ValueError("composition_report contract failed")
    detection = registry["detection_base"]
    if (
        report.get("detection_repository") != detection["repository"]
        or report.get("detection_revision") != detection["revision"]
        or int(report.get("detection_rows", -1)) != int(detection["rows"])
    ):
        raise ValueError("composition_report detection contract drifted")

    integrity_name = str(report.get("composition_integrity_receipt") or "")
    if not integrity_name:
        raise ValueError("composition_report has no integrity receipt")
    integrity_path = _find_unique(composition_root, PurePosixPath(integrity_name).name)
    integrity_sha = _sha256(integrity_path)
    if integrity_sha != report.get("composition_integrity_receipt_sha256"):
        raise ValueError("composition integrity receipt hash mismatch")
    integrity = _load_json(integrity_path)
    if (
        integrity.get("schema_version") != 1
        or integrity.get("campaign_id") != registry.get("campaign_id")
        or integrity.get("composition_registry_sha256") != registry_sha
        or integrity.get("detection_revision") != detection["revision"]
        or integrity.get("integrity_gates_passed") is not True
        or integrity.get("publication_allowed") is not False
        or int(integrity.get("composition_rows", -1)) != int(report.get("composition_rows", -2))
    ):
        raise ValueError("composition integrity receipt contract failed")

    detection_receipts = report.get("detection_manifest_receipts")
    if not isinstance(detection_receipts, list) or (
        len(detection_receipts) != EXPECTED_DETECTION_MANIFESTS
    ):
        raise ValueError("composition detection receipts are missing")
    detection_by_name = {str(item.get("name") or ""): item for item in detection_receipts}
    expected_detection_names = set(detection["manifest_paths"])
    if set(detection_by_name) != expected_detection_names:
        raise ValueError("composition detection receipt set drifted")
    for name, relative in detection["manifest_paths"].items():
        receipt = detection_by_name[name]
        if (
            receipt.get("repository_path") != relative
            or receipt.get("revision") != detection["revision"]
            or not _valid_sha(receipt.get("sha256"))
            or int(receipt.get("bytes", 0)) <= 0
        ):
            raise ValueError(f"composition detection receipt is incomplete: {name}")
    receipt_detection_hashes = {
        name: str(receipt["sha256"]) for name, receipt in detection_by_name.items()
    }
    if integrity.get("detection_manifest_sha256") != receipt_detection_hashes:
        raise ValueError("integrity receipt detection hashes drifted")

    overlay_receipts = report.get("overlay_receipts")
    if not isinstance(overlay_receipts, list) or (
        len(overlay_receipts) != len(EXPECTED_POINTING_NAMES)
    ):
        raise ValueError("composition overlay receipts are missing")
    overlay_by_name = {str(item.get("name") or ""): item for item in overlay_receipts}
    if set(overlay_by_name) != EXPECTED_POINTING_NAMES:
        raise ValueError("composition overlay receipt set drifted")
    registry_overlay = {str(item["name"]): item for item in registry["overlay_sources"]}
    for name, contract in registry_overlay.items():
        receipt = overlay_by_name[name]
        if (
            receipt.get("manifest_sha256") != contract["manifest_sha256"]
            or receipt.get("report_sha256") != contract["report_sha256"]
            or not int(contract["strict_rows_min"])
            <= int(receipt.get("rows", -1))
            <= int(contract["strict_rows_max"])
            or receipt.get("source_gate_passed") is not True
        ):
            raise ValueError(f"composition overlay receipt contract failed: {name}")
    receipt_overlay_hashes = {
        name: str(receipt["manifest_sha256"]) for name, receipt in overlay_by_name.items()
    }
    if integrity.get("overlay_manifest_sha256") != receipt_overlay_hashes:
        raise ValueError("integrity receipt overlay hashes drifted")
    return registry, report, integrity, {
        "registry_sha256": registry_sha,
        "composition_report_sha256": report_sha,
        "composition_integrity_receipt_sha256": integrity_sha,
        "detection_receipts": detection_by_name,
        "overlay_receipts": overlay_by_name,
    }


def _download_detection_manifests(
    *,
    registry: dict[str, Any],
    composition_contract: dict[str, Any],
    work_dir: Path,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    detection = registry["detection_base"]
    repository = str(detection["repository"])
    revision = str(detection["revision"])
    paths: dict[str, Path] = {}
    observed_receipts: list[dict[str, Any]] = []
    for name, relative in sorted(detection["manifest_paths"].items()):
        expected = composition_contract["detection_receipts"][name]
        destination = work_dir / f"{name}.jsonl"
        observed = _http_download_to_path(
            _hf_url(repository, revision, str(relative)),
            destination,
            MAX_MANIFEST_BYTES,
        )
        if (
            observed["sha256"] != expected["sha256"]
            or observed["bytes"] != int(expected["bytes"])
        ):
            raise ValueError(f"downloaded detection manifest receipt mismatch: {name}")
        paths[name] = destination
        observed_receipts.append(
            {
                "name": name,
                "repository_path": relative,
                "revision": revision,
                "bytes": observed["bytes"],
                "sha256": observed["sha256"],
            }
        )
    return paths, observed_receipts


def _detection_index_rows(
    *,
    registry: dict[str, Any],
    manifests: dict[str, Path],
    publication_metadata: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detection = registry["detection_base"]
    validation_profile = str(detection["validation_profile"])
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_sha: set[str] = set()
    parquet_phash_rows = 0
    verified_manifest_phash_rows = 0
    per_manifest_rows: dict[str, int] = {}
    for name, path in sorted(manifests.items()):
        manifest_rows = 0
        relative_manifest = str(detection["manifest_paths"][name])
        for line_number, source in _iter_jsonl(path):
            sample_id = str(source.get("sample_id") or "")
            digest = str(source.get("sha256") or "").casefold()
            image_relative = str(source.get("image_relpath") or "")
            if (
                not sample_id
                or sample_id in seen_samples
                or not _valid_sha(digest)
                or digest in seen_sha
                or source.get("sample_validation_status") != "strict_automated_validated"
                or source.get("validation_profile") != validation_profile
                or source.get("split") not in VALID_SPLITS
                or not source.get("split_group")
                or not source.get("license")
                or not isinstance(source.get("consent_basis"), dict)
            ):
                raise ValueError(f"invalid strict detection row: {name}:{line_number}:{sample_id}")
            _safe_relative(image_relative)
            authoritative = publication_metadata.get(sample_id)
            if authoritative is None or authoritative["sha256"] != digest:
                raise ValueError(
                    f"detection manifest and Parquet metadata identity diverged: "
                    f"{name}:{line_number}:{sample_id}"
                )
            phash = authoritative["phash"]
            manifest_phash = _manifest_phash(source)
            if manifest_phash is None:
                phash_origin = "versioned_s3_parquet_metadata_projection"
                parquet_phash_rows += 1
            else:
                if manifest_phash != phash:
                    raise ValueError(
                        f"detection manifest and Parquet metadata pHash diverged: "
                        f"{name}:{line_number}:{sample_id}"
                    )
                phash_origin = "manifest_field_verified_against_versioned_parquet_metadata"
                verified_manifest_phash_rows += 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_sha256": digest,
                    "phash": phash,
                    "index_partition": "detection",
                    "source_manifest": relative_manifest,
                    "phash_origin": phash_origin,
                }
            )
            seen_samples.add(sample_id)
            seen_sha.add(digest)
            manifest_rows += 1
        if manifest_rows == 0:
            raise ValueError(f"detection manifest is empty: {name}")
        per_manifest_rows[name] = manifest_rows
    if len(rows) != int(detection["rows"]):
        raise ValueError(f"detection row-count mismatch: {len(rows)} != {detection['rows']}")
    if seen_samples != set(publication_metadata):
        raise ValueError("detection manifests do not cover the authoritative Parquet metadata")
    return rows, {
        "rows": len(rows),
        "unique_sample_ids": len(seen_samples),
        "unique_image_sha256": len(seen_sha),
        "phash_manifest_rows_verified_against_parquet": verified_manifest_phash_rows,
        "phash_parquet_projection_rows": parquet_phash_rows,
        "http_media_download_rows": 0,
        "publication_metadata_identity_rows_verified": len(rows),
        "manifest_row_counts": dict(sorted(per_manifest_rows.items())),
    }


def _pointing_index_rows(
    *,
    registry: dict[str, Any],
    composition_contract: dict[str, Any],
    pointing_roots: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if set(pointing_roots) != EXPECTED_POINTING_NAMES:
        raise ValueError("exactly Boreal, Camp Swift and KIT pointing roots are required")
    contracts = {str(item["name"]): item for item in registry["overlay_sources"]}
    rows: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_sha: set[str] = set()
    for name in sorted(EXPECTED_POINTING_NAMES):
        contract = contracts[name]
        root = pointing_roots[name]
        manifest = _find_unique(root, str(contract["manifest_filename"]))
        report_path = _find_unique(root, str(contract["report_filename"]))
        manifest_sha = _sha256(manifest)
        report_sha = _sha256(report_path)
        if (
            manifest_sha != contract["manifest_sha256"]
            or report_sha != contract["report_sha256"]
        ):
            raise ValueError(f"pointing source receipt hash mismatch: {name}")
        composition_receipt = composition_contract["overlay_receipts"][name]
        if (
            composition_receipt["manifest_sha256"] != manifest_sha
            or composition_receipt["report_sha256"] != report_sha
        ):
            raise ValueError(f"composition pointing receipt mismatch: {name}")
        report = _load_json(report_path)
        source_rows = list(_iter_jsonl(manifest))
        row_count = len(source_rows)
        if (
            not int(contract["strict_rows_min"])
            <= row_count
            <= int(contract["strict_rows_max"])
            or int(report.get("strict_automated_validated_rows", -1)) != row_count
            or report.get("source_gate_passed") is not True
            or report.get("gate_errors") != []
            or report.get("reviews_admitted") is not False
            or report.get("publication_allowed") is not False
            or report.get("decode_or_payload_errors", []) != []
            or report.get("split_group_leakage", []) != []
        ):
            raise ValueError(f"pointing source gate or row count failed: {name}")
        allowed_profiles = set(contract["allowed_validation_profiles"])
        source_seen = 0
        for line_number, source in source_rows:
            sample_id = str(source.get("sample_id") or "")
            digest = str(
                source.get("image_sha256") or source.get("source_image_sha256") or ""
            ).casefold()
            relative = str(source.get("image_relpath") or "")
            if (
                not sample_id
                or sample_id in seen_samples
                or not _valid_sha(digest)
                or digest in seen_sha
                or source.get("sample_validation_status") != "strict_automated_validated"
                or source.get("strict_keep") is not True
                or source.get("training_eligible") is not True
                or source.get("reviews_admitted") is not False
                or source.get("source_id") != contract["source_id"]
                or source.get("source_revision") != contract["source_revision"]
                or source.get("validation_profile") not in allowed_profiles
            ):
                raise ValueError(f"invalid strict pointing row: {name}:{line_number}:{sample_id}")
            image_path = _safe_local(root, relative)
            if not image_path.is_file():
                raise ValueError(f"mounted pointing image is missing: {name}:{sample_id}")
            if not 0 < image_path.stat().st_size <= MAX_IMAGE_BYTES:
                raise ValueError(f"mounted pointing image has invalid size: {name}:{sample_id}")
            payload = image_path.read_bytes()
            phash = _image_phash(payload, expected_sha256=digest)
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_sha256": digest,
                    "phash": phash,
                    "index_partition": "pointing",
                    "source_name": name,
                    "phash_origin": "mounted_payload_sha256_verified_in_memory",
                }
            )
            seen_samples.add(sample_id)
            seen_sha.add(digest)
            source_seen += 1
        if source_seen == 0:
            raise ValueError(f"pointing manifest is empty: {name}")
        source_receipts.append(
            {
                "name": name,
                "rows": source_seen,
                "manifest_sha256": manifest_sha,
                "manifest_bytes": manifest.stat().st_size,
                "report_sha256": report_sha,
                "report_bytes": report_path.stat().st_size,
                "source_gate_passed": True,
                "mounted_images_rehashed": source_seen,
            }
        )
    if not rows:
        raise ValueError("pointing exclusion index would be empty")
    return rows, source_receipts


def _benchmark_index_rows(
    *,
    benchmark_guard_root: Path,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    guard = _find_unique(benchmark_guard_root, BENCHMARK_GUARD_FILENAME)
    observed_sha = _sha256(guard)
    if observed_sha != expected_sha256:
        raise ValueError("benchmark guard SHA-256 mismatch")
    source_rows = list(_iter_jsonl(guard))
    if len(source_rows) != expected_rows:
        raise ValueError(
            f"benchmark guard row-count mismatch: {len(source_rows)} != {expected_rows}"
        )
    rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_sha: set[str] = set()
    seen_signatures: set[tuple[str, str]] = set()
    for line_number, source in source_rows:
        sample_id = str(source.get("sample_id") or "")
        digest = str(source.get("sha256") or "").casefold()
        phash = str(source.get("phash64") or "").casefold()
        flipped = str(source.get("phash64_flipped") or "").casefold()
        signature_pair = (phash, flipped)
        if (
            source.get("corpus_id") != BENCHMARK_CORPUS_ID
            or not sample_id
            or sample_id in seen_samples
            or not _valid_sha(digest)
            or digest in seen_sha
            or not _valid_phash(phash)
            or not _valid_phash(flipped)
        ):
            raise ValueError(f"invalid or duplicate benchmark guard row: {line_number}:{sample_id}")
        rows.append(
            {
                "sample_id": sample_id,
                "image_sha256": digest,
                "phash": phash,
                "phash64_flipped": flipped,
                "index_partition": "benchmark",
                "corpus_id": BENCHMARK_CORPUS_ID,
                "phash_origin": "immutable_external_guard",
            }
        )
        seen_samples.add(sample_id)
        seen_sha.add(digest)
        seen_signatures.add(signature_pair)
    if not rows:
        raise ValueError("benchmark exclusion index would be empty")
    return rows, {
        "guard_filename": BENCHMARK_GUARD_FILENAME,
        "guard_sha256": observed_sha,
        "guard_bytes": guard.stat().st_size,
        "rows": len(rows),
        "unique_sample_ids": len(seen_samples),
        "unique_image_sha256": len(seen_sha),
        "unique_phash_pairs": len(seen_signatures),
        "corpus_id": BENCHMARK_CORPUS_ID,
    }


def _validate_index_rows(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{name} exclusion index is empty")
    seen_samples: set[str] = set()
    seen_sha: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        digest = str(row.get("image_sha256") or "")
        phash = str(row.get("phash") or "")
        if (
            not sample_id
            or sample_id in seen_samples
            or not _valid_sha(digest)
            or digest in seen_sha
            or not _valid_phash(phash)
        ):
            raise ValueError(f"{name} exclusion index contains an incomplete or duplicate row")
        if name == "benchmark" and not _valid_phash(row.get("phash64_flipped")):
            raise ValueError("benchmark exclusion index lacks a valid flipped pHash")
        seen_samples.add(sample_id)
        seen_sha.add(digest)


def _write_partition(
    staging_output: Path,
    *,
    name: str,
    rows: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    _validate_index_rows(name, rows)
    partition = staging_output / name
    index_path = partition / "hash-index.jsonl"
    _write_jsonl(index_path, rows)
    media_files = [
        path
        for path in partition.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    ]
    if media_files:
        raise ValueError(f"{name} output unexpectedly contains media")
    complete_receipt = {
        "schema_version": 1,
        "partition": name,
        **receipt,
        "index_filename": index_path.name,
        "index_rows": len(rows),
        "index_bytes": index_path.stat().st_size,
        "index_sha256": _sha256(index_path),
        "incomplete_rows": 0,
        "media_files_output": 0,
        "source_gate_passed": True,
        "gate_errors": [],
        "reviews_admitted": False,
        "publication_allowed": False,
        "submitted_job": False,
    }
    receipt_path = partition / "receipt.json"
    _write_json(receipt_path, complete_receipt)
    return {
        "rows": len(rows),
        "index_sha256": complete_receipt["index_sha256"],
        "index_bytes": complete_receipt["index_bytes"],
        "receipt_sha256": _sha256(receipt_path),
        "receipt_bytes": receipt_path.stat().st_size,
    }


def _build_exclusion_indexes(
    *,
    registry_path: Path,
    composition_root: Path,
    expected_composition_report_sha256: str,
    publication_manifest_path: Path,
    pointing_roots: dict[str, Path],
    benchmark_guard_root: Path,
    work_dir: Path,
    output_dir: Path,
    s3_client: Any | None = None,
    parquet_filesystem: Any | None = None,
    expected_publication_manifest_sha256: str = PUBLICATION_MANIFEST_SHA256,
    expected_publication_rows: int = PUBLICATION_ROWS,
    expected_publication_shards: int = PUBLICATION_SHARDS,
    expected_benchmark_rows: int = BENCHMARK_ROWS,
    expected_benchmark_sha256: str = BENCHMARK_GUARD_SHA256,
) -> dict[str, Any]:
    """Build all indexes transactionally; test hooks never alter CLI production pins."""
    if expected_benchmark_rows <= 0 or not _valid_sha(expected_benchmark_sha256):
        raise ValueError("invalid benchmark guard expectations")
    if (
        expected_publication_rows <= 0
        or expected_publication_shards <= 0
        or not _valid_sha(expected_publication_manifest_sha256)
    ):
        raise ValueError("invalid publication metadata expectations")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    registry, _, _, contract = _validate_composition(
        registry_path=registry_path,
        composition_root=composition_root,
        expected_report_sha256=expected_composition_report_sha256,
    )
    resolved_s3_client = s3_client or boto3.client("s3", region_name=S3_REGION)
    resolved_parquet_filesystem = parquet_filesystem or pafs.S3FileSystem(region=S3_REGION)
    publication_metadata, publication_receipt = _load_publication_metadata(
        publication_manifest_path=publication_manifest_path,
        registry=registry,
        composition_contract=contract,
        s3_client=resolved_s3_client,
        parquet_filesystem=resolved_parquet_filesystem,
        expected_sha256=expected_publication_manifest_sha256,
        expected_rows=expected_publication_rows,
        expected_shards=expected_publication_shards,
    )
    with tempfile.TemporaryDirectory(prefix="pointing-exclusion-", dir=work_dir) as temporary:
        temporary_root = Path(temporary)
        manifests, detection_manifest_receipts = _download_detection_manifests(
            registry=registry,
            composition_contract=contract,
            work_dir=temporary_root / "detection-manifests",
        )
        detection_rows, detection_stats = _detection_index_rows(
            registry=registry,
            manifests=manifests,
            publication_metadata=publication_metadata,
        )
        pointing_rows, pointing_receipts = _pointing_index_rows(
            registry=registry,
            composition_contract=contract,
            pointing_roots=pointing_roots,
        )
        benchmark_rows, benchmark_receipt = _benchmark_index_rows(
            benchmark_guard_root=benchmark_guard_root,
            expected_sha256=expected_benchmark_sha256,
            expected_rows=expected_benchmark_rows,
        )
        staging_output = temporary_root / "output"
        partitions = {
            "detection": _write_partition(
                staging_output,
                name="detection",
                rows=detection_rows,
                receipt={
                    **detection_stats,
                    "repository": registry["detection_base"]["repository"],
                    "revision": registry["detection_base"]["revision"],
                    "downloaded_manifest_receipts": detection_manifest_receipts,
                    "publication_metadata_receipt": publication_receipt,
                },
            ),
            "pointing": _write_partition(
                staging_output,
                name="pointing",
                rows=pointing_rows,
                receipt={"source_receipts": pointing_receipts},
            ),
            "benchmark": _write_partition(
                staging_output,
                name="benchmark",
                rows=benchmark_rows,
                receipt=benchmark_receipt,
            ),
        }
        global_receipt = {
            "schema_version": 1,
            "artifact": "fireviewer-pointing-exclusion-indexes-v1",
            "composition_registry_sha256": contract["registry_sha256"],
            "composition_report_sha256": contract["composition_report_sha256"],
            "composition_integrity_receipt_sha256": contract[
                "composition_integrity_receipt_sha256"
            ],
            "publication_manifest_sha256": publication_receipt[
                "publication_manifest_sha256"
            ],
            "benchmark_guard_sha256": benchmark_receipt["guard_sha256"],
            "partitions": partitions,
            "total_index_rows": sum(int(value["rows"]) for value in partitions.values()),
            "incomplete_rows": 0,
            "media_files_output": 0,
            "source_gate_passed": True,
            "gate_errors": [],
            "reviews_admitted": False,
            "publication_allowed": False,
            "submitted_job": False,
        }
        global_path = staging_output / "exclusion_indexes_receipt.json"
        _write_json(global_path, global_receipt)
        if any(
            path.suffix.casefold() in IMAGE_SUFFIXES
            for path in staging_output.rglob("*")
            if path.is_file()
        ):
            raise ValueError("exclusion-index output contains media")

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        publish_root = Path(
            tempfile.mkdtemp(prefix=".pointing-exclusion-publish-", dir=output_dir.parent)
        )
        try:
            for source in sorted(staging_output.iterdir(), key=lambda path: path.name):
                destination = publish_root / source.name
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            if output_dir.exists():
                output_dir.rmdir()
            publish_root.replace(output_dir)
        except Exception:
            shutil.rmtree(publish_root, ignore_errors=True)
            raise
    final_global = output_dir / "exclusion_indexes_receipt.json"
    return {
        **global_receipt,
        "global_receipt_sha256": _sha256(final_global),
        "global_receipt_bytes": final_global.stat().st_size,
    }


def build_exclusion_indexes(
    *,
    registry_path: Path,
    composition_root: Path,
    expected_composition_report_sha256: str,
    publication_manifest_path: Path,
    pointing_roots: dict[str, Path],
    benchmark_guard_root: Path,
    work_dir: Path,
    output_dir: Path,
    s3_client: Any | None = None,
    parquet_filesystem: Any | None = None,
    expected_publication_manifest_sha256: str = PUBLICATION_MANIFEST_SHA256,
    expected_publication_rows: int = PUBLICATION_ROWS,
    expected_publication_shards: int = PUBLICATION_SHARDS,
    expected_benchmark_rows: int = BENCHMARK_ROWS,
    expected_benchmark_sha256: str = BENCHMARK_GUARD_SHA256,
) -> dict[str, Any]:
    """Build all indexes or emit one fail-closed receipt without partial indexes."""
    try:
        return _build_exclusion_indexes(
            registry_path=registry_path,
            composition_root=composition_root,
            expected_composition_report_sha256=expected_composition_report_sha256,
            publication_manifest_path=publication_manifest_path,
            pointing_roots=pointing_roots,
            benchmark_guard_root=benchmark_guard_root,
            work_dir=work_dir,
            output_dir=output_dir,
            s3_client=s3_client,
            parquet_filesystem=parquet_filesystem,
            expected_publication_manifest_sha256=expected_publication_manifest_sha256,
            expected_publication_rows=expected_publication_rows,
            expected_publication_shards=expected_publication_shards,
            expected_benchmark_rows=expected_benchmark_rows,
            expected_benchmark_sha256=expected_benchmark_sha256,
        )
    except Exception as exc:
        if not output_dir.exists() or not any(output_dir.iterdir()):
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                output_dir / "exclusion_indexes_failure_receipt.json",
                {
                    "schema_version": 1,
                    "artifact": "fireviewer-pointing-exclusion-indexes-v1",
                    "expected_composition_report_sha256": (
                        expected_composition_report_sha256.casefold()
                        if _valid_sha(expected_composition_report_sha256)
                        else None
                    ),
                    "expected_benchmark_guard_sha256": (
                        expected_benchmark_sha256.casefold()
                        if _valid_sha(expected_benchmark_sha256)
                        else None
                    ),
                    "expected_publication_manifest_sha256": (
                        expected_publication_manifest_sha256.casefold()
                        if _valid_sha(expected_publication_manifest_sha256)
                        else None
                    ),
                    "source_gate_passed": False,
                    "gate_errors": [f"{type(exc).__name__}:{exc}"],
                    "indexes_emitted": 0,
                    "incomplete_rows": 0,
                    "media_files_output": 0,
                    "reviews_admitted": False,
                    "publication_allowed": False,
                    "submitted_job": False,
                },
            )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--composition-root", type=Path, required=True)
    parser.add_argument("--composition-report-sha256", required=True)
    parser.add_argument("--publication-manifest", type=Path, required=True)
    parser.add_argument("--boreal-root", type=Path, required=True)
    parser.add_argument("--camp-swift-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--benchmark-guard-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_exclusion_indexes(
        registry_path=args.registry,
        composition_root=args.composition_root,
        expected_composition_report_sha256=args.composition_report_sha256,
        publication_manifest_path=args.publication_manifest,
        pointing_roots={
            "boreal": args.boreal_root,
            "camp-swift": args.camp_swift_root,
            "kit": args.kit_root,
        },
        benchmark_guard_root=args.benchmark_guard_root,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
