from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

UPSTREAM_REVISION = "d194cca045aa6b0dc0e53b0ee45573324efa5c4f"
UPSTREAM_ROOT = (
    f"https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires/resolve/{UPSTREAM_REVISION}"
)
DEFAULT_ARCHIVE = f"{UPSTREAM_ROOT}/eo4wildfires.tar.gz?download=true"
DEFAULT_SPLITS = {
    "train": f"{UPSTREAM_ROOT}/files_train.csv.gz?download=true",
    "validation": f"{UPSTREAM_ROOT}/files_val.csv.gz?download=true",
    "test": f"{UPSTREAM_ROOT}/files_test.csv.gz?download=true",
}
EXPECTED_SPLITS = {
    "train": {
        "count": 20_307,
        "sha256": "bd5f077d86aaab0f0406bcdb86aba04669b0367ff2d65ad4d4b74bb0e713cccd",
    },
    "validation": {
        "count": 5_077,
        "sha256": "b55ac8f0e42cff789f4f393710b980fe7a1a275fbea82354d09f5d74f241ca1c",
    },
    "test": {
        "count": 6_346,
        "sha256": "ce07dfac670e5553d8c868fd26b8bdbd0346eca8286e12dc6ab5a96b09195557",
    },
}
EXPECTED_ARCHIVE_SHA256 = "f9939c74fa11d90cf4a0e51f3079161311cc218db1f42deedd7ed1d7b167c5c3"
EXPECTED_ARCHIVE_BYTES = 25_474_671_500
EXPECTED_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
MAX_SPLIT_BYTES = 1_000_000
MAX_MEMBER_BYTES = 16 * 1024 * 1024
SOURCE_VALIDATOR_ID = "eo4-official-source-contract-v1"
MATERIALIZED_VALIDATOR_ID = "geotiff-geospatial-reopen-v1"
VALIDATION_POLICY = "dual_automated_official_source_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value, usedforsecurity=False).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_binary(location: str) -> BinaryIO:
    if location.startswith(("https://", "http://")):
        request = urllib.request.Request(  # noqa: S310
            location,
            headers={"User-Agent": "firewarning-prithvi-geographic-test/1"},
        )
        return urllib.request.urlopen(request, timeout=180)  # noqa: S310
    return Path(location).open("rb")


def _read_bounded(handle: BinaryIO, limit: int) -> bytes:
    payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"payload exceeds byte limit: {limit}")
    return payload


def _load_split(location: str, expected: dict[str, Any]) -> tuple[bytes, list[str]]:
    with closing(_open_binary(location)) as handle:
        compressed = _read_bounded(handle, MAX_SPLIT_BYTES)
    digest = _sha256_bytes(compressed)
    if digest != expected["sha256"]:
        raise ValueError(f"official split digest mismatch: {location}")
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as archive:
        text = io.TextIOWrapper(archive, encoding="utf-8-sig")
        rows = [row[0].strip() for row in csv.reader(text) if row and row[0].strip()]
    if len(rows) != expected["count"] or len(rows) != len(set(rows)):
        raise ValueError(f"official split inventory mismatch: {location}")
    if not all(PurePosixPath(row).name == row and row.endswith(".nc") for row in rows):
        raise ValueError(f"unsafe official split member: {location}")
    return compressed, rows


def _load_official_splits(
    output: Path,
    locations: dict[str, str],
) -> dict[str, set[str]]:
    source_dir = output / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    assignments: dict[str, set[str]] = {}
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        compressed, rows = _load_split(locations[split], EXPECTED_SPLITS[split])
        overlap = seen.intersection(rows)
        if overlap:
            raise ValueError(f"official split leakage detected: {split}")
        seen.update(rows)
        assignments[split] = set(rows)
        destination = source_dir / f"files_{split}.csv.gz"
        destination.write_bytes(compressed)
    return assignments


def _decode_netcdf_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tobytes"):
        return value.tobytes().decode("utf-8").rstrip("\x00")
    return str(value)


def _safe_member(member: tarfile.TarInfo) -> str | None:
    if not member.isfile() or member.size <= 0 or member.size > MAX_MEMBER_BYTES:
        return None
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".nc":
        return None
    return path.name


def _stream_test_members(
    archive_location: str,
    test_ids: set[str],
) -> Iterator[tuple[str, bytes]]:
    with (
        closing(_open_binary(archive_location)) as raw,
        tarfile.open(fileobj=raw, mode="r|gz") as archive,
    ):
        for member in archive:
            name = _safe_member(member)
            if name is None or name not in test_ids:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"unable to read archive member: {member.name}")
            payload = _read_bounded(extracted, MAX_MEMBER_BYTES)
            if len(payload) != member.size:
                raise ValueError(f"truncated archive member: {member.name}")
            yield name, payload


def _decode_candidate(payload: bytes) -> dict[str, Any]:
    try:
        import h5py
        import numpy as np
        import rasterio
        from rasterio.transform import Affine, array_bounds, xy
    except ImportError as exc:  # pragma: no cover - exercised by the real environment
        raise RuntimeError("h5py, numpy and rasterio are required") from exc

    with h5py.File(io.BytesIO(payload), "r") as dataset:
        image = np.asarray(dataset["S2A"][...], dtype=np.float32)
        raw_mask = np.asarray(dataset["burned_mask"][...], dtype=np.float32)
        x = np.asarray(dataset["x"][...], dtype=np.float64)
        y = np.asarray(dataset["y"][...], dtype=np.float64)
        spatial_ref = dataset.get("spatial_ref")
        if spatial_ref is None:
            raise ValueError("spatial_ref is missing")
        geotransform = _decode_netcdf_text(spatial_ref.attrs.get("GeoTransform"))
        crs_wkt = _decode_netcdf_text(
            spatial_ref.attrs.get("crs_wkt", spatial_ref.attrs.get("spatial_ref"))
        )

    if image.ndim != 3 or image.shape[0] != 6 or raw_mask.shape != image.shape[1:]:
        raise ValueError("EO4 image/mask shape mismatch")
    if x.shape != (image.shape[2],) or y.shape != (image.shape[1],):
        raise ValueError("EO4 coordinate vector shape mismatch")
    values = tuple(float(value) for value in geotransform.split())
    if len(values) != 6 or not crs_wkt:
        raise ValueError("EO4 georeferencing metadata is invalid")
    transform = Affine.from_gdal(*values)
    crs = rasterio.crs.CRS.from_wkt(crs_wkt)
    if crs.to_epsg() != 4326:
        raise ValueError(f"unexpected EO4 CRS: {crs}")

    expected_x = np.asarray(xy(transform, 0, range(image.shape[2]), offset="center")[0])
    expected_y = np.asarray(xy(transform, range(image.shape[1]), 0, offset="center")[1])
    tolerance = max(abs(transform.a), abs(transform.e), 1e-7) * 0.05
    if not np.allclose(expected_x, x, rtol=0.0, atol=tolerance):
        raise ValueError("EO4 x coordinates disagree with GeoTransform")
    if not np.allclose(expected_y, y, rtol=0.0, atol=tolerance):
        raise ValueError("EO4 y coordinates disagree with GeoTransform")

    if np.isinf(raw_mask).any():
        raise ValueError("EO4 burn mask contains infinite values")
    # The upstream EO4Wildfires loader explicitly maps NaN to zero before
    # producing patches. Preserve that published semantic here: NaN is the
    # non-burned background, not an ignored pixel.
    normalized_mask = np.nan_to_num(raw_mask, nan=0.0)
    positive = normalized_mask > 0
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("critical sample must contain burned and non-burned pixels")
    finite_image = np.isfinite(image)
    if not finite_image.any() or float(np.nanstd(image)) <= 0:
        raise ValueError("critical sample image has no usable signal")

    image[~finite_image] = 0.0
    mask = np.full(raw_mask.shape, -1, dtype=np.int16)
    mask[negative] = 0
    mask[positive] = 1
    bounds = array_bounds(image.shape[1], image.shape[2], transform)
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    site_id = f"eo4-grid-{math.floor(center_lat):+03d}-{math.floor(center_lon):+04d}"
    return {
        "image": image,
        "mask": mask,
        "transform": transform,
        "crs": crs,
        "bounds": [float(value) for value in bounds],
        "site_id": site_id,
        "positive_pixels": int(positive.sum()),
        "valid_pixels": int(raw_mask.size),
    }


def _write_candidate(
    output: Path,
    source_name: str,
    source_payload: bytes,
    decoded: dict[str, Any],
) -> dict[str, Any]:
    import rasterio

    event_token = Path(source_name).stem
    sample_id = f"eo4-test-{event_token}"
    image_relpath = Path("data") / f"{sample_id}_merged.tif"
    mask_relpath = Path("data") / f"{sample_id}.mask.tif"
    image_path = output / image_relpath
    mask_path = output / mask_relpath
    image_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": decoded["image"].shape[1],
        "width": decoded["image"].shape[2],
        "transform": decoded["transform"],
        "crs": decoded["crs"],
        "compress": "deflate",
    }
    with rasterio.open(
        image_path,
        "w",
        count=6,
        dtype="float32",
        predictor=3,
        **profile,
    ) as target:
        target.write(decoded["image"])
        target.descriptions = tuple(EXPECTED_BANDS)
    with rasterio.open(
        mask_path,
        "w",
        count=1,
        dtype="int16",
        nodata=-1,
        predictor=2,
        **profile,
    ) as target:
        target.write(decoded["mask"], 1)
        target.set_band_description(1, "BURNED_AREA")
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "event_id": f"eo4wildfires:{event_token}",
        "site_id": decoded["site_id"],
        "split": "test",
        "source_split": "test",
        "source_member": f"eo4wildfires/{source_name}",
        "source_payload_sha256": _sha256_bytes(source_payload),
        "source_revision": UPSTREAM_REVISION,
        "image_relpath": image_relpath.as_posix(),
        "mask_relpath": mask_relpath.as_posix(),
        "image_sha256": _sha256_file(image_path),
        "mask_sha256": _sha256_file(mask_path),
        "crs": decoded["crs"].to_string(),
        "bounds": decoded["bounds"],
        "bands": EXPECTED_BANDS,
        "mask_values": {"burned": 1, "not_burned": 0, "ignore": -1},
        "positive_pixels": decoded["positive_pixels"],
        "valid_pixels": decoded["valid_pixels"],
        "reference_authority": "EFFIS annotations distributed by EO4Wildfires",
        "georeferencing_check": "pixel_centers_match_embedded_xy_vectors",
        "validation_status": "source_contract_validated",
        "automated_validators": [SOURCE_VALIDATOR_ID],
        "validator_count": 1,
        "human_review_required": False,
    }


def _audit_materialized_candidate(
    output: Path,
    row: dict[str, Any],
) -> list[str]:
    import numpy as np
    import rasterio

    errors: list[str] = []
    image_path = output / row["image_relpath"]
    mask_path = output / row["mask_relpath"]
    if not image_path.is_file() or _sha256_file(image_path) != row["image_sha256"]:
        errors.append("image_digest")
    if not mask_path.is_file() or _sha256_file(mask_path) != row["mask_sha256"]:
        errors.append("mask_digest")
    if errors:
        return errors

    with rasterio.open(image_path) as image, rasterio.open(mask_path) as mask:
        if image.count != 6 or list(image.descriptions) != EXPECTED_BANDS:
            errors.append("image_bands")
        if image.crs is None or image.crs.to_string() != row["crs"]:
            errors.append("image_crs")
        if mask.crs != image.crs or mask.transform != image.transform:
            errors.append("image_mask_georeferencing")
        if mask.width != image.width or mask.height != image.height or mask.count != 1:
            errors.append("image_mask_shape")
        expected_bounds = np.asarray(row["bounds"], dtype=np.float64)
        if not np.allclose(
            np.asarray(image.bounds, dtype=np.float64),
            expected_bounds,
            rtol=0.0,
            atol=1e-10,
        ):
            errors.append("image_bounds")
        if (
            not (-180 <= image.bounds.left < image.bounds.right <= 180)
            or not (-90 <= image.bounds.bottom < image.bounds.top <= 90)
            or image.transform.a <= 0
            or image.transform.e >= 0
        ):
            errors.append("geographic_bounds")
        image_values = image.read()
        if (
            not np.isfinite(image_values).all()
            or float(np.std(image_values, dtype=np.float64)) <= 0
        ):
            errors.append("image_signal")
        mask_values = mask.read(1)
        unique = set(int(value) for value in np.unique(mask_values))
        if unique != {0, 1}:
            errors.append("mask_values")
        if int((mask_values == 1).sum()) != row["positive_pixels"]:
            errors.append("mask_positive_pixels")
        if mask_values.size != row["valid_pixels"]:
            errors.append("mask_valid_pixels")
    return sorted(set(errors))


def _mark_dual_automated_validation(row: dict[str, Any]) -> None:
    row["validation_status"] = "dual_automated_validation_passed"
    row["automated_validators"] = [
        SOURCE_VALIDATOR_ID,
        MATERIALIZED_VALIDATOR_ID,
    ]
    row["validator_count"] = 2
    row["human_review_required"] = False


def _selection_sha256(rows: list[dict[str, Any]]) -> str:
    selection = [
        {
            "event_id": row["event_id"],
            "image_sha256": row["image_sha256"],
            "mask_sha256": row["mask_sha256"],
            "sample_id": row["sample_id"],
            "site_id": row["site_id"],
        }
        for row in rows
    ]
    return hashlib.sha256(
        (
            json.dumps(
                sorted(selection, key=lambda row: row["sample_id"]),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_anomaly_queue(
    output: Path,
    anomalies: list[dict[str, Any]],
) -> None:
    fields = [
        "sample_id",
        "event_id",
        "site_id",
        "automated_errors",
        "human_decision",
        "human_comment",
    ]
    with (output / "anomaly-review.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for anomaly in anomalies:
            row = anomaly["row"]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "event_id": row["event_id"],
                    "site_id": row["site_id"],
                    "automated_errors": "|".join(anomaly["errors"]),
                    "human_decision": "",
                    "human_comment": "",
                }
            )


def _existing_rows(staging: Path) -> list[dict[str, Any]]:
    pending = staging / "manifest.pending.jsonl"
    if not pending.is_file():
        return []
    rows = [
        json.loads(line)
        for line in pending.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        if not (staging / row["image_relpath"]).is_file():
            raise ValueError(f"resumed image is missing: {row['sample_id']}")
        if not (staging / row["mask_relpath"]).is_file():
            raise ValueError(f"resumed mask is missing: {row['sample_id']}")
    return rows


def build(
    output: Path,
    *,
    archive_location: str,
    split_locations: dict[str, str],
    target_samples: int,
    max_per_site: int,
    resume: bool,
) -> dict[str, Any]:
    if target_samples < 100:
        raise ValueError("the Prithvi geographic gate requires at least 100 samples")
    if max_per_site <= 0:
        raise ValueError("max_per_site must be positive")
    output = output.resolve()
    staging = output.with_name(f"{output.name}.staging")
    if output.exists():
        raise FileExistsError(f"final output already exists: {output}")
    if staging.exists() and not resume:
        raise FileExistsError(f"staging output already exists; pass --resume: {staging}")
    staging.mkdir(parents=True, exist_ok=True)

    splits = _load_official_splits(staging, split_locations)
    rows = _existing_rows(staging)
    selected_ids = {f"{row['event_id'].split(':', 1)[1]}.nc" for row in rows}
    site_counts: Counter[str] = Counter(row["site_id"] for row in rows)
    rejection_counts: Counter[str] = Counter()
    rejection_examples: dict[str, str] = {}

    pending_manifest = staging / "manifest.pending.jsonl"
    for source_name, payload in _stream_test_members(
        archive_location,
        splits["test"] - selected_ids,
    ):
        if len(rows) >= target_samples:
            break
        try:
            decoded = _decode_candidate(payload)
        except (KeyError, OSError, ValueError) as exc:
            rejection_key = type(exc).__name__
            rejection_counts[rejection_key] += 1
            rejection_examples.setdefault(rejection_key, str(exc))
            rejected = sum(rejection_counts.values())
            if rejected <= 5 or rejected % 25 == 0:
                print(
                    "prithvi geographic critical test "
                    f"rejected={rejected} source={source_name} "
                    f"reason={rejection_key}:{exc}",
                    flush=True,
                )
            continue
        if site_counts[decoded["site_id"]] >= max_per_site:
            rejection_counts["site_cap"] += 1
            continue
        row = _write_candidate(staging, source_name, payload, decoded)
        if source_name in splits["train"] or source_name in splits["validation"]:
            raise ValueError(f"selected event leaks into training: {source_name}")
        materialized_errors = _audit_materialized_candidate(staging, row)
        if materialized_errors:
            (staging / row["image_relpath"]).unlink(missing_ok=True)
            (staging / row["mask_relpath"]).unlink(missing_ok=True)
            rejection_counts["materialized_audit"] += 1
            rejection_examples.setdefault(
                "materialized_audit",
                "|".join(materialized_errors),
            )
            continue
        _mark_dual_automated_validation(row)
        with pending_manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        rows.append(row)
        site_counts[row["site_id"]] += 1
        if len(rows) % 10 == 0:
            print(
                "prithvi geographic critical test "
                f"samples={len(rows)}/{target_samples} sites={len(site_counts)}",
                flush=True,
            )

    if len(rows) < target_samples:
        raise RuntimeError(
            f"archive stream ended with only {len(rows)}/{target_samples} eligible samples"
        )
    if len(site_counts) < 3:
        raise RuntimeError(f"geographic diversity is insufficient: {len(site_counts)} sites")

    rows = sorted(rows[:target_samples], key=lambda row: row["sample_id"])
    selected_site_counts = Counter(row["site_id"] for row in rows)
    manifest = staging / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_anomaly_queue(staging, [])
    provenance = {
        "schema_version": 1,
        "dataset": "AUA-Informatics-Lab/eo4wildfires",
        "revision": UPSTREAM_REVISION,
        "license": "CC-BY-SA-4.0",
        "annotations": "EFFIS",
        "sensors": ["Sentinel-2"],
        "period": "2018-2022",
        "official_split": "test",
        "canonical_url": "https://huggingface.co/datasets/AUA-Informatics-Lab/eo4wildfires",
        "archive_url": archive_location,
        "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "expected_archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "archive_full_digest_verified_in_this_partial_stream": False,
        "split_sha256": {split: EXPECTED_SPLITS[split]["sha256"] for split in EXPECTED_SPLITS},
    }
    _write_json(staging / "sources.json", provenance)
    report = {
        "schema_version": 1,
        "dataset_id": "prithvi-geographic-critical-test-v1",
        "manifest_path": "manifest.jsonl",
        "manifest_sha256": _sha256_file(manifest),
        "selection_sha256": _selection_sha256(rows),
        "sample_count": len(rows),
        "event_count": len({row["event_id"] for row in rows}),
        "site_count": len(selected_site_counts),
        "site_counts": dict(sorted(selected_site_counts.items())),
        "split": "test",
        "independent_from_training": True,
        "training_group_overlap": 0,
        "split_leakage": 0,
        "georeferencing_verified": True,
        "official_reference_verified": True,
        "independent_validation_complete": True,
        "validation_policy": VALIDATION_POLICY,
        "automated_validator_count": 2,
        "human_review_required": False,
        "anomaly_count": 0,
        "training_ready": True,
        "blocking_reasons": [],
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejection_examples": dict(sorted(rejection_examples.items())),
    }
    _write_json(staging / "report.json", report)
    pending_manifest.unlink(missing_ok=True)
    os.replace(staging, output)
    return report


def audit_existing(output: Path) -> dict[str, Any]:
    output = output.resolve()
    report_path = output / "report.json"
    manifest_path = output / "manifest.jsonl"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"incomplete geographic critical test: {output}")

    split_locations = {
        "train": str(output / "sources" / "files_train.csv.gz"),
        "validation": str(output / "sources" / "files_validation.csv.gz"),
        "test": str(output / "sources" / "files_test.csv.gz"),
    }
    splits: dict[str, set[str]] = {}
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        _compressed, names = _load_split(
            split_locations[split],
            EXPECTED_SPLITS[split],
        )
        if seen.intersection(names):
            raise ValueError(f"official split leakage detected: {split}")
        seen.update(names)
        splits[split] = set(names)

    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anomalies: list[dict[str, Any]] = []
    for row in rows:
        source_name = PurePosixPath(row["source_member"]).name
        errors: list[str] = []
        if source_name not in splits["test"]:
            errors.append("source_not_in_official_test")
        if source_name in splits["train"] or source_name in splits["validation"]:
            errors.append("source_split_leakage")
        errors.extend(_audit_materialized_candidate(output, row))
        if errors:
            anomalies.append({"row": row, "errors": sorted(set(errors))})
        else:
            _mark_dual_automated_validation(row)

    _write_anomaly_queue(output, anomalies)
    if anomalies:
        return {
            "training_ready": False,
            "sample_count": len(rows),
            "anomaly_count": len(anomalies),
            "blocking_reasons": ["automated_geographic_validation_failed"],
        }

    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "manifest_sha256": _sha256_file(manifest_path),
            "selection_sha256": _selection_sha256(rows),
            "independent_validation_complete": True,
            "validation_policy": VALIDATION_POLICY,
            "automated_validator_count": 2,
            "human_review_required": False,
            "anomaly_count": 0,
            "training_ready": True,
            "blocking_reasons": [],
        }
    )
    for obsolete in ("double_human_validation", "pending_human_reviews"):
        report.pop(obsolete, None)
    (output / "review-queue.csv").unlink(missing_ok=True)
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the independent EO4Wildfires geographic critical test "
            "required before Prithvi training."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--train-split", default=DEFAULT_SPLITS["train"])
    parser.add_argument("--validation-split", default=DEFAULT_SPLITS["validation"])
    parser.add_argument("--test-split", default=DEFAULT_SPLITS["test"])
    parser.add_argument("--target-samples", type=int, default=100)
    parser.add_argument("--max-per-site", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()
    if args.audit_existing:
        print(
            json.dumps(
                audit_existing(args.output),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    report = build(
        args.output,
        archive_location=args.archive,
        split_locations={
            "train": args.train_split,
            "validation": args.validation_split,
            "test": args.test_split,
        },
        target_samples=args.target_samples,
        max_per_site=args.max_per_site,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
