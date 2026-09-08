from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PRITHVI_MODEL = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
PRITHVI_MODEL_REVISION = "a3f2c410e45b8ac7417976614528a872f024d831"
TERRAMIND_MODEL = "ibm-esa-geospatial/TerraMind-base-Fire"
TERRAMIND_MODEL_REVISION = "6eb5178aac4f8a4191796258ae26e796195cc00d"
EO4_SOURCE_REVISION = "d194cca045aa6b0dc0e53b0ee45573324efa5c4f"
EXPECTED_HLS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
EXPECTED_EO4_SPLITS = {"train": 20_307, "validation": 5_077, "test": 6_346}
GEOGRAPHIC_TEST_MIN_SAMPLES = 100
GEOGRAPHIC_TEST_MIN_EVENTS = 3
GEOGRAPHIC_TEST_MIN_SITES = 3
HLS_MEANS = [
    0.033349706741586264,
    0.05701185520536176,
    0.05889748132001316,
    0.2323245113436119,
    0.1972854853760658,
    0.11944914225186566,
]
HLS_STDS = [
    0.02269135568823774,
    0.026807560223070237,
    0.04004109844362779,
    0.07791732423672691,
    0.08708738838140137,
    0.07241979477437814,
]


def _resolve_resume_checkpoint(
    output_dir: Path,
    requested: str | None,
) -> Path | None:
    if requested is None:
        return None
    if requested == "auto":
        checkpoint = output_dir / "checkpoints" / "last.ckpt"
        return checkpoint.resolve() if checkpoint.is_file() else None
    checkpoint = Path(requested).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Prithvi checkpoint is missing: {checkpoint}")
    return checkpoint


def _resolve_terratorch_executable() -> str | None:
    executable = shutil.which("terratorch")
    if executable is not None:
        return executable
    python_dir = Path(sys.executable).absolute().parent
    for filename in ("terratorch", "terratorch.exe"):
        candidate = python_dir / filename
        if candidate.is_file():
            return str(candidate)
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_geographic_test_report(
    report_path: Path | None,
    *,
    verify_files: bool,
) -> dict[str, Any]:
    missing_error = "independent_geographic_critical_test_missing"
    if report_path is None or not report_path.is_file():
        return {"ready": False, "errors": [missing_error]}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "errors": [f"independent_geographic_critical_test_invalid:{type(exc).__name__}"],
        }
    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append("geographic_test_schema_version_invalid")
    for field in (
        "independent_from_training",
        "georeferencing_verified",
        "official_reference_verified",
        "independent_validation_complete",
    ):
        if report.get(field) is not True:
            errors.append(f"geographic_test_{field}_missing")
    if report.get("validation_policy") != "dual_automated_official_source_v1":
        errors.append("geographic_test_validation_policy_invalid")
    if int(report.get("automated_validator_count", 0)) < 2:
        errors.append("geographic_test_automated_validator_count_insufficient")
    if report.get("training_group_overlap") != 0:
        errors.append("geographic_test_training_group_overlap")
    if report.get("split_leakage") != 0:
        errors.append("geographic_test_split_leakage")

    manifest_name = report.get("manifest_path")
    manifest_path: Path | None = None
    rows: list[dict[str, Any]] = []
    if not isinstance(manifest_name, str) or not manifest_name:
        errors.append("geographic_test_manifest_path_missing")
    else:
        candidate = (report_path.parent / manifest_name).resolve()
        try:
            candidate.relative_to(report_path.parent.resolve())
        except ValueError:
            errors.append("geographic_test_manifest_path_unsafe")
        else:
            manifest_path = candidate
    if manifest_path is None or not manifest_path.is_file():
        errors.append("geographic_test_manifest_missing")
    else:
        expected_manifest_sha = report.get("manifest_sha256")
        if (
            not isinstance(expected_manifest_sha, str)
            or len(expected_manifest_sha) != 64
            or _sha256_file(manifest_path) != expected_manifest_sha.lower()
        ):
            errors.append("geographic_test_manifest_sha256_mismatch")
        try:
            for line_number, line in enumerate(
                manifest_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"row {line_number} is not an object")
                rows.append(row)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"geographic_test_manifest_invalid:{type(exc).__name__}")

    sample_ids: set[str] = set()
    event_ids: set[str] = set()
    site_ids: set[str] = set()
    selection_rows: list[dict[str, Any]] = []
    if rows:
        for row in rows:
            sample_id = row.get("sample_id")
            event_id = row.get("event_id")
            site_id = row.get("site_id")
            if not all(
                isinstance(value, str) and value for value in (sample_id, event_id, site_id)
            ):
                errors.append("geographic_test_row_identity_missing")
                continue
            if sample_id in sample_ids:
                errors.append("geographic_test_duplicate_sample_id")
            sample_ids.add(sample_id)
            event_ids.add(event_id)
            site_ids.add(site_id)
            if row.get("split") != "test":
                errors.append("geographic_test_non_test_row")
            bounds = row.get("bounds")
            if (
                not isinstance(row.get("crs"), str)
                or not isinstance(bounds, list)
                or len(bounds) != 4
                or not all(isinstance(value, (int, float)) for value in bounds)
            ):
                errors.append("geographic_test_row_georeferencing_missing")
            image_sha = row.get("image_sha256")
            mask_sha = row.get("mask_sha256")
            if not all(
                isinstance(value, str) and len(value) == 64 for value in (image_sha, mask_sha)
            ):
                errors.append("geographic_test_row_digest_missing")
                continue
            validators = row.get("automated_validators")
            if (
                row.get("validation_status") != "dual_automated_validation_passed"
                or int(row.get("validator_count", 0)) < 2
                or not isinstance(validators, list)
                or len(set(validators)) < 2
            ):
                errors.append("geographic_test_row_dual_validation_missing")
            selection_rows.append(
                {
                    "event_id": event_id,
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "sample_id": sample_id,
                    "site_id": site_id,
                }
            )
            if verify_files and manifest_path is not None:
                for rel_field, digest in (
                    ("image_relpath", image_sha),
                    ("mask_relpath", mask_sha),
                ):
                    relpath = row.get(rel_field)
                    if not isinstance(relpath, str) or not relpath:
                        errors.append(f"geographic_test_row_{rel_field}_missing")
                        continue
                    payload = (manifest_path.parent / relpath).resolve()
                    try:
                        payload.relative_to(manifest_path.parent.resolve())
                    except ValueError:
                        errors.append("geographic_test_payload_path_unsafe")
                        continue
                    if not payload.is_file() or _sha256_file(payload) != digest.lower():
                        errors.append(f"geographic_test_payload_invalid:{rel_field}")

    if len(sample_ids) < GEOGRAPHIC_TEST_MIN_SAMPLES:
        errors.append("geographic_test_sample_count_insufficient")
    if len(event_ids) < GEOGRAPHIC_TEST_MIN_EVENTS:
        errors.append("geographic_test_event_count_insufficient")
    if len(site_ids) < GEOGRAPHIC_TEST_MIN_SITES:
        errors.append("geographic_test_site_count_insufficient")
    selection_sha = hashlib.sha256(
        (
            json.dumps(
                sorted(selection_rows, key=lambda row: row["sample_id"]),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    if report.get("selection_sha256") != selection_sha:
        errors.append("geographic_test_selection_sha256_mismatch")
    errors = sorted(set(errors))
    return {
        "ready": not errors,
        "errors": errors,
        "report_path": str(report_path.resolve()),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "sample_count": len(sample_ids),
        "event_count": len(event_ids),
        "site_count": len(site_ids),
        "selection_sha256": selection_sha,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _validate_hls(
    manifest: Path,
    *,
    verify_files: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(manifest)
    split_counts: Counter[str] = Counter()
    split_groups: dict[str, set[str]] = defaultdict(set)
    seen_images: set[str] = set()
    seen_masks: set[str] = set()
    burned_pixels = 0
    for line_number, row in enumerate(rows, start=1):
        split = str(row["split"])
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported HLS split at line {line_number}: {split}")
        if row.get("mask_values") != {"burned": 1, "not_burned": 0, "ignore": -1}:
            raise ValueError(f"Unexpected HLS mask contract at line {line_number}")
        source_asset = row.get("source_asset")
        if not isinstance(source_asset, dict) or source_asset.get("bands") != [
            "B02",
            "B03",
            "B04",
            "B8A",
            "B11",
            "B12",
        ]:
            raise ValueError(f"Unexpected HLS band order at line {line_number}")
        image_digest = str(row["sha256"])
        mask_digest = str(row["mask_sha256"])
        if image_digest in seen_images or mask_digest in seen_masks:
            raise ValueError(f"Duplicate HLS image or mask at line {line_number}")
        seen_images.add(image_digest)
        seen_masks.add(mask_digest)
        split_counts[split] += 1
        split_groups[str(row["split_group"])].add(split)
        burned_pixels += int(row.get("raster", {}).get("burned_pixels", 0))
        if verify_files:
            image = (manifest.parent / str(row["image_relpath"])).resolve()
            mask = (manifest.parent / str(row["mask_relpath"])).resolve()
            if manifest.parent.resolve() not in image.parents:
                raise ValueError(f"HLS image escapes corpus at line {line_number}")
            if manifest.parent.resolve() not in mask.parents:
                raise ValueError(f"HLS mask escapes corpus at line {line_number}")
            if _sha256_file(image) != image_digest or _sha256_file(mask) != mask_digest:
                raise ValueError(f"HLS payload digest mismatch at line {line_number}")
        if line_number % 500 == 0:
            print(f"prithvi preflight HLS rows={line_number}", flush=True)
    leakage = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leakage:
        raise ValueError(f"HLS split-group leakage: {len(leakage)} groups")
    if split_counts["train"] == 0 or split_counts["validation"] == 0:
        raise ValueError("HLS must contain train and validation scenes")
    return rows, {
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "split_group_leakage": 0,
        "burned_pixels": burned_pixels,
    }


def _validate_eo4(
    manifest: Path,
    *,
    verify_files: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(manifest)
    split_counts: Counter[str] = Counter()
    seen_digests: set[str] = set()
    positive_scenes = 0
    for line_number, row in enumerate(rows, start=1):
        split = str(row["split"])
        if split not in EXPECTED_EO4_SPLITS:
            raise ValueError(f"Unsupported EO4 split at line {line_number}: {split}")
        if str(row.get("source_revision")) != EO4_SOURCE_REVISION:
            raise ValueError(f"EO4 source revision drift at line {line_number}")
        variables = row.get("variables")
        if not isinstance(variables, dict):
            raise ValueError(f"EO4 variable evidence is missing at line {line_number}")
        s2a = variables.get("S2A")
        mask = variables.get("burned_mask")
        if not isinstance(s2a, dict) or list(s2a.get("shape", []))[0:1] != [6]:
            raise ValueError(f"EO4 S2A must expose six channels at line {line_number}")
        if not isinstance(mask, dict) or len(list(mask.get("shape", []))) != 2:
            raise ValueError(f"EO4 burned_mask must be two-dimensional at line {line_number}")
        digest = str(row["sha256"])
        if digest in seen_digests:
            raise ValueError(f"Duplicate EO4 payload at line {line_number}")
        seen_digests.add(digest)
        split_counts[split] += 1
        positive_scenes += int(row.get("burned_mask", {}).get("positive_pixels", 0) > 0)
        if verify_files:
            payload = (manifest.parent / str(row["source_member"])).resolve()
            if manifest.parent.resolve() not in payload.parents:
                raise ValueError(f"EO4 payload escapes corpus at line {line_number}")
            if _sha256_file(payload) != digest:
                raise ValueError(f"EO4 payload digest mismatch at line {line_number}")
        if line_number % 5_000 == 0:
            print(f"prithvi preflight EO4 rows={line_number}", flush=True)
    if len(rows) == sum(EXPECTED_EO4_SPLITS.values()) and split_counts != Counter(
        EXPECTED_EO4_SPLITS
    ):
        raise ValueError(f"EO4 official split-count drift: {dict(split_counts)}")
    return rows, {
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_payloads": len(seen_digests),
        "positive_scenes": positive_scenes,
        "complete_official_corpus": len(rows) == sum(EXPECTED_EO4_SPLITS.values()),
    }


def build_preflight_report(
    hls_manifest: Path,
    eo4_manifest: Path,
    *,
    verify_files: bool,
    geographic_test_report: Path | None = None,
) -> dict[str, Any]:
    _hls_rows, hls_report = _validate_hls(hls_manifest, verify_files=verify_files)
    _eo4_rows, eo4_report = _validate_eo4(eo4_manifest, verify_files=verify_files)
    dataset_errors: list[str] = []
    if not eo4_report["complete_official_corpus"]:
        dataset_errors.append("eo4wildfires_official_corpus_incomplete")
    geographic_test = _validate_geographic_test_report(
        geographic_test_report,
        verify_files=verify_files,
    )
    training_errors = dataset_errors + geographic_test["errors"]
    promotion_errors: list[str] = ["trained_model_independent_evaluation_missing"]
    return {
        "schema_version": 1,
        "trainer": "prithvi_eo_2_300m_burnscars",
        "base_model": PRITHVI_MODEL,
        "base_model_revision": PRITHVI_MODEL_REVISION,
        "benchmark_model": TERRAMIND_MODEL,
        "benchmark_model_revision": TERRAMIND_MODEL_REVISION,
        "hls": hls_report,
        "eo4wildfires": eo4_report,
        "dataset_errors": dataset_errors,
        "geographic_test": geographic_test,
        "training_errors": training_errors,
        "promotion_errors": promotion_errors,
        "training_ready": not training_errors,
        "promotion_ready": not training_errors and not promotion_errors,
    }


def build_materialized_preflight_report(
    dataset_root: Path,
    *,
    geographic_test_report: Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    materialized = _validate_materialized_dataset(dataset_root)
    geographic_test = _validate_geographic_test_report(
        geographic_test_report,
        verify_files=verify_files,
    )
    training_errors = list(geographic_test["errors"])
    return {
        "schema_version": 1,
        "trainer": "prithvi_eo_2_300m_burnscars",
        "base_model": PRITHVI_MODEL,
        "base_model_revision": PRITHVI_MODEL_REVISION,
        "benchmark_model": TERRAMIND_MODEL,
        "benchmark_model_revision": TERRAMIND_MODEL_REVISION,
        "materialized_dataset": materialized,
        "dataset_errors": [],
        "geographic_test": geographic_test,
        "training_errors": training_errors,
        "promotion_errors": ["trained_model_independent_evaluation_missing"],
        "training_ready": not training_errors,
        "promotion_ready": False,
    }


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _decode_netcdf_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _reflectance_scale_compatible(
    minimums: Any,
    standard_deviations: Any,
    mean_delta_in_hls_std: Any,
    ratios_above_threshold: Any,
) -> bool:
    return bool(
        all(float(value) >= -0.25 for value in minimums)
        and all(float(value) > 0 for value in standard_deviations)
        and all(float(value) <= 2.5 for value in mean_delta_in_hls_std)
        and all(float(row[0]) <= 0.005 for row in ratios_above_threshold)
        and all(float(row[2]) <= 0.000001 for row in ratios_above_threshold)
    )


def _materialize_hls(rows: list[dict[str, Any]], source_root: Path, output: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        token = f"hls_{index:06d}"
        _link_or_copy(
            (source_root / str(row["image_relpath"])).resolve(),
            output / "data" / f"{token}_merged.tif",
        )
        _link_or_copy(
            (source_root / str(row["mask_relpath"])).resolve(),
            output / "data" / f"{token}.mask.tif",
        )
        split = str(row["split"])
        (output / "splits" / f"{split}.txt").parent.mkdir(parents=True, exist_ok=True)
        with (output / "splits" / f"{split}.txt").open("a", encoding="utf-8") as handle:
            handle.write(token + "\n")
        counts[split] += 1
        if (index + 1) % 500 == 0:
            print(f"prithvi materialize HLS rows={index + 1}", flush=True)
    return counts


def _materialize_eo4(
    rows: list[dict[str, Any]],
    source_root: Path,
    output: Path,
) -> tuple[Counter[str], dict[str, Any]]:
    try:
        import h5py
        import numpy as np
        import rasterio
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError("EO4 materialization requires h5py, numpy and rasterio") from exc
    counts: Counter[str] = Counter()
    finite_counts = np.zeros(6, dtype=np.int64)
    channel_sums = np.zeros(6, dtype=np.float64)
    channel_squares = np.zeros(6, dtype=np.float64)
    channel_minimums = np.full(6, np.inf, dtype=np.float64)
    channel_maximums = np.full(6, -np.inf, dtype=np.float64)
    reflectance_thresholds = np.asarray([1.0, 1.5, 2.0], dtype=np.float32)
    values_above_threshold = np.zeros((6, len(reflectance_thresholds)), dtype=np.int64)
    source_crs_digests: set[str] = set()
    for index, row in enumerate(rows):
        source = (source_root / str(row["source_member"])).resolve()
        with h5py.File(source, "r") as dataset:
            image = np.asarray(dataset["S2A"][...], dtype=np.float32)
            raw_mask = np.asarray(dataset["burned_mask"][...], dtype=np.float32)
            spatial_ref = dataset.get("spatial_ref")
            if spatial_ref is None:
                raise ValueError(f"EO4 spatial_ref is missing: {source}")
            geo_transform = _decode_netcdf_text(spatial_ref.attrs.get("GeoTransform"))
            crs_wkt = _decode_netcdf_text(
                spatial_ref.attrs.get("crs_wkt", spatial_ref.attrs.get("spatial_ref"))
            )
        transform_values = tuple(float(value) for value in geo_transform.split())
        if len(transform_values) != 6 or not crs_wkt:
            raise ValueError(f"EO4 georeferencing metadata is invalid: {source}")
        source_crs_digests.add(hashlib.sha256(crs_wkt.encode("utf-8")).hexdigest())
        if image.ndim != 3 or image.shape[0] != 6 or raw_mask.shape != image.shape[1:]:
            raise ValueError(f"EO4 payload shape drift: {source}")
        for channel_index in range(6):
            channel = image[channel_index]
            finite_values = channel[np.isfinite(channel)].astype(np.float64, copy=False)
            if finite_values.size:
                finite_counts[channel_index] += finite_values.size
                channel_sums[channel_index] += finite_values.sum()
                channel_squares[channel_index] += np.square(finite_values).sum()
                channel_minimums[channel_index] = min(
                    channel_minimums[channel_index],
                    float(finite_values.min()),
                )
                channel_maximums[channel_index] = max(
                    channel_maximums[channel_index],
                    float(finite_values.max()),
                )
                values_above_threshold[channel_index] += np.asarray(
                    [
                        np.count_nonzero(finite_values > threshold)
                        for threshold in reflectance_thresholds
                    ],
                    dtype=np.int64,
                )
        image[~np.isfinite(image)] = 0.0
        mask = np.full(raw_mask.shape, -1, dtype=np.int16)
        finite = np.isfinite(raw_mask)
        mask[finite & (raw_mask <= 0)] = 0
        mask[finite & (raw_mask > 0)] = 1
        token = f"eo4_{index:06d}"
        image_path = output / "data" / f"{token}_merged.tif"
        mask_path = output / "data" / f"{token}.mask.tif"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": image.shape[1],
            "width": image.shape[2],
            "transform": Affine.from_gdal(*transform_values),
            "crs": crs_wkt,
        }
        with rasterio.open(image_path, "w", count=6, dtype="float32", **profile) as target:
            target.write(image)
        with rasterio.open(
            mask_path,
            "w",
            count=1,
            dtype="int16",
            nodata=-1,
            **profile,
        ) as target:
            target.write(mask, 1)
        split = str(row["split"])
        with (output / "splits" / f"{split}.txt").open("a", encoding="utf-8") as handle:
            handle.write(token + "\n")
        counts[split] += 1
        if (index + 1) % 1_000 == 0:
            print(f"prithvi materialize EO4 rows={index + 1}", flush=True)
    if np.any(finite_counts == 0):
        raise ValueError("EO4 contains an empty Sentinel-2 channel")
    means = channel_sums / finite_counts
    variances = np.maximum(channel_squares / finite_counts - np.square(means), 0.0)
    standard_deviations = np.sqrt(variances)
    threshold_ratios = values_above_threshold / finite_counts[:, None]
    mean_delta_in_hls_std = np.abs(
        (means - np.asarray(HLS_MEANS, dtype=np.float64)) / np.asarray(HLS_STDS, dtype=np.float64)
    )
    normalization_compatible = _reflectance_scale_compatible(
        channel_minimums,
        standard_deviations,
        mean_delta_in_hls_std,
        threshold_ratios,
    )
    return counts, {
        "policy": (
            "eo4_s2a_hls_scale_requires_nonnegative_reflectance,nonzero_variance,"
            "mean_delta_lte_2.5_hls_std,above_1_lte_0.5pct,above_2_lte_0.0001pct"
        ),
        "bands": EXPECTED_HLS_BANDS,
        "finite_values": finite_counts.tolist(),
        "minimums": channel_minimums.tolist(),
        "maximums": channel_maximums.tolist(),
        "means": means.tolist(),
        "standard_deviations": standard_deviations.tolist(),
        "mean_delta_in_hls_std": mean_delta_in_hls_std.tolist(),
        "reflectance_thresholds": reflectance_thresholds.tolist(),
        "values_above_threshold": values_above_threshold.tolist(),
        "ratios_above_threshold": threshold_ratios.tolist(),
        "georeferencing": {
            "all_scenes_georeferenced": True,
            "scene_count": len(rows),
            "crs_wkt_sha256": sorted(source_crs_digests),
            "policy": "exact_source_geotransform_and_crs_no_reprojection",
        },
        "compatible_with_hls_normalization": normalization_compatible,
    }


def materialize_terratorch_dataset(
    hls_manifest: Path,
    eo4_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to merge into non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    hls_rows = _read_jsonl(hls_manifest)
    eo4_rows = _read_jsonl(eo4_manifest)
    hls_counts = _materialize_hls(hls_rows, hls_manifest.parent, output)
    eo4_counts, eo4_normalization = _materialize_eo4(
        eo4_rows,
        eo4_manifest.parent,
        output,
    )
    return {
        "schema_version": 1,
        "hls_split_counts": dict(sorted(hls_counts.items())),
        "eo4_split_counts": dict(sorted(eo4_counts.items())),
        "combined_split_counts": dict(sorted((hls_counts + eo4_counts).items())),
        "band_order": EXPECTED_HLS_BANDS,
        "mask_values": {"burned": 1, "not_burned": 0, "ignore": -1},
        "normalization": {
            "means": HLS_MEANS,
            "stds": HLS_STDS,
            "source": "Prithvi-EO-2.0-300M-BurnScars official HLS configuration",
            "eo4_audit": eo4_normalization,
        },
    }


def _validate_materialized_dataset(dataset_root: Path) -> dict[str, Any]:
    report_path = dataset_root / "materialization-report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            "Prithvi training requires materialization-report.json; run materialize first"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    eo4_audit = report.get("normalization", {}).get("eo4_audit", {})
    if eo4_audit.get("compatible_with_hls_normalization") is not True:
        raise RuntimeError(
            "EO4 reflectance statistics are incompatible with the frozen HLS normalization"
        )
    split_counts = report.get("combined_split_counts")
    if not isinstance(split_counts, dict):
        raise ValueError("Materialization report does not declare combined split counts")
    for split in ("train", "validation", "test"):
        split_file = dataset_root / "splits" / f"{split}.txt"
        expected = int(split_counts.get(split, 0))
        if not split_file.is_file() or expected <= 0:
            raise ValueError(f"Materialized Prithvi split is missing or empty: {split}")
        observed = sum(1 for line in split_file.read_text(encoding="utf-8").splitlines() if line)
        if observed != expected:
            raise ValueError(
                f"Materialized Prithvi split-count drift for {split}: {observed} != {expected}"
            )
    return {
        "report_sha256": _sha256_file(report_path),
        "split_counts": split_counts,
        "eo4_normalization_compatible": True,
    }


def build_terratorch_config(
    dataset_root: Path,
    output_root: Path,
    *,
    batch_size: int,
    workers: int,
    epochs: int,
    checkpoint_steps: int,
    smoke: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "seed_everything": 42,
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "precision": "bf16-mixed",
            "max_epochs": epochs,
            "log_every_n_steps": 5,
            "limit_val_batches": 0,
            "num_sanity_val_steps": 0,
            "default_root_dir": str(output_root.resolve()),
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "monitor": "step",
                        "mode": "max",
                        "save_top_k": 4,
                        "save_last": True,
                        "every_n_train_steps": checkpoint_steps,
                        "save_on_train_epoch_end": False,
                        "dirpath": str((output_root / "checkpoints").resolve()),
                    },
                },
            ],
        },
        "model": {
            "class_path": "terratorch.tasks.SemanticSegmentationTask",
            "init_args": {
                "model_factory": "EncoderDecoderFactory",
                "model_args": {
                    "backbone": "prithvi_eo_v2_300",
                    "backbone_pretrained": True,
                    "backbone_bands": EXPECTED_HLS_BANDS,
                    "necks": [
                        {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
                        {"name": "ReshapeTokensToImage"},
                        {"name": "LearnedInterpolateToPyramidal"},
                    ],
                    "decoder": "UNetDecoder",
                    "decoder_channels": [512, 256, 128, 64],
                    "num_classes": 2,
                },
                "loss": "ce",
                "ignore_index": -1,
                "freeze_backbone": False,
                "plot_on_val": False,
                "class_names": ["Not burned", "Burn scar"],
            },
        },
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 0.0001},
        },
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "init_args": {"T_max": epochs, "eta_min": 0.000001},
        },
        "data": {
            "class_path": (
                "fireviewer_model_lab.training.prithvi_optimized_datamodule.OptimizedGenericNonGeoSegmentationDataModule"
            ),
            "init_args": {
                "batch_size": batch_size,
                "num_workers": workers,
                "check_stackability": False,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
                "dataset_bands": EXPECTED_HLS_BANDS,
                "output_bands": EXPECTED_HLS_BANDS,
                "rgb_indices": [2, 1, 0],
                "train_data_root": str((dataset_root / "data").resolve()),
                "val_data_root": str((dataset_root / "data").resolve()),
                "test_data_root": str((dataset_root / "data").resolve()),
                "train_split": str((dataset_root / "splits" / "train.txt").resolve()),
                "val_split": str((dataset_root / "splits" / "validation.txt").resolve()),
                "test_split": str((dataset_root / "splits" / "test.txt").resolve()),
                "img_grep": "*_merged.tif",
                "label_grep": "*.mask.tif",
                "means": HLS_MEANS,
                "stds": HLS_STDS,
                "num_classes": 2,
                "train_transform": [
                    {
                        "class_path": "albumentations.PadIfNeeded",
                        "init_args": {
                            "min_height": 512,
                            "min_width": 512,
                            "border_mode": 0,
                            "fill": 0,
                            "fill_mask": -1,
                        },
                    },
                    {
                        "class_path": "albumentations.RandomCrop",
                        "init_args": {"height": 512, "width": 512},
                    },
                    {"class_path": "albumentations.D4"},
                    {"class_path": "albumentations.pytorch.ToTensorV2"},
                ],
                "val_transform": [
                    {
                        "class_path": "albumentations.PadIfNeeded",
                        "init_args": {
                            "min_height": 512,
                            "min_width": 512,
                            "border_mode": 0,
                            "fill": 0,
                            "fill_mask": -1,
                        },
                    },
                    {
                        "class_path": "albumentations.CenterCrop",
                        "init_args": {"height": 512, "width": 512},
                    },
                    {"class_path": "albumentations.pytorch.ToTensorV2"},
                ],
                "test_transform": [
                    {
                        "class_path": "albumentations.PadIfNeeded",
                        "init_args": {
                            "min_height": 512,
                            "min_width": 512,
                            "border_mode": 0,
                            "fill": 0,
                            "fill_mask": -1,
                        },
                    },
                    {
                        "class_path": "albumentations.CenterCrop",
                        "init_args": {"height": 512, "width": 512},
                    },
                    {"class_path": "albumentations.pytorch.ToTensorV2"},
                ],
                "no_data_replace": 0,
                "no_label_replace": -1,
            },
        },
    }
    if smoke:
        config["trainer"].update(
            {
                "max_epochs": 1,
                "limit_train_batches": 2,
                "limit_val_batches": 0,
                "limit_test_batches": 0,
                "num_sanity_val_steps": 0,
            }
        )
    return config


def build_full_evaluation_config(training_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(training_config)
    config["trainer"]["limit_val_batches"] = 1.0
    config["trainer"]["limit_test_batches"] = 1.0
    config["trainer"]["num_sanity_val_steps"] = 0
    config["trainer"]["callbacks"] = []
    return config


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to generate the TerraTorch config") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _require_a40_class_gpu() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for the Prithvi training gate") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Prithvi BurnScars training requires a CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if total_gib < 40:
        raise RuntimeError(
            f"Prithvi BurnScars requires at least 40 GiB VRAM, found {total_gib:.1f}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Prithvi BurnScars requires BF16 support")
    return {"device": properties.name, "vram_gib": round(total_gib, 2), "bf16": True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight, materialize and train Prithvi-EO-2.0 BurnScars"
    )
    parser.add_argument(
        "command",
        choices=("preflight", "materialize", "plan", "smoke", "train"),
    )
    parser.add_argument("--hls-manifest", type=Path)
    parser.add_argument("--eo4-manifest", type=Path)
    parser.add_argument(
        "--geographic-test-report",
        type=Path,
        help=(
            "Validated independent geographic test report. Defaults to "
            "<dataset-root>/../geographic-critical-test/report.json."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Checkpoint file to resume, or 'auto' to use "
            "<output>/checkpoints/last.ckpt. Auto starts fresh when none exists."
        ),
    )
    parser.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-training", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0 or args.checkpoint_steps <= 0:
        raise ValueError("batch-size, epochs and checkpoint-steps must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")

    if (args.hls_manifest is None) != (args.eo4_manifest is None):
        raise ValueError("hls-manifest and eo4-manifest must be provided together")
    if args.command == "materialize" and args.hls_manifest is None:
        raise ValueError("materialize requires hls-manifest and eo4-manifest")
    geographic_test_report = args.geographic_test_report or (
        args.dataset_root.parent / "geographic-critical-test" / "report.json"
    )
    report = (
        build_preflight_report(
            args.hls_manifest,
            args.eo4_manifest,
            verify_files=args.verify_files,
            geographic_test_report=geographic_test_report,
        )
        if args.hls_manifest is not None and args.eo4_manifest is not None
        else build_materialized_preflight_report(
            args.dataset_root,
            geographic_test_report=geographic_test_report,
            verify_files=args.verify_files,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "preflight":
        if not report["training_ready"]:
            raise SystemExit(2)
        return

    if args.command == "materialize":
        if report["dataset_errors"]:
            raise RuntimeError("Prithvi materialization dataset gate failed")
        staging = args.dataset_root.with_name(args.dataset_root.name + ".partial")
        if args.dataset_root.exists():
            raise FileExistsError(args.dataset_root)
        if staging.exists():
            raise FileExistsError(
                f"Interrupted materialization must be reviewed before retry: {staging}"
            )
        assert args.hls_manifest is not None
        assert args.eo4_manifest is not None
        materialization = materialize_terratorch_dataset(
            args.hls_manifest, args.eo4_manifest, staging
        )
        (staging / "materialization-report.json").write_text(
            json.dumps(materialization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, args.dataset_root)
        print(json.dumps(materialization, ensure_ascii=False, sort_keys=True))
        return
    if not report["training_ready"]:
        raise RuntimeError("Prithvi training gate failed")

    smoke = args.command == "smoke"
    config = build_terratorch_config(
        args.dataset_root,
        args.output,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        checkpoint_steps=args.checkpoint_steps,
        smoke=smoke,
    )
    config_path = args.output / (
        "prithvi-burnscars-smoke.yaml" if smoke else "prithvi-burnscars.yaml"
    )
    _write_yaml(config_path, config)
    evaluation_config_path = args.output / "prithvi-burnscars-eval.yaml"
    if not smoke:
        _write_yaml(evaluation_config_path, build_full_evaluation_config(config))
    benchmark_contract = {
        "schema_version": 1,
        "primary": {
            "model_id": PRITHVI_MODEL,
            "revision": PRITHVI_MODEL_REVISION,
        },
        "challenger": {
            "model_id": TERRAMIND_MODEL,
            "revision": TERRAMIND_MODEL_REVISION,
            "role": "benchmark_only",
            "required_modalities": ["S2L2A", "S1RTC", "DEM"],
        },
        "same_geographic_test_required": True,
        "promotion_blocked_until": ["independent_geographic_critical_test_missing"],
    }
    (args.output / "burnscar-benchmark-contract.json").write_text(
        json.dumps(benchmark_contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.command == "plan":
        print(
            json.dumps(
                {"config": str(config_path.resolve()), **benchmark_contract},
                sort_keys=True,
            )
        )
        return
    if not args.confirm_training:
        raise RuntimeError(f"Refusing Prithvi {args.command} without --confirm-training")
    materialized = _validate_materialized_dataset(args.dataset_root)
    gpu = _require_a40_class_gpu()
    executable = _resolve_terratorch_executable()
    if executable is None:
        raise RuntimeError(
            "terratorch executable is missing; install the burned-area-training extra"
        )
    resume_checkpoint = _resolve_resume_checkpoint(
        args.output,
        args.resume_from_checkpoint,
    )
    provenance = {
        "schema_version": 1,
        "model_id": PRITHVI_MODEL,
        "model_revision": PRITHVI_MODEL_REVISION,
        "config_sha256": _sha256_file(config_path),
        "hls_manifest_sha256": (
            _sha256_file(args.hls_manifest) if args.hls_manifest is not None else None
        ),
        "eo4_manifest_sha256": (
            _sha256_file(args.eo4_manifest) if args.eo4_manifest is not None else None
        ),
        "materialized_dataset": materialized,
        "gpu": gpu,
        "run_mode": args.command,
        "checkpoint_steps": args.checkpoint_steps,
        "resume_from_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "evaluation_policy": (
            "bounded_smoke_during_fit" if smoke else "full_validation_test_once_after_training"
        ),
    }
    (args.output / f"{args.command}-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fit_command = [executable, "fit", "-c", str(config_path)]
    if resume_checkpoint is not None:
        fit_command.extend(["--ckpt_path", str(resume_checkpoint)])
    subprocess.run(fit_command, check=True)  # noqa: S603
    if not smoke:
        checkpoint = args.output / "checkpoints" / "last.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Prithvi final checkpoint is missing after training: {checkpoint}"
            )
        for command in ("validate", "test"):
            subprocess.run(  # noqa: S603
                [
                    executable,
                    command,
                    "-c",
                    str(evaluation_config_path),
                    "--ckpt_path",
                    str(checkpoint),
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
