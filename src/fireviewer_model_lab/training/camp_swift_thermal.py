"""Build EO fire masks from the georeferenced Camp Swift infrared campaign.

The output materializes only the selected EO images and deterministic derived
labels. DINOv3 and SegFormer share this single canonical payload.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SOURCE_REVISION = "RDS-2018-0046+RDS-2018-0047"
BLOCK_SPLITS = {1: "validation", 2: "train", 3: "test"}
TIMESTAMP_RE = re.compile(r"_(\d{9})(?:_modified|_?rect)?$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    root = root.resolve()
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escapes campaign root: {path}")
    return path.relative_to(root).as_posix()


def _timestamp(subdataset: str) -> int:
    match = TIMESTAMP_RE.search(subdataset)
    if not match:
        raise ValueError(f"Camp Swift layer has no video timestamp: {subdataset}")
    return int(match.group(1))


def _layer_name(subdataset: str) -> str:
    return subdataset.rsplit(":", 1)[-1]


def pair_nearest_layers(
    eo_layers: list[str], ir_layers: list[str], max_delta_ms: int
) -> list[tuple[str, str, int]]:
    """Create a deterministic one-to-one nearest-timestamp pairing."""

    eo = sorted(((_timestamp(value), value) for value in eo_layers), key=lambda item: item[0])
    ir = sorted(((_timestamp(value), value) for value in ir_layers), key=lambda item: item[0])
    ir_times = [item[0] for item in ir]
    candidates: list[tuple[int, int, int]] = []
    for eo_index, (eo_time, _) in enumerate(eo):
        insertion = bisect.bisect_left(ir_times, eo_time)
        left = insertion - 1
        while left >= 0 and eo_time - ir_times[left] <= max_delta_ms:
            candidates.append((abs(eo_time - ir_times[left]), eo_index, left))
            left -= 1
        right = insertion
        while right < len(ir) and ir_times[right] - eo_time <= max_delta_ms:
            candidates.append((abs(eo_time - ir_times[right]), eo_index, right))
            right += 1
    used_eo: set[int] = set()
    used_ir: set[int] = set()
    selected: list[tuple[str, str, int]] = []
    for delta, eo_index, ir_index in sorted(candidates):
        if eo_index in used_eo or ir_index in used_ir:
            continue
        used_eo.add(eo_index)
        used_ir.add(ir_index)
        selected.append((eo[eo_index][1], ir[ir_index][1], delta))
    return sorted(selected, key=lambda item: _timestamp(item[0]))


def thermal_hot_mask(
    ir_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    red_threshold: int,
    minimum_component_pixels: int,
) -> np.ndarray:
    """Extract hot fire cores from the Camp Swift cyan-to-white IR palette."""

    if ir_rgb.shape[0] < 3 or valid_mask.shape != ir_rgb.shape[1:]:
        raise ValueError("invalid Camp Swift IR raster shape")
    hot = ((ir_rgb[0] >= red_threshold) & (valid_mask > 0)).astype(np.uint8)
    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    kept = np.zeros_like(hot)
    for component in range(1, component_count):
        if int(statistics[component, cv2.CC_STAT_AREA]) >= minimum_component_pixels:
            kept[labels == component] = 255
    return kept


def _save_image(path: Path, array: np.ndarray, *, jpeg_quality: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    kwargs: dict[str, Any] = {}
    if jpeg_quality is not None:
        kwargs.update(quality=jpeg_quality, optimize=True, subsampling=0)
    Image.fromarray(array).save(temporary, **kwargs)
    os.replace(temporary, path)


def _anchor(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask > 0)
    if not len(xs):
        return None
    height, width = mask.shape
    return float(xs.mean()) / width, float(ys.mean()) / height


def _subdatasets(path: Path) -> list[str]:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(str(path)) as container:
            return list(container.subdatasets)


def _prepare_selected_pair(
    *,
    campaign_root: Path,
    output_root: Path,
    block: int,
    split: str,
    pair: tuple[str, str, int],
    red_threshold: int,
    minimum_component_pixels: int,
) -> dict[str, Any]:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    eo_layer, ir_layer, delta_ms = pair
    with rasterio.open(eo_layer) as eo, rasterio.open(ir_layer) as ir:
        eo_rgb = np.clip(eo.read([1, 2, 3]), 0, 255).astype(np.uint8)
        eo_valid = (eo.dataset_mask() > 0).astype(np.uint8) * 255
        ir_rgb = np.clip(ir.read([1, 2, 3]), 0, 255).astype(np.uint8)
        ir_valid = ir.dataset_mask()
        ir_hot = thermal_hot_mask(
            ir_rgb,
            ir_valid,
            red_threshold=red_threshold,
            minimum_component_pixels=minimum_component_pixels,
        )
        projected = np.zeros((eo.height, eo.width), dtype=np.uint8)
        reproject(
            source=ir_hot,
            destination=projected,
            src_transform=ir.transform,
            src_crs=ir.crs,
            dst_transform=eo.transform,
            dst_crs=eo.crs,
            resampling=Resampling.nearest,
        )
        projected = np.where(eo_valid > 0, projected, 0).astype(np.uint8)
    sample_stem = f"block-{block}-eo-{_timestamp(eo_layer):09d}-ir-{_timestamp(ir_layer):09d}"
    image_path = output_root / "images" / f"block-{block}" / f"{sample_stem}.jpg"
    mask_path = output_root / "masks" / f"block-{block}" / f"{sample_stem}.png"
    valid_path = output_root / "valid-masks" / f"block-{block}" / f"{sample_stem}.png"
    image_array = np.moveaxis(eo_rgb, 0, -1)
    image_array[eo_valid == 0] = 0
    _save_image(image_path, image_array, jpeg_quality=92)
    _save_image(mask_path, projected)
    _save_image(valid_path, eo_valid)
    anchor = _anchor(projected)
    return {
        "sample_id": f"camp-swift:{sample_stem}",
        "source_id": "Camp Swift Fire Experiment 2014",
        "source_revision": SOURCE_REVISION,
        "split": split,
        "split_group": f"camp-swift:burn-block-{block}",
        "image_relpath": _relative(campaign_root, image_path),
        "image_sha256": _sha256(image_path),
        "mask_relpath": _relative(campaign_root, mask_path),
        "mask_sha256": _sha256(mask_path),
        "valid_mask_relpath": _relative(campaign_root, valid_path),
        "valid_mask_sha256": _sha256(valid_path),
        "mask_quality": "sensor_derived_thermal_reprojection",
        "mask_semantics": "thermal_hot_fire_core",
        "annotation_strength": "strong",
        "sample_validation_status": "sensor_derived",
        "anchor_points": (
            [{"kind": "fire_centroid", "x": anchor[0], "y": anchor[1]}]
            if anchor is not None
            else []
        ),
        "visual_abstention_reason": (
            None if anchor is not None else "no_thermal_fire_core_in_overlap"
        ),
        "license": "CC-BY-4.0",
        "redistribution_allowed": True,
        "is_operational_incident": False,
        "eo_layer": _layer_name(eo_layer),
        "ir_layer": _layer_name(ir_layer),
        "eo_timestamp_ms": _timestamp(eo_layer),
        "ir_timestamp_ms": _timestamp(ir_layer),
        "pair_delta_ms": delta_ms,
    }


def _pair_has_georeference(pair: tuple[str, str, int]) -> bool:
    import rasterio

    eo_layer, ir_layer, _ = pair
    with rasterio.open(eo_layer) as eo, rasterio.open(ir_layer) as ir:
        return (
            eo.crs is not None
            and ir.crs is not None
            and not eo.transform.is_identity
            and not ir.transform.is_identity
        )


def build_camp_swift_thermal_corpus(
    *,
    campaign_root: Path,
    geodatabase_root: Path,
    output_root: Path,
    frame_stride: int = 3,
    max_delta_ms: int = 2000,
    red_threshold: int = 40,
    minimum_component_pixels: int = 20,
    jobs: int = 8,
) -> dict[str, Any]:

    if frame_stride <= 0 or max_delta_ms < 0 or minimum_component_pixels <= 0 or jobs <= 0:
        raise ValueError("invalid Camp Swift preparation parameters")
    campaign_root = campaign_root.resolve()
    geodatabase_root = geodatabase_root.resolve()
    output_root = output_root.resolve()
    _relative(campaign_root, geodatabase_root)
    _relative(campaign_root, output_root)
    rows: list[dict[str, Any]] = []
    pairing_counts: dict[str, int] = {}
    skipped_missing_georeference: Counter[str] = Counter()
    for block, split in BLOCK_SPLITS.items():
        eo_layers = _subdatasets(geodatabase_root / f"BurnBlock{block}EO.gdb")
        ir_layers = _subdatasets(geodatabase_root / f"BurnBlock{block}IR.gdb")
        pairs = pair_nearest_layers(eo_layers, ir_layers, max_delta_ms)
        pairing_counts[f"block_{block}_within_delta"] = len(pairs)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            georeference_flags = list(executor.map(_pair_has_georeference, pairs))
        georeferenced_pairs = [
            pair for pair, is_valid in zip(pairs, georeference_flags, strict=True) if is_valid
        ]
        skipped_missing_georeference[f"block_{block}"] += len(pairs) - len(georeferenced_pairs)
        pairing_counts[f"block_{block}_georeferenced"] = len(georeferenced_pairs)
        selected = georeferenced_pairs[::frame_stride]
        pairing_counts[f"block_{block}_selected"] = len(selected)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            rows.extend(
                executor.map(
                    lambda pair, block=block, split=split: _prepare_selected_pair(
                        campaign_root=campaign_root,
                        output_root=output_root,
                        block=block,
                        split=split,
                        pair=pair,
                        red_threshold=red_threshold,
                        minimum_component_pixels=minimum_component_pixels,
                    ),
                    selected,
                )
            )
    rows.sort(key=lambda row: str(row["sample_id"]))
    manifest = output_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".jsonl.partial")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, manifest)
    split_counts = Counter(str(row["split"]) for row in rows)
    positive_rows = sum(bool(row["anchor_points"]) for row in rows)
    report = {
        "schema_version": 1,
        "dataset_family": "camp-swift-eo-ir-thermal-v1",
        "source_revision": SOURCE_REVISION,
        "manifest": _relative(campaign_root, manifest),
        "manifest_sha256": _sha256(manifest),
        "rows": len(rows),
        "positive_rows": positive_rows,
        "abstention_rows": len(rows) - positive_rows,
        "split_counts": dict(sorted(split_counts.items())),
        "pairing_counts": pairing_counts,
        "skipped_missing_georeference": dict(sorted(skipped_missing_georeference.items())),
        "parameters": {
            "frame_stride": frame_stride,
            "max_delta_ms": max_delta_ms,
            "red_threshold": red_threshold,
            "minimum_component_pixels": minimum_component_pixels,
            "jobs": jobs,
        },
        "consumers": ["dinov3_multitask", "segformer_baseline"],
    }
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
