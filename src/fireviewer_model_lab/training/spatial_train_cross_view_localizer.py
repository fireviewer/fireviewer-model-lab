"""Fine-tune the FireWarning coarse cross-view adapter on the selective JustZoomIn corpus.

This stage narrows a known incident locality from roughly ten kilometres to one final
4x4 cell at each of four zoom steps.  It does not replace RoMa, PnP or the MNT raycast:
those geometry stages consume the narrowed search window only after this model passes its
held-out distance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from PIL import Image
from pyproj import Geod, Transformer
from torch import nn

from fireviewer_model_lab.training.cross_view_contract import (
    CORPUS_ID,
    DATASET_LICENSE,
    REQUIRED_SATELLITE_LEVELS,
)
from fireviewer_model_lab.training.spatial_training_setup import (
    SetupError,
    _deny_operational_path,
    _require_production_license,
    _write_json,
)

MODEL_ID = "facebook/dinov2-base"
MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
MODEL_LICENSE = "Apache-2.0"
RUN_ID = "cross-view-coarse-localizer-dinov2-v0.2.0"
PROJECTED_CRS = "EPSG:26985"
IMAGE_SIZE = 224
GRID_SIZE = 4
STEPS = 4
EMBEDDING_DIMENSION = 768
CHECKPOINT_MILESTONES = (50, 60, 70, 80, 90, 100)
STATE_OFFSETS = (0, 1, 17, 273)
TOTAL_STATES = 4_369
UPSTREAM_GEOGRAPHIC_CENTER_LATLON = (38.8936, -77.0116)
UPSTREAM_REGION_BOUNDS_METERS = (-3_000.0, 7_000.0, -5_000.0, 5_000.0)
DEFAULT_TILE_RGB = (0, 0, 0)
DEFAULT_REGION_ID = "washington-dc-benchmark"
DEFAULT_SATELLITE_LAYOUT_RELPATH = "sources/justzoomin-selective/extracted/satellite/layout.yaml"
REGIONS_REGISTRY_FILENAME = "regions.yaml"
_REGION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CRS_PATTERN = re.compile(r"^EPSG:\d{4,5}$")


@dataclass(frozen=True)
class RegionSpec:
    """Fixed geographic contract of one coarse-localization region.

    Every region owns an independent 4x4^STEPS action grid, its own projected CRS
    and its own satellite tile pyramid.  ``bounds_meters`` is the signed offset box
    (min_east, max_east, min_north, max_north) around ``center_*`` and must project
    to a square.
    """

    region_id: str
    crs: str
    center_latitude: float
    center_longitude: float
    bounds_meters: tuple[float, float, float, float]
    satellite_layout_relpath: str


DEFAULT_REGION_SPEC = RegionSpec(
    region_id=DEFAULT_REGION_ID,
    crs=PROJECTED_CRS,
    center_latitude=UPSTREAM_GEOGRAPHIC_CENTER_LATLON[0],
    center_longitude=UPSTREAM_GEOGRAPHIC_CENTER_LATLON[1],
    bounds_meters=UPSTREAM_REGION_BOUNDS_METERS,
    satellite_layout_relpath=DEFAULT_SATELLITE_LAYOUT_RELPATH,
)


@dataclass(frozen=True)
class CorpusRow:
    sample_id: str
    image_path: Path
    latitude: float
    longitude: float
    actions: tuple[int, int, int, int]
    region_id: str = DEFAULT_REGION_ID


@dataclass(frozen=True)
class RegionCalibration:
    west: float
    east: float
    south: float
    north: float
    crs: str
    train_cell_agreement: float
    train_within_one_cell_agreement: float
    maximum_cell_error: int

    @property
    def width(self) -> float:
        return self.east - self.west

    @property
    def height(self) -> float:
        return self.north - self.south


@dataclass(frozen=True)
class RenderedTileWindow:
    image: Image.Image
    valid_cells: np.ndarray
    present_tiles: int
    required_tiles: int


def _manifest_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / "corpus" / CORPUS_ID / f"{split}.jsonl"


def region_registry_path(dataset_root: Path) -> Path:
    return dataset_root / "corpus" / CORPUS_ID / REGIONS_REGISTRY_FILENAME


def validate_region_spec(spec: RegionSpec) -> RegionSpec:
    if not _REGION_ID_PATTERN.match(spec.region_id):
        raise SetupError(f"invalid region id: {spec.region_id!r}")
    if not _CRS_PATTERN.match(spec.crs.upper()):
        raise SetupError(f"invalid region CRS: {spec.crs!r}")
    bounds = spec.bounds_meters
    if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
        raise SetupError(f"invalid region bounds: {spec.region_id}={bounds!r}")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise SetupError(f"inverted region bounds: {spec.region_id}={bounds!r}")
    if not (
        math.isfinite(spec.center_latitude)
        and math.isfinite(spec.center_longitude)
        and -90.0 <= spec.center_latitude <= 90.0
        and -180.0 <= spec.center_longitude <= 180.0
    ):
        raise SetupError(f"invalid region center: {spec.region_id}")
    layout = Path(spec.satellite_layout_relpath)
    if layout.is_absolute() or any(part == ".." for part in layout.parts):
        raise SetupError(f"region satellite layout escapes dataset root: {spec.region_id}")
    _deny_operational_path(spec.satellite_layout_relpath)
    region_bounds(spec)
    return spec


def region_bounds(spec: RegionSpec) -> tuple[float, float, float, float]:
    """Project the fixed region declared by a region specification."""

    minimum_east, maximum_east, minimum_north, maximum_north = spec.bounds_meters
    center_east = (minimum_east + maximum_east) / 2.0
    center_north = (minimum_north + maximum_north) / 2.0
    geod = Geod(ellps="WGS84")
    longitude, latitude, _ = geod.fwd(
        spec.center_longitude, spec.center_latitude, 90.0, center_east
    )
    longitude, latitude, _ = geod.fwd(longitude, latitude, 0.0, center_north)
    center_x, center_y = Transformer.from_crs("EPSG:4326", spec.crs, always_xy=True).transform(
        longitude, latitude
    )
    width = maximum_east - minimum_east
    height = maximum_north - minimum_north
    if not math.isclose(width, height):
        raise SetupError(f"region is not square: {spec.region_id}={width}x{height}m")
    return (
        center_x - width / 2.0,
        center_x + width / 2.0,
        center_y - height / 2.0,
        center_y + height / 2.0,
    )


def load_region_registry(dataset_root: Path) -> tuple[RegionSpec, ...]:
    """Load the region registry, falling back to the single DC bootstrap region."""

    path = region_registry_path(dataset_root)
    if not path.is_file():
        return (DEFAULT_REGION_SPEC,)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SetupError("PyYAML is required to read the region registry") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("regions"), list):
        raise SetupError(f"invalid region registry: {path}")
    specs: list[RegionSpec] = []
    seen: set[str] = set()
    for entry in raw["regions"]:
        if not isinstance(entry, dict):
            raise SetupError(f"invalid region registry entry: {entry!r}")
        bounds_raw = entry.get("bounds_meters")
        if not isinstance(bounds_raw, list) or len(bounds_raw) != 4:
            raise SetupError(f"invalid region bounds entry: {entry!r}")
        spec = RegionSpec(
            region_id=str(entry.get("region_id", "")),
            crs=str(entry.get("crs", "")).upper(),
            center_latitude=float(entry.get("center_latitude")),
            center_longitude=float(entry.get("center_longitude")),
            bounds_meters=tuple(float(value) for value in bounds_raw),
            satellite_layout_relpath=str(entry.get("satellite_layout", "")),
        )
        validate_region_spec(spec)
        if spec.region_id in seen:
            raise SetupError(f"duplicate region id: {spec.region_id}")
        seen.add(spec.region_id)
        specs.append(spec)
    if not specs:
        raise SetupError(f"region registry declares no region: {path}")
    return tuple(specs)


def _region_by_id(registry: tuple[RegionSpec, ...]) -> dict[str, RegionSpec]:
    return {spec.region_id: spec for spec in registry}


def _load_rows(
    dataset_root: Path,
    split: str,
    registry: tuple[RegionSpec, ...] = (DEFAULT_REGION_SPEC,),
) -> list[CorpusRow]:
    known_regions = _region_by_id(registry)
    manifest = _manifest_path(dataset_root, split)
    if not manifest.is_file():
        raise SetupError(f"missing coarse-localizer manifest: {manifest}")
    rows: list[CorpusRow] = []
    sample_ids: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SetupError(f"invalid JSONL at {manifest}:{line_number}") from exc
            if raw.get("operational_incident") is not False:
                raise SetupError(f"operational incident denied at {manifest}:{line_number}")
            if raw.get("training_membership") is not True:
                raise SetupError(f"non-training row denied at {manifest}:{line_number}")
            _require_production_license(str(raw["license"]), source_id=str(raw["source_id"]))
            sample_id = str(raw["sample_id"])
            if sample_id in sample_ids:
                raise SetupError(f"duplicate coarse-localizer sample: {sample_id}")
            sample_ids.add(sample_id)
            region_id = str(raw.get("region_id") or DEFAULT_REGION_ID)
            if region_id not in known_regions:
                raise SetupError(f"row references unknown region: {sample_id}={region_id}")
            source_view_relpath = Path(str(raw["source_view_relpath"]))
            if source_view_relpath.is_absolute():
                raise SetupError(f"source image path must be relative: {sample_id}")
            if any(part == ".." for part in source_view_relpath.parts):
                raise SetupError(f"source image escapes dataset root: {sample_id}")
            image_path = dataset_root / source_view_relpath
            if not image_path.is_file():
                raise SetupError(f"missing source image: {image_path}")
            actions_raw = raw.get("action_sequence")
            if not isinstance(actions_raw, list) or len(actions_raw) != STEPS:
                raise SetupError(f"invalid action sequence: {sample_id}")
            actions = tuple(int(action) for action in actions_raw)
            if any(action < 0 or action >= GRID_SIZE**2 for action in actions):
                raise SetupError(f"action outside 4x4 grid: {sample_id}")
            rows.append(
                CorpusRow(
                    sample_id=sample_id,
                    image_path=image_path,
                    latitude=float(raw["latitude"]),
                    longitude=float(raw["longitude"]),
                    actions=(actions[0], actions[1], actions[2], actions[3]),
                    region_id=region_id,
                )
            )
    if not rows:
        raise SetupError(f"empty coarse-localizer split: {split}")
    return rows


def _final_grid_index(actions: tuple[int, int, int, int]) -> tuple[int, int]:
    row_index = 0
    column_index = 0
    for action in actions:
        action_row, action_column = divmod(action, GRID_SIZE)
        row_index = row_index * GRID_SIZE + action_row
        column_index = column_index * GRID_SIZE + action_column
    return row_index, column_index


def _project_rows(rows: list[CorpusRow], crs: str = PROJECTED_CRS) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected = [transformer.transform(row.longitude, row.latitude) for row in rows]
    return np.asarray(projected, dtype=np.float64)


def _project_rows_by_region(
    rows: list[CorpusRow], calibrations: dict[str, RegionCalibration]
) -> np.ndarray:
    transformers = {
        region_id: Transformer.from_crs("EPSG:4326", calibration.crs, always_xy=True)
        for region_id, calibration in calibrations.items()
    }
    projected = np.empty((len(rows), 2), dtype=np.float64)
    for index, row in enumerate(rows):
        projected[index] = transformers[row.region_id].transform(row.longitude, row.latitude)
    return projected


def _sequence_geometry_report(
    rows: list[CorpusRow],
    *,
    west: float,
    east: float,
    south: float,
    north: float,
    crs: str = PROJECTED_CRS,
) -> dict[str, float | int]:
    projected = _project_rows(rows, crs)
    expected = np.asarray([_final_grid_index(row.actions) for row in rows], dtype=np.int64)
    cell_count = GRID_SIZE**STEPS
    columns = np.clip(
        np.floor((projected[:, 0] - west) / (east - west) * cell_count),
        0,
        cell_count - 1,
    ).astype(np.int64)
    grid_rows = np.clip(
        np.floor((north - projected[:, 1]) / (north - south) * cell_count),
        0,
        cell_count - 1,
    ).astype(np.int64)
    errors = np.column_stack([grid_rows, columns]) - expected
    absolute_errors = np.abs(errors)
    return {
        "rows": len(rows),
        "cell_agreement": float(np.all(errors == 0, axis=1).mean()),
        "within_one_cell_agreement": float(np.all(absolute_errors <= 1, axis=1).mean()),
        "maximum_cell_error": int(absolute_errors.max()),
    }


def derive_region_calibration(
    rows: list[CorpusRow], region: RegionSpec = DEFAULT_REGION_SPEC
) -> RegionCalibration:
    """Load and verify one region's fixed, source-declared region contract."""

    west, east, south, north = region_bounds(region)
    report = _sequence_geometry_report(
        rows, west=west, east=east, south=south, north=north, crs=region.crs
    )
    if report["within_one_cell_agreement"] < 0.999 or report["maximum_cell_error"] > 1:
        raise SetupError(
            f"action sequences violate the fixed geographic region {region.region_id}: {report}"
        )
    return RegionCalibration(
        west=west,
        east=east,
        south=south,
        north=north,
        crs=region.crs,
        train_cell_agreement=float(report["cell_agreement"]),
        train_within_one_cell_agreement=float(report["within_one_cell_agreement"]),
        maximum_cell_error=int(report["maximum_cell_error"]),
    )


def derive_region_calibrations(
    rows: list[CorpusRow], registry: tuple[RegionSpec, ...]
) -> dict[str, RegionCalibration]:
    """Derive one calibration per region referenced by the given rows."""

    referenced = {row.region_id for row in rows}
    calibrations: dict[str, RegionCalibration] = {}
    for spec in registry:
        region_rows = [row for row in rows if row.region_id == spec.region_id]
        if not region_rows:
            continue
        calibrations[spec.region_id] = derive_region_calibration(region_rows, spec)
    missing = referenced - set(calibrations)
    if missing:
        raise SetupError(f"regions without training rows: {sorted(missing)}")
    return calibrations


def _validate_rows_against_region(
    rows: list[CorpusRow], calibration: RegionCalibration
) -> dict[str, float | int]:
    report = _sequence_geometry_report(
        rows,
        west=calibration.west,
        east=calibration.east,
        south=calibration.south,
        north=calibration.north,
        crs=calibration.crs,
    )
    if report["within_one_cell_agreement"] < 0.999 or report["maximum_cell_error"] > 1:
        raise SetupError(f"validation split violates region contract: {report}")
    return report


def _validate_rows_against_regions(
    rows: list[CorpusRow], calibrations: dict[str, RegionCalibration]
) -> dict[str, dict[str, float | int]]:
    reports: dict[str, dict[str, float | int]] = {}
    for region_id, calibration in calibrations.items():
        region_rows = [row for row in rows if row.region_id == region_id]
        if not region_rows:
            continue
        reports[region_id] = _validate_rows_against_region(region_rows, calibration)
    unchecked = {row.region_id for row in rows} - set(reports)
    if unchecked:
        raise SetupError(f"validation rows without region calibration: {sorted(unchecked)}")
    return reports


def _decode_prefix(step: int, prefix: int) -> list[int]:
    actions = [0] * step
    value = prefix
    for index in range(step - 1, -1, -1):
        actions[index] = value % (GRID_SIZE**2)
        value //= GRID_SIZE**2
    if value:
        raise SetupError(f"state prefix exceeds step capacity: step={step} prefix={prefix}")
    return actions


def state_bounds(
    calibration: RegionCalibration, step: int, prefix: int
) -> tuple[float, float, float, float]:
    west, east, south, north = (
        calibration.west,
        calibration.east,
        calibration.south,
        calibration.north,
    )
    for action in _decode_prefix(step, prefix):
        action_row, action_column = divmod(action, GRID_SIZE)
        cell_width = (east - west) / GRID_SIZE
        cell_height = (north - south) / GRID_SIZE
        west = west + action_column * cell_width
        east = west + cell_width
        north = north - action_row * cell_height
        south = north - cell_height
    return west, east, south, north


class TilePyramidRenderer:
    def __init__(self, layout_path: Path, expected_crs: str = PROJECTED_CRS) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise SetupError("PyYAML is required to read the satellite layout") from exc
        layout = yaml.safe_load(layout_path.read_text(encoding="utf-8"))
        if str(layout.get("crs", "")).upper() != expected_crs.upper():
            raise SetupError(
                f"unsupported satellite CRS: {layout.get('crs')} expected={expected_crs}"
            )
        if layout.get("tile_axes") != ["east", "north"]:
            raise SetupError(f"unsupported satellite axes: {layout.get('tile_axes')}")
        self.root = layout_path.parent
        self.origin_east = float(layout["origin_crs"][0])
        self.origin_north = float(layout["origin_crs"][1])
        self.base_width = float(layout["tile_shape_crs"][0])
        self.base_height = float(layout["tile_shape_crs"][1])
        self.pixel_width = int(layout["tile_shape_px"][0])
        self.pixel_height = int(layout["tile_shape_px"][1])
        self.path_template = str(layout["path"])
        probe = Path(self.path_template.format(zoom=0, x=0, y=0))
        if probe.is_absolute() or any(part == ".." for part in probe.parts):
            raise SetupError(f"unsafe satellite tile path template: {self.path_template}")

    def _tile_path(self, zoom: int, x: int, y: int) -> Path:
        return self.root / self.path_template.format(zoom=zoom, x=x, y=y)

    def _tile_range(
        self, zoom: int, bounds: tuple[float, float, float, float]
    ) -> tuple[int, int, int, int, float, float]:
        west, east, south, north = bounds
        scale = 2.0 ** (-zoom)
        tile_width = self.base_width * scale
        tile_height = self.base_height * scale
        epsilon = 1e-8
        minimum_x = math.floor((west - self.origin_east) / tile_width)
        maximum_x = math.floor((east - epsilon - self.origin_east) / tile_width)
        minimum_y = math.floor((south - self.origin_north) / tile_height)
        maximum_y = math.floor((north - epsilon - self.origin_north) / tile_height)
        columns = maximum_x - minimum_x + 1
        rows = maximum_y - minimum_y + 1
        if columns < 1 or rows < 1 or columns > 4 or rows > 4:
            raise SetupError(
                f"invalid satellite mosaic at zoom {zoom}: columns={columns} rows={rows}"
            )
        return minimum_x, maximum_x, minimum_y, maximum_y, tile_width, tile_height

    def required_tile_paths(
        self, zoom: int, bounds: tuple[float, float, float, float]
    ) -> list[Path]:
        minimum_x, maximum_x, minimum_y, maximum_y, _, _ = self._tile_range(zoom, bounds)
        return [
            self._tile_path(zoom, tile_x, tile_y)
            for tile_y in range(maximum_y, minimum_y - 1, -1)
            for tile_x in range(minimum_x, maximum_x + 1)
        ]

    def valid_cells(self, zoom: int, bounds: tuple[float, float, float, float]) -> np.ndarray:
        west, east, south, north = bounds
        (
            minimum_x,
            maximum_x,
            minimum_y,
            maximum_y,
            tile_width,
            tile_height,
        ) = self._tile_range(zoom, bounds)
        columns = maximum_x - minimum_x + 1
        rows = maximum_y - minimum_y + 1
        coverage = Image.new("L", (columns * self.pixel_width, rows * self.pixel_height), 0)
        for tile_y in range(maximum_y, minimum_y - 1, -1):
            for tile_x in range(minimum_x, maximum_x + 1):
                if not self._tile_path(zoom, tile_x, tile_y).is_file():
                    continue
                column = tile_x - minimum_x
                row = maximum_y - tile_y
                offset = (column * self.pixel_width, row * self.pixel_height)
                coverage.paste(
                    255,
                    (
                        *offset,
                        offset[0] + self.pixel_width,
                        offset[1] + self.pixel_height,
                    ),
                )
        mosaic_west = self.origin_east + minimum_x * tile_width
        mosaic_north = self.origin_north + (maximum_y + 1) * tile_height
        pixels_per_east = self.pixel_width / tile_width
        pixels_per_north = self.pixel_height / tile_height
        crop = (
            (west - mosaic_west) * pixels_per_east,
            (mosaic_north - north) * pixels_per_north,
            (east - mosaic_west) * pixels_per_east,
            (mosaic_north - south) * pixels_per_north,
        )
        coverage_array = np.asarray(
            coverage.crop(crop).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        )
        cell_size = IMAGE_SIZE // GRID_SIZE
        return (
            coverage_array.reshape(GRID_SIZE, cell_size, GRID_SIZE, cell_size)
            .transpose(0, 2, 1, 3)
            .any(axis=(2, 3))
            .reshape(GRID_SIZE**2)
        )

    def render(self, zoom: int, bounds: tuple[float, float, float, float]) -> Image.Image:
        return self.render_with_coverage(zoom, bounds).image

    def render_with_coverage(
        self, zoom: int, bounds: tuple[float, float, float, float]
    ) -> RenderedTileWindow:
        west, east, south, north = bounds
        (
            minimum_x,
            maximum_x,
            minimum_y,
            maximum_y,
            tile_width,
            tile_height,
        ) = self._tile_range(zoom, bounds)
        columns = maximum_x - minimum_x + 1
        rows = maximum_y - minimum_y + 1
        mosaic_size = (columns * self.pixel_width, rows * self.pixel_height)
        mosaic = Image.new("RGB", mosaic_size, DEFAULT_TILE_RGB)
        coverage = Image.new("L", mosaic_size, 0)
        present_tiles = 0
        for tile_y in range(maximum_y, minimum_y - 1, -1):
            for tile_x in range(minimum_x, maximum_x + 1):
                path = self._tile_path(zoom, tile_x, tile_y)
                if not path.is_file():
                    continue
                with Image.open(path) as tile:
                    tile_rgb = tile.convert("RGB")
                    column = tile_x - minimum_x
                    row = maximum_y - tile_y
                    offset = (column * self.pixel_width, row * self.pixel_height)
                    mosaic.paste(tile_rgb, offset)
                    coverage.paste(
                        255,
                        (
                            *offset,
                            offset[0] + self.pixel_width,
                            offset[1] + self.pixel_height,
                        ),
                    )
                    present_tiles += 1

        mosaic_west = self.origin_east + minimum_x * tile_width
        mosaic_north = self.origin_north + (maximum_y + 1) * tile_height
        pixels_per_east = self.pixel_width / tile_width
        pixels_per_north = self.pixel_height / tile_height
        crop = (
            (west - mosaic_west) * pixels_per_east,
            (mosaic_north - north) * pixels_per_north,
            (east - mosaic_west) * pixels_per_east,
            (mosaic_north - south) * pixels_per_north,
        )
        image = mosaic.crop(crop).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        valid_cells = self.valid_cells(zoom, bounds)
        return RenderedTileWindow(
            image=image,
            valid_cells=valid_cells,
            present_tiles=present_tiles,
            required_tiles=rows * columns,
        )


def _state_keys() -> list[tuple[int, int]]:
    states: list[tuple[int, int]] = []
    for step in range(STEPS):
        for prefix in range((GRID_SIZE**2) ** step):
            states.append((step, prefix))
    if len(states) != TOTAL_STATES:
        raise AssertionError(len(states))
    return states


def _state_valid_actions(
    renderer: TilePyramidRenderer,
    calibration: RegionCalibration,
    step: int,
    prefix: int,
    valid_cells: np.ndarray,
) -> np.ndarray:
    if step == STEPS - 1:
        return valid_cells.copy()
    valid = np.zeros(GRID_SIZE**2, dtype=np.bool_)
    for action in range(GRID_SIZE**2):
        child_prefix = prefix * (GRID_SIZE**2) + action
        child_bounds = state_bounds(calibration, step + 1, child_prefix)
        paths = renderer.required_tile_paths(REQUIRED_SATELLITE_LEVELS[step + 1], child_bounds)
        valid[action] = any(path.is_file() for path in paths)
    return valid


def _to_float32_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _materialize_embeddings(
    array: np.ndarray,
    *,
    keep_on_gpu: bool,
    max_vram_bytes: int | None = None,
) -> torch.Tensor:
    tensor = _to_float32_tensor(array).contiguous()
    if keep_on_gpu:
        if max_vram_bytes is None:
            max_vram_bytes = int(torch.cuda.get_device_properties(0).total_memory)
        tensor_bytes = tensor.element_size() * tensor.numel()
        # Keep a margin for model/optimizer/activation memory.
        if tensor_bytes * 3 > max_vram_bytes:
            raise RuntimeError("embedding cache too large for this GPU cap")
        return tensor.to("cuda", non_blocking=True)
    return tensor.pin_memory()


def _prepare_training_indices(
    train_examples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return tuple(
        torch.as_tensor(values, dtype=torch.long, device=device) for values in train_examples
    )


def _ensure_tensor_on_device(
    values: np.ndarray | torch.Tensor,
    *,
    device: torch.device | None,
) -> torch.Tensor:
    if torch.is_tensor(values):
        return values.to(device=device or values.device, non_blocking=True)
    return torch.as_tensor(values, dtype=torch.long, device=device, non_blocking=True)


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


class DinoFeatureExtractor:
    def __init__(
        self,
        *,
        cache_dir: Path,
        max_vram_bytes: int,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise SetupError("transformers is required for DINOv2 fine-tuning") from exc
        if not torch.cuda.is_available():
            raise SetupError("CUDA is required for local cross-view fine-tuning")
        if not torch.cuda.is_bf16_supported():
            raise SetupError("the local GPU does not support BF16")
        _require_production_license(MODEL_LICENSE, source_id=MODEL_ID)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            use_fast=True,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        ).to("cuda")
        self.model.eval()
        self.model.requires_grad_(False)
        self.max_vram_bytes = max_vram_bytes

    def _forward(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda", non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.model(pixel_values=pixel_values).last_hidden_state
        self._check_vram()
        return tokens.float().cpu()

    def ground(self, images: list[Image.Image]) -> np.ndarray:
        tokens = self._forward(images)
        return tokens[:, 0].numpy().astype(np.float16)

    def satellite_cells(self, images: list[Image.Image]) -> np.ndarray:
        tokens = self._forward(images)[:, 1:]
        batch, patches, dimension = tokens.shape
        side = math.isqrt(patches)
        if side * side != patches or side % GRID_SIZE:
            raise SetupError(f"unexpected DINOv2 patch grid: {patches}")
        block = side // GRID_SIZE
        grid = tokens.reshape(batch, side, side, dimension)
        cells = (
            grid.reshape(batch, GRID_SIZE, block, GRID_SIZE, block, dimension)
            .permute(0, 1, 3, 2, 4, 5)
            .mean(dim=(3, 4))
            .reshape(batch, GRID_SIZE**2, dimension)
        )
        return cells.numpy().astype(np.float16)

    def _check_vram(self) -> None:
        allocated = torch.cuda.max_memory_allocated()
        if allocated > self.max_vram_bytes:
            raise SetupError(
                f"VRAM limit exceeded: allocated={allocated} limit={self.max_vram_bytes}"
            )

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()


def _region_layout_path(dataset_root: Path, spec: RegionSpec) -> Path:
    layout = dataset_root / Path(spec.satellite_layout_relpath)
    _deny_operational_path(layout)
    if not layout.is_file():
        raise SetupError(f"missing satellite layout for region {spec.region_id}: {layout}")
    return layout


def _cache_key(
    train_manifest: Path,
    validation_manifest: Path,
    calibrations: dict[str, RegionCalibration],
    registry: tuple[RegionSpec, ...],
    dataset_root: Path,
) -> str:
    digest = hashlib.sha256()
    digest.update(train_manifest.read_bytes())
    digest.update(validation_manifest.read_bytes())
    digest.update(
        json.dumps(
            {region_id: asdict(calibration) for region_id, calibration in calibrations.items()},
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(json.dumps([asdict(spec) for spec in registry], sort_keys=True).encode("utf-8"))
    for spec in registry:
        digest.update(_region_layout_path(dataset_root, spec).read_bytes())
    digest.update(MODEL_REVISION.encode("ascii"))
    return digest.hexdigest()


def _open_embedding_memmap(
    path: Path, shape: tuple[int, ...], *, dtype: np.dtype[Any] = np.float16
) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _temporary_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}.{os.getpid()}.partial{path.suffix}")


def _complete_array_cache(path: Path, *, shape: tuple[int, ...], dtype: np.dtype[Any]) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    if array.shape != shape or array.dtype != np.dtype(dtype):
        return False
    if array.size:
        samples = np.asarray([array.reshape(-1)[0], array.reshape(-1)[-1]])
        if not np.isfinite(samples).all():
            return False
    return True


def _check_ram(max_ram_bytes: int) -> None:
    rss = psutil.Process().memory_info().rss
    if rss > max_ram_bytes:
        raise SetupError(f"RAM limit exceeded: rss={rss} limit={max_ram_bytes}")


def _resolve_training_batching(
    requested_batch_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[int, int, int]:
    if requested_batch_size <= 0:
        raise SetupError("batch-size must be > 0")
    if micro_batch_size <= 0:
        micro_batch_size = requested_batch_size
    micro_batch_size = min(micro_batch_size, requested_batch_size)
    if gradient_accumulation_steps <= 0:
        raise SetupError("gradient-accumulation-steps must be > 0")
    effective_batch_size = requested_batch_size * gradient_accumulation_steps
    return requested_batch_size, micro_batch_size, effective_batch_size


def _precompute_features(
    dataset_root: Path,
    output_root: Path,
    train_rows: list[CorpusRow],
    validation_rows: list[CorpusRow],
    calibrations: dict[str, RegionCalibration],
    registry: tuple[RegionSpec, ...],
    *,
    embedding_batch_size: int,
    max_vram_bytes: int,
    max_ram_bytes: int,
) -> dict[str, Path]:
    cache_root = output_root / "feature-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": cache_root / "train-ground.fp16.npy",
        "validation": cache_root / "validation-ground.fp16.npy",
        "satellite": cache_root / "satellite-cells.fp16.npy",
        "satellite_valid": cache_root / "satellite-valid-actions.uint8.npy",
    }
    metadata_path = cache_root / "metadata.json"
    expected_key = _cache_key(
        _manifest_path(dataset_root, "train"),
        _manifest_path(dataset_root, "validation"),
        calibrations,
        registry,
        dataset_root,
    )
    if metadata_path.is_file() and all(path.is_file() for path in paths.values()):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") == expected_key:
            return paths

    feature_extractor = DinoFeatureExtractor(
        cache_dir=dataset_root / "models" / "huggingface-cache",
        max_vram_bytes=max_vram_bytes,
    )
    try:
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            expected_shape = (len(rows), EMBEDDING_DIMENSION)
            if _complete_array_cache(paths[split], shape=expected_shape, dtype=np.float16):
                print(
                    f"firewarning cross-view features split={split} cache=complete",
                    flush=True,
                )
                continue
            temporary = _temporary_output(paths[split])
            embeddings = _open_embedding_memmap(temporary, expected_shape)
            for start in range(0, len(rows), embedding_batch_size):
                end = min(start + embedding_batch_size, len(rows))
                images = [_load_rgb(row.image_path) for row in rows[start:end]]
                embeddings[start:end] = feature_extractor.ground(images)
                embeddings.flush()
                _check_ram(max_ram_bytes)
                if start == 0 or end == len(rows) or end % 1_000 < embedding_batch_size:
                    print(
                        f"firewarning cross-view features split={split} encoded={end}/{len(rows)}",
                        flush=True,
                    )
            del embeddings
            os.replace(temporary, paths[split])

        states = _state_keys()
        region_count = len(registry)
        satellite_shape = (region_count * len(states), GRID_SIZE**2, EMBEDDING_DIMENSION)
        valid_shape = (region_count * len(states), GRID_SIZE**2)
        satellite_complete = _complete_array_cache(
            paths["satellite"], shape=satellite_shape, dtype=np.float16
        ) and _complete_array_cache(paths["satellite_valid"], shape=valid_shape, dtype=np.uint8)
        if satellite_complete:
            print("firewarning cross-view satellite cache=complete", flush=True)
        else:
            temporary = _temporary_output(paths["satellite"])
            valid_temporary = _temporary_output(paths["satellite_valid"])
            satellite_embeddings = _open_embedding_memmap(temporary, satellite_shape)
            satellite_valid = _open_embedding_memmap(valid_temporary, valid_shape, dtype=np.uint8)
            for region_index, spec in enumerate(registry):
                calibration = calibrations[spec.region_id]
                renderer = TilePyramidRenderer(
                    _region_layout_path(dataset_root, spec), expected_crs=spec.crs
                )
                region_offset = region_index * len(states)
                for start in range(0, len(states), embedding_batch_size):
                    end = min(start + embedding_batch_size, len(states))
                    windows = [
                        renderer.render_with_coverage(
                            REQUIRED_SATELLITE_LEVELS[step],
                            state_bounds(calibration, step, prefix),
                        )
                        for step, prefix in states[start:end]
                    ]
                    images = [window.image for window in windows]
                    satellite_embeddings[region_offset + start : region_offset + end] = (
                        feature_extractor.satellite_cells(images)
                    )
                    satellite_valid[region_offset + start : region_offset + end] = np.stack(
                        [
                            _state_valid_actions(
                                renderer,
                                calibration,
                                step,
                                prefix,
                                window.valid_cells,
                            )
                            for (step, prefix), window in zip(
                                states[start:end], windows, strict=True
                            )
                        ]
                    ).astype(np.uint8)
                    satellite_embeddings.flush()
                    satellite_valid.flush()
                    _check_ram(max_ram_bytes)
                    if start == 0 or end == len(states) or end % 500 < embedding_batch_size:
                        print(
                            f"firewarning cross-view satellite region={spec.region_id} "
                            f"states encoded={end}/{len(states)}",
                            flush=True,
                        )
            del satellite_embeddings
            del satellite_valid
            os.replace(temporary, paths["satellite"])
            os.replace(valid_temporary, paths["satellite_valid"])
    finally:
        feature_extractor.close()

    metadata = {
        "schema_version": 1,
        "cache_key": expected_key,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "float16_storage_bfloat16_inference",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "satellite_states": len(registry) * TOTAL_STATES,
        "satellite_validity_mask": True,
        "region_order": [spec.region_id for spec in registry],
        "regions": {
            region_id: asdict(calibration) for region_id, calibration in calibrations.items()
        },
        "calibration": asdict(calibrations[registry[0].region_id]),
    }
    _write_json(metadata_path, metadata)
    return paths


class CrossViewProjectionHead(nn.Module):
    def __init__(self, embedding_dimension: int = EMBEDDING_DIMENSION) -> None:
        super().__init__()
        projection_dimension = 256
        self.ground = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, projection_dimension),
        )
        self.satellite = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, projection_dimension),
        )
        self.step_bias = nn.Parameter(torch.zeros(STEPS, GRID_SIZE**2))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(
        self,
        ground_embeddings: torch.Tensor,
        satellite_cells: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        ground = nn.functional.normalize(self.ground(ground_embeddings), dim=-1)
        satellite = nn.functional.normalize(self.satellite(satellite_cells), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = torch.einsum("bd,bkd->bk", ground, satellite) * scale
        return logits + self.step_bias[steps]


def _region_state_offsets(registry: tuple[RegionSpec, ...]) -> dict[str, int]:
    return {spec.region_id: index * TOTAL_STATES for index, spec in enumerate(registry)}


def _examples(
    rows: list[CorpusRow],
    region_state_offsets: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if region_state_offsets is None:
        region_state_offsets = {row.region_id: 0 for row in rows}
    ground_indices = np.empty(len(rows) * STEPS, dtype=np.int64)
    satellite_indices = np.empty(len(rows) * STEPS, dtype=np.int64)
    labels = np.empty(len(rows) * STEPS, dtype=np.int64)
    steps = np.empty(len(rows) * STEPS, dtype=np.int64)
    cursor = 0
    for ground_index, row in enumerate(rows):
        prefix = 0
        state_base = region_state_offsets[row.region_id]
        for step, action in enumerate(row.actions):
            ground_indices[cursor] = ground_index
            satellite_indices[cursor] = state_base + STATE_OFFSETS[step] + prefix
            labels[cursor] = action
            steps[cursor] = step
            prefix = prefix * (GRID_SIZE**2) + action
            cursor += 1
    return ground_indices, satellite_indices, labels, steps


def _embedding_batch(
    ground: np.ndarray | torch.Tensor,
    satellite: np.ndarray | torch.Tensor,
    satellite_valid: np.ndarray | torch.Tensor,
    ground_indices: np.ndarray | torch.Tensor,
    satellite_indices: np.ndarray | torch.Tensor,
    steps: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.is_tensor(ground):
        device = ground.device
        if not torch.is_tensor(ground_indices):
            ground_indices = _ensure_tensor_on_device(ground_indices, device=device).to(torch.long)
        if not torch.is_tensor(satellite_indices):
            satellite_indices = _ensure_tensor_on_device(satellite_indices, device=device).to(
                torch.long
            )
        step_tensor = (
            steps if torch.is_tensor(steps) else _ensure_tensor_on_device(steps, device=device)
        ).to(torch.long)
        ground_tensor = ground.index_select(0, ground_indices)
        if torch.is_tensor(satellite):
            satellite_tensor = satellite.index_select(0, satellite_indices)
        else:
            raise SetupError("satellite must be a tensor when ground is a tensor")
        if torch.is_tensor(satellite_valid):
            valid_tensor = satellite_valid.index_select(0, satellite_indices)
        else:
            raise SetupError("satellite_valid must be a tensor when ground is a tensor")
        return (
            ground_tensor,
            satellite_tensor,
            step_tensor,
            valid_tensor.bool(),
        )

    ground_tensor = torch.from_numpy(np.asarray(ground[ground_indices], dtype=np.float32)).to(
        "cuda", non_blocking=True
    )
    satellite_tensor = torch.from_numpy(
        np.asarray(satellite[satellite_indices], dtype=np.float32)
    ).to("cuda", non_blocking=True)
    valid_tensor = torch.from_numpy(
        np.asarray(satellite_valid[satellite_indices], dtype=np.bool_)
    ).to("cuda", non_blocking=True)
    step_tensor = torch.from_numpy(steps).to("cuda", non_blocking=True)
    return ground_tensor, satellite_tensor, step_tensor, valid_tensor


def _mask_invalid_actions(logits: torch.Tensor, valid_actions: torch.Tensor) -> torch.Tensor:
    safe_valid = valid_actions.bool()
    empty_rows = ~safe_valid.any(dim=1)
    if empty_rows.any():
        safe_valid = safe_valid.clone()
        safe_valid[empty_rows] = True
    return logits.masked_fill(~safe_valid, torch.finfo(logits.dtype).min)


def _masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_actions: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    safe_valid = valid_actions.bool()
    empty_rows = ~safe_valid.any(dim=1)
    if empty_rows.any():
        safe_valid = safe_valid.clone()
        safe_valid[empty_rows] = True
    if not safe_valid.gather(1, labels[:, None]).all():
        raise SetupError("teacher-forced label points to missing satellite data")
    masked_logits = _mask_invalid_actions(logits, safe_valid).float()
    log_probabilities = nn.functional.log_softmax(masked_logits, dim=-1)
    negative_log_likelihood = -log_probabilities.gather(1, labels[:, None]).squeeze(1)
    smooth_loss = -(
        log_probabilities.masked_fill(~safe_valid, 0.0).sum(dim=-1) / safe_valid.sum(dim=-1)
    )
    return (
        (1.0 - label_smoothing) * negative_log_likelihood + label_smoothing * smooth_loss
    ).mean()


def _true_prefix(row: CorpusRow) -> int:
    prefix = 0
    for action in row.actions:
        prefix = prefix * (GRID_SIZE**2) + action
    return prefix


def _prefix_centers(prefixes: np.ndarray, calibration: RegionCalibration) -> np.ndarray:
    centers = np.empty((len(prefixes), 2), dtype=np.float64)
    for index, prefix in enumerate(prefixes.tolist()):
        west, east, south, north = state_bounds(calibration, STEPS, int(prefix))
        centers[index] = ((west + east) / 2.0, (south + north) / 2.0)
    return centers


def _distance_metrics(distances: np.ndarray) -> dict[str, float]:
    return {
        "median_error_m": float(np.median(distances)),
        "p90_error_m": float(np.percentile(distances, 90)),
        "recall_50m": float((distances <= 50).mean()),
        "recall_100m": float((distances <= 100).mean()),
        "recall_250m": float((distances <= 250).mean()),
    }


def _prefix_centers_by_region(
    prefixes: np.ndarray,
    rows: list[CorpusRow],
    calibrations: dict[str, RegionCalibration],
) -> np.ndarray:
    centers = np.empty((len(rows), 2), dtype=np.float64)
    region_ids = np.asarray([row.region_id for row in rows])
    for region_id, calibration in calibrations.items():
        mask = region_ids == region_id
        if not mask.any():
            continue
        centers[mask] = _prefix_centers(prefixes[mask], calibration)
    return centers


def _prior_baseline(
    train_rows: list[CorpusRow],
    validation_rows: list[CorpusRow],
    calibrations: dict[str, RegionCalibration],
) -> dict[str, float]:
    actions = []
    for step in range(STEPS):
        counts = Counter(row.actions[step] for row in train_rows)
        actions.append(counts.most_common(1)[0][0])
    prefix = 0
    for action in actions:
        prefix = prefix * (GRID_SIZE**2) + action
    predicted = np.full(len(validation_rows), prefix, dtype=np.int64)
    truth = _project_rows_by_region(validation_rows, calibrations)
    distances = np.linalg.norm(
        _prefix_centers_by_region(predicted, validation_rows, calibrations) - truth,
        axis=1,
    )
    report = _distance_metrics(distances)
    report["path_exact_accuracy"] = float(
        np.mean([prefix == _true_prefix(row) for row in validation_rows])
    )
    return report


def _evaluate(
    model: CrossViewProjectionHead,
    validation_rows: list[CorpusRow],
    ground_embeddings: np.ndarray | torch.Tensor,
    satellite_embeddings: np.ndarray | torch.Tensor,
    satellite_valid: np.ndarray | torch.Tensor,
    calibrations: dict[str, RegionCalibration],
    region_state_offsets: dict[str, int] | None = None,
    *,
    batch_size: int,
) -> dict[str, Any]:
    if region_state_offsets is None:
        region_state_offsets = {row.region_id: 0 for row in validation_rows}
    row_state_base = np.asarray(
        [region_state_offsets[row.region_id] for row in validation_rows],
        dtype=np.int64,
    )
    model.eval()
    use_tensors = torch.is_tensor(ground_embeddings)
    if use_tensors:
        device = ground_embeddings.device
        validation_ground_indices = torch.arange(
            len(validation_rows), dtype=torch.long, device=device
        )
        state_base: np.ndarray | torch.Tensor = torch.as_tensor(
            row_state_base, dtype=torch.long, device=device
        )
        prefixes: np.ndarray | torch.Tensor = torch.zeros(
            len(validation_rows), dtype=torch.long, device=device
        )
    else:
        validation_ground_indices = np.arange(len(validation_rows), dtype=np.int64)
        state_base = row_state_base
        prefixes = np.zeros(len(validation_rows), dtype=np.int64)
    with torch.inference_mode():
        for step in range(STEPS):
            for start in range(0, len(validation_rows), batch_size):
                end = min(start + batch_size, len(validation_rows))
                if use_tensors:
                    ground_indices = validation_ground_indices[start:end]
                    current_prefix = prefixes[start:end]
                else:
                    ground_indices = np.arange(start, end, dtype=np.int64)
                    current_prefix = prefixes[start:end]
                step_values = (
                    torch.full(
                        (end - start,),
                        step,
                        dtype=torch.long,
                        device=device if use_tensors else None,
                    )
                    if use_tensors
                    else np.full(end - start, step, dtype=np.int64)
                )
                satellite_indices = state_base[start:end] + STATE_OFFSETS[step] + current_prefix
                ground, satellite, steps, valid_actions = _embedding_batch(
                    ground_embeddings,
                    satellite_embeddings,
                    satellite_valid,
                    ground_indices,
                    satellite_indices,
                    step_values,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ground, satellite, steps)
                    predicted = _mask_invalid_actions(logits, valid_actions).argmax(dim=-1)
                if use_tensors:
                    prefixes[start:end] = prefixes[start:end] * (GRID_SIZE**2) + predicted.to(
                        prefixes.device, non_blocking=True
                    )
                else:
                    prefixes[start:end] = (
                        prefixes[start:end] * (GRID_SIZE**2) + predicted.cpu().numpy()
                    )
    prefixes_np = prefixes.cpu().numpy() if torch.is_tensor(prefixes) else prefixes
    expected = np.asarray([_true_prefix(row) for row in validation_rows], dtype=np.int64)
    truth = _project_rows_by_region(validation_rows, calibrations)
    predicted_centers = _prefix_centers_by_region(prefixes_np, validation_rows, calibrations)
    distances = np.linalg.norm(predicted_centers - truth, axis=1)
    report: dict[str, Any] = _distance_metrics(distances)
    report["path_exact_accuracy"] = float((prefixes_np == expected).mean())
    region_ids = np.asarray([row.region_id for row in validation_rows])
    per_region: dict[str, Any] = {}
    for region_id in calibrations:
        mask = region_ids == region_id
        if not mask.any():
            continue
        region_report: dict[str, Any] = _distance_metrics(distances[mask])
        region_report["path_exact_accuracy"] = float((prefixes_np[mask] == expected[mask]).mean())
        region_report["rows"] = int(mask.sum())
        per_region[region_id] = region_report
    report["per_region"] = per_region
    return report


def _save_checkpoint(
    output_root: Path,
    model: CrossViewProjectionHead,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    milestone: int,
    metrics: dict[str, float],
) -> Path:
    path = output_root / "checkpoints" / f"checkpoint-{milestone:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.pt")
    torch.save(
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "epoch": epoch,
            "milestone_percent": milestone,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        temporary,
    )
    os.replace(temporary, path)
    return path


def train(
    dataset_root: Path,
    *,
    epochs: int,
    batch_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    embedding_batch_size: int,
    learning_rate: float,
    max_vram_gb: float,
    max_ram_gb: float,
    output_root: Path | None = None,
    keep_embeddings_on_gpu: bool = True,
    warmup_ratio: float = 0.03,
    min_lr_ratio: float = 0.1,
    early_stop_patience: int = 5,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    _deny_operational_path(dataset_root)
    if epochs < 2:
        raise SetupError("at least two epochs are required for checkpoint scheduling")
    if max_vram_gb <= 0 or max_ram_gb <= 0:
        raise SetupError("training VRAM and RAM budgets must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise SetupError("warmup-ratio must be in [0, 1)")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise SetupError("min-lr-ratio must be in (0, 1]")
    if early_stop_patience < 0:
        raise SetupError("early-stop-patience must be >= 0 (0 disables early stop)")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SetupError("CUDA with BF16 support is required")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    total_ram_gb = psutil.virtual_memory().total / 1024**3
    if max_vram_gb > total_vram_gb * 0.95:
        raise SetupError(
            "configured VRAM budget must leave at least five percent of physical VRAM free"
        )
    if max_ram_gb > total_ram_gb * 0.90:
        raise SetupError(
            "configured RAM budget must leave at least ten percent of physical RAM free"
        )
    torch.manual_seed(20260721)
    np.random.seed(20260721)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    requested_batch_size, micro_batch_size, effective_batch_size = _resolve_training_batching(
        requested_batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    if output_root is None:
        output_root = dataset_root / "training" / RUN_ID
    else:
        output_root = output_root.resolve()
        _deny_operational_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    registry = load_region_registry(dataset_root)
    train_rows = _load_rows(dataset_root, "train", registry)
    validation_rows = _load_rows(dataset_root, "validation", registry)
    train_ids = {row.sample_id for row in train_rows}
    validation_ids = {row.sample_id for row in validation_rows}
    if train_ids & validation_ids:
        raise SetupError("train/validation sample overlap")
    referenced_regions = {row.region_id for row in train_rows + validation_rows}
    registry = tuple(spec for spec in registry if spec.region_id in referenced_regions)
    calibrations = derive_region_calibrations(train_rows, registry)
    validation_geometry = _validate_rows_against_regions(validation_rows, calibrations)
    region_state_offsets = _region_state_offsets(registry)
    primary_calibration = calibrations[
        max(calibrations, key=lambda rid: sum(row.region_id == rid for row in train_rows))
    ]
    max_vram_bytes = int(max_vram_gb * 1024**3)
    max_ram_bytes = int(max_ram_gb * 1024**3)
    cache_paths = _precompute_features(
        dataset_root,
        output_root,
        train_rows,
        validation_rows,
        calibrations,
        registry,
        embedding_batch_size=embedding_batch_size,
        max_vram_bytes=max_vram_bytes,
        max_ram_bytes=max_ram_bytes,
    )
    train_ground_np = np.load(cache_paths["train"], mmap_mode="r")
    validation_ground_np = np.load(cache_paths["validation"], mmap_mode="r")
    satellite_np = np.load(cache_paths["satellite"], mmap_mode="r")
    satellite_valid_np = np.load(cache_paths["satellite_valid"], mmap_mode="r")
    if keep_embeddings_on_gpu:
        try:
            train_ground = _materialize_embeddings(
                train_ground_np, keep_on_gpu=True, max_vram_bytes=max_vram_bytes
            )
            validation_ground = _materialize_embeddings(
                validation_ground_np, keep_on_gpu=True, max_vram_bytes=max_vram_bytes
            )
            satellite = _materialize_embeddings(
                satellite_np, keep_on_gpu=True, max_vram_bytes=max_vram_bytes
            )
            satellite_valid = torch.from_numpy(np.asarray(satellite_valid_np, dtype=np.bool_)).to(
                "cuda", non_blocking=True
            )
        except RuntimeError:
            # Keep CPU fallback to ensure train still runs with lower VRAM availability.
            train_ground = _to_float32_tensor(train_ground_np).pin_memory()
            validation_ground = _to_float32_tensor(validation_ground_np).pin_memory()
            satellite = _to_float32_tensor(satellite_np).pin_memory()
            satellite_valid = torch.from_numpy(
                np.asarray(satellite_valid_np, dtype=np.bool_)
            ).pin_memory()
            keep_embeddings_on_gpu = False
    else:
        train_ground = _to_float32_tensor(train_ground_np).pin_memory()
        validation_ground = _to_float32_tensor(validation_ground_np).pin_memory()
        satellite = _to_float32_tensor(satellite_np).pin_memory()
        satellite_valid = torch.from_numpy(
            np.asarray(satellite_valid_np, dtype=np.bool_)
        ).pin_memory()
    train_examples = _examples(train_rows, region_state_offsets)
    if keep_embeddings_on_gpu:
        train_example_tensors = _prepare_training_indices(
            train_examples, device=torch.device("cuda")
        )
    else:
        train_example_tensors = train_examples

    model = CrossViewProjectionHead().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    example_count = len(train_examples[0])
    steps_per_epoch = max(1, math.ceil(example_count / effective_batch_size))
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0

    def _lr_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_factor)
    baseline = _prior_baseline(train_rows, validation_rows, calibrations)
    history: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    emitted_milestones: set[int] = set()
    best: dict[str, Any] | None = None
    epochs_without_improvement = 0
    early_stopped = False
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        if torch.is_tensor(train_example_tensors[0]):
            permutation = torch.randperm(len(train_example_tensors[0]), device="cuda")
        else:
            permutation = np.random.permutation(len(train_example_tensors[0]))
        total_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(permutation), effective_batch_size):
            optimizer.zero_grad(set_to_none=True)
            update_stop = min(start + effective_batch_size, len(permutation))
            accumulator_steps = max(1, math.ceil((update_stop - start) / micro_batch_size))
            for micro_start in range(start, update_stop, micro_batch_size):
                micro_end = min(micro_start + micro_batch_size, update_stop)
                indices = permutation[micro_start:micro_end]
                ground, cells, steps, valid_actions = _embedding_batch(
                    train_ground,
                    satellite,
                    satellite_valid,
                    train_example_tensors[0][indices],
                    train_example_tensors[1][indices],
                    train_example_tensors[3][indices],
                )
                labels = (
                    train_example_tensors[2][indices]
                    if torch.is_tensor(train_example_tensors[2])
                    else torch.from_numpy(train_example_tensors[2][indices]).to(
                        "cuda", non_blocking=True
                    )
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ground, cells, steps)
                    loss = _masked_cross_entropy(
                        logits,
                        labels,
                        valid_actions,
                        label_smoothing=0.02,
                    )
                (loss / accumulator_steps).backward()
                masked_logits = _mask_invalid_actions(logits.detach(), valid_actions)
                count = len(indices)
                total_loss += float(loss.detach()) * count
                correct += int((masked_logits.argmax(dim=-1) == labels).sum())
                seen += count
            optimizer.step()
            scheduler.step()
        metrics = _evaluate(
            model,
            validation_rows,
            validation_ground,
            satellite,
            satellite_valid,
            calibrations,
            region_state_offsets,
            batch_size=batch_size,
        )
        epoch_report: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_action_accuracy": correct / seen,
            "validation": metrics,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "rss_bytes": psutil.Process().memory_info().rss,
            "batch_size": requested_batch_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report, sort_keys=True), flush=True)
        if epoch_report["peak_vram_bytes"] > max_vram_bytes:
            raise SetupError("VRAM ceiling exceeded during adapter training")
        _check_ram(max_ram_bytes)

        score = (metrics["recall_100m"], -metrics["median_error_m"])
        if best is None or score > tuple(best["score"]):
            epochs_without_improvement = 0
            best = {"epoch": epoch, "score": list(score), "metrics": metrics}
            best_path = output_root / "best-adapter.pt"
            temporary = best_path.with_suffix(".partial.pt")
            torch.save(
                {
                    "schema_version": 1,
                    "run_id": RUN_ID,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "model": model.state_dict(),
                    "calibration": asdict(primary_calibration),
                    "regions": {
                        region_id: asdict(calibration)
                        for region_id, calibration in calibrations.items()
                    },
                    "region_order": [spec.region_id for spec in registry],
                    "metrics": metrics,
                },
                temporary,
            )
            os.replace(temporary, best_path)
        else:
            epochs_without_improvement += 1

        progress = math.floor(epoch / epochs * 100)
        for milestone in CHECKPOINT_MILESTONES:
            if progress >= milestone and milestone not in emitted_milestones:
                checkpoint = _save_checkpoint(
                    output_root,
                    model,
                    optimizer,
                    epoch=epoch,
                    milestone=milestone,
                    metrics=metrics,
                )
                if checkpoint.is_relative_to(output_root):
                    checkpoints.append(checkpoint.relative_to(output_root).as_posix())
                elif checkpoint.is_absolute():
                    checkpoints.append(checkpoint.as_posix())
                else:
                    checkpoints.append(str(checkpoint))
                emitted_milestones.add(milestone)

        if early_stop_patience and epochs_without_improvement >= early_stop_patience:
            early_stopped = True
            print(
                f"firewarning cross-view early stop epoch={epoch} patience={early_stop_patience}",
                flush=True,
            )
            break

    if best is None:  # pragma: no cover - epochs is constrained above
        raise AssertionError("training produced no epoch")
    final_metrics = history[-1]["validation"]
    best_metrics = best["metrics"]
    bootstrap_benchmark_passed = bool(
        best_metrics["recall_100m"] > baseline["recall_100m"]
        and best_metrics["median_error_m"] < baseline["median_error_m"]
    )
    report = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "training_complete": True,
        "training_kind": "frozen_dinov2_cross_view_projection_adapter",
        "quantization": None,
        "compute_dtype": "bfloat16",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_license": MODEL_LICENSE,
        "dataset_license": DATASET_LICENSE,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "region_order": [spec.region_id for spec in registry],
        "regions": {
            region_id: asdict(calibration) for region_id, calibration in calibrations.items()
        },
        "calibration": asdict(primary_calibration),
        "validation_geometry": validation_geometry,
        "lr_schedule": {
            "kind": "linear_warmup_cosine",
            "base_learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "min_lr_ratio": min_lr_ratio,
            "total_steps": total_steps,
        },
        "early_stopped": early_stopped,
        "early_stop_patience": early_stop_patience,
        "baseline": baseline,
        "best": best,
        "final": final_metrics,
        "bootstrap_benchmark_passed": bootstrap_benchmark_passed,
        "production_promotion_ready": False,
        "production_blockers": [
            "France rural and mountain domain adaptation is not yet trained",
            "independent double-validated geographic critical test is missing",
            "RoMa/PnP/MNT downstream benchmark must pass after narrowed-window integration",
        ],
        "resource_limits": {
            "max_vram_gb": max_vram_gb,
            "max_ram_gb": max_ram_gb,
        },
        "checkpoints": checkpoints,
        "history": history,
    }
    _write_json(output_root / "training-report.json", report)
    return report


def preflight(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    registry = load_region_registry(dataset_root)
    train_rows = _load_rows(dataset_root, "train", registry)
    validation_rows = _load_rows(dataset_root, "validation", registry)
    referenced_regions = {row.region_id for row in train_rows + validation_rows}
    registry = tuple(spec for spec in registry if spec.region_id in referenced_regions)
    calibrations = derive_region_calibrations(train_rows, registry)
    validation_geometry = _validate_rows_against_regions(validation_rows, calibrations)
    unique_tile_paths: set[Path] = set()
    missing_tile_paths: set[Path] = set()
    incomplete_states = 0
    empty_states = 0
    render_checks: list[dict[str, Any]] = []
    per_region: dict[str, dict[str, Any]] = {}
    valid_actions_by_region: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for spec in registry:
        calibration = calibrations[spec.region_id]
        renderer = TilePyramidRenderer(
            _region_layout_path(dataset_root, spec), expected_crs=spec.crs
        )
        region_unique: set[Path] = set()
        region_missing: set[Path] = set()
        region_incomplete = 0
        region_empty = 0
        valid_actions_by_state: dict[tuple[int, int], np.ndarray] = {}
        for step, prefix in _state_keys():
            bounds = state_bounds(calibration, step, prefix)
            required = renderer.required_tile_paths(REQUIRED_SATELLITE_LEVELS[step], bounds)
            missing = [path for path in required if not path.is_file()]
            if missing:
                region_incomplete += 1
                region_missing.update(missing)
            if len(missing) == len(required):
                region_empty += 1
            region_unique.update(required)
            valid_cells = renderer.valid_cells(REQUIRED_SATELLITE_LEVELS[step], bounds)
            valid_actions_by_state[(step, prefix)] = _state_valid_actions(
                renderer, calibration, step, prefix, valid_cells
            )
            if prefix == 0:
                image = renderer.render(REQUIRED_SATELLITE_LEVELS[step], bounds)
                render_checks.append(
                    {
                        "region_id": spec.region_id,
                        "step": step,
                        "prefix": prefix,
                        "size": list(image.size),
                    }
                )
        valid_actions_by_region[spec.region_id] = valid_actions_by_state
        unique_tile_paths.update(region_unique)
        missing_tile_paths.update(region_missing)
        incomplete_states += region_incomplete
        empty_states += region_empty
        per_region[spec.region_id] = {
            "required_unique_satellite_tiles": len(region_unique),
            "missing_unique_satellite_tiles": len(region_missing),
            "incomplete_autoregressive_states": region_incomplete,
            "empty_autoregressive_states": region_empty,
        }

    invalid_teacher_labels: list[str] = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            prefix = 0
            valid_actions_by_state = valid_actions_by_region[row.region_id]
            for step, action in enumerate(row.actions):
                if not valid_actions_by_state[(step, prefix)][action]:
                    invalid_teacher_labels.append(
                        f"{split}:{row.sample_id}:step={step}:prefix={prefix}:action={action}"
                    )
                prefix = prefix * (GRID_SIZE**2) + action
    if invalid_teacher_labels:
        raise SetupError(
            f"teacher-forced labels point to missing satellite data: {invalid_teacher_labels[0]}"
        )
    return {
        "schema_version": 1,
        "preflight_passed": True,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "region_order": [spec.region_id for spec in registry],
        "regions": {
            region_id: asdict(calibration) for region_id, calibration in calibrations.items()
        },
        "calibration": asdict(calibrations[registry[0].region_id]),
        "validation_geometry": validation_geometry,
        "render_checks": render_checks,
        "covered_autoregressive_states": TOTAL_STATES * len(registry),
        "required_unique_satellite_tiles": len(unique_tile_paths),
        "missing_unique_satellite_tiles": len(missing_tile_paths),
        "incomplete_autoregressive_states": incomplete_states,
        "empty_autoregressive_states": empty_states,
        "per_region": per_region,
        "invalid_teacher_forced_labels": 0,
        "missing_tile_policy": "upstream_with_default_plus_action_mask",
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "quantization": None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "train"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--dataset-root", type=Path, required=True)
    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4_096)
    train_parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="Per-step mini-batch size used for forward/backward. 0 = same as --batch-size.",
    )
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of mini-batches accumulated per optimizer step.",
    )
    train_parser.add_argument("--embedding-batch-size", type=int, default=48)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Fraction of total optimizer steps used for linear LR warmup.",
    )
    train_parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="Final cosine LR as a fraction of --learning-rate.",
    )
    train_parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop after this many epochs without validation improvement. 0 disables.",
    )
    train_parser.add_argument("--max-vram-gb", type=float, default=14.0)
    train_parser.add_argument("--max-ram-gb", type=float, default=10.0)
    train_parser.add_argument("--output-root", type=Path, default=None)
    train_parser.add_argument(
        "--embeddings-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep feature embeddings in GPU memory when memory budget allows",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "preflight":
        report = preflight(args.dataset_root)
    elif args.command == "train":
        report = train(
            args.dataset_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            embedding_batch_size=args.embedding_batch_size,
            learning_rate=args.learning_rate,
            max_vram_gb=args.max_vram_gb,
            max_ram_gb=args.max_ram_gb,
            output_root=args.output_root,
            keep_embeddings_on_gpu=args.embeddings_on_gpu,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
            early_stop_patience=args.early_stop_patience,
        )
    else:  # pragma: no cover - argparse rejects this
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
