"""Extend the coarse cross-view localizer with synthetic fire-region data.

Three explicit steps, mirroring the bootstrap setup contract:

1. ``register-region`` declares a new geographic region (French forest massif) in
   ``corpus/<corpus>/regions.yaml`` without touching the bootstrap Washington DC entry.
2. ``build-pyramid`` derives the four-level satellite tile pyramid of the region from a
   georeferenced orthophoto (IGN BD ORTHO export) with an explicit bounds sidecar.
3. ``prepare-views`` ingests externally rendered ground views (Blender/Unity batch with
   exact camera poses) and merges them into the corpus manifests with deterministic
   render-group split assignment.

No operational incident zone may enter any step: the same deny-list as the rest of the
training pipeline applies, and synthetic rows keep ``production_promotion_gate=false``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from PIL import Image
from pyproj import Transformer

from fireviewer_model_lab.training.cross_view_localizer_setup import CORPUS_ID, REQUIRED_SATELLITE_LEVELS
from fireviewer_model_lab.training.spatial_train_cross_view_localizer import (
    DEFAULT_REGION_SPEC,
    GRID_SIZE,
    STEPS,
    RegionSpec,
    load_region_registry,
    region_bounds,
    region_registry_path,
    validate_region_spec,
)
from fireviewer_model_lab.training.spatial_training_setup import (
    SetupError,
    _deny_operational_path,
    _require_production_license,
    _sha256_file,
    _write_json,
)

SYNTHETIC_SOURCE_ID = "fireviewer_synthetic_forest_v1"
SYNTHETIC_LICENSE = "FireViewer-Synthetic-1.0"
TILE_CRS_METERS = 20.0
TILE_PIXELS = 250
JPEG_QUALITY = 90
MAX_ORTHOPHOTO_PIXELS = 2_000_000_000
POSES_COLUMNS = ("image_relpath", "latitude", "longitude", "render_group")


def _registry_specs_payload(specs: list[RegionSpec]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "regions": [
            {
                "region_id": spec.region_id,
                "crs": spec.crs,
                "center_latitude": spec.center_latitude,
                "center_longitude": spec.center_longitude,
                "bounds_meters": list(spec.bounds_meters),
                "satellite_layout": spec.satellite_layout_relpath,
            }
            for spec in specs
        ],
    }


def register_region(
    dataset_root: Path,
    *,
    region_id: str,
    crs: str,
    center_latitude: float,
    center_longitude: float,
    bounds_meters: tuple[float, float, float, float],
    satellite_layout_relpath: str,
) -> dict[str, Any]:
    """Upsert one region in the corpus registry, preserving the bootstrap region."""

    dataset_root = dataset_root.resolve()
    _deny_operational_path(dataset_root)
    if region_id == DEFAULT_REGION_SPEC.region_id:
        raise SetupError("the bootstrap region entry is managed by the trainer defaults")
    _deny_operational_path(region_id)
    spec = RegionSpec(
        region_id=region_id,
        crs=crs.upper(),
        center_latitude=center_latitude,
        center_longitude=center_longitude,
        bounds_meters=tuple(float(value) for value in bounds_meters),
        satellite_layout_relpath=satellite_layout_relpath,
    )
    validate_region_spec(spec)
    west, east, south, north = region_bounds(spec)

    registry_path = region_registry_path(dataset_root)
    if registry_path.is_file():
        specs = list(load_region_registry(dataset_root))
    else:
        specs = [DEFAULT_REGION_SPEC]
    specs = [existing for existing in specs if existing.region_id != region_id]
    specs.append(spec)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SetupError("PyYAML is required to write the region registry") from exc
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_suffix(".yaml.partial")
    temporary.write_text(
        yaml.safe_dump(_registry_specs_payload(specs), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, registry_path)
    report = {
        "schema_version": 1,
        "registry_path": registry_path.relative_to(dataset_root).as_posix(),
        "region_id": region_id,
        "crs": spec.crs,
        "projected_bounds": {"west": west, "east": east, "south": south, "north": north},
        "square_meters": east - west,
        "regions_declared": [existing.region_id for existing in specs],
    }
    return report


def _synthetic_source_root(dataset_root: Path, region_id: str) -> Path:
    root = dataset_root.resolve() / "sources" / f"synthetic-{region_id}"
    _deny_operational_path(root)
    return root


def _load_region_spec(dataset_root: Path, region_id: str) -> RegionSpec:
    registry = load_region_registry(dataset_root)
    for spec in registry:
        if spec.region_id == region_id:
            return spec
    raise SetupError(f"region is not registered: {region_id}")


def build_pyramid(
    dataset_root: Path,
    *,
    region_id: str,
    orthophoto: Path,
    orthophoto_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    """Render the region's four-level satellite pyramid from an orthophoto.

    ``orthophoto`` is a north-up RGB image spanning ``orthophoto_bounds``
    (west, east, south, north) expressed in the region CRS.  Only the tile levels
    required by the localizer are written; missing tiles stay absent so the
    action-masking contract still applies.
    """

    dataset_root = dataset_root.resolve()
    _deny_operational_path(dataset_root)
    spec = _load_region_spec(dataset_root, region_id)
    orthophoto = orthophoto.resolve()
    _deny_operational_path(orthophoto)
    if not orthophoto.is_file():
        raise SetupError(f"missing orthophoto: {orthophoto}")
    west_o, east_o, south_o, north_o = (float(v) for v in orthophoto_bounds)
    if not (west_o < east_o and south_o < north_o):
        raise SetupError(f"inverted orthophoto bounds: {orthophoto_bounds!r}")
    west_r, east_r, south_r, north_r = region_bounds(spec)
    if west_r < west_o or east_r > east_o or south_r < south_o or north_r > north_o:
        raise SetupError(
            f"orthophoto does not cover region {region_id}: "
            f"region=({west_r}, {east_r}, {south_r}, {north_r}) "
            f"orthophoto=({west_o}, {east_o}, {south_o}, {north_o})"
        )
    source_sha256 = _sha256_file(orthophoto)

    satellite_root = _synthetic_source_root(dataset_root, region_id) / "satellite"
    layout = {
        "crs": spec.crs.lower(),
        "min_zoom": min(REQUIRED_SATELLITE_LEVELS),
        "max_zoom": max(REQUIRED_SATELLITE_LEVELS),
        "origin_crs": [west_r, south_r],
        "path": "{zoom}/{x}/{y}.jpg",
        "tile_axes": ["east", "north"],
        "tile_shape_crs": [TILE_CRS_METERS, TILE_CRS_METERS],
        "tile_shape_px": [TILE_PIXELS, TILE_PIXELS],
    }
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SetupError("PyYAML is required to write the satellite layout") from exc

    # The orthophoto legitimately exceeds the default PIL decompression-bomb
    # threshold; a generous explicit cap is kept instead of disabling the guard.
    Image.MAX_IMAGE_PIXELS = MAX_ORTHOPHOTO_PIXELS
    with Image.open(orthophoto) as source:
        source_rgb = source.convert("RGB")
        source_width, source_height = source_rgb.size
        px_per_meter_x = source_width / (east_o - west_o)
        px_per_meter_y = source_height / (north_o - south_o)

        tiles_written: dict[str, int] = {}
        for zoom in REQUIRED_SATELLITE_LEVELS:
            tile_meters = TILE_CRS_METERS * (2.0 ** (-zoom))
            min_x = 0
            max_x = math.floor((east_r - 1e-8 - west_r) / tile_meters)
            min_y = 0
            max_y = math.floor((north_r - 1e-8 - south_r) / tile_meters)
            written = 0
            for tile_y in range(min_y, max_y + 1):
                for tile_x in range(min_x, max_x + 1):
                    tile_west = west_r + tile_x * tile_meters
                    tile_east = tile_west + tile_meters
                    tile_south = south_r + tile_y * tile_meters
                    tile_north = tile_south + tile_meters
                    crop = (
                        round((tile_west - west_o) * px_per_meter_x),
                        round((north_o - tile_north) * px_per_meter_y),
                        round((tile_east - west_o) * px_per_meter_x),
                        round((north_o - tile_south) * px_per_meter_y),
                    )
                    crop = (
                        max(0, crop[0]),
                        max(0, crop[1]),
                        min(source_width, crop[2]),
                        min(source_height, crop[3]),
                    )
                    if crop[2] <= crop[0] or crop[3] <= crop[1]:
                        continue
                    tile = source_rgb.crop(crop).resize(
                        (TILE_PIXELS, TILE_PIXELS), Image.Resampling.BICUBIC
                    )
                    target = satellite_root / str(zoom) / str(tile_x) / f"{tile_y}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".jpg.partial")
                    tile.save(temporary, format="JPEG", quality=JPEG_QUALITY)
                    os.replace(temporary, target)
                    written += 1
            tiles_written[str(zoom)] = written

    satellite_root.mkdir(parents=True, exist_ok=True)
    layout_path = satellite_root / "layout.yaml"
    temporary = layout_path.with_suffix(".yaml.partial")
    temporary.write_text(yaml.safe_dump(layout, sort_keys=True), encoding="utf-8")
    os.replace(temporary, layout_path)

    expected_layout_relpath = satellite_root.relative_to(dataset_root) / "layout.yaml"
    if spec.satellite_layout_relpath != expected_layout_relpath.as_posix():
        raise SetupError(
            f"region {region_id} declares satellite_layout="
            f"{spec.satellite_layout_relpath} but the pyramid writes "
            f"{expected_layout_relpath.as_posix()}; update the registry entry"
        )
    report = {
        "schema_version": 1,
        "region_id": region_id,
        "crs": spec.crs,
        "orthophoto": str(orthophoto),
        "orthophoto_sha256": source_sha256,
        "orthophoto_bounds": {
            "west": west_o,
            "east": east_o,
            "south": south_o,
            "north": north_o,
        },
        "levels": list(REQUIRED_SATELLITE_LEVELS),
        "tiles_written": tiles_written,
        "layout": expected_layout_relpath.as_posix(),
        "license": "Licence-Ouverte-2.0",
        "attribution_required": "IGN BD ORTHO",
    }
    _write_json(satellite_root / "firewarning-pyramid-report.json", report)
    return report


def _actions_for_position(
    bounds: tuple[float, float, float, float],
    transformer: Transformer,
    latitude: float,
    longitude: float,
) -> tuple[int, int, int, int] | None:
    west, east, south, north = bounds
    x, y = transformer.transform(longitude, latitude)
    if not (west <= x <= east and south <= y <= north):
        return None
    cell_count = GRID_SIZE**STEPS
    column = min(cell_count - 1, max(0, int((x - west) / (east - west) * cell_count)))
    row = min(cell_count - 1, max(0, int((north - y) / (north - south) * cell_count)))
    actions = []
    for step in range(STEPS):
        shift = 2 * (STEPS - 1 - step)
        action_row = (row >> shift) & (GRID_SIZE - 1)
        action_column = (column >> shift) & (GRID_SIZE - 1)
        actions.append(action_row * GRID_SIZE + action_column)
    return actions[0], actions[1], actions[2], actions[3]


def _split_for_group(render_group: str, validation_fraction: float) -> str:
    digest = hashlib.sha256(render_group.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < validation_fraction else "train"


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SetupError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _stage_manifest_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return temporary


def prepare_views(
    dataset_root: Path,
    *,
    region_id: str,
    poses_csv: Path,
    source_id: str = SYNTHETIC_SOURCE_ID,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    """Merge externally rendered ground views into the corpus manifests.

    The poses CSV carries ``image_relpath,latitude,longitude,render_group`` per
    rendered view.  Splits are assigned by deterministic hash of the render group so
    a whole render batch stays in a single split (no near-duplicate leakage).
    Existing rows of the same ``source_id`` and region are replaced atomically.
    """

    dataset_root = dataset_root.resolve()
    _deny_operational_path(dataset_root)
    if not 0.0 < validation_fraction < 0.5:
        raise SetupError("validation-fraction must be in (0, 0.5)")
    spec = _load_region_spec(dataset_root, region_id)
    _require_production_license(SYNTHETIC_LICENSE, source_id=source_id)
    poses_csv = poses_csv.resolve()
    _deny_operational_path(poses_csv)
    if not poses_csv.is_file():
        raise SetupError(f"missing poses CSV: {poses_csv}")
    poses_sha256 = _sha256_file(poses_csv)
    bounds = region_bounds(spec)
    transformer = Transformer.from_crs("EPSG:4326", spec.crs, always_xy=True)

    new_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen_relpaths: set[str] = set()
    seen_image_hashes: set[str] = set()
    outside_region = 0
    parsed = 0
    with poses_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(
            column not in reader.fieldnames for column in POSES_COLUMNS
        ):
            raise SetupError(f"poses CSV must declare columns {POSES_COLUMNS}: {poses_csv}")
        for line_number, record in enumerate(reader, start=2):
            relpath = str(record["image_relpath"]).strip()
            render_group = str(record["render_group"]).strip()
            if not relpath or not render_group:
                raise SetupError(f"empty relpath or render group at {poses_csv}:{line_number}")
            relative = Path(relpath)
            if relative.is_absolute() or any(part == ".." for part in relative.parts):
                raise SetupError(f"pose image escapes dataset root: {relpath}")
            _deny_operational_path(relpath)
            image_path = dataset_root / relative
            if not image_path.is_file():
                raise SetupError(f"missing rendered view: {image_path}")
            if relpath in seen_relpaths:
                raise SetupError(f"duplicate rendered view: {relpath}")
            seen_relpaths.add(relpath)
            latitude = float(record["latitude"])
            longitude = float(record["longitude"])
            actions = _actions_for_position(bounds, transformer, latitude, longitude)
            if actions is None:
                outside_region += 1
                continue
            image_sha256 = _sha256_file(image_path)
            if image_sha256 in seen_image_hashes:
                raise SetupError(
                    f"duplicate rendered view content: {relpath} sha256={image_sha256}"
                )
            seen_image_hashes.add(image_sha256)
            split = _split_for_group(render_group, validation_fraction)
            new_rows[split].append(
                {
                    "schema_version": "1.0",
                    "family": "coarse_cross_view_localization",
                    "sample_id": f"{source_id}:{image_sha256[:24]}",
                    "source_id": source_id,
                    "source_revision": f"synthetic-poses-{poses_sha256[:12]}",
                    "source_view_relpath": relative.as_posix(),
                    "source_view_sha256": image_sha256,
                    "latitude": latitude,
                    "longitude": longitude,
                    "action_sequence": list(actions),
                    "satellite_levels": list(REQUIRED_SATELLITE_LEVELS),
                    "region_id": region_id,
                    "split": split,
                    "split_basis": "synthetic_render_group_hash",
                    "split_group": f"{source_id}:{render_group}",
                    "license": SYNTHETIC_LICENSE,
                    "training_membership": True,
                    "critical_test_membership": False,
                    "production_promotion_gate": False,
                    "operational_incident": False,
                }
            )
            parsed += 1
    if not parsed:
        raise SetupError(f"poses CSV contains no usable view: {poses_csv}")

    corpus_root = dataset_root / "corpus" / CORPUS_ID
    merged_counts: dict[str, int] = {}
    replaced = 0
    staged: list[tuple[Path, Path]] = []
    for split in ("train", "validation"):
        manifest = corpus_root / f"{split}.jsonl"
        existing = _manifest_rows(manifest)
        kept = []
        for row in existing:
            if row.get("source_id") == source_id and row.get("region_id", "") == region_id:
                replaced += 1
                continue
            kept.append(row)
        kept.extend(new_rows[split])
        staged.append((_stage_manifest_rows(manifest, kept), manifest))
        merged_counts[split] = len(kept)
    for temporary, manifest in staged:
        os.replace(temporary, manifest)

    report = {
        "schema_version": 1,
        "region_id": region_id,
        "source_id": source_id,
        "poses_csv": str(poses_csv),
        "views_parsed": parsed,
        "views_outside_region_rejected": outside_region,
        "rows_written": {split: len(rows) for split, rows in new_rows.items()},
        "rows_replaced": replaced,
        "manifest_rows_total": merged_counts,
        "validation_fraction": validation_fraction,
        "split_basis": "synthetic_render_group_hash",
        "production_promotion_gate": False,
    }
    _write_json(corpus_root / f"synthetic-{region_id}-prepare-report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register-region")
    register.add_argument("--dataset-root", type=Path, required=True)
    register.add_argument("--region-id", required=True)
    register.add_argument("--crs", required=True)
    register.add_argument("--center-latitude", type=float, required=True)
    register.add_argument("--center-longitude", type=float, required=True)
    register.add_argument(
        "--bounds-meters",
        type=float,
        nargs=4,
        required=True,
        metavar=("MIN_EAST", "MAX_EAST", "MIN_NORTH", "MAX_NORTH"),
    )
    register.add_argument("--satellite-layout", required=True)

    pyramid = subparsers.add_parser("build-pyramid")
    pyramid.add_argument("--dataset-root", type=Path, required=True)
    pyramid.add_argument("--region-id", required=True)
    pyramid.add_argument("--orthophoto", type=Path, required=True)
    pyramid.add_argument(
        "--orthophoto-bounds",
        type=float,
        nargs=4,
        required=True,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
    )

    views = subparsers.add_parser("prepare-views")
    views.add_argument("--dataset-root", type=Path, required=True)
    views.add_argument("--region-id", required=True)
    views.add_argument("--poses-csv", type=Path, required=True)
    views.add_argument("--source-id", default=SYNTHETIC_SOURCE_ID)
    views.add_argument("--validation-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "register-region":
        report = register_region(
            args.dataset_root,
            region_id=args.region_id,
            crs=args.crs,
            center_latitude=args.center_latitude,
            center_longitude=args.center_longitude,
            bounds_meters=tuple(args.bounds_meters),
            satellite_layout_relpath=args.satellite_layout,
        )
    elif args.command == "build-pyramid":
        report = build_pyramid(
            args.dataset_root,
            region_id=args.region_id,
            orthophoto=args.orthophoto,
            orthophoto_bounds=tuple(args.orthophoto_bounds),
        )
    elif args.command == "prepare-views":
        report = prepare_views(
            args.dataset_root,
            region_id=args.region_id,
            poses_csv=args.poses_csv,
            source_id=args.source_id,
            validation_fraction=args.validation_fraction,
        )
    else:  # pragma: no cover - argparse rejects unknown commands
        raise AssertionError(args.command)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
