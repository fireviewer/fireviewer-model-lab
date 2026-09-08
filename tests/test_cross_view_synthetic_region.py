from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image
from pyproj import Transformer
from fireviewer_model_lab.training.cross_view_synthetic_region import (
    SYNTHETIC_LICENSE,
    _actions_for_position,
    _split_for_group,
    build_pyramid,
    prepare_views,
    register_region,
)
from fireviewer_model_lab.training.spatial_train_cross_view_localizer import (
    _final_grid_index,
    load_region_registry,
    region_bounds,
    region_registry_path,
)
from fireviewer_model_lab.training.spatial_training_setup import SetupError

REGION_ID = "corbieres-synthetic-v1"
CENTER = (43.05, 2.77)
BOUNDS_METERS = (-5_000.0, 5_000.0, -5_000.0, 5_000.0)
LAYOUT_RELPATH = f"sources/synthetic-{REGION_ID}/satellite/layout.yaml"
DC_LAYOUT_RELPATH = "sources/justzoomin-selective/extracted/satellite/layout.yaml"


def _bootstrap_dc_layout(tmp_path: Path) -> None:
    layout = tmp_path / DC_LAYOUT_RELPATH
    layout.parent.mkdir(parents=True, exist_ok=True)
    layout.write_text("crs: epsg:26985\n", encoding="utf-8")


def _register(tmp_path: Path):
    _bootstrap_dc_layout(tmp_path)
    return register_region(
        tmp_path,
        region_id=REGION_ID,
        crs="EPSG:2154",
        center_latitude=CENTER[0],
        center_longitude=CENTER[1],
        bounds_meters=BOUNDS_METERS,
        satellite_layout_relpath=LAYOUT_RELPATH,
    )


def test_register_region_preserves_bootstrap_entry(tmp_path: Path) -> None:
    report = _register(tmp_path)

    registry = load_region_registry(tmp_path)
    assert [spec.region_id for spec in registry] == [
        "washington-dc-benchmark",
        REGION_ID,
    ]
    assert report["square_meters"] == pytest.approx(10_000.0, abs=0.01)
    raw = yaml.safe_load(region_registry_path(tmp_path).read_text(encoding="utf-8"))
    assert raw["regions"][1]["crs"] == "EPSG:2154"


def test_register_region_rejects_bootstrap_override(tmp_path: Path) -> None:
    with pytest.raises(SetupError, match="bootstrap region"):
        register_region(
            tmp_path,
            region_id="washington-dc-benchmark",
            crs="EPSG:26985",
            center_latitude=38.8936,
            center_longitude=-77.0116,
            bounds_meters=(-3_000.0, 7_000.0, -5_000.0, 5_000.0),
            satellite_layout_relpath=LAYOUT_RELPATH,
        )


def test_register_region_rejects_non_square_bounds(tmp_path: Path) -> None:
    with pytest.raises(SetupError, match="not square"):
        register_region(
            tmp_path,
            region_id=REGION_ID,
            crs="EPSG:2154",
            center_latitude=CENTER[0],
            center_longitude=CENTER[1],
            bounds_meters=(-5_000.0, 6_000.0, -5_000.0, 5_000.0),
            satellite_layout_relpath=LAYOUT_RELPATH,
        )


def test_actions_for_position_round_trips_grid_cell(tmp_path: Path) -> None:
    _register(tmp_path)
    spec = next(s for s in load_region_registry(tmp_path) if s.region_id == REGION_ID)
    bounds = region_bounds(spec)
    west, east, _, north = bounds
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

    for grid_row, grid_column in ((0, 0), (20, 30), (120, 200), (255, 255)):
        size = east - west
        longitude, latitude = inverse.transform(
            west + (grid_column + 0.5) * size / 256,
            north - (grid_row + 0.5) * size / 256,
        )
        actions = _actions_for_position(bounds, transformer, latitude, longitude)

        assert actions is not None
        assert _final_grid_index(actions) == (grid_row, grid_column)


def test_actions_for_position_rejects_points_outside_region(tmp_path: Path) -> None:
    _register(tmp_path)
    spec = next(s for s in load_region_registry(tmp_path) if s.region_id == REGION_ID)
    bounds = region_bounds(spec)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

    assert _actions_for_position(bounds, transformer, 48.85, 2.35) is None


def test_split_for_group_is_deterministic_and_group_safe() -> None:
    first = _split_for_group("batch-a", 0.1)
    second = _split_for_group("batch-a", 0.1)

    assert first == second
    assert first in {"train", "validation"}


def _write_pose_image(root: Path, relpath: str, color: tuple[int, int, int]) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path)


def _write_poses_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_relpath", "latitude", "longitude", "render_group"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _pyramid_layout(tmp_path: Path) -> Path:
    layout = tmp_path / LAYOUT_RELPATH
    layout.parent.mkdir(parents=True, exist_ok=True)
    layout.write_text(
        "crs: epsg:2154\n"
        "max_zoom: -2\n"
        "min_zoom: -8\n"
        "origin_crs: [0.0, 0.0]\n"
        "path: '{zoom}/{x}/{y}.jpg'\n"
        "tile_axes: [east, north]\n"
        "tile_shape_crs: [20.0, 20.0]\n"
        "tile_shape_px: [250, 250]\n",
        encoding="utf-8",
    )
    return layout


def test_prepare_views_merges_and_replaces_source_rows(tmp_path: Path) -> None:
    _register(tmp_path)
    _pyramid_layout(tmp_path)
    spec = next(s for s in load_region_registry(tmp_path) if s.region_id == REGION_ID)
    west, east, south, north = region_bounds(spec)
    inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    center_lon, center_lat = inverse.transform((west + east) / 2.0, (south + north) / 2.0)
    _write_pose_image(tmp_path, "renders/batch-a/0001.jpg", (10, 20, 30))
    _write_pose_image(tmp_path, "renders/batch-b/0002.jpg", (40, 50, 60))
    poses = tmp_path / "poses.csv"
    _write_poses_csv(
        poses,
        [
            {
                "image_relpath": "renders/batch-a/0001.jpg",
                "latitude": str(center_lat),
                "longitude": str(center_lon),
                "render_group": "batch-a",
            },
            {
                "image_relpath": "renders/batch-b/0002.jpg",
                "latitude": str(center_lat),
                "longitude": str(center_lon),
                "render_group": "batch-b",
            },
        ],
    )

    report = prepare_views(
        tmp_path,
        region_id=REGION_ID,
        poses_csv=poses,
        validation_fraction=0.4,
    )

    assert report["views_parsed"] == 2
    assert report["rows_written"]["train"] + report["rows_written"]["validation"] == 2
    manifest_rows = []
    for split in ("train", "validation"):
        manifest = tmp_path / "corpus" / "cross-view-coarse-localizer-v0.1.0" / f"{split}.jsonl"
        with manifest.open(encoding="utf-8") as handle:
            manifest_rows.extend(json.loads(line) for line in handle if line.strip())
    assert len(manifest_rows) == 2
    for row in manifest_rows:
        assert row["region_id"] == REGION_ID
        assert row["license"] == SYNTHETIC_LICENSE
        assert row["operational_incident"] is False
        assert row["training_membership"] is True
        assert len(row["action_sequence"]) == 4

    second = prepare_views(
        tmp_path,
        region_id=REGION_ID,
        poses_csv=poses,
        validation_fraction=0.4,
    )
    assert second["rows_replaced"] == 2
    total = 0
    for split in ("train", "validation"):
        manifest = tmp_path / "corpus" / "cross-view-coarse-localizer-v0.1.0" / f"{split}.jsonl"
        with manifest.open(encoding="utf-8") as handle:
            total += sum(1 for line in handle if line.strip())
    assert total == 2


def test_prepare_views_rejects_pose_outside_region(tmp_path: Path) -> None:
    _register(tmp_path)
    _pyramid_layout(tmp_path)
    spec = next(s for s in load_region_registry(tmp_path) if s.region_id == REGION_ID)
    west, east, south, north = region_bounds(spec)
    inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    center_lon, center_lat = inverse.transform((west + east) / 2.0, (south + north) / 2.0)
    _write_pose_image(tmp_path, "renders/batch-a/0001.jpg", (10, 20, 30))
    _write_pose_image(tmp_path, "renders/batch-a/0002.jpg", (70, 80, 90))
    poses = tmp_path / "poses.csv"
    _write_poses_csv(
        poses,
        [
            {
                "image_relpath": "renders/batch-a/0001.jpg",
                "latitude": str(center_lat),
                "longitude": str(center_lon),
                "render_group": "batch-a",
            },
            {
                "image_relpath": "renders/batch-a/0002.jpg",
                "latitude": "48.8566",
                "longitude": "2.3522",
                "render_group": "batch-a",
            },
        ],
    )

    report = prepare_views(tmp_path, region_id=REGION_ID, poses_csv=poses)

    assert report["views_parsed"] == 1
    assert report["views_outside_region_rejected"] == 1


def test_prepare_views_rejects_empty_poses_csv(tmp_path: Path) -> None:
    _register(tmp_path)
    _pyramid_layout(tmp_path)
    _write_pose_image(tmp_path, "renders/batch-a/0001.jpg", (10, 20, 30))
    poses = tmp_path / "poses.csv"
    _write_poses_csv(
        poses,
        [
            {
                "image_relpath": "renders/batch-a/0001.jpg",
                "latitude": "48.8566",
                "longitude": "2.3522",
                "render_group": "batch-a",
            }
        ],
    )

    with pytest.raises(SetupError, match="no usable view"):
        prepare_views(tmp_path, region_id=REGION_ID, poses_csv=poses)


def test_prepare_views_rejects_duplicate_image_content(tmp_path: Path) -> None:
    _register(tmp_path)
    _pyramid_layout(tmp_path)
    spec = next(s for s in load_region_registry(tmp_path) if s.region_id == REGION_ID)
    west, east, south, north = region_bounds(spec)
    inverse = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    center_lon, center_lat = inverse.transform((west + east) / 2.0, (south + north) / 2.0)
    _write_pose_image(tmp_path, "renders/batch-a/0001.jpg", (10, 20, 30))
    source = tmp_path / "renders/batch-a/0001.jpg"
    duplicate = tmp_path / "renders/batch-b/0002.jpg"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_bytes(source.read_bytes())
    poses = tmp_path / "poses.csv"
    _write_poses_csv(
        poses,
        [
            {
                "image_relpath": "renders/batch-a/0001.jpg",
                "latitude": str(center_lat),
                "longitude": str(center_lon),
                "render_group": "batch-a",
            },
            {
                "image_relpath": "renders/batch-b/0002.jpg",
                "latitude": str(center_lat),
                "longitude": str(center_lon),
                "render_group": "batch-b",
            },
        ],
    )

    with pytest.raises(SetupError, match="duplicate rendered view content"):
        prepare_views(tmp_path, region_id=REGION_ID, poses_csv=poses)


def test_build_pyramid_writes_four_levels(tmp_path: Path) -> None:
    _bootstrap_dc_layout(tmp_path)
    small_region = "test-forest-v1"
    register_region(
        tmp_path,
        region_id=small_region,
        crs="EPSG:2154",
        center_latitude=CENTER[0],
        center_longitude=CENTER[1],
        bounds_meters=(-500.0, 500.0, -500.0, 500.0),
        satellite_layout_relpath=f"sources/synthetic-{small_region}/satellite/layout.yaml",
    )
    spec = next(s for s in load_region_registry(tmp_path) if s.region_id == small_region)
    west, east, south, north = region_bounds(spec)
    orthophoto = tmp_path / "orthophoto.jpg"
    Image.new("RGB", (200, 200), (80, 120, 60)).save(orthophoto, quality=95)

    report = build_pyramid(
        tmp_path,
        region_id=small_region,
        orthophoto=orthophoto,
        orthophoto_bounds=(west, east, south, north),
    )

    assert report["tiles_written"] == {
        "-8": 1,
        "-6": 1,
        "-4": 16,
        "-2": 169,
    }
    layout = yaml.safe_load(
        (tmp_path / f"sources/synthetic-{small_region}/satellite/layout.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert layout["crs"] == "epsg:2154"
    assert layout["origin_crs"] == [west, south]
    sample = tmp_path / f"sources/synthetic-{small_region}/satellite/-2/6/6.jpg"
    assert sample.is_file()
    with Image.open(sample) as tile:
        assert tile.size == (250, 250)


def test_build_pyramid_rejects_orthophoto_not_covering_region(tmp_path: Path) -> None:
    _register(tmp_path)
    orthophoto = tmp_path / "orthophoto.jpg"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(orthophoto)

    with pytest.raises(SetupError, match="does not cover"):
        build_pyramid(
            tmp_path,
            region_id=REGION_ID,
            orthophoto=orthophoto,
            orthophoto_bounds=(0.0, 1_000.0, 0.0, 1_000.0),
        )
