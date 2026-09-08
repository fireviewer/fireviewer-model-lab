from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from pyproj import Transformer
from fireviewer_model_lab.training.spatial_train_cross_view_localizer import (
    DEFAULT_REGION_SPEC,
    TOTAL_STATES,
    CorpusRow,
    CrossViewProjectionHead,
    RegionCalibration,
    RegionSpec,
    TilePyramidRenderer,
    _examples,
    _final_grid_index,
    _mask_invalid_actions,
    _masked_cross_entropy,
    _region_state_offsets,
    _state_keys,
    derive_region_calibration,
    derive_region_calibrations,
    load_region_registry,
    region_bounds,
    state_bounds,
)
from fireviewer_model_lab.training.spatial_training_setup import SetupError


def _actions_for_cell(row_index: int, column_index: int) -> tuple[int, int, int, int]:
    actions = [0, 0, 0, 0]
    for index in range(3, -1, -1):
        row_digit = row_index % 4
        column_digit = column_index % 4
        actions[index] = row_digit * 4 + column_digit
        row_index //= 4
        column_index //= 4
    return actions[0], actions[1], actions[2], actions[3]


def test_region_calibration_recovers_fixed_geographic_contract(tmp_path: Path) -> None:
    inverse = Transformer.from_crs("EPSG:26985", "EPSG:4326", always_xy=True)
    west, east, south, north = region_bounds(DEFAULT_REGION_SPEC)
    size = east - west
    rows = []
    for index, (grid_row, grid_column) in enumerate(
        ((0, 0), (20, 30), (64, 100), (120, 150), (180, 200), (255, 255))
    ):
        projected_east = west + (grid_column + 0.5) * size / 256
        projected_north = north - (grid_row + 0.5) * size / 256
        longitude, latitude = inverse.transform(projected_east, projected_north)
        rows.append(
            CorpusRow(
                sample_id=f"sample-{index}",
                image_path=tmp_path / f"{index}.jpg",
                latitude=latitude,
                longitude=longitude,
                actions=_actions_for_cell(grid_row, grid_column),
            )
        )

    calibration = derive_region_calibration(rows)

    assert calibration.west == pytest.approx(west, abs=0.01)
    assert calibration.east == pytest.approx(east, abs=0.01)
    assert calibration.south == pytest.approx(south, abs=0.01)
    assert calibration.north == pytest.approx(north, abs=0.01)
    assert calibration.train_cell_agreement == 1.0
    assert calibration.train_within_one_cell_agreement == 1.0


def test_state_bounds_and_state_offsets_cover_every_autoregressive_prefix() -> None:
    calibration = RegionCalibration(
        west=0,
        east=10_000,
        south=0,
        north=10_000,
        crs="EPSG:26985",
        train_cell_agreement=1.0,
        train_within_one_cell_agreement=1.0,
        maximum_cell_error=0,
    )
    action = 6  # row 1, column 2

    bounds = state_bounds(calibration, 1, action)

    assert bounds == pytest.approx((5_000, 7_500, 5_000, 7_500))
    assert len(_state_keys()) == 4_369
    assert _final_grid_index((0, 0, 0, 0)) == (0, 0)
    assert _final_grid_index((15, 15, 15, 15)) == (255, 255)


def test_renderer_preserves_north_up_tile_order(tmp_path: Path) -> None:
    satellite = tmp_path / "satellite"
    layout = satellite / "layout.yaml"
    layout.parent.mkdir(parents=True)
    layout.write_text(
        "crs: epsg:26985\n"
        "max_zoom: 0\n"
        "min_zoom: 0\n"
        "origin_crs: [0.0, 0.0]\n"
        "path: '{zoom}/{x}/{y}.jpg'\n"
        "tile_axes: [east, north]\n"
        "tile_shape_crs: [20.0, 20.0]\n"
        "tile_shape_px: [20, 20]\n",
        encoding="utf-8",
    )
    colors = {
        (0, 1): (255, 0, 0),
        (1, 1): (0, 255, 0),
        (0, 0): (0, 0, 255),
        (1, 0): (255, 255, 0),
    }
    for (x, y), color in colors.items():
        path = satellite / "0" / str(x) / f"{y}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), color).save(path, quality=100, subsampling=0)

    renderer = TilePyramidRenderer(layout)
    required = renderer.required_tile_paths(0, (0, 40, 0, 40))
    rendered = renderer.render(0, (0, 40, 0, 40))

    assert len(required) == 4
    assert rendered.getpixel((28, 28))[0] > 240
    assert rendered.getpixel((196, 28))[1] > 240
    assert rendered.getpixel((28, 196))[2] > 240
    bottom_right = rendered.getpixel((196, 196))
    assert bottom_right[0] > 240 and bottom_right[1] > 240


def test_renderer_uses_explicit_default_and_marks_missing_cells(tmp_path: Path) -> None:
    satellite = tmp_path / "satellite"
    layout = satellite / "layout.yaml"
    layout.parent.mkdir(parents=True)
    layout.write_text(
        "crs: epsg:26985\n"
        "max_zoom: 0\n"
        "min_zoom: 0\n"
        "origin_crs: [0.0, 0.0]\n"
        "path: '{zoom}/{x}/{y}.jpg'\n"
        "tile_axes: [east, north]\n"
        "tile_shape_crs: [20.0, 20.0]\n"
        "tile_shape_px: [20, 20]\n",
        encoding="utf-8",
    )
    present = satellite / "0" / "0" / "0.jpg"
    present.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), (200, 100, 50)).save(present)

    rendered = TilePyramidRenderer(layout).render_with_coverage(0, (0, 40, 0, 40))

    assert rendered.present_tiles == 1
    assert rendered.required_tiles == 4
    assert rendered.image.getpixel((196, 28)) == (0, 0, 0)
    assert rendered.valid_cells[:8].sum() == 0
    assert rendered.valid_cells[8:].sum() == 4


def test_renderer_rejects_unsafe_path_template(tmp_path: Path) -> None:
    layout = tmp_path / "layout.yaml"
    layout.write_text(
        "crs: epsg:26985\n"
        "max_zoom: 0\n"
        "min_zoom: 0\n"
        "origin_crs: [0.0, 0.0]\n"
        "path: '../{zoom}/{x}/{y}.jpg'\n"
        "tile_axes: [east, north]\n"
        "tile_shape_crs: [20.0, 20.0]\n"
        "tile_shape_px: [20, 20]\n",
        encoding="utf-8",
    )

    with pytest.raises(SetupError, match="unsafe satellite tile path template"):
        TilePyramidRenderer(layout)


def test_projection_head_scores_sixteen_cells() -> None:
    model = CrossViewProjectionHead(embedding_dimension=8)
    ground = torch.randn(3, 8)
    satellite = torch.randn(3, 16, 8)
    steps = torch.tensor([0, 1, 3])

    logits = model(ground, satellite, steps)

    assert logits.shape == (3, 16)


def test_action_mask_blocks_missing_cells_but_keeps_empty_branch_finite() -> None:
    logits = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    valid = torch.zeros((2, 16), dtype=torch.bool)
    valid[0, 3] = True

    masked = _mask_invalid_actions(logits, valid)

    assert masked[0].argmax().item() == 3
    assert torch.equal(masked[1], logits[1])


def test_masked_cross_entropy_smooths_only_over_available_cells() -> None:
    logits = torch.zeros((1, 16), dtype=torch.float32, requires_grad=True)
    valid = torch.zeros((1, 16), dtype=torch.bool)
    valid[0, :2] = True

    loss = _masked_cross_entropy(logits, torch.tensor([0]), valid, label_smoothing=0.02)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.equal(logits.grad[0, 2:], torch.zeros(14))


FRANCE_REGION = RegionSpec(
    region_id="corbieres-synthetic-v1",
    crs="EPSG:2154",
    center_latitude=43.05,
    center_longitude=2.77,
    bounds_meters=(-5_000.0, 5_000.0, -5_000.0, 5_000.0),
    satellite_layout_relpath="sources/synthetic-corbieres-synthetic-v1/satellite/layout.yaml",
)


def test_region_registry_defaults_to_bootstrap_region(tmp_path: Path) -> None:
    registry = load_region_registry(tmp_path)

    assert registry == (DEFAULT_REGION_SPEC,)


def test_region_registry_loads_declared_regions(tmp_path: Path) -> None:
    layout = tmp_path / FRANCE_REGION.satellite_layout_relpath
    layout.parent.mkdir(parents=True)
    layout.write_text("crs: epsg:2154\n", encoding="utf-8")
    registry_path = tmp_path / "corpus" / "cross-view-coarse-localizer-v0.1.0"
    registry_path.mkdir(parents=True)
    (registry_path / "regions.yaml").write_text(
        "schema_version: 1\n"
        "regions:\n"
        "  - region_id: washington-dc-benchmark\n"
        "    crs: EPSG:26985\n"
        "    center_latitude: 38.8936\n"
        "    center_longitude: -77.0116\n"
        "    bounds_meters: [-3000.0, 7000.0, -5000.0, 5000.0]\n"
        "    satellite_layout: sources/justzoomin-selective/extracted/satellite/layout.yaml\n"
        f"  - region_id: {FRANCE_REGION.region_id}\n"
        "    crs: EPSG:2154\n"
        f"    center_latitude: {FRANCE_REGION.center_latitude}\n"
        f"    center_longitude: {FRANCE_REGION.center_longitude}\n"
        "    bounds_meters: [-5000.0, 5000.0, -5000.0, 5000.0]\n"
        f"    satellite_layout: {FRANCE_REGION.satellite_layout_relpath}\n",
        encoding="utf-8",
    )

    registry = load_region_registry(tmp_path)

    assert [spec.region_id for spec in registry] == [
        "washington-dc-benchmark",
        FRANCE_REGION.region_id,
    ]


def test_region_registry_rejects_duplicate_region_ids(tmp_path: Path) -> None:
    layout = tmp_path / "sources/justzoomin-selective/extracted/satellite/layout.yaml"
    layout.parent.mkdir(parents=True)
    layout.write_text("crs: epsg:26985\n", encoding="utf-8")
    registry_path = tmp_path / "corpus" / "cross-view-coarse-localizer-v0.1.0"
    registry_path.mkdir(parents=True)
    entry = (
        "  - region_id: washington-dc-benchmark\n"
        "    crs: EPSG:26985\n"
        "    center_latitude: 38.8936\n"
        "    center_longitude: -77.0116\n"
        "    bounds_meters: [-3000.0, 7000.0, -5000.0, 5000.0]\n"
        "    satellite_layout: sources/justzoomin-selective/extracted/satellite/layout.yaml\n"
    )
    (registry_path / "regions.yaml").write_text(
        f"schema_version: 1\nregions:\n{entry}{entry}",
        encoding="utf-8",
    )

    with pytest.raises(SetupError, match="duplicate region id"):
        load_region_registry(tmp_path)


def test_region_bounds_projects_french_square() -> None:
    west, east, south, north = region_bounds(FRANCE_REGION)

    assert east - west == pytest.approx(10_000.0, abs=0.01)
    assert north - south == pytest.approx(10_000.0, abs=0.01)


def test_derive_region_calibrations_covers_each_referenced_region(tmp_path: Path) -> None:
    rows = []
    for region, inverse_crs in (
        (DEFAULT_REGION_SPEC, "EPSG:26985"),
        (FRANCE_REGION, "EPSG:2154"),
    ):
        west, east, _south, north = region_bounds(region)
        inverse = Transformer.from_crs(inverse_crs, "EPSG:4326", always_xy=True)
        for index, (grid_row, grid_column) in enumerate(((0, 0), (255, 255))):
            size = east - west
            projected_east = west + (grid_column + 0.5) * size / 256
            projected_north = north - (grid_row + 0.5) * size / 256
            longitude, latitude = inverse.transform(projected_east, projected_north)
            rows.append(
                CorpusRow(
                    sample_id=f"{region.region_id}-{index}",
                    image_path=tmp_path / f"{index}.jpg",
                    latitude=latitude,
                    longitude=longitude,
                    actions=_actions_for_cell(grid_row, grid_column),
                    region_id=region.region_id,
                )
            )

    calibrations = derive_region_calibrations(rows, (DEFAULT_REGION_SPEC, FRANCE_REGION))

    assert set(calibrations) == {DEFAULT_REGION_SPEC.region_id, FRANCE_REGION.region_id}
    assert calibrations[FRANCE_REGION.region_id].crs == "EPSG:2154"
    assert calibrations[FRANCE_REGION.region_id].train_cell_agreement == 1.0


def test_examples_offset_satellite_states_per_region(tmp_path: Path) -> None:
    rows = [
        CorpusRow(
            sample_id="dc",
            image_path=tmp_path / "dc.jpg",
            latitude=38.9,
            longitude=-77.0,
            actions=(1, 2, 3, 4),
            region_id=DEFAULT_REGION_SPEC.region_id,
        ),
        CorpusRow(
            sample_id="fr",
            image_path=tmp_path / "fr.jpg",
            latitude=43.05,
            longitude=2.77,
            actions=(4, 3, 2, 1),
            region_id=FRANCE_REGION.region_id,
        ),
    ]
    offsets = _region_state_offsets((DEFAULT_REGION_SPEC, FRANCE_REGION))

    ground, satellite, labels, steps = _examples(rows, offsets)

    assert offsets[FRANCE_REGION.region_id] == TOTAL_STATES
    dc_first = int(satellite[0])
    fr_first = int(satellite[4])
    assert dc_first == 0
    assert fr_first == TOTAL_STATES
    assert int(satellite[1]) == 1 + 1  # step 1 offset plus first action prefix
    assert int(satellite[5]) == TOTAL_STATES + 1 + 4
    assert int(labels[0]) == 1 and int(labels[4]) == 4
    assert ground.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert steps.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
