from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from fireviewer_model_lab.training.dino_complements import (
    build_plan,
    extract_archives,
    find_source,
    load_registry,
)


def test_registry_keeps_sources_out_of_training_until_labels_are_derived() -> None:
    registry = load_registry()
    plan = build_plan(registry)

    assert set(plan["ready_downloads"]) == {"hpwren-figlib", "firestereo-firesgl"}
    assert all(source["multitask_status"] != "training_ready" for source in plan["sources"])


def test_wit_uses_asset_boxes_as_exclusions_and_groups_by_burn_site() -> None:
    source = find_source(load_registry(), "wit-uas")

    recipe = source["multitask_role"]["annotation_recipe"]
    assert "exclude_person_and_vehicle_boxes_with_safety_margin" in recipe
    assert source["split_policy"] == "hold_out_complete_burn_site_and_flight_groups"
    assert "unsuitable_for_precise_mapping" in source["cross_view_role"]["geometry_limitations"]


def test_firesgl_is_bounded_and_not_promoted_to_geographic_training() -> None:
    source = find_source(load_registry(), "firestereo-firesgl")

    assert sum(asset["expected_bytes"] for asset in source["assets"]) == 43_614_968_427
    assert source["sampling_policy"]["maximum_selected_frames_per_sequence"] == 1500
    assert source["cross_view_role"]["status"] == "robustness_validation_only"
    assert source["split_policy"] == "never_split_archives_from_same_sequence_group"


def test_figlib_uses_one_bulk_bundle() -> None:
    source = find_source(load_registry(), "hpwren-figlib")

    assert source["acquisition"]["strategy"] == "resumable_direct_http"
    assert len(source["assets"]) == 1
    assert source["assets"][0]["expected_bytes"] == 12_020_094_336


def test_extraction_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive = archive_root / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "forbidden")
    source = {
        "source_id": "unsafe",
        "assets": [
            {
                "filename": archive.name,
                "expected_bytes": archive.stat().st_size,
                "sequence_group": "unsafe",
            }
        ],
    }

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        extract_archives(source, archive_root, tmp_path / "output")
