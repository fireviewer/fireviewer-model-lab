from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from PIL import Image
from fireviewer_model_lab.training.pointing_new_source_flame2 import inventory_source, validate_zip_members


def test_zip_member_validation_rejects_path_escape(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.jpg", b"x")
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(ValueError, match="unsafe ZIP member"),
    ):
        validate_zip_members(archive)


def test_rgb_inventory_is_pending_strict_automatic_mask_conversion(tmp_path: Path) -> None:
    image_path = tmp_path / "FLAME2" / "rgb" / "scene_rgb_001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (24, 12), (255, 80, 20)).save(image_path)

    rows, report = inventory_source(tmp_path)

    assert report["decode_errors"] == []
    assert report["asset_kind_counts"] == {"rgb": 1}
    assert rows[0]["point_annotation_status"] == (
        "source_semantic_mask_pair_pending_strict_conversion"
    )
    assert rows[0]["mask_to_point_conversion"] == ("deterministic_class_mask_bottom_band_median")
    assert rows[0]["corpus_disposition"] == "pending_strict_automated_validation"
    assert not rows[0]["training_eligible"]
