from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from PIL import Image
from fireviewer_model_lab.training.pointing_activefire_inventory import inventory_archive, validate_zip_members


def test_activefire_zip_validation_rejects_path_escape(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.png", b"x")
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(ValueError, match="unsafe ZIP member"),
    ):
        validate_zip_members(archive)


def test_activefire_inventory_records_mask_values(tmp_path: Path) -> None:
    mask = tmp_path / "mask.png"
    image = Image.new("L", (8, 4), 0)
    image.putpixel((3, 2), 255)
    image.save(mask)
    archive_path = tmp_path / "annotations.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(mask, "annotations/scene_mask.png")

    report = inventory_archive(archive_path, tmp_path / "extracted")

    assert report["image_members"] == 1
    assert report["decoded_images"] == 1
    assert report["decode_errors"] == []
    assert report["scalar_value_sets"] == {"0,255": 1}
    assert report["normalized_stems"] == ["scene"]
