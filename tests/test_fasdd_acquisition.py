from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from fireviewer_model_lab.training.fasdd_acquisition import (
    ARCHIVES,
    ArchiveSpec,
    _coco_annotations,
    curate_archive,
    fasdd_split_group,
    inspect_archive,
    logical_zip_member,
    reconcile_fasdd_visual_splits,
    safe_zip_member,
    visual_fingerprint,
)


def test_selected_archives_avoid_the_combined_and_non_rgb_payloads() -> None:
    assert set(ARCHIVES) == {"CV", "RS", "UAV"}
    assert sum(spec.expected_bytes for spec in ARCHIVES.values()) == 31_921_782_564


@pytest.mark.parametrize(
    "member",
    ("../escape.jpg", "/absolute.jpg", "C:/absolute.jpg", "folder\\..\\escape.jpg"),
)
def test_safe_zip_member_rejects_paths_outside_the_archive_root(member: str) -> None:
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_zip_member(member)


def test_logical_zip_member_accepts_optional_fasdd_archive_root() -> None:
    assert logical_zip_member("FASDD_CV/images/example.jpg").as_posix() == "images/example.jpg"
    assert logical_zip_member("images/example.jpg").as_posix() == "images/example.jpg"


def test_inspect_archive_reports_structure_without_extracting(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("FASDD_RS/images/example.jpg", b"image")
        archive.writestr("FASDD_RS/labels/example.txt", b"0 0.5 0.5 0.2 0.2\n")

    report = inspect_archive(archive_path)

    assert report["member_count"] == 2
    assert report["suffix_counts"] == {".jpg": 1, ".txt": 1}
    assert report["member_samples"] == [
        "images/example.jpg",
        "labels/example.txt",
    ]


def test_fasdd_split_group_keeps_neighboring_source_numbers_together() -> None:
    assert fasdd_split_group("RS", "smoke_RS000024.tif") == fasdd_split_group(
        "RS", "smoke_RS000000.tif"
    )


def test_coco_annotations_clamp_one_pixel_overflow_and_drop_zero_area() -> None:
    from collections import Counter

    fixes: Counter[str] = Counter()
    annotations = _coco_annotations(
        [
            {"category_id": 0, "bbox": [10, 10, 91, 20]},
            {"category_id": 1, "bbox": [5, 5, 0, 1]},
        ],
        {0: "fire", 1: "smoke"},
        width=100,
        height=100,
        lot="CV",
        source_name="example.jpg",
        quality_fixes=fixes,
    )

    assert annotations[0]["bbox_xywh"] == [10.0, 10.0, 90.0, 20.0]
    assert fixes == {"clamped_one_pixel_bbox": 1, "dropped_non_positive_bbox": 1}
    assert fasdd_split_group("RS", "smoke_RS000025.tif") != fasdd_split_group(
        "RS", "smoke_RS000024.tif"
    )


def test_visual_fingerprint_is_stable_and_color_sensitive(tmp_path: Path) -> None:
    red = tmp_path / "red.png"
    orange = tmp_path / "orange.png"
    Image.new("RGB", (64, 48), color=(240, 20, 20)).save(red)
    Image.new("RGB", (64, 48), color=(240, 80, 20)).save(orange)

    assert visual_fingerprint(red.read_bytes()) == visual_fingerprint(red.read_bytes())
    assert visual_fingerprint(red.read_bytes()) != visual_fingerprint(orange.read_bytes())


def test_fasdd_reconciliation_avoids_phash_collision_chains_and_balances() -> None:
    records = []
    for index in range(100):
        records.append(
            {
                "sha256": f"{index:064x}",
                "phash": "0123456789abcdef",
                "visual_fingerprint": f"rgb32q5-v1:{index:064x}",
                "width": 640,
                "height": 480,
                "split_group": f"group-{index}",
                "split": "train",
                "near_duplicate_of": f"{max(0, index - 1):064x}" if index else None,
            }
        )
    records[1]["visual_fingerprint"] = records[0]["visual_fingerprint"]

    report = reconcile_fasdd_visual_splits(records)

    assert records[1]["near_duplicate_of"] == records[0]["sha256"]
    assert records[2]["near_duplicate_of"] is None
    assert records[0]["split"] == records[1]["split"]
    assert report["largest_component_rows"] == 2
    assert report["cross_split_visual_duplicates_after"] == 0
    assert report["split_row_counts"] == {"train": 70, "validation": 15, "test": 15}


def test_curate_archive_keeps_images_and_coco_only(tmp_path: Path) -> None:
    source_image = tmp_path / "image.png"
    Image.new("RGB", (64, 48), color=(220, 80, 20)).save(source_image)
    archive_path = tmp_path / "sample.zip"
    documents = {
        "train": {
            "images": [{"id": 1, "file_name": "fire000001.png", "width": 64, "height": 48}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [2, 3, 20, 15]}],
            "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}],
        },
        "val": {"images": [], "annotations": [], "categories": []},
        "test": {"images": [], "annotations": [], "categories": []},
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source_image, "images/fire000001.png")
        for split, document in documents.items():
            archive.writestr(
                f"annotations/COCO_TEST/Annotations/{split}.json",
                json.dumps(document),
            )
        archive.writestr("annotations/VOC_TEST/Annotations/fire000001.xml", "unused")
    payload = archive_path.read_bytes()
    spec = ArchiveSpec(
        lot="TEST",
        filename=archive_path.name,
        file_id="unused",
        expected_bytes=len(payload),
        expected_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )

    report = curate_archive(
        tmp_path,
        archive_path,
        spec,
        expected_rows=1,
        verify_source=True,
    )

    assert report["source_rows"] == 1
    assert report["appended_rows"] == 1
    manifest = (tmp_path / "corpus" / "fasdd" / "manifest.partial.jsonl").read_text()
    assert "flame_visible" in manifest
    assert not (tmp_path / "corpus" / "fasdd" / "annotations").exists()
