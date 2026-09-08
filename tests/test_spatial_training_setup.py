from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training.spatial_training_setup import (
    CameraRecord,
    SetupError,
    _camera_center,
    _deny_operational_path,
    _spatial_group,
    _spatial_split,
    build_pointing,
)


def _source_row(*, sample_id: str, sha256: str, kind: str) -> dict[str, object]:
    annotations: list[dict[str, object]] = []
    negative_tags: list[str] = []
    if kind == "positive":
        annotations = [
            {
                "class_name": "smoke_visible",
                "bbox_xywh": [10.0, 20.0, 40.0, 30.0],
            }
        ]
    elif kind == "negative":
        negative_tags = ["no_target_visible"]
    return {
        "sample_id": sample_id,
        "source_id": f"source-{sample_id}",
        "source_record_id": f"record-{sample_id}",
        "image_relpath": f"images/{sha256[:2]}/{sha256}.jpg",
        "sha256": sha256,
        "width": 100,
        "height": 100,
        "annotations": annotations,
        "negative_tags": negative_tags,
        "license": "CC-BY-4.0",
        "consent_basis": {"kind": "source_license", "reference": "test"},
        "event_id": f"event-{sample_id}",
        "sequence_id": f"sequence-{sample_id}",
        "split_group": f"group-{sample_id}",
        "split": "train",
    }


def test_pointing_build_references_media_without_copying(tmp_path: Path) -> None:
    sources = (
        ("fasdd", "positive"),
        ("pyro-sdis-v0.1.0", "negative"),
        ("wikimedia-candidates-v0.1.0", "candidate"),
    )
    for index, (directory_name, kind) in enumerate(sources):
        payload = f"image-{index}".encode()
        sha256 = hashlib.sha256(payload).hexdigest()
        source_root = tmp_path / "corpus" / directory_name
        image = source_root / "images" / sha256[:2] / f"{sha256}.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(payload)
        row = _source_row(sample_id=str(index), sha256=sha256, kind=kind)
        (source_root / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = build_pointing(tmp_path, verify_files=True)

    assert report["rows"] == 3
    assert report["media_files_copied"] == 0
    assert report["gates"]["setup_ready"] is True
    assert report["gates"]["training_ready"] is False
    output_root = tmp_path / "corpus" / "fire-pointing-v0.1.0"
    assert not list(output_root.rglob("*.jpg"))
    rows = [json.loads(line) for line in (output_root / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["targets"][0]["source_pixel_normalized"] == [0.3, 0.5]
    assert {row["pointing_status"] for row in rows} == {
        "point_candidate",
        "non_fire_negative_candidate",
        "annotation_candidate",
    }


def test_active_incident_paths_are_denied() -> None:
    with pytest.raises(SetupError, match="operational incident source denied"):
        _deny_operational_path("D:/data/fireviewer-operational-reference-r1-v4/terrain")


def test_camera_center_uses_world_to_camera_pose() -> None:
    assert _camera_center((1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)) == (
        -1.0,
        -2.0,
        -3.0,
    )


def test_spatial_cells_never_cross_splits() -> None:
    cameras: list[CameraRecord] = []
    for cell in range(20):
        for offset in (5.0, 15.0):
            cameras.append(
                CameraRecord(
                    name=f"camera-{cell}-{offset}",
                    quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                    translation_w2c=(0.0, 0.0, 0.0),
                    center_xyz=(cell * 100.0 + offset, 1000.0, 50.0),
                    intrinsic=(1.0, 1.0, 0.5, 0.5, 100, 100),
                )
            )

    splits = _spatial_split(cameras)
    group_splits: dict[str, set[str]] = {}
    for camera in cameras:
        group_splits.setdefault(_spatial_group(camera), set()).add(splits[camera.name])

    assert all(len(values) == 1 for values in group_splits.values())
    assert set(splits.values()) == {"train", "validation", "test"}
