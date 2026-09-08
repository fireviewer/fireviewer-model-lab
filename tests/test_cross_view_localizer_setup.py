from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
from fireviewer_model_lab.training.cross_view_localizer_setup import (
    REQUIRED_SATELLITE_LEVELS,
    SELECTED_ARTIFACTS,
    _safe_extract_tar,
    plan_acquisition,
    prepare,
)
from fireviewer_model_lab.training.spatial_training_setup import SetupError


def test_selective_plan_is_pinned_and_excludes_the_full_dataset(tmp_path: Path) -> None:
    report = plan_acquisition(tmp_path / "dataset")

    selected = {item["relative_path"] for item in report["selected_artifacts"]}
    assert report["planned_payload_bytes"] == 6_772_694_114
    assert report["remaining_download_bytes"] == 6_772_694_114
    assert report["full_dataset_downloaded"] is False
    assert selected == {artifact.relative_path for artifact in SELECTED_ARTIFACTS}
    assert "archives/streetview_images_001.tar" not in selected
    assert "archives/satellite_level_0_000.tar" not in selected
    assert len(selected) == 9


def _write_tar(path: Path, member_name: str, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as handle:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))


def test_safe_extract_accepts_regular_files_and_is_resumable(tmp_path: Path) -> None:
    archive = tmp_path / "valid.tar"
    destination = tmp_path / "output"
    _write_tar(archive, "satellite/-8/77/26.jpg", b"tile")

    first = _safe_extract_tar(archive, destination)
    second = _safe_extract_tar(archive, destination)

    assert (destination / "satellite/-8/77/26.jpg").read_bytes() == b"tile"
    assert first == second
    assert first["extracted_files"] == 1


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "malicious.tar"
    _write_tar(archive, "../escape.txt", b"blocked")

    with pytest.raises(SetupError, match="unsafe archive path"):
        _safe_extract_tar(archive, tmp_path / "output")

    assert not (tmp_path / "escape.txt").exists()


def _write_metadata(path: Path, image_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'image_id,latitude,longitude,sequence\n{image_id},38.9,-77.0,"{sequence}"\n',
        encoding="utf-8",
    )


def test_prepare_keeps_upstream_splits_disjoint_and_paths_portable(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    source_root = dataset_root / "sources/justzoomin-selective"
    repository = source_root / "repository"
    extracted = source_root / "extracted"
    ground = extracted / "streetview/images"
    ground.mkdir(parents=True)
    (ground / "train-image_undistorted.jpg").write_bytes(b"train")
    (ground / "validation-image_undistorted.jpg").write_bytes(b"validation")
    for level in REQUIRED_SATELLITE_LEVELS:
        (extracted / "satellite" / str(level)).mkdir(parents=True)
    _write_metadata(
        repository / "metadata/large_area_train_map.csv",
        "train-image",
        "[7, 6, 6, 5]",
    )
    _write_metadata(
        repository / "metadata/large_area_val_map.csv",
        "validation-image",
        "[7, 6, 2, 13]",
    )

    report = prepare(
        dataset_root,
        minimum_train_rows=1,
        minimum_validation_rows=1,
    )

    assert report["bootstrap_training_ready"] is True
    assert report["production_training_ready"] is False
    assert report["rows"] == {"train": 1, "validation": 1}
    train_row = json.loads(
        (dataset_root / "corpus/cross-view-coarse-localizer-v0.1.0/train.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert not Path(train_row["source_view_relpath"]).is_absolute()
    assert train_row["source_view_relpath"].endswith("train-image_undistorted.jpg")
