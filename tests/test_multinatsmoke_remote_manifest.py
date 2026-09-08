from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from fireviewer_model_lab.training.multinatsmoke_remote_manifest import build_remote_manifest


def _archive(path: Path, *, missing_mask: bool = False) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("MultiNatSmokeDataset/Train/D-Fire/images/a.jpg", b"image-a")
        if not missing_mask:
            archive.writestr("MultiNatSmokeDataset/Train/D-Fire/masks/a.png", b"mask-a")
        archive.writestr("MultiNatSmokeDataset/Test/WSDataset/images/b.jpg", b"image-b")
        archive.writestr("MultiNatSmokeDataset/Test/WSDataset/masks/b.png", b"mask-b")
        archive.writestr("MultiNatSmokeDataset/Test/blocked/images/c.jpg", b"ignored")


def _run(archive: Path, output: Path) -> dict:
    return build_remote_manifest(
        source=archive,
        output_dir=output,
        repository="owner/repo",
        revision="a" * 40,
        archive_lfs_sha256="b" * 64,
        expected_archive_bytes=archive.stat().st_size,
        allowed_sources={"D-Fire", "WSDataset"},
    )


def test_builds_paired_allowlisted_manifest_without_admission(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _archive(archive)

    report = _run(archive, tmp_path / "output")

    assert report["selected_pairs"] == 2
    assert report["missing_pair_count"] == 0
    assert report["payload_downloaded"] is False
    rows = [
        json.loads(line)
        for line in (tmp_path / "output/multinatsmoke_selected_members.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["source_family"] for row in rows} == {"D-Fire", "WSDataset"}
    assert all(row["training_eligible"] is False for row in rows)


def test_fails_closed_when_a_selected_pair_is_incomplete(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _archive(archive, missing_mask=True)

    with pytest.raises(ValueError, match="missing_image_mask_pairs"):
        _run(archive, tmp_path / "output")
