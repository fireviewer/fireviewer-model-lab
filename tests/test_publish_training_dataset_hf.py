from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fireviewer_model_lab.tools.publish_training_dataset_hf import collect_asset_relpaths, stage_dataset


def test_collect_asset_relpaths_recurses_and_deduplicates() -> None:
    rows = [
        {
            "image_relpath": "sources/a.jpg",
            "source_view": {"image_relpath": "sources/a.jpg"},
            "map_view": {"image_relpath": "prepared/b.jpg"},
            "anchor_points": [{"x": 0.5, "y": 0.5}],
        }
    ]
    assert collect_asset_relpaths(rows) == ["prepared/b.jpg", "sources/a.jpg"]


def test_stage_dataset_uses_hardlinks_and_sanitized_metadata(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    asset = data_root / "sources" / "frame.jpg"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"frame")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "image_relpath": "sources/frame.jpg",
                "source_id": "source-a",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    staging = tmp_path / "stage"
    info = stage_dataset(
        data_root=data_root,
        manifest=manifest,
        staging=staging,
        dataset_id="fireviewer/test-dataset",
        title="Test Dataset",
        private=True,
        notice="Restricted test corpus.",
    )

    linked = staging / "sources" / "frame.jpg"
    assert linked.read_bytes() == b"frame"
    assert os.stat(asset).st_ino == os.stat(linked).st_ino
    assert info["rows"] == 1
    assert info["asset_files"] == 1
    rendered = (staging / "dataset-info.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in rendered
    assert "fireviewer/test-dataset" in rendered


def test_stage_rejects_manifest_path_escape(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image_relpath": "../secret.txt"}) + "\n")

    with pytest.raises(ValueError, match="Unsafe manifest path"):
        stage_dataset(
            data_root=data_root,
            manifest=manifest,
            staging=tmp_path / "stage",
            dataset_id="fireviewer/test",
            title="Test",
            private=True,
            notice="",
        )
