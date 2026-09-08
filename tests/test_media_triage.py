from __future__ import annotations

import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training import media_triage


def _write_fixture(root: Path) -> None:
    rows = [
        ("fasdd", "train", "fire", ["fire"]),
        ("pyro", "validation", "smoke", ["smoke"]),
        ("pyro", "test", "normal", ["normal"]),
    ]
    for source, split, primary_class, labels in rows:
        relative = media_triage.MANIFESTS[0 if source == "fasdd" else 1]
        manifest = root / relative
        manifest.parent.mkdir(parents=True, exist_ok=True)
        image = manifest.parent / f"{source}-{split}.jpg"
        image.write_bytes(f"{source}-{split}".encode())
        digest = media_triage._sha256_file(image)
        row = {
            "image_relpath": image.name,
            "labels": labels,
            "primary_class": primary_class,
            "sample_id": f"{source}-{split}",
            "sha256": digest,
            "source_id": source,
            "split": split,
            "split_group": f"{source}-{split}",
        }
        with manifest.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")


def test_preflight_validates_labels_splits_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture(tmp_path)
    monkeypatch.setattr(media_triage, "EXPECTED_ROWS", 3)
    monkeypatch.setattr(
        media_triage,
        "EXPECTED_SPLITS",
        {"test": 1, "train": 1, "validation": 1},
    )
    monkeypatch.setattr(
        media_triage,
        "EXPECTED_CLASSES",
        {"fire": 1, "normal": 1, "smoke": 1},
    )
    report = media_triage.preflight(tmp_path, verify_hashes=True)
    assert report["dataset_ready"] is True
    assert report["training_ready"] is False
    assert report["verified_media"] == 3


def test_preflight_rejects_a_split_group_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fixture(tmp_path)
    manifests = [tmp_path / relative for relative in media_triage.MANIFESTS]
    row = json.loads(manifests[1].read_text(encoding="utf-8").splitlines()[0])
    row["split_group"] = "fasdd-train"
    row["source_id"] = "fasdd"
    with manifests[1].open("w", encoding="utf-8") as output:
        output.write(json.dumps(row) + "\n")
    monkeypatch.setattr(media_triage, "EXPECTED_ROWS", 2)
    with pytest.raises(media_triage.MediaTriageError, match="split-group leakage"):
        media_triage.preflight(tmp_path)
