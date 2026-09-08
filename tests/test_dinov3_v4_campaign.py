from __future__ import annotations

import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training.dinov3_v4_campaign import load_registry, merge_manifests


def _row(sample_id: str, digest: str, split: str, group: str, image: str, mask: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_id": "test",
        "split": split,
        "split_group": group,
        "image_relpath": image,
        "image_sha256": digest,
        "mask_relpath": mask,
        "mask_sha256": "mask-" + digest,
        "mask_quality": "weak",
        "annotation_strength": "weak",
        "anchor_points": [],
        "visual_abstention_reason": None,
    }


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_weak_v4_registry_is_fail_closed_after_supersession() -> None:
    with pytest.raises(ValueError, match="weak-label campaign is superseded"):
        load_registry()


def test_merge_rejects_group_leakage(tmp_path: Path) -> None:
    for name in ("a.jpg", "a.png", "b.jpg", "b.png"):
        (tmp_path / name).write_bytes(b"x")
    manifest = tmp_path / "source.jsonl"
    _write_manifest(
        manifest,
        [
            _row("a", "a", "train", "shared", "a.jpg", "a.png"),
            _row("b", "b", "test", "shared", "b.jpg", "b.png"),
        ],
    )

    with pytest.raises(ValueError, match="split-group leakage"):
        merge_manifests(
            campaign_root=tmp_path,
            manifests=[manifest],
            output_root=tmp_path / "output",
        )


def test_merge_removes_exact_duplicate_image(tmp_path: Path) -> None:
    for name in ("a.jpg", "a.png", "b.jpg", "b.png"):
        (tmp_path / name).write_bytes(b"x")
    manifest = tmp_path / "source.jsonl"
    weak = _row("a", "same", "train", "a", "a.jpg", "a.png")
    strong = _row("b", "same", "validation", "b", "b.jpg", "b.png")
    strong["annotation_strength"] = "human_strong"
    _write_manifest(manifest, [weak, strong])

    report = merge_manifests(
        campaign_root=tmp_path,
        manifests=[manifest],
        output_root=tmp_path / "output",
    )

    assert report["rows"] == 1
    assert report["duplicate_images_removed"] == 1
