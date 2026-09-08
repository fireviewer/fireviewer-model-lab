from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training.challenger_training import load_records, preflight_report


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _row(index: int, split: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "sample_id": f"sample-{index}",
        "source_id": "fireviewer-synthetic-v1",
        "source_revision": "scene-contract-1",
        "split": split,
        "split_group": f"incident-{index}",
        "image_relpath": f"images/{index}.png",
        "sha256": _digest(f"image-{index}"),
        "license": "FireViewer-Synthetic-1.0",
        "sample_validation_status": "double_validated",
        "annotations": [],
    }
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_preflight_accepts_held_out_challenger_records(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            _row(
                1,
                "train",
                mask_relpath="masks/1.png",
                mask_sha256=_digest("mask-1"),
                anchor_points=[{"kind": "fire_base", "x": 0.5, "y": 0.5}],
            ),
            _row(
                2,
                "validation",
                mask_relpath="masks/2.png",
                mask_sha256=_digest("mask-2"),
                anchor_points=[{"kind": "fire_base", "x": 0.5, "y": 0.5}],
            ),
            _row(
                3,
                "test",
                mask_relpath="masks/3.png",
                mask_sha256=_digest("mask-3"),
                anchor_points=[{"kind": "fire_base", "x": 0.5, "y": 0.5}],
                visual_abstention_reason="no_visible_ground_origin",
            ),
        ],
    )

    report = preflight_report(
        load_records([manifest]),
        model_family="DINOv3 multi-task",
        requires_masks=True,
        requires_anchors=True,
    )

    assert report["training_ready"] is True
    assert report["promotion_ready"] is True
    assert report["split_group_leakage"] == 0


def test_preflight_rejects_split_leakage_and_operational_incidents(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            _row(1, "train", split_group="same", is_operational_incident=True),
            _row(2, "validation", split_group="same"),
        ],
    )

    report = preflight_report(
        load_records([manifest]),
        model_family="RF-DETR Large",
        requires_masks=False,
        requires_anchors=False,
    )

    assert report["training_ready"] is False
    assert "operational_incident_forbidden:sample-1" in report["training_errors"]
    assert "split_group_leakage:same" in report["training_errors"]


def test_records_reject_duplicate_images_across_manifests(tmp_path: Path) -> None:
    shared = _digest("same-image")
    first = _write_manifest(tmp_path / "one.jsonl", [_row(1, "train", sha256=shared)])
    second = _write_manifest(tmp_path / "two.jsonl", [_row(2, "validation", sha256=shared)])

    with pytest.raises(ValueError, match="duplicate image SHA-256"):
        load_records([first, second])
