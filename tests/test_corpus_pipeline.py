from __future__ import annotations

import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training.corpus_pipeline import (
    deterministic_split,
    parse_yolo_annotations,
    reconcile_near_duplicate_splits,
    sha256_bytes,
    validate_manifest,
)


def test_split_is_stable_for_a_camera_group() -> None:
    group = "pyro_sdis_a1e553e:sdis-07:brison-200"
    assert deterministic_split(group) == deterministic_split(group)
    assert deterministic_split(group) in {"train", "validation", "test"}


def test_yolo_parser_preserves_negatives_and_converts_smoke_boxes() -> None:
    assert parse_yolo_annotations("", width=1280, height=720) == []

    annotations = parse_yolo_annotations(
        "1 0.5 0.5 0.25 0.5\n1 0.2 0.2 0.1 0.1",
        width=1000,
        height=500,
    )
    assert len(annotations) == 2
    assert annotations[0]["class_name"] == "smoke_visible"
    assert annotations[0]["bbox_xywh"] == [375.0, 125.0, 250.0, 250.0]


def test_yolo_parser_rejects_unmapped_or_out_of_bounds_boxes() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        parse_yolo_annotations("0 0.5 0.5 0.2 0.2", width=640, height=640)
    with pytest.raises(ValueError, match="outside"):
        parse_yolo_annotations("1 0.02 0.5 0.2 0.2", width=640, height=640)


def _record(*, digest: str, split: str, split_group: str) -> dict[str, object]:
    return {
        "sample_id": f"sample:{digest[:16]}",
        "source_id": "test-source",
        "source_record_id": digest[:16],
        "corpus_role": "detector_training",
        "image_relpath": f"images/{digest[:2]}/{digest}.jpg",
        "sha256": digest,
        "phash": "0123456789abcdef",
        "near_duplicate_of": None,
        "width": 64,
        "height": 64,
        "event_id": split_group,
        "sequence_id": split_group,
        "split_group": split_group,
        "captured_at_literal": None,
        "split": split,
        "license": "Apache-2.0",
        "consent_basis": {"kind": "source_license", "reference": "test"},
        "sample_validation_status": "source_provided",
        "candidate_classes": [],
        "annotations": [],
        "negative_tags": ["no_target_visible"],
        "location": None,
    }


def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_manifest_validation_detects_group_leakage(tmp_path: Path) -> None:
    first = sha256_bytes(b"first")
    second = sha256_bytes(b"second")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            _record(digest=first, split="train", split_group="same-camera"),
            _record(digest=second, split="test", split_group="same-camera"),
        ],
    )

    with pytest.raises(ValueError, match="Split leakage"):
        validate_manifest(manifest)


def test_manifest_validation_can_verify_image_digest(tmp_path: Path) -> None:
    payload = b"not-a-decoded-image-but-content-addressed"
    digest = sha256_bytes(payload)
    image_path = tmp_path / "images" / digest[:2] / f"{digest}.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(payload)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [_record(digest=digest, split="train", split_group="camera-a")],
    )

    report = validate_manifest(manifest, verify_files=True)
    assert report["rows"] == 1
    assert report["negative_rows"] == 1
    assert report["files_verified"] is True
    assert report["four_class_training_ready"] is False


def test_manifest_reports_cross_split_near_duplicates(tmp_path: Path) -> None:
    first = sha256_bytes(b"first-near")
    second = sha256_bytes(b"second-near")
    first_record = _record(digest=first, split="train", split_group="camera-a")
    second_record = _record(digest=second, split="test", split_group="camera-b")
    second_record["near_duplicate_of"] = first
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [first_record, second_record])

    report = validate_manifest(manifest)

    assert report["cross_split_near_duplicates"] == 1
    assert report["four_class_training_ready"] is False


def test_near_duplicate_reconciliation_keeps_linked_groups_together() -> None:
    first = sha256_bytes(b"first-component")
    second = sha256_bytes(b"second-component")
    records = [
        _record(digest=first, split="train", split_group="camera-a"),
        _record(digest=second, split="test", split_group="camera-b"),
    ]
    records[1]["near_duplicate_of"] = first

    report = reconcile_near_duplicate_splits(records)

    assert records[0]["split"] == records[1]["split"]
    assert report["cross_split_near_duplicates_before"] == 1
    assert report["cross_split_near_duplicates_after"] == 0
    assert report["merged_group_components"] == 1
