from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fireviewer_model_lab.training.dinov3_dataset_quality_profile import profile_manifest


def _write_fixture(root: Path) -> str:
    rows = [
        {
            "sample_id": "point-1",
            "split": "validation",
            "split_group": "event:1",
            "source_id": "source-a",
            "source_family_id": "family-a",
            "image_sha256": "a" * 64,
            "image_locator": {"kind": "s3_object", "uri": "s3://bucket/image.jpg", "sha256": "a" * 64},
            "license": "CC-BY-4.0",
            "consent_basis": {"kind": "source_license", "reference": "source-a:CC-BY-4.0"},
            "anchor_points": [{"kind": "fire_base", "x": 0.4, "y": 0.8}],
            "point_supervised": True,
            "presence_supervised": True,
            "segmentation_supervised": True,
            "abstention_supervised": True,
            "bbox_to_point": False,
            "point_derivation": "mask_bottom_band_median",
            "annotation_provenance": "human_pixel_mask",
        },
        {
            "sample_id": "negative-1",
            "split": "train",
            "split_group": "event:2",
            "source_id": "source-b",
            "source_family_id": "family-b",
            "image_sha256": "b" * 64,
            "image_locator": {"kind": "hf_dataset_row", "sha256": "b" * 64},
            "license": "Apache-2.0",
            "consent_basis": {"kind": "source_license", "reference": "source-b:Apache-2.0"},
            "anchor_points": [],
            "point_supervised": True,
            "presence_supervised": True,
            "segmentation_supervised": True,
            "abstention_supervised": True,
            "bbox_to_point": False,
            "point_derivation": "explicit_absence_no_point",
        },
    ]
    manifest = root / "composition_candidate_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "composition_report.json").write_text(
        json.dumps({"manifest_sha256": digest, "professional_corpus_ready": False, "quality_gate_deficits": {"point_source_families": {"actual": 2, "minimum": 8}}}),
        encoding="utf-8",
    )
    return digest


def test_profile_emits_canvas_and_automated_audit_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    digest = _write_fixture(input_dir)

    report = profile_manifest(input_dir=input_dir, output_dir=output_dir, expected_manifest_sha256=digest)

    assert report["integrity_passed"] is True
    assert report["rows"] == 2
    assert report["automated_point_audit_queue_rows"] == 1
    assert report["ground_truth_import_rows"] == 0
    assert report["ground_truth_ready"] is False
    assert report["sagemaker_ground_truth_allowed"] is False
    assert report["training_ready"] is False
    assert (output_dir / "canvas_source_split_quality.csv").is_file()
    assert (output_dir / "automated_point_audit_queue.jsonl").is_file()
    assert not (output_dir / "ground_truth_import.manifest").exists()
    policy = json.loads((output_dir / "human_annotation_policy.json").read_text(encoding="utf-8"))
    assert policy["allowed"] is False
    assert policy["ground_truth_import_rows"] == 0


def test_profile_fails_closed_on_unknown_rights(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    digest = _write_fixture(input_dir)
    manifest = input_dir / "composition_candidate_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["license"] = ""
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (input_dir / "composition_report.json").write_text(json.dumps({"manifest_sha256": digest}), encoding="utf-8")

    report = profile_manifest(input_dir=input_dir, output_dir=output_dir, expected_manifest_sha256=digest)

    assert report["integrity_passed"] is False
    assert report["unknown_rights_rows"] == 1
