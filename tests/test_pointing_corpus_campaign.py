from __future__ import annotations

import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training.pointing_corpus_campaign import (
    assert_input_isolation,
    audit_box_reservoir,
    audit_point_seed,
    load_registry,
)

REGISTRY = Path(__import__("fireviewer_model_lab.training", fromlist=["_"]).__file__).parent / Path("registries/pointing-corpus-v2.json")


def _point_row(sample_id: str, *, split: str = "train", variant: str = "clean") -> dict:
    return {
        "ground_view": True,
        "sample_id": sample_id,
        "schema_version": 1,
        "source": {
            "height": 100,
            "image_relpath": f"images/{split}/{sample_id}.jpg",
            "source_id": f"source:{sample_id}",
            "source_sha256": (sample_id * 64)[:64],
            "width": 200,
        },
        "split": split,
        "split_group": f"group:{sample_id}",
        "targets": [
            {
                "point_normalized": [0.25, 0.5],
                "point_origin": "source_colored_marker_centroid",
                "point_pixel": [50.0, 50.0],
                "semantic_anchor": "fire_base",
            }
        ],
        "training_eligible": True,
        "variant": variant,
    }


def _box_row(sample_id: str, *, split: str = "train", negative: bool = False) -> dict:
    return {
        "file_name": f"images/{sample_id}.jpg",
        "height": 100,
        "image_id": sample_id,
        "is_negative": negative,
        "objects": {
            "bbox": [] if negative else [[10, 20, 30, 40]],
            "category": [] if negative else [0],
        },
        "scene_bin": "negative" if negative else "fire_only",
        "sha256": (sample_id * 64)[:64],
        "source_dataset": "test-source",
        "source_family": "Test Source",
        "source_group_id": f"event:{sample_id}",
        "source_record_id": sample_id,
        "source_revision": "revision",
        "split": split,
        "width": 200,
    }


def test_registry_forbids_detection_and_benchmark_inputs() -> None:
    registry = load_registry(REGISTRY)
    with pytest.raises(ValueError, match="forbidden non-pointing corpus"):
        assert_input_isolation(
            registry,
            "s3://bucket/fire-smoke-detection-corpus-v1/raw",
        )
    with pytest.raises(ValueError, match="forbidden non-pointing corpus"):
        assert_input_isolation(registry, "D:/benchdata/fireviewer_bench")


def test_seed_keeps_only_clean_views_and_requires_strict_validation() -> None:
    rows = [_point_row("a"), _point_row("b"), _point_row("c", variant="hflip")]
    report, canonical, issues, _hashes = audit_point_seed(
        rows,
        source_s3_prefix="s3://bucket/pointing-ground-v1/raw",
        materialized_hashes={row["source"]["image_relpath"]: ("f" * 64) for row in rows},
    )

    assert report["materialized_rows"] == 3
    assert report["canonical_clean_rows"] == 2
    assert report["materialized_hashes_resolved"] == 2
    assert not issues
    assert all(not row["training_eligible"] for row in canonical)
    assert all(
        row["admission_status"] == "excluded_until_rights_and_strict_validation_pass"
        for row in canonical
    )


def test_box_reservoir_never_derives_points() -> None:
    report, candidates, issues, _hashes = audit_box_reservoir(
        [_box_row("a"), _box_row("b", negative=True)],
        source_s3_prefix="s3://bucket/pointing-v8-reservoir-v1/raw",
    )

    assert report["point_ground_truth_rows"] == 0
    assert not issues
    assert all(row["box_to_point_conversion"] == "prohibited" for row in candidates)
    assert all("anchor_points" not in row and "targets" not in row for row in candidates)
    assert all("review_priority" not in row for row in candidates)
    assert all(not row["training_eligible"] for row in candidates)


def test_box_reservoir_normalizes_legacy_basename_to_images_directory() -> None:
    row = _box_row("legacy")
    row["file_name"] = "legacy.jpg"

    _report, candidates, issues, _hashes = audit_box_reservoir(
        [row], source_s3_prefix="s3://bucket/pointing-v8-reservoir-v1/raw"
    )

    assert not issues
    assert candidates[0]["image_s3_uri"].endswith("/coco/train/images/legacy.jpg")


def test_registry_json_is_stable_json() -> None:
    value = load_registry(REGISTRY)
    assert json.loads(REGISTRY.read_text(encoding="utf-8")) == value
    assert value["strict_automation"]["reviews_admitted"] is False
    assert "review_batch_size" not in REGISTRY.read_text(encoding="utf-8")
    plan = value["professional_extension_plan"]
    assert plan["status"] == "active_ready_data_only_intake"
    assert plan["ready_source_families"]
    gates = value["quality_gates"]
    assert len(set(plan["ready_source_families"])) >= gates["source_families_min"]
    assert (
        plan["target_capacity_after_strict_admission"]["unique_point_images_min"]
        >= gates["strict_automated_validated_point_images_min"]
    )
    assert plan["annotation_quality"]["box_center_conversion_allowed"] is False
    assert plan["annotation_quality"]["human_annotation_allowed"] is False
    assert plan["annotation_quality"]["sagemaker_ground_truth_job_allowed"] is False
    assert "automated_silver_extension" not in plan
    assert plan["source_policy"] == "published_native_masks_or_sensor_geometry_only"
    assert plan["training_allowed"] is False
