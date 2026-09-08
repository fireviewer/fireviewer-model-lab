from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from fireviewer_model_lab.training.pointing_camp_swift_audit import (
    _cross_split_near_exclusions,
    audit_camp_swift,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_camp_swift_sensor_mask_is_converted_without_review(tmp_path: Path) -> None:
    source = tmp_path / "source"
    image_path = source / "source" / "image.jpg"
    mask_path = source / "source" / "mask.png"
    valid_path = source / "source" / "valid.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 20), (210, 70, 20)).save(image_path)
    mask = Image.new("L", (32, 20), 0)
    ImageDraw.Draw(mask).rectangle((8, 5, 23, 18), fill=255)
    mask.save(mask_path)
    Image.new("L", (32, 20), 255).save(valid_path)
    _jsonl(
        source / "candidate_manifest.jsonl",
        [
            {
                "sample_id": "camp-swift:test",
                "source_id": "Camp Swift Fire Experiment 2014",
                "source_revision": "RDS-2018-0046+RDS-2018-0047",
                "source_repository": "fireviewer/dinov3-multitask-fireviewer-v3-dataset",
                "source_repository_revision": "06dad028c4e65fde36f36bb3a97c6fec766a270d",
                "split": "train",
                "split_group": "camp-swift:block-test",
                "image_relpath": "source/image.jpg",
                "mask_relpath": "source/mask.png",
                "valid_mask_relpath": "source/valid.png",
                "image_sha256": _sha(image_path),
                "mask_sha256": _sha(mask_path),
                "valid_mask_sha256": _sha(valid_path),
                "sample_validation_status": "sensor_derived",
                "annotation_strength": "strong",
                "mask_quality": "sensor_derived_thermal_reprojection",
                "mask_semantics": "thermal_hot_fire_core",
                "pair_delta_ms": 250,
                "license": "CC-BY-4.0",
                "redistribution_allowed": True,
                "image_s3_uri": "s3://bucket/camp-swift/source/image.jpg",
                "mask_s3_uri": "s3://bucket/camp-swift/source/mask.png",
                "valid_mask_s3_uri": "s3://bucket/camp-swift/source/valid.png",
            }
        ],
    )
    baseline = tmp_path / "baseline"
    _jsonl(
        baseline / "strict_combined_manifest.jsonl",
        [
            {
                "sample_id": "baseline:1",
                "image_sha256": "f" * 64,
                "dhash": "f" * 16,
            }
        ],
    )
    output = tmp_path / "output"

    report = audit_camp_swift(
        source_root=source,
        baseline_root=baseline,
        output_dir=output,
        output_s3_prefix="s3://bucket/pointing/camp-swift-run",
    )
    validated = [
        json.loads(line)
        for line in (output / "camp_swift_strict_validated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert report["strict_automated_validated_rows"] == 1
    assert report["strict_payload_artifacts_materialized"] == 3
    assert report["reviews_admitted"] is False
    assert validated[0]["anchor_points"][0]["kind"] == "fire_base"
    assert validated[0]["sample_validation_status"] == "strict_automated_validated"
    assert validated[0]["training_eligible"]
    assert validated[0]["schema_version"] == 1
    assert validated[0]["segmentation_supervised"] is True
    assert validated[0]["point_supervised"] is True
    assert validated[0]["presence_supervised"] is True
    assert validated[0]["abstention_supervised"] is True
    assert validated[0]["presence_targets"] == {
        "flame_visible": True,
        "smoke_visible": False,
    }
    for path_field, sha_field in (
        ("image_relpath", "image_sha256"),
        ("mask_relpath", "mask_sha256"),
        ("valid_mask_relpath", "valid_mask_sha256"),
    ):
        payload = output / validated[0][path_field]
        assert payload.is_file()
        assert _sha(payload) == validated[0][sha_field]
        assert validated[0][path_field].startswith("strict-payload/")
    assert validated[0]["image_s3_uri"].startswith(
        "s3://bucket/pointing/camp-swift-run/strict-payload/"
    )


def test_cross_split_near_duplicate_keeps_test_sample() -> None:
    rows = [
        {
            "sample_id": "camp-swift:train-near",
            "split": "train",
            "dhash": "0000000000000000",
            "exclusion_reasons": [],
        },
        {
            "sample_id": "camp-swift:test-near",
            "split": "test",
            "dhash": "0000000000000001",
            "exclusion_reasons": [],
        },
        {
            "sample_id": "camp-swift:unrelated",
            "split": "validation",
            "dhash": "ffffffffffffffff",
            "exclusion_reasons": [],
        },
    ]

    pairs, excluded = _cross_split_near_exclusions(rows, maximum_distance=4)

    assert len(pairs) == 1
    assert pairs[0]["dhash_distance"] == 1
    assert excluded == {"camp-swift:train-near"}
