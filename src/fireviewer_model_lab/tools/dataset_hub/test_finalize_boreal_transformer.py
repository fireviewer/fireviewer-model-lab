from __future__ import annotations

import json
from pathlib import Path

from dataset_archive_validation import sha256_file, validate_firewarning_dataset
from finalize_train_bundle import _normalize_boreal_detection
from PIL import Image


def test_boreal_yolo_source_is_converted_to_firewarning_detection_contract(
    tmp_path: Path,
) -> None:
    media = tmp_path / "payload" / "sample.jpg"
    media.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), color=(20, 30, 40)).save(media)
    label = tmp_path / "labels" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.4 0.25\n", encoding="utf-8")
    artifact = tmp_path / "samples" / "sample.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "annotation": {
                    "path": "labels/sample.txt",
                    "sha256": sha256_file(label),
                },
                "box_count": 1,
                "image": {
                    "path": "payload/sample.jpg",
                    "sha256": sha256_file(media),
                },
                "negative": False,
                "split": "train",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(
            {
                "artifact": {
                    "path": "samples/sample.json",
                    "sha256": sha256_file(artifact),
                },
                "sample_id": "boreal:sample",
                "source_id": "boreal-forest-fire-detection-v1",
                "source_record_id": "sample.jpg",
                "split_group": "boreal-site:evo",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = _normalize_boreal_detection(tmp_path)
    validation = validate_firewarning_dataset(tmp_path)

    assert report["rows"] == 1
    assert report["annotation_count"] == 1
    assert validation["role_counts"] == {"detector_training": 1}
    row = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["annotations"][0]["class_name"] == "smoke_visible"
    assert row["annotations"][0]["class_id"] == 0
    assert row["image_relpath"].startswith("images/")
