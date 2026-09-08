from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fireviewer_model_lab.training.detector_benchmark import (
    _operational_detection_metrics,
    freeze_selection,
    validate_prediction_payload,
    validate_selection,
)
from fireviewer_model_lab.training.detector_benchmark_infer import _firewarning_class_id
from fireviewer_model_lab.training.train_rtdetr import LoadedRecord


def _record(identifier: str, split: str, class_id: int | None) -> LoadedRecord:
    annotations = (
        []
        if class_id is None
        else [
            {
                "bbox_xywh": [2.0, 3.0, 10.0, 8.0],
                "class_id": class_id,
                "class_name": ["smoke_visible", "flame_visible"][class_id],
            }
        ]
    )
    return LoadedRecord(
        {
            "annotations": annotations,
            "corpus_role": "detector_training",
            "height": 64,
            "image_relpath": f"images/{identifier}.jpg",
            "sample_id": identifier,
            "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            "split": split,
            "width": 96,
        },
        Path("."),
    )


def _records() -> list[LoadedRecord]:
    return [
        _record("validation-fire", "validation", 0),
        _record("validation-smoke", "validation", 1),
        _record("validation-negative", "validation", None),
        _record("test-fire", "test", 0),
        _record("test-smoke", "test", 1),
        _record("test-negative", "test", None),
    ]


def test_frozen_detector_benchmark_binds_classes_images_annotations_and_order() -> None:
    selection = freeze_selection(
        _records(),
        allowed_class_ids=frozenset({0, 1}),
        limit=6,
    )

    assert selection["class_ids"] == [0, 1]
    assert len(selection["entries"]) == 6
    resolved = validate_selection(selection, _records())
    assert [loaded.record["sample_id"] for loaded in resolved] == [
        entry["sample_id"] for entry in selection["entries"]
    ]


def test_prediction_payload_must_target_the_exact_frozen_order() -> None:
    selection = freeze_selection(
        _records(),
        allowed_class_ids=frozenset({0, 1}),
        limit=6,
    )
    predictions = [
        {
            "sample_id": entry["sample_id"],
            "boxes_xyxy": [],
            "labels": [],
            "scores": [],
        }
        for entry in selection["entries"]
    ]
    payload = {
        "schema_version": 1,
        "selection_sha256": selection["selection_sha256"],
        "timing": {
            "total_seconds": 1.0,
            "images_per_second": 6.0,
            "milliseconds_per_image": 166.67,
        },
        "predictions": predictions,
    }
    validate_prediction_payload(payload, selection)

    payload["predictions"] = list(reversed(predictions))
    with pytest.raises(ValueError, match="reordered"):
        validate_prediction_payload(payload, selection)


def test_selection_rejects_annotation_drift() -> None:
    records = _records()
    selection = freeze_selection(
        records,
        allowed_class_ids=frozenset({0, 1}),
        limit=6,
    )
    records[0].record["annotations"][0]["bbox_xywh"][0] = 4.0

    with pytest.raises(ValueError, match="annotation drift"):
        validate_selection(selection, records)


@pytest.mark.parametrize(
    ("label", "expected"),
    (
        ("smoke", 0),
        ("fumée visible", 0),
        ("fire", 1),
        ("flame_visible", 1),
    ),
)
def test_detector_labels_use_the_canonical_firewarning_ids(
    label: str,
    expected: int,
) -> None:
    assert _firewarning_class_id(label) == expected


def test_detector_score_reports_class_recall_false_positives_and_timing() -> None:
    selection = freeze_selection(
        _records(),
        allowed_class_ids=frozenset({0, 1}),
        limit=6,
    )
    predictions = []
    for entry in selection["entries"]:
        if entry["annotations"]:
            annotation = entry["annotations"][0]
            x, y, width, height = annotation["bbox_xywh"]
            boxes = [[x, y, x + width, y + height]]
            labels = [annotation["class_id"]]
        else:
            boxes = [[1.0, 1.0, 4.0, 4.0]]
            labels = [0]
        predictions.append(
            {
                "sample_id": entry["sample_id"],
                "boxes_xyxy": boxes,
                "labels": labels,
                "scores": [0.9],
            }
        )
    payload = {
        "schema_version": 1,
        "candidate_id": "perfect-with-negative-fp",
        "selection_sha256": selection["selection_sha256"],
        "timing": {
            "total_seconds": 0.6,
            "images_per_second": 10.0,
            "milliseconds_per_image": 100.0,
        },
        "predictions": predictions,
    }

    candidate = _operational_detection_metrics(selection, payload)

    assert candidate["recall_by_class"] == {
        "smoke_visible": 1.0,
        "flame_visible": 1.0,
    }
    assert candidate["false_positives"]["detections_on_negative_images"] == 2
    assert candidate["false_positives"]["negative_image_rate"] == 1.0
