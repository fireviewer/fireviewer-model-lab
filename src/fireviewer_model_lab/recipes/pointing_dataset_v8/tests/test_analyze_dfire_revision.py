from training.pointing_dataset_v8.analyze_dfire_revision import analyze, iou_xywh


def test_iou_xywh():
    assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou_xywh([0, 0, 10, 10], [20, 20, 2, 2]) == 0


def test_analysis_flags_unsupported_labels_and_unmatched_predictions():
    rows = [{"sha256": "a", "revision_review_index": 7, "source_record_id": "x.jpg", "source_group_id": "g"}]
    coco = {"images": [{"id": 1, "fireviewer_sha256": "a"}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}]}
    predictions = [{"image_id": 1, "category_id": 0, "bbox": [20, 20, 5, 5], "score": 0.9}]
    result = analyze(rows, coco, predictions)
    assert len(result[0]["unsupported_annotations"]) == 1
    assert len(result[0]["unmatched_high_confidence_predictions"]) == 1
    assert result[0]["screening_only"] is True
