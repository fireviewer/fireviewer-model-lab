from __future__ import annotations

import numpy as np
from fireviewer_model_lab.training.pyro_sdis_multitask import (
    assign_split,
    boxes_to_roi,
    build_session_groups,
    filter_teacher_mask,
    parse_captured_at,
    parse_yolo_boxes,
    smoke_base,
    station_from_camera,
)


def test_parse_captured_at_accepts_upstream_hyphenated_time() -> None:
    assert parse_captured_at("2024-01-15T14-32-36").isoformat() == "2024-01-15T14:32:36"


def test_parse_yolo_boxes_and_station() -> None:
    boxes = parse_yolo_boxes("1 0.5 0.6 0.2 0.1\n")

    assert boxes == [(0.5, 0.6, 0.2, 0.1)]
    assert station_from_camera("croix-augas-212") == "croix-augas"
    assert station_from_camera("fixed-camera") == "fixed-camera"


def test_sessions_break_on_large_temporal_gap() -> None:
    rows = [
        {
            "partner": "sdis-77",
            "camera": "croix-augas-10",
            "image_name": "a.jpg",
            "date": "2024-01-01T10:00:00",
        },
        {
            "partner": "sdis-77",
            "camera": "croix-augas-20",
            "image_name": "b.jpg",
            "date": "2024-01-01T10:10:00",
        },
        {
            "partner": "sdis-77",
            "camera": "croix-augas-20",
            "image_name": "c.jpg",
            "date": "2024-01-01T12:00:00",
        },
    ]

    groups = build_session_groups(rows)

    assert groups["a.jpg"] == groups["b.jpg"]
    assert groups["a.jpg"] != groups["c.jpg"]
    assert assign_split(groups["a.jpg"]) in {"train", "validation", "test"}


def test_teacher_mask_must_overlap_source_box() -> None:
    probability = np.zeros((100, 100), dtype=np.float32)
    probability[45:65, 45:65] = 0.9
    probability[5:15, 5:15] = 0.99
    boxes = [(0.55, 0.55, 0.3, 0.3)]

    mask, valid = filter_teacher_mask(probability, boxes, threshold=0.55, minimum_pixels=20)
    anchor = smoke_base(mask)

    assert mask[50, 50] == 255
    assert mask[10, 10] == 0
    assert valid[50, 50] == 255
    assert anchor is not None


def test_roi_expands_but_preserves_core() -> None:
    core, expanded = boxes_to_roi([(0.5, 0.5, 0.2, 0.2)], (100, 100))

    assert np.count_nonzero(expanded) > np.count_nonzero(core)
    assert np.all(expanded[core > 0] == 1)
