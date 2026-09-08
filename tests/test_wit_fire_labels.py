from __future__ import annotations

import numpy as np
from fireviewer_model_lab.training.wit_fire_labels import (
    calibrate_hot_threshold,
    derive_wit_fire_label,
    temporal_vote,
)


def test_calibration_uses_pre_ignition_distribution() -> None:
    background = np.full((4, 12, 12), 20.0, dtype=np.float32)
    background[:, 2, 2] = 25.0

    threshold = calibrate_hot_threshold(background)

    assert threshold >= 25.0


def test_asset_box_is_excluded_from_fire_mask() -> None:
    frame = np.full((24, 24), 20.0, dtype=np.float32)
    frame[5:10, 4:9] = 120.0
    frame[12:22, 14:20] = 150.0
    teacher = np.zeros_like(frame, dtype=bool)
    teacher[5:10, 4:9] = True
    teacher[12:22, 14:20] = True
    temporal = teacher.copy()

    label = derive_wit_fire_label(
        frame,
        hot_threshold=80.0,
        asset_boxes_xyxy=[(4, 5, 9, 10)],
        asset_margin_pixels=1,
        teacher_mask=teacher,
        temporal_support_mask=temporal,
    )

    assert not label.mask[7, 6]
    assert label.mask[18, 17]
    assert label.fire_base_xy == (16.5, 21.0)
    assert not label.abstain


def test_unresolved_hot_asset_becomes_abstention() -> None:
    frame = np.full((16, 16), 20.0, dtype=np.float32)
    frame[4:12, 4:12] = 100.0

    label = derive_wit_fire_label(
        frame,
        hot_threshold=80.0,
        asset_boxes_xyxy=[(3, 3, 13, 13)],
        minimum_pixels=4,
    )

    assert label.abstain
    assert label.fire_base_xy is None
    assert not label.mask.any()


def test_temporal_vote_requires_persistence() -> None:
    masks = np.zeros((3, 8, 8), dtype=bool)
    masks[0, 2:4, 2:4] = True
    masks[1, 2:4, 2:4] = True
    masks[2, 6, 6] = True

    stable = temporal_vote(masks, minimum_votes=2)

    assert stable[2, 2]
    assert not stable[6, 6]


def test_hot_region_without_independent_consensus_is_rejected() -> None:
    frame = np.full((16, 16), 20.0, dtype=np.float32)
    frame[4:12, 4:12] = 100.0

    label = derive_wit_fire_label(frame, hot_threshold=80.0, minimum_pixels=4)

    assert label.abstain
    assert label.fire_base_xy is None
    assert not label.mask.any()
    assert "missing_teacher_consensus+temporal_consensus" in label.quality


def test_teacher_without_temporal_persistence_is_rejected() -> None:
    frame = np.full((16, 16), 20.0, dtype=np.float32)
    frame[4:12, 4:12] = 100.0
    teacher = frame >= 80.0

    label = derive_wit_fire_label(
        frame,
        hot_threshold=80.0,
        teacher_mask=teacher,
        minimum_pixels=4,
    )

    assert label.abstain
    assert not label.mask.any()
    assert "missing_temporal_consensus" in label.quality
