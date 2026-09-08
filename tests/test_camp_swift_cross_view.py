from __future__ import annotations

from pathlib import Path

import numpy as np
from fireviewer_model_lab.training.camp_swift_cross_view import common_timeline, contained_audio_offset


def test_audio_alignment_recovers_known_offset() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(size=400).astype(np.float32)
    query = reference[123:263].copy()

    offset, score = contained_audio_offset(reference, query, hop_seconds=0.05)

    assert abs(offset - 6.15) < 1e-6
    assert score > 0.999


def test_common_timeline_requires_distinct_overlapping_cameras() -> None:
    first = Path("first.mpg")
    second = Path("second.mpg")
    third = Path("third.mpg")
    timeline = common_timeline(
        {first: 0.0, second: 2.0, third: 8.0},
        {first: 12.0, second: 4.0, third: 3.0},
        stride_seconds=2.0,
    )

    assert [time for time, _ in timeline] == [2.0, 4.0, 8.0, 10.0]
    assert all(len(cameras) >= 2 for _, cameras in timeline)
