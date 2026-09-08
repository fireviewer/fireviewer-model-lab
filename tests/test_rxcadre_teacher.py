from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from fireviewer_model_lab.training.rxcadre_teacher import select_by_interval


def test_select_by_interval_is_temporal_and_deterministic() -> None:
    start = datetime(2012, 11, 11, 12, 0, 0)
    observations = [
        (start + timedelta(seconds=value), Path(f"{value}.jpg")) for value in (61, 0, 59, 120, 121)
    ]
    selected = select_by_interval(observations, 60)
    assert [timestamp for timestamp, _ in selected] == [
        start,
        start + timedelta(seconds=61),
        start + timedelta(seconds=121),
    ]
