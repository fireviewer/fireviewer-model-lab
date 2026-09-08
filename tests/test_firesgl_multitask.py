from __future__ import annotations

import numpy as np
from fireviewer_model_lab.training.firesgl_multitask import (
    StereoPair,
    even_sample,
    stereo_candidate_mask,
    stereo_pairs,
    temporal_consensus_mask,
)
from fireviewer_model_lab.training.remote_zip import RemoteZipEntry


def _entry(name: str) -> RemoteZipEntry:
    return RemoteZipEntry(name, 10, 10, 0, 0, 0)


def test_pairs_only_matching_stereo_members() -> None:
    entries = [
        _entry("run/img_left/00001.png"),
        _entry("run/img_right/00001.png"),
        _entry("run/img_left/00002.png"),
    ]
    pairs = stereo_pairs(entries, sequence_group="firesgl-1", archive_name="one.zip")
    assert [pair.frame_key for pair in pairs] == ["run/00001.png"]


def test_even_sample_preserves_boundaries() -> None:
    entry = _entry("x.png")
    pairs = [
        StereoPair("firesgl-1", "one.zip", f"{index:05d}.png", entry, entry) for index in range(10)
    ]
    sampled = even_sample(pairs, 4)
    assert [pair.frame_key for pair in sampled] == [
        "00000.png",
        "00003.png",
        "00006.png",
        "00009.png",
    ]


def test_stereo_and_temporal_consensus_keeps_persistent_hot_region() -> None:
    left = np.full((100, 120), 30, dtype=np.uint8)
    right = left.copy()
    left[40:50, 60:70] = 250
    right[40:50, 50:60] = 250
    stereo = stereo_candidate_mask(left, right, maximum_disparity=16, minimum_pixels=16)
    assert int(stereo.sum()) >= 100
    masks = [stereo.copy(), stereo.copy(), np.zeros_like(stereo)]
    accepted = temporal_consensus_mask(masks, 0)
    rejected = temporal_consensus_mask(masks, 2)
    assert int(accepted.sum()) > 0
    assert int(rejected.sum()) == 0
