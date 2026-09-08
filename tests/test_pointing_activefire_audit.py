from __future__ import annotations

import numpy as np
from fireviewer_model_lab.training.pointing_activefire_audit import (
    POINTING_SEMANTIC_EXCLUSION,
    assign_scene_splits,
    normalize_manual_stem,
    render_false_color_762,
    source_scene,
)


def test_activefire_manual_pair_normalization_and_scene_group() -> None:
    path = "manual/LC08_L1TP_025033_20200921_20200921_01_RT_v1_p00677.tif"
    stem = normalize_manual_stem(path)

    assert stem == "lc08_l1tp_025033_20200921_20200921_01_rt_p00677"
    assert source_scene(stem) == "lc08_l1tp_025033_20200921_20200921_01_rt"


def test_activefire_scene_split_is_grouped_and_complete() -> None:
    scenes = {f"scene-{index}" for index in range(10)}
    splits = assign_scene_splits(scenes)

    assert set(splits) == scenes
    assert set(splits.values()) == {"train", "validation", "test"}
    assert list(splits.values()).count("train") == 6


def test_activefire_render_is_auxiliary_not_pointing() -> None:
    data = np.zeros((8, 8, 10), dtype=np.uint16)
    data[..., 6] = 20
    data[..., 5] = 30
    data[..., 1] = 40
    image = render_false_color_762(data, [(0, 40), (0, 60), (0, 80)])
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (128, 128, 128)
    assert POINTING_SEMANTIC_EXCLUSION == "top_down_hotspot_mask_has_no_ground_base_semantics"
