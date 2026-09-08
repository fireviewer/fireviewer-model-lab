from __future__ import annotations

import numpy as np
from fireviewer_model_lab.training.camp_swift_thermal import pair_nearest_layers, thermal_hot_mask


def test_pair_nearest_layers_is_one_to_one_and_bounded() -> None:
    eo = ["x:EO_000001000", "x:EO_000002000", "x:EO_000003000"]
    ir = ["x:IR_000001050_rect", "x:IR_000002100rect"]
    pairs = pair_nearest_layers(eo, ir, max_delta_ms=150)
    assert [(item[0], item[1], item[2]) for item in pairs] == [
        (eo[0], ir[0], 50),
        (eo[1], ir[1], 100),
    ]


def test_thermal_hot_mask_rejects_small_palette_noise() -> None:
    rgb = np.zeros((3, 12, 12), dtype=np.uint8)
    rgb[0, 2:7, 3:8] = 80
    rgb[0, 10, 10] = 255
    valid = np.full((12, 12), 255, dtype=np.uint8)
    mask = thermal_hot_mask(rgb, valid, red_threshold=40, minimum_component_pixels=10)
    assert int((mask > 0).sum()) == 25
    assert int(mask[10, 10]) == 0
