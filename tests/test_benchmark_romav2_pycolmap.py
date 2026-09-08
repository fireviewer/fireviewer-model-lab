import numpy as np
from fireviewer_model_lab.training.benchmark_romav2_pycolmap import _pair_assets, _static_match_mask


def test_static_match_mask_rejects_either_transient_view() -> None:
    points = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mask_a = np.zeros((5, 5), dtype=bool)
    mask_b = np.zeros((5, 5), dtype=bool)
    mask_a[1, 1] = True
    mask_b[2, 2] = True
    assert _static_match_mask(points, points, mask_a, mask_b).tolist() == [False, False, True]


def test_pair_assets_supports_cross_view_schema() -> None:
    row = {
        "sample_id": "pair",
        "source_view": {"image_relpath": "a.jpg"},
        "map_view": {"image_relpath": "b.jpg"},
        "source_transient_mask_relpath": "a.png",
        "map_transient_mask_relpath": "b.png",
    }
    assert _pair_assets(row) == ("pair", "a.jpg", "b.jpg", "a.png", "b.png")
