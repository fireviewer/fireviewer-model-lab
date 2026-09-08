from __future__ import annotations

import numpy as np
from fireviewer_model_lab.training.gaussians_moge import sparse_depth_from_points


def test_sparse_depth_keeps_nearest_projected_point() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [1.0, 0.0, -1.0]])
    depth, valid = sparse_depth_from_points(
        points=points,
        rotation=np.eye(3),
        translation=np.zeros(3),
        intrinsics=np.array([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]),
        width=8,
        height=8,
    )
    assert float(depth[4, 4]) == 2.0
    assert int(valid[4, 4]) == 255
    assert int((valid > 0).sum()) == 1
