from __future__ import annotations

import numpy as np
from fireviewer_model_lab.training.gaussians_cross_view import select_shared_landmark


def test_select_shared_landmark_projects_into_map_view() -> None:
    points = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [-1.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )
    rotation = np.eye(3)
    source_translation = np.zeros(3)
    map_translation = np.array([0.5, 0.0, 0.0])
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    index, target = select_shared_landmark(
        points=points,
        source_rotation=rotation,
        source_translation=source_translation,
        source_k=intrinsics,
        source_size=(100, 100),
        map_rotation=rotation,
        map_translation=map_translation,
        map_k=intrinsics,
        map_size=(100, 100),
    )
    assert index == 0
    assert 0.5 < target[0] < 0.7
    assert 0.45 < target[1] < 0.55
