from __future__ import annotations

import unittest

from fireviewer_model_lab.synthetic.geometry import camera_contract, project_aabb, project_point


class GeometryTests(unittest.TestCase):
    def test_optical_axis_projects_to_principal_point(self) -> None:
        camera = camera_contract(
            position=[0.0, -10.0, 2.0],
            look_at=[0.0, 0.0, 2.0],
            width=1024,
            height=768,
        )
        point = project_point([0.0, 0.0, 2.0], camera)
        self.assertAlmostEqual(point["x_normalized"], 0.5)
        self.assertAlmostEqual(point["y_normalized"], 0.5)
        self.assertEqual(point["depth_m"], 10.0)

    def test_world_aabb_produces_positive_normalized_box(self) -> None:
        camera = camera_contract(
            position=[0.0, -20.0, 5.0],
            look_at=[0.0, 0.0, 2.0],
            width=1024,
            height=768,
        )
        box = project_aabb([-2.0, -1.0, 0.0], [2.0, 1.0, 4.0], camera)
        self.assertGreater(box["x_max"], box["x_min"])
        self.assertGreater(box["y_max"], box["y_min"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in box.values()))


if __name__ == "__main__":
    unittest.main()
