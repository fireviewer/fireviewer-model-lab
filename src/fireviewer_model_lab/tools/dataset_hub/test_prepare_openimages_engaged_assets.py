from __future__ import annotations

import unittest

from prepare_openimages_engaged_assets import (
    IMAGE_OBJECT_BASE_URL,
    _balanced_selection,
    _valid_box,
)


class OpenImagesEngagedAssetsTests(unittest.TestCase):
    def test_pixels_use_the_official_public_cvdf_s3_bucket(self) -> None:
        self.assertEqual(
            IMAGE_OBJECT_BASE_URL,
            "https://open-images-dataset.s3.amazonaws.com",
        )

    def test_depictions_inside_views_and_groups_are_excluded(self) -> None:
        base = {
            "LabelName": "/m/012n7d",
            "IsDepiction": "0",
            "IsInside": "0",
            "IsGroupOf": "0",
            "XMin": "0.1",
            "XMax": "0.9",
            "YMin": "0.2",
            "YMax": "0.8",
        }
        self.assertTrue(_valid_box(base))
        for key in ("IsDepiction", "IsInside", "IsGroupOf"):
            self.assertFalse(_valid_box({**base, key: "1"}))

    def test_balancing_keeps_only_licensed_candidates(self) -> None:
        boxes = {
            "a": [{"class_name": "ambulance"}],
            "b": [{"class_name": "ambulance"}],
        }
        selected = _balanced_selection(boxes, {"a": {"license": "ok"}}, 10)
        self.assertEqual(selected, {"a"})


if __name__ == "__main__":
    unittest.main()
