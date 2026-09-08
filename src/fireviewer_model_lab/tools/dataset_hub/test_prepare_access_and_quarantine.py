from __future__ import annotations

import unittest

from prepare_access_and_quarantine import _classify_mcped_file


class AccessAndQuarantineTests(unittest.TestCase):
    def test_mcped_standalone_views_are_blocked_from_republication(self) -> None:
        for name in ("2023-08-09-13-30.png", "2023-08-06-13-34-18.jpg"):
            status, _reason = _classify_mcped_file(name)
            self.assertEqual(status, "blocked_from_republication")

    def test_mcped_ground_archive_remains_private_audit_candidate(self) -> None:
        status, reason = _classify_mcped_file("2023-08-06-13-34image.zip")
        self.assertEqual(status, "candidate_private_audit")
        self.assertIn("ground_view", reason)

    def test_mcped_rejects_unsafe_filename(self) -> None:
        with self.assertRaises(ValueError):
            _classify_mcped_file("../satellite.png")


if __name__ == "__main__":
    unittest.main()
