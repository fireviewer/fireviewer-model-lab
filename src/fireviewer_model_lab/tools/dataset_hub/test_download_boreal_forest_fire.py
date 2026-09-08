from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from download_boreal_forest_fire import (
    OfficialFile,
    _authorize,
    _download_one,
    parse_official_files,
    select_profile,
)


class BorealDownloaderTests(unittest.TestCase):
    def test_authorize_accepts_official_https_host_with_explicit_port(self) -> None:
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                return {"url": "https://download.fairdata.fi:443/download?token=abc"}

        class Session:
            def post(self, *args: object, **kwargs: object) -> Response:
                return Response()

        self.assertEqual(
            _authorize(Session(), "/Boreal-Forest-Fire/example.jpg"),
            "https://download.fairdata.fi:443/download?token=abc",
        )

    def test_parse_official_files_requires_sha256_and_unique_safe_paths(self) -> None:
        files = parse_official_files(
            [
                {
                    "pathname": "/Boreal-Forest-Fire/example.jpg",
                    "size": 3,
                    "checksum": "sha256:" + "a" * 64,
                    "storage_identifier": "pid-1",
                }
            ]
        )
        self.assertEqual(files[0].relative_path.as_posix(), "Boreal-Forest-Fire/example.jpg")
        with self.assertRaises(ValueError):
            parse_official_files(
                [
                    {
                        "pathname": "/../escape.jpg",
                        "size": 3,
                        "checksum": "sha256:" + "a" * 64,
                    }
                ]
            )

    def test_profiles_separate_video_subset_from_images_and_metadata(self) -> None:
        checksum = "a" * 64
        image = OfficialFile("/root/Boreal-Forest-Fire-Subset-A/a.jpg", 1, checksum, "1")
        video = OfficialFile("/root/Boreal-Forest-Fire-Subset-B/a.mp4", 1, checksum, "2")
        notebook = OfficialFile("/root/notebook.ipynb", 1, checksum, "3")
        self.assertEqual(select_profile([image, video, notebook], "images"), [image, notebook])
        self.assertEqual(select_profile([image, video, notebook], "videos"), [video])
        self.assertEqual(select_profile([image, video, notebook], "all"), [image, video, notebook])

    def test_existing_verified_file_is_a_cache_hit_without_network(self) -> None:
        content = b"verified"
        item = OfficialFile(
            pathname="/Boreal-Forest-Fire/file.txt",
            size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            storage_identifier="pid",
        )

        def forbidden_session() -> None:
            raise AssertionError("network session must not be created for a verified cache hit")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "Boreal-Forest-Fire" / "file.txt"
            target.parent.mkdir(parents=True)
            target.write_bytes(content)
            result = _download_one(item, root, session_factory=forbidden_session)
        self.assertEqual(result["status"], "cache_hit")


if __name__ == "__main__":
    unittest.main()
