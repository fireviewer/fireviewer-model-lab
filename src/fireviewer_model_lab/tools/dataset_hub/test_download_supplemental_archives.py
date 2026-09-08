from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from download_supplemental_archives import (
    DIODE_ARCHIVES,
    TARTANAIR_ARCHIVES,
    ArchiveSpec,
    _validate_download_response,
    download_archive,
)


class Response:
    def __init__(self, status_code: int, content_range: str = "") -> None:
        self.status_code = status_code
        self.headers = {"Content-Range": content_range}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class SupplementalArchiveDownloadTests(unittest.TestCase):
    def test_pinned_inventories_have_unique_safe_paths_and_checksums(self) -> None:
        all_specs = (*TARTANAIR_ARCHIVES, *DIODE_ARCHIVES)
        paths = [spec.safe_path.as_posix() for spec in all_specs]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(TARTANAIR_ARCHIVES), 10)
        self.assertEqual(len(DIODE_ARCHIVES), 2)
        for spec in all_specs:
            expected_length = 64 if spec.checksum_algorithm == "sha256" else 32
            self.assertEqual(len(spec.checksum), expected_length)
            self.assertGreater(spec.size, 0)

    def test_resume_requires_matching_content_range(self) -> None:
        self.assertEqual(
            _validate_download_response(Response(206, "bytes 12-19/20"), 12),
            ("ab", 12),
        )
        with self.assertRaisesRegex(ValueError, "Content-Range"):
            _validate_download_response(Response(206, "bytes 0-19/20"), 12)
        self.assertEqual(_validate_download_response(Response(200), 12), ("wb", 0))

    def test_verified_destination_is_a_cache_hit_without_network(self) -> None:
        content = b"verified"
        spec = ArchiveSpec(
            relative_path="archives/example.bin",
            url="https://invalid.example/example.bin",
            size=len(content),
            checksum=hashlib.sha256(content).hexdigest(),
            checksum_algorithm="sha256",
            source_id="test",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "archives" / "example.bin"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(content)
            result = download_archive(spec, root)
        self.assertEqual(result["status"], "cache_hit")


if __name__ == "__main__":
    unittest.main()
