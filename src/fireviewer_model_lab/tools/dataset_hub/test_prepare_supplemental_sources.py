from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from prepare_supplemental_sources import (
    BOREAL_SITE_SPLITS,
    CRISISFACTS_WILDFIRE_SPLITS,
    _boreal_detection_pairs,
    _boreal_segmentation_pairs,
    _boreal_site,
    _imsr_event_proxy,
    _normalize_tar_member_name,
    _tartanair_members,
    build_train_bundle,
    deterministic_group_splits,
    validate_normalized_source,
)


def row(pathname: str, content: bytes = b"x") -> dict[str, object]:
    return {
        "pathname": pathname,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


class SupplementalSourceTests(unittest.TestCase):
    def test_bundle_reuses_inventory_hashes_and_verifies_written_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "source"
            rows = []
            for split in ("train", "validation", "test"):
                payload_bytes = f"payload-{split}".encode()
                payload = source / "payload" / f"{split}.jpg"
                payload.parent.mkdir(parents=True, exist_ok=True)
                payload.write_bytes(payload_bytes)
                artifact_bytes = (json.dumps({"split": split}) + "\n").encode()
                artifact = source / "samples" / f"{split}.json"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(artifact_bytes)
                rows.append(
                    {
                        "sample_id": split,
                        "source_id": "fixture",
                        "task": "fixture",
                        "split": split,
                        "split_group": f"site:{split}",
                        "license": "CC-BY-4.0",
                        "provenance": {},
                        "artifact": {
                            "path": artifact.relative_to(source).as_posix(),
                            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                        },
                        "referenced_payloads": [
                            {
                                "path": payload.relative_to(source).as_posix(),
                                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                                "size_bytes": len(payload_bytes),
                                "role": "media",
                            }
                        ],
                    }
                )
            (source / "manifest.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
            )
            (source / "SOURCE_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "source_id": "fixture",
                        "license": "CC-BY-4.0",
                        "landing_page": "https://example.test/fixture",
                    }
                ),
                encoding="utf-8",
            )
            (source / "VALIDATION_REPORT.json").write_text("{}", encoding="utf-8")
            output = base / "ready"
            report = build_train_bundle(
                train_id="fixture-train",
                source_roots=[source],
                output_dir=output,
                entrypoints=[],
                training_ready=False,
                blocking_reasons=["fixture_only"],
                force=False,
            )
            self.assertTrue(report["zip_validation"]["entry_sha256_verified"])
            with zipfile.ZipFile(output / "fixture-train.zip") as archive:
                self.assertIsNone(archive.testzip())

    def test_tartanair_environment_split_is_site_isolated(self) -> None:
        from prepare_supplemental_sources import TARTANAIR_ENVIRONMENT_SPLITS

        self.assertEqual(
            set(TARTANAIR_ENVIRONMENT_SPLITS.values()),
            {
                "train",
                "validation",
                "test",
            },
        )
        self.assertEqual(len(TARTANAIR_ENVIRONMENT_SPLITS), 5)

    def test_tartanair_members_pair_frames_and_poses(self) -> None:
        import zipfile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "DesertGasStation/Data_easy/P000/image_lcam_front/000000_lcam_front.png",
                    b"png",
                )
                archive.writestr(
                    "DesertGasStation/Data_easy/P000/pose_lcam_front.txt",
                    "0 0 0 0 0 0 1\n",
                )
            with zipfile.ZipFile(path) as archive:
                frames, poses = _tartanair_members(archive, "DesertGasStation", "image")
        self.assertIn(("P000", 0), frames)
        self.assertIn("P000", poses)

    def test_diode_site_assignment_never_splits_a_scene(self) -> None:
        assignments = deterministic_group_splits(
            [f"train:scene_{index:05d}" for index in range(15)]
        )
        self.assertEqual(len(assignments), 15)
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})
        self.assertEqual(_normalize_tar_member_name("./train/outdoor/a.png"), "train/outdoor/a.png")

    def test_crisisfacts_wildfires_are_split_only_by_event(self) -> None:
        self.assertEqual(
            set(CRISISFACTS_WILDFIRE_SPLITS),
            {
                "CrisisFACTS-001",
                "CrisisFACTS-002",
                "CrisisFACTS-003",
                "CrisisFACTS-006",
            },
        )
        self.assertEqual(
            set(CRISISFACTS_WILDFIRE_SPLITS.values()),
            {
                "train",
                "validation",
                "test",
            },
        )

    def test_imsr_event_proxy_keeps_same_incident_days_together(self) -> None:
        first = {
            "imsr_date": "2019-08-01",
            "unit": "CA-ABC",
            "fire_name": " Example   Fire ",
        }
        second = {**first, "imsr_date": "2019-08-02"}
        different_year = {**first, "imsr_date": "2020-08-01"}
        self.assertEqual(_imsr_event_proxy(first), _imsr_event_proxy(second))
        self.assertNotEqual(_imsr_event_proxy(first), _imsr_event_proxy(different_year))

    def test_boreal_site_split_is_site_isolated_and_complete(self) -> None:
        self.assertEqual(_boreal_site("evoDJI_0001_frame0.jpg"), "evo")
        self.assertEqual(_boreal_site("RUOKOLAHTI_DJI_0089_frame1.jpg"), "ruokolahti")
        self.assertEqual(set(BOREAL_SITE_SPLITS.values()), {"train", "validation", "test"})
        self.assertEqual(len(BOREAL_SITE_SPLITS), 4)

    def test_boreal_detection_excludes_missing_labels_without_treating_them_as_negative(
        self,
    ) -> None:
        image_path = (
            "/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C/images/train/evoDJI_0001_frame0.jpg"
        )
        rows = [row(image_path)]
        pairs, excluded = _boreal_detection_pairs(rows, {image_path: rows[0]})
        self.assertEqual(pairs, [])
        self.assertEqual(excluded[0]["reason"], "official_detection_label_missing")

    def test_boreal_detection_quarantines_empty_labels_outside_documented_negatives(self) -> None:
        image_path = (
            "/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-A/Evo-Images/evoDJI_0001_frame23.jpg"
        )
        label_path = image_path.replace("-Images/", "-Labels/").replace(".jpg", ".txt")
        image = row(image_path, b"image")
        empty_label = row(label_path, b"")
        pairs, excluded = _boreal_detection_pairs(
            [image, empty_label],
            {image_path: image, label_path: empty_label},
        )
        self.assertEqual(pairs, [])
        self.assertEqual(
            excluded[0]["reason"],
            "official_empty_detection_label_outside_documented_negative_set",
        )

    def test_boreal_detection_keeps_documented_empty_image_negative(self) -> None:
        image_path = (
            "/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-A/Empty-Images/"
            "heinolaDJI_0001_frame1.jpg"
        )
        label_path = image_path.replace("-Images/", "-Labels/").replace(".jpg", ".txt")
        image = row(image_path, b"image")
        empty_label = row(label_path, b"")
        pairs, excluded = _boreal_detection_pairs(
            [image, empty_label],
            {image_path: image, label_path: empty_label},
        )
        self.assertEqual(excluded, [])
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0]["negative"])

    def test_boreal_segmentation_prefers_human_mask_and_marks_it_strong(self) -> None:
        image_path = (
            "/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C/images/test/evoDJI_0001_frame0.jpg"
        )
        sam_path = image_path.replace("/images/", "/sam_masks/").replace(".jpg", ".png")
        manual_path = image_path.replace("/images/", "/manual_masks/").replace(".jpg", ".png")
        rows = [row(image_path), row(sam_path, b"sam"), row(manual_path, b"human")]
        pairs, excluded = _boreal_segmentation_pairs(
            rows, {str(item["pathname"]): item for item in rows}
        )
        self.assertEqual(excluded, [])
        self.assertEqual(pairs[0]["mask"]["pathname"], manual_path)
        self.assertEqual(pairs[0]["annotation_strength"], "strong")
        self.assertEqual(pairs[0]["annotation_provenance"], "human_pixel_mask")

    def test_normalized_validator_checks_referenced_payload_hash_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "payload" / "image.jpg"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"image")
            artifact = root / "samples" / "sample.json"
            artifact.parent.mkdir()
            artifact.write_bytes(b"{}\n")
            manifest_row = {
                "sample_id": "sample",
                "source_id": "source",
                "task": "task",
                "split": "train",
                "split_group": "site:a",
                "license": "CC-BY-4.0",
                "provenance": {},
                "artifact": {
                    "path": "samples/sample.json",
                    "sha256": hashlib.sha256(b"{}\n").hexdigest(),
                },
                "referenced_payloads": [
                    {
                        "path": "payload/image.jpg",
                        "sha256": hashlib.sha256(b"image").hexdigest(),
                        "size_bytes": 5,
                        "role": "media",
                    }
                ],
            }
            (root / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
            (root / "SOURCE_MANIFEST.json").write_text("{}", encoding="utf-8")
            (root / "VALIDATION_REPORT.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty train/validation/test"):
                validate_normalized_source(root)


if __name__ == "__main__":
    unittest.main()
