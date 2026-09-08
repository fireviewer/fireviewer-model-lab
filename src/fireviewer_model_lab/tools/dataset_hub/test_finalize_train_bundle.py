from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path

from PIL import Image

TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from dataset_archive_validation import validate_optional_location  # noqa: E402
from finalize_train_bundle import (  # noqa: E402
    _alarmod_box_to_absolute,
    _derive_image_triage_manifest,
    _extract_train_bundle_subtree,
    _normalize_hls_split_groups,
    _validate_image_triage_bundle,
    _validate_prithvi_materialized_dataset,
    finalize_train_bundle,
    repair_alarmod_detection_manifest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_prithvi_materialized_validator_accepts_variable_eo4_shapes(
    tmp_path: Path,
) -> None:
    import numpy as np
    import rasterio

    data = tmp_path / "data"
    splits = tmp_path / "splits"
    data.mkdir()
    splits.mkdir()
    samples = {
        "train": ("hls_000000", 512, 512),
        "validation": ("eo4_000000", 17, 27),
        "test": ("eo4_000001", 33, 49),
    }
    for split, (sample_id, height, width) in samples.items():
        (splits / f"{split}.txt").write_text(sample_id + "\n", encoding="utf-8")
        with rasterio.open(
            data / f"{sample_id}_merged.tif",
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=6,
            dtype="float32",
        ) as image:
            image.write(np.zeros((6, height, width), dtype=np.float32))
        with rasterio.open(
            data / f"{sample_id}.mask.tif",
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="int16",
        ) as mask:
            mask.write(np.zeros((1, height, width), dtype=np.int16))
    (tmp_path / "materialization-report.json").write_text(
        json.dumps(
            {
                "combined_split_counts": {
                    "train": 1,
                    "validation": 1,
                    "test": 1,
                },
                "normalization": {"eo4_audit": {"compatible_with_hls_normalization": True}},
            }
        ),
        encoding="utf-8",
    )

    report = _validate_prithvi_materialized_dataset(
        tmp_path,
        expected_samples=3,
        expected_source_counts={"hls": 1, "eo4": 2},
    )

    assert report["samples"] == 3
    assert report["source_counts"] == {"eo4": 2, "hls": 1}
    assert report["raster_header_samples"]["eo4"]["shape_min"] == [17, 27]
    assert report["raster_header_samples"]["eo4"]["shape_max"] == [33, 49]
    assert report["raster_header_samples"]["hls"]["shape_min"] == [512, 512]
    assert report["runtime_tensor_contract"]["height"] == 512


def test_location_context_without_coordinates_is_explicitly_supported() -> None:
    assert (
        validate_optional_location(
            {
                "latitude": None,
                "longitude": None,
                "massif_id": "wikidata:Q1115037",
                "precision": "massif",
            },
            line_number=1,
        )
        is False
    )


def test_location_rejects_partial_coordinate_pairs() -> None:
    try:
        validate_optional_location(
            {"latitude": 44.75, "longitude": None},
            line_number=2,
        )
    except ValueError as error:
        assert "Incomplete coordinate pair" in str(error)
    else:
        raise AssertionError("A partial coordinate pair was accepted")


def test_hls_split_group_is_derived_without_changing_official_split(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    source_record_id = "subsetted_512x512_HLS.S30.T10SDH.2020248.v1.4_merged"
    manifest.write_text(
        json.dumps(
            {
                "source_record_id": source_record_id,
                "split": "validation",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = _normalize_hls_split_groups(tmp_path)
    normalized = json.loads(manifest.read_text(encoding="utf-8"))
    assert normalized["split"] == "validation"
    assert normalized["split_group"] == f"hls-scene:{source_record_id}"
    assert report["derived_rows"] == 1


def test_detection_manifest_is_derived_and_validated_for_image_triage(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "corpus" / "source"
    images = dataset / "images"
    images.mkdir(parents=True)
    rows = []
    examples = (
        ("fire", [{"class_name": "flame_visible"}], [], ["fire"]),
        (
            "fire_and_smoke",
            [{"class_name": "flame_visible"}, {"class_name": "smoke_visible"}],
            [],
            ["fire", "smoke"],
        ),
        ("normal", [], ["no_target_visible"], ["normal"]),
        ("smoke", [{"class_name": "smoke_visible"}], [], ["smoke"]),
    )
    for index, (_primary_class, source_annotations, negative_tags, _labels) in enumerate(examples):
        image = images / f"{index}.jpg"
        Image.new("RGB", (8, 6), color=(index * 40, 20, 10)).save(image, format="JPEG")
        rows.append(
            {
                "annotations": source_annotations,
                "height": 6,
                "image_relpath": f"images/{index}.jpg",
                "negative_tags": negative_tags,
                "sample_id": f"sample-{index}",
                "sha256": _sha256(image),
                "source_id": "source",
                "source_record_id": str(index),
                "split": "train",
                "split_group": f"group-{index}",
                "width": 8,
            }
        )
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    report = _derive_image_triage_manifest(dataset)
    assert report["primary_class_counts"] == {
        "fire": 1,
        "fire_and_smoke": 1,
        "normal": 1,
        "smoke": 1,
    }
    triage_rows = [
        json.loads(line)
        for line in (dataset / "triage-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["labels"] for row in triage_rows] == [
        ["fire"],
        ["fire", "smoke"],
        ["normal"],
        ["smoke"],
    ]
    validation = _validate_image_triage_bundle(
        bundle_root=tmp_path,
        validator={
            "manifests": ["corpus/source/triage-manifest.jsonl"],
            "expected_rows": 4,
            "expected_split_counts": {"train": 4},
            "expected_primary_class_counts": {
                "fire": 1,
                "fire_and_smoke": 1,
                "normal": 1,
                "smoke": 1,
            },
        },
    )
    assert validation["verified_image_paths"] == 4


def _make_archived_dataset(root: Path, source_id: str, color: tuple[int, int, int]) -> Path:
    archive_dir = root / "datasets" / "corpus" / source_id
    archive_dir.mkdir(parents=True)
    payload = root / f"payload-{source_id}"
    images = payload / "images"
    images.mkdir(parents=True)
    image = images / "image.jpg"
    Image.new("RGB", (10, 8), color=color).save(image, format="JPEG")
    digest = _sha256(image)
    final_image = image.with_name(f"{digest}.jpg")
    image.rename(final_image)
    row = {
        "annotations": [{"bbox_xywh": [1, 1, 4, 3], "class_name": "flame_visible"}],
        "corpus_role": "detector_training",
        "height": 8,
        "image_relpath": f"images/{digest}.jpg",
        "sample_id": f"{source_id}:1",
        "sha256": digest,
        "source_id": source_id,
        "split": "train",
        "split_group": f"{source_id}-group",
        "width": 10,
    }
    (payload / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    shard = archive_dir / "shard-00000.tar"
    with tarfile.open(shard, "w") as tar:
        for path in sorted(payload.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=f"payload/{path.relative_to(payload).as_posix()}")
    files = [path for path in payload.rglob("*") if path.is_file()]
    manifest = {
        "dataset_id": f"corpus/{source_id}",
        "file_count": len(files),
        "source_bytes": sum(path.stat().st_size for path in files),
        "shards": [
            {
                "path": f"datasets/corpus/{source_id}/shard-00000.tar",
                "sha256": _sha256(shard),
                "size_bytes": shard.stat().st_size,
                "file_count": len(files),
                "source_bytes": sum(path.stat().st_size for path in files),
            }
        ],
    }
    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_spec(root: Path, duplicate_mount: bool = False) -> Path:
    spec = {
        "schema_version": 1,
        "train_id": "media-filter-fire-smoke-v1",
        "training_ready": True,
        "entrypoints": [
            {
                "name": "rtdetr",
                "command": (
                    "python -m fireviewer_model_lab.training.train_rtdetr train "
                    "--manifest corpus/one/manifest.jsonl "
                    "--manifest corpus/two/manifest.jsonl"
                ),
            }
        ],
        "sources": [
            {
                "source_id": "one",
                "dataset_id": "corpus/one",
                "archive_manifest": "datasets/corpus/one/manifest.json",
                "mount_path": "corpus/one",
                "license": "test",
                "validator": "firewarning_image_manifest",
            },
            {
                "source_id": "two",
                "dataset_id": "corpus/two",
                "archive_manifest": "datasets/corpus/two/manifest.json",
                "mount_path": "corpus/one" if duplicate_mount else "corpus/two",
                "license": "test",
                "validator": "firewarning_image_manifest",
            },
        ],
        "excluded_evaluation_sets": ["operational-reference-a", "operational-reference-b"],
    }
    path = root / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_finalize_train_bundle_contains_all_sources_and_one_root(tmp_path: Path) -> None:
    _make_archived_dataset(tmp_path, "one", (200, 10, 10))
    _make_archived_dataset(tmp_path, "two", (20, 100, 200))
    report = finalize_train_bundle(
        spec_path=_write_spec(tmp_path),
        source_root=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        force=False,
    )
    zip_path = tmp_path / "output" / "media-filter-fire-smoke-v1.zip"
    assert report["integrity"]["unique_image_sha256"] == 2
    assert report["zip_validation"]["single_train_root"] == "media-filter-fire-smoke-v1"
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        assert all(name.startswith("media-filter-fire-smoke-v1/") for name in names)
        assert "media-filter-fire-smoke-v1/corpus/one/manifest.jsonl" in names
        assert "media-filter-fire-smoke-v1/corpus/two/manifest.jsonl" in names
        assert "media-filter-fire-smoke-v1/TRAIN_BUNDLE.json" in names


def test_finalize_train_bundle_rejects_duplicate_mount_paths(tmp_path: Path) -> None:
    _make_archived_dataset(tmp_path, "one", (200, 10, 10))
    _make_archived_dataset(tmp_path, "two", (20, 100, 200))
    try:
        finalize_train_bundle(
            spec_path=_write_spec(tmp_path, duplicate_mount=True),
            source_root=tmp_path,
            work_dir=tmp_path / "work",
            output_dir=tmp_path / "output",
            force=False,
        )
    except ValueError as error:
        assert "Duplicate mount_path" in str(error)
    else:
        raise AssertionError("Duplicate mount path was accepted")


def test_finalize_train_bundle_validates_receipt_tar_gz_sets(tmp_path: Path) -> None:
    shard_root = tmp_path / "eo4" / "shards" / "train"
    shard_root.mkdir(parents=True)
    payload = tmp_path / "scene.nc"
    payload.write_bytes(b"netcdf-fixture")
    payload_digest = _sha256(payload)
    archive_path = shard_root / "eo4wildfires-train-00000.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(payload, arcname="eo4wildfires/scene.nc")
    row = {
        "filename": "scene.nc",
        "id": "scene",
        "official_ordinal": 1,
        "sha256": payload_digest,
        "size_bytes": payload.stat().st_size,
        "source_member": "eo4wildfires/scene.nc",
        "split": "train",
    }
    manifest_path = shard_root / "eo4wildfires-train-00000.manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    receipt = {
        "archive": {
            "path": archive_path.name,
            "sha256": _sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
        },
        "manifest": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "samples": 1,
        "shard": 0,
        "split": "train",
    }
    (shard_root / "eo4wildfires-train-00000.receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    spec = {
        "schema_version": 1,
        "train_id": "burned-area-segmentation-v1",
        "entrypoints": [{"name": "preflight", "command": "validate"}],
        "sources": [
            {
                "kind": "receipt_tar_gz_set",
                "source_id": "eo4",
                "dataset_id": "additional/eo4",
                "receipt_glob": "eo4/shards/**/*.receipt.json",
                "expected_samples": 1,
                "mount_path": "additional/eo4",
            }
        ],
    }
    spec_path = tmp_path / "eo4-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    report = finalize_train_bundle(
        spec_path=spec_path,
        source_root=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        force=False,
    )
    assert report["source_validation"][0]["dataset_validation"]["rows"] == 1
    assert report["source_validation"][0]["dataset_validation"]["files_verified"] is True


def test_finalize_train_bundle_normalizes_alarmod_to_canonical_detection(tmp_path: Path) -> None:
    archive_dir = tmp_path / "additional" / "alarmod"
    archive_dir.mkdir(parents=True)
    payload = tmp_path / "alarmod-payload"
    image_dir = payload / "train" / "images"
    label_dir = payload / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image_path = image_dir / "image_3_0_0.jpg"
    Image.new("RGB", (20, 10), color=(230, 80, 10)).save(image_path, format="JPEG")
    digest = _sha256(image_path)
    (label_dir / "image_3_0_0.txt").write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    row = {
        "annotations": [
            {
                "bbox_xywh_normalized": [0.5, 0.5, 0.4, 0.2],
                "class_id": 0,
                "class_name": "fire",
                "point_xy_normalized": [0.5, 0.5],
            }
        ],
        "height": 10,
        "image_relpath": "train/images/image_3_0_0.jpg",
        "label_relpath": "train/labels/image_3_0_0.txt",
        "negative": False,
        "sample_id": "alarmod:1",
        "sha256": digest,
        "source_id": "alarmod",
        "split": "train",
        "split_group": "alarmod-frame:3",
        "width": 20,
    }
    (payload / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    shard = archive_dir / "shard-00000.tar"
    with tarfile.open(shard, "w") as archive:
        for path in sorted(payload.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=f"payload/{path.relative_to(payload).as_posix()}")
    files = [path for path in payload.rglob("*") if path.is_file()]
    archive_manifest = {
        "dataset_id": "additional/alarmod",
        "file_count": len(files),
        "source_bytes": sum(path.stat().st_size for path in files),
        "shards": [
            {
                "path": "additional/alarmod/shard-00000.tar",
                "sha256": _sha256(shard),
                "size_bytes": shard.stat().st_size,
                "file_count": len(files),
                "source_bytes": sum(path.stat().st_size for path in files),
            }
        ],
    }
    (archive_dir / "manifest.json").write_text(json.dumps(archive_manifest), encoding="utf-8")
    spec = {
        "schema_version": 1,
        "train_id": "alarmod-train",
        "entrypoints": [{"name": "train", "command": "train"}],
        "sources": [
            {
                "source_id": "alarmod",
                "dataset_id": "additional/alarmod",
                "archive_manifest": "additional/alarmod/manifest.json",
                "mount_path": "additional/alarmod",
                "transformer": "alarmod_to_firewarning_detection_v1",
                "validator": "firewarning_image_manifest",
            }
        ],
    }
    spec_path = tmp_path / "alarmod-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    report = finalize_train_bundle(
        spec_path=spec_path,
        source_root=tmp_path,
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        force=False,
    )
    source_report = report["source_validation"][0]
    assert source_report["transformation"]["annotation_count"] == 1
    assert source_report["dataset_validation"]["rows"] == 1
    canonical_manifest = (
        tmp_path / "work" / "alarmod-train" / "additional" / "alarmod" / "manifest.jsonl"
    )
    canonical = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    assert canonical["annotations"][0]["class_name"] == "flame_visible"
    assert canonical["annotations"][0]["class_id"] == 1
    assert canonical["annotations"][0]["bbox_xywh"] == [6.0, 4.0, 8.0, 2.0]
    assert canonical["annotations"][0]["validation_status"] == "source_provided"
    assert canonical["sample_validation_status"] == "source_provided"
    assert canonical["consent_basis"]["kind"] == "source_license"


def test_repair_alarmod_manifest_preserves_source_and_emits_rtdetr_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "manifest.jsonl"
    output = tmp_path / "manifest.rtdetr.jsonl"
    source.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "bbox_xywh": [1.0, 2.0, 3.0, 4.0],
                        "class_name": "flame_visible",
                        "point_xy_normalized": [0.25, 0.5],
                        "source_class_name": "fire",
                    }
                ],
                "corpus_role": "detector_training",
                "height": 10,
                "image_relpath": "images/example.jpg",
                "negative": False,
                "sample_id": "alarmod:1",
                "sha256": "0" * 64,
                "source_asset": {
                    "dataset": "alarmod/forest_fire",
                    "license": "GPL-3.0",
                    "revision": "revision",
                },
                "source_id": "alarmod",
                "source_record_id": "1",
                "split": "train",
                "split_group": "frame:1",
                "width": 20,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = repair_alarmod_detection_manifest(source, output)

    assert report["rows"] == 1
    assert source.is_file()
    repaired = json.loads(output.read_text(encoding="utf-8"))
    assert repaired["annotations"][0]["class_id"] == 1
    assert repaired["annotations"][0]["validation_status"] == "source_provided"
    assert repaired["sample_validation_status"] == "source_provided"
    assert repaired["negative_tags"] == []


def test_alarmod_box_repairs_only_subpixel_rounding_overflow() -> None:
    repaired = _alarmod_box_to_absolute(
        center_x=0.157031,
        center_y=0.006944,
        box_width=0.017188,
        box_height=0.013889,
        image_width=1280,
        image_height=720,
        line_number=27,
    )
    assert repaired[0] == 189.99936000000002
    assert repaired[1] == 0.0
    assert abs(repaired[2] - 22.00064) < 1e-9
    assert 9.999 < repaired[3] < 10.001


def test_alarmod_box_rejects_material_boundary_overflow() -> None:
    try:
        _alarmod_box_to_absolute(
            center_x=0.01,
            center_y=0.5,
            box_width=0.2,
            box_height=0.2,
            image_width=100,
            image_height=100,
            line_number=1,
        )
    except ValueError as error:
        assert "materially exceeds" in str(error)
    else:
        raise AssertionError("Material Alarmod overflow was accepted")


def test_verified_train_bundle_subtree_extracts_only_selected_payload(tmp_path: Path) -> None:
    archive_path = tmp_path / "source.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("coarse-v1/sources/aerial/rgb/example.jpg", b"image")
        archive.writestr("coarse-v1/sources/aerial/metadata.json", b"{}")
        archive.writestr("coarse-v1/sources/other/ignored.jpg", b"ignored")

    destination = tmp_path / "destination"
    destination.mkdir()
    report = _extract_train_bundle_subtree(
        source={
            "source_id": "aerial",
            "dataset_id": "sources/aerial",
            "bundle_zip": "source.zip",
            "bundle_train_id": "coarse-v1",
            "bundle_size_bytes": archive_path.stat().st_size,
            "bundle_sha256": _sha256(archive_path),
            "subtree": "sources/aerial",
            "mount_path": "sources/aerial",
        },
        source_root=tmp_path,
        destination=destination,
    )

    assert report["archive_validation"]["files"] == 2
    assert (destination / "rgb" / "example.jpg").read_bytes() == b"image"
    assert (destination / "metadata.json").read_text(encoding="utf-8") == "{}"
    assert not (destination / "ignored.jpg").exists()
