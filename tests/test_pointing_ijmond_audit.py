from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

import imagehash
import pytest
from PIL import Image, ImageDraw
from fireviewer_model_lab.tools.launch_sagemaker_pointing_ijmond import build_request
from fireviewer_model_lab.tools.launch_sagemaker_pointing_ijmond_materialize import (
    build_request as build_materialization_request,
)
from fireviewer_model_lab.training.pointing_ijmond_audit import (
    ARCHIVE_DOWNLOAD_URL,
    ARCHIVE_FILENAME,
    EXPECTED_ARCHIVE_BYTES,
    EXPECTED_ARCHIVE_MD5,
    FIGSHARE_ARTICLE_ID,
    FIGSHARE_FILE_ID,
    FIGSHARE_VERSION,
    SOURCE_DOI,
    SOURCE_LICENSE,
    SOURCE_LICENSE_URL,
    _perceptual_hash,
    audit_ijmond_archive,
    validate_figshare_article,
)
from fireviewer_model_lab.training.pointing_ijmond_materialize import materialize_ijmond_archive


def _contract() -> dict:
    return validate_figshare_article(
        {
            "id": FIGSHARE_ARTICLE_ID,
            "version": FIGSHARE_VERSION,
            "doi": SOURCE_DOI,
            "is_public": True,
            "license": {"name": SOURCE_LICENSE, "url": SOURCE_LICENSE_URL},
            "files": [
                {
                    "id": FIGSHARE_FILE_ID,
                    "name": ARCHIVE_FILENAME,
                    "size": EXPECTED_ARCHIVE_BYTES,
                    "computed_md5": EXPECTED_ARCHIVE_MD5,
                    "download_url": ARCHIVE_DOWNLOAD_URL,
                }
            ],
        }
    )


def _image_bytes(style: int, *, visible_source: bool = False) -> bytes:
    image = Image.new("RGB", (64, 48), (170 - style * 20, 190 - style * 10, 205))
    draw = ImageDraw.Draw(image)
    if style == 1:
        for x in range(0, 64, 8):
            draw.rectangle((x, 0, x + 3, 47), fill=(45, 115, 65))
    elif style == 2:
        for y in range(0, 48, 6):
            draw.line((0, y, 63, y), fill=(35, 55, 120), width=2)
    elif style == 3:
        draw.ellipse((4, 5, 31, 31), fill=(210, 80, 35))
        draw.rectangle((42, 4, 61, 42), fill=(25, 90, 145))
    if visible_source:
        draw.rectangle((30, 24, 36, 47), fill=(38, 38, 38))
        draw.polygon(
            [(14, 5), (48, 6), (38, 15), (35, 24), (31, 24), (27, 16)],
            fill=(205, 205, 198),
        )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, subsampling=0)
    return output.getvalue()


def _mask_bytes(kind: str) -> bytes:
    mask = Image.new("L", (64, 48), 0)
    draw = ImageDraw.Draw(mask)
    if kind == "connected":
        draw.polygon(
            [(14, 5), (48, 6), (38, 15), (35, 24), (31, 24), (27, 16)],
            fill=155,
        )
    elif kind == "detached":
        draw.ellipse((8, 4, 48, 18), fill=255)
    output = io.BytesIO()
    mask.save(output, format="PNG")
    return output.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _archive(
    path: Path,
    rows: list[tuple[str, bytes, bytes, bool]],
    *,
    coco_metadata: list[dict] | None = None,
) -> tuple[list[bytes], list[bytes]]:
    images = []
    masks = []
    coco_images = []
    annotations = []
    for index, (name, image, mask, positive) in enumerate(rows, 1):
        images.append(image)
        masks.append(mask)
        coco_image = {"id": index, "file_name": f"{name}.jpg"}
        if coco_metadata is not None:
            coco_image.update(coco_metadata[index - 1])
        coco_images.append(coco_image)
        if positive:
            annotations.append(
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 1,
                    "segmentation": [[8, 4, 48, 4, 48, 24, 8, 24]],
                }
            )
    coco = json.dumps(
        {
            "images": coco_images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "smoke"}],
        }
    ).encode()
    with tarfile.open(path, mode="w:gz") as archive:
        for (name, _, _, _), image, mask in zip(rows, images, masks, strict=True):
            _add_bytes(archive, f"ijmond_seg/test/raw/images/{name}.jpg", image)
            _add_bytes(archive, f"ijmond_seg/test/raw/masks/{name}.png", mask)
        _add_bytes(archive, "ijmond_seg/test/annotations/instances_raw.json", coco)
        _add_bytes(archive, "ijmond_seg/test/cropped/images/ignored.jpg", images[0])
    return images, masks


def _far_phash(signatures: list[str], seed: str) -> str:
    for nonce in range(10_000):
        candidate = hashlib.sha256(f"{seed}:{nonce}".encode()).hexdigest()[:16]
        if all((int(candidate, 16) ^ int(value, 16)).bit_count() > 20 for value in signatures):
            return candidate
    raise AssertionError("could not construct a distant pHash")


def _index(root: Path, rows: list[dict]) -> dict[str, object]:
    root.mkdir(parents=True)
    normalized_rows = []
    for source in rows:
        row = {**source, "index_partition": root.name}
        if root.name == "benchmark":
            row.setdefault("phash64_flipped", row.get("phash"))
        normalized_rows.append(row)
    index = root / "hash-index.jsonl"
    index.write_text(
        "".join(json.dumps(row) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 1,
        "partition": root.name,
        "index_filename": index.name,
        "index_rows": len(normalized_rows),
        "index_bytes": index.stat().st_size,
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "incomplete_rows": 0,
        "media_files_output": 0,
        "source_gate_passed": True,
        "gate_errors": [],
        "publication_allowed": False,
    }
    receipt_path = root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return {
        "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "rows": len(normalized_rows),
    }


def _audit(
    archive: Path,
    output: Path,
    detection: Path,
    pointing: Path,
    benchmark: Path,
    *,
    raw_images: int,
    positive_images: int,
    raw_polygons: int | None = None,
    expected_index_receipts: dict[str, dict[str, object]] | None = None,
) -> dict:
    roots = {
        "detection": detection,
        "pointing": pointing,
        "benchmark": benchmark,
    }
    pinned = expected_index_receipts or {
        name: {
            "receipt_sha256": hashlib.sha256((root / "receipt.json").read_bytes()).hexdigest(),
            "rows": len((root / "hash-index.jsonl").read_text(encoding="utf-8").splitlines()),
        }
        for name, root in roots.items()
    }
    return audit_ijmond_archive(
        archive_path=archive,
        detection_index_root=detection,
        pointing_index_root=pointing,
        benchmark_index_root=benchmark,
        output_dir=output,
        output_s3_prefix="s3://bucket/pointing-corpus-v2/reports/ijmond/job",
        source_contract_receipt=_contract(),
        expected_index_receipts=pinned,
        expected_archive_bytes=archive.stat().st_size,
        expected_archive_md5=hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest(),
        expected_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        expected_raw_images=raw_images,
        expected_raw_positive_images=positive_images,
        expected_raw_polygons=positive_images if raw_polygons is None else raw_polygons,
    )


def _unit_split_receipt(rows: list[dict]) -> dict:
    for row in rows:
        row["split"] = "train"
        row["final_split"] = "train"
    return {
        "strategy": "unit_test_existing_episode_groups",
        "observed_ratios": {"train": 1.0, "validation": 0.0, "test": 0.0},
        "gate_errors": [],
    }


def test_ijmond_requires_visible_source_but_keeps_valid_abstentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fireviewer_model_lab.training.pointing_ijmond_audit._assign_group_splits", _unit_split_receipt
    )
    archive = tmp_path / "ijmond.tar.gz"
    rows = [
        (
            "kooks_1_20260101T120000",
            _image_bytes(0, visible_source=True),
            _mask_bytes("connected"),
            True,
        ),
        (
            "kooks_1_20260101T121000",
            _image_bytes(1),
            _mask_bytes("detached"),
            True,
        ),
        (
            "kooks_2_20260101T140000",
            _image_bytes(2),
            _mask_bytes("negative"),
            False,
        ),
    ]
    images, _ = _archive(archive, rows)
    signatures = [_perceptual_hash(Image.open(io.BytesIO(value))) for value in images]
    assert signatures[0] == str(imagehash.phash(Image.open(io.BytesIO(images[0]))))
    for name in ("detection", "pointing", "benchmark"):
        _index(
            tmp_path / name,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash(signatures, name),
                }
            ],
        )

    output = tmp_path / "output"
    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=3,
        positive_images=2,
    )
    dispositions = [
        json.loads(line)
        for line in (output / "ijmond_automatic_dispositions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_time = {row["captured_at"][11:19]: row for row in dispositions}
    connected = by_time["12:00:00"]
    detached = by_time["12:10:00"]
    negative = by_time["14:00:00"]

    assert report["source_gate_passed"] is True
    assert report["gate_errors"] == []
    assert report["cropped_assets_admitted"] == 0
    assert report["smoke_column_base_points"] == 1
    assert report["strict_defensible_point_ceiling"] == 1
    assert connected["anchor_points"][0]["kind"] == "smoke_column_base"
    assert connected["point_supervised"] is True
    assert detached["training_eligible"] is True
    assert detached["point_supervised"] is False
    assert detached["anchor_points"] == []
    assert detached["corpus_role"] == "segmentation_presence_abstention_auxiliary"
    assert negative["presence_targets"]["smoke_visible"] is False
    assert negative["point_supervised"] is False
    assert connected["split_group"] == detached["split_group"]
    assert connected["split"] == detached["split"]
    assert report["reviews_admitted"] is False
    assert report["publication_allowed"] is False
    assert report["benchmark_hash_exclusion_only"] is True
    assert report["strict_payload_artifacts_materialized"] == 0


def test_ijmond_excludes_detection_pointing_and_benchmark_overlaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fireviewer_model_lab.training.pointing_ijmond_audit._assign_group_splits", _unit_split_receipt
    )
    archive = tmp_path / "ijmond.tar.gz"
    rows = [
        (
            "kooks_1_20260101T100000",
            _image_bytes(0, visible_source=True),
            _mask_bytes("connected"),
            True,
        ),
        (
            "kooks_1_20260101T110000",
            _image_bytes(1),
            _mask_bytes("detached"),
            True,
        ),
        (
            "kooks_2_20260101T120000",
            _image_bytes(2),
            _mask_bytes("detached"),
            True,
        ),
        (
            "hoogovens_6_7_20260101T130000",
            _image_bytes(3),
            _mask_bytes("detached"),
            True,
        ),
    ]
    images, _ = _archive(archive, rows)
    signatures = [_perceptual_hash(Image.open(io.BytesIO(value))) for value in images]
    hashes = [hashlib.sha256(value).hexdigest() for value in images]
    _index(
        tmp_path / "detection",
        [{"sample_id": "det:exact", "image_sha256": hashes[0], "phash": signatures[0]}],
    )
    _index(
        tmp_path / "pointing",
        [
            {
                "sample_id": "point:phash",
                "image_sha256": "a" * 64,
                "phash": signatures[1],
            }
        ],
    )
    _index(
        tmp_path / "benchmark",
        [
            {
                "sample_id": "bench:phash",
                "image_sha256": "b" * 64,
                "phash": _far_phash(signatures, "benchmark-primary"),
                "phash64_flipped": signatures[2],
            }
        ],
    )

    output = tmp_path / "output"
    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=4,
        positive_images=4,
    )
    dispositions = [
        json.loads(line)
        for line in (output / "ijmond_automatic_dispositions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_time = {row["captured_at"][11:19]: row for row in dispositions}

    assert report["source_gate_passed"] is True
    assert "detection_exact_sha_overlap" in by_time["10:00:00"]["exclusion_reasons"]
    assert "detection_phash_overlap" in by_time["10:00:00"]["exclusion_reasons"]
    assert "pointing_phash_overlap" in by_time["11:00:00"]["exclusion_reasons"]
    assert "benchmark_phash_overlap" in by_time["12:00:00"]["exclusion_reasons"]
    assert by_time["13:00:00"]["training_eligible"] is True
    assert report["independent_benchmark_used_for_training"] is False


def test_ijmond_uses_variable_filename_times_when_date_captured_is_constant_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fireviewer_model_lab.training.pointing_ijmond_audit._assign_group_splits", _unit_split_receipt
    )
    archive = tmp_path / "constant-export-date.tar.gz"
    rows = [
        (
            "kooks_1__2024-08-28T10-06-56Z",
            _image_bytes(0, visible_source=True),
            _mask_bytes("connected"),
            True,
        ),
        (
            "kooks_1__2024-08-28T11-06-56Z",
            _image_bytes(1),
            _mask_bytes("detached"),
            True,
        ),
        (
            "kooks_2__2024-08-28T12-06-56Z",
            _image_bytes(3),
            _mask_bytes("negative"),
            False,
        ),
    ]
    images, _ = _archive(
        archive,
        rows,
        coco_metadata=[{"date_captured": "2025-02-13T22:21:42Z"}] * len(rows),
    )
    signatures = [_perceptual_hash(Image.open(io.BytesIO(value))) for value in images]
    for name in ("detection", "pointing", "benchmark"):
        _index(
            tmp_path / name,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash(signatures, f"constant-date-{name}"),
                }
            ],
        )
    output = tmp_path / "constant-export-output"

    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=3,
        positive_images=2,
    )

    assert report["source_gate_passed"] is True
    assert report["date_captured_classification"] == "dataset_constant_export_metadata"
    assert report["date_captured_ignored_for_captured_at"] is True
    assert report["captured_at_from_filename_rows"] == 3
    receipt = json.loads((output / report["date_captured_receipt"]).read_text())
    assert receipt["constant_value"] == "2025-02-13T22:21:42Z"
    assert receipt["filename_timestamp_valid_rows"] == 3
    assert receipt["filename_timestamp_distinct_values"] == 3
    dispositions = [
        json.loads(line)
        for line in (output / "ijmond_automatic_dispositions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["captured_at"] for row in dispositions} == {
        "2024-08-28T10:06:56+00:00",
        "2024-08-28T11:06:56+00:00",
        "2024-08-28T12:06:56+00:00",
    }
    assert all(
        "timestamp_filename_metadata_mismatch" not in row["exclusion_reasons"]
        for row in dispositions
    )


def test_ijmond_constant_export_date_does_not_relax_explicit_capture_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fireviewer_model_lab.training.pointing_ijmond_audit._assign_group_splits", _unit_split_receipt
    )
    archive = tmp_path / "explicit-conflict.tar.gz"
    rows = [
        (
            "kooks_1__2024-08-28T10-06-56Z",
            _image_bytes(0, visible_source=True),
            _mask_bytes("connected"),
            True,
        ),
        (
            "kooks_1__2024-08-28T11-06-56Z",
            _image_bytes(1),
            _mask_bytes("detached"),
            True,
        ),
        (
            "kooks_2__2024-08-28T12-06-56Z",
            _image_bytes(3),
            _mask_bytes("negative"),
            False,
        ),
    ]
    metadata = [{"date_captured": "2025-02-13T22:21:42Z"} for _ in rows]
    metadata[1]["captured_at"] = "2024-08-28T15:06:56Z"
    images, _ = _archive(archive, rows, coco_metadata=metadata)
    signatures = [_perceptual_hash(Image.open(io.BytesIO(value))) for value in images]
    for name in ("detection", "pointing", "benchmark"):
        _index(
            tmp_path / name,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash(signatures, f"explicit-conflict-{name}"),
                }
            ],
        )
    output = tmp_path / "explicit-conflict-output"

    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=3,
        positive_images=2,
    )

    assert report["source_gate_passed"] is True
    assert report["date_captured_ignored_for_captured_at"] is True
    assert report["exclusions_by_reason"]["timestamp_filename_metadata_mismatch"] == 1


def test_ijmond_nonconstant_date_captured_remains_capture_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fireviewer_model_lab.training.pointing_ijmond_audit._assign_group_splits", _unit_split_receipt
    )
    archive = tmp_path / "variable-capture-date.tar.gz"
    rows = [
        (
            "kooks_1__2024-08-28T10-06-56Z",
            _image_bytes(0, visible_source=True),
            _mask_bytes("connected"),
            True,
        ),
        (
            "kooks_1__2024-08-28T11-06-56Z",
            _image_bytes(1),
            _mask_bytes("detached"),
            True,
        ),
        (
            "kooks_2__2024-08-28T12-06-56Z",
            _image_bytes(3),
            _mask_bytes("negative"),
            False,
        ),
    ]
    metadata = [
        {"date_captured": "2024-08-28T10:06:56Z"},
        {"date_captured": "2024-08-28T15:06:56Z"},
        {"date_captured": "2024-08-28T16:06:56Z"},
    ]
    images, _ = _archive(archive, rows, coco_metadata=metadata)
    signatures = [_perceptual_hash(Image.open(io.BytesIO(value))) for value in images]
    for name in ("detection", "pointing", "benchmark"):
        _index(
            tmp_path / name,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash(signatures, f"variable-date-{name}"),
                }
            ],
        )
    output = tmp_path / "variable-capture-output"

    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=3,
        positive_images=2,
    )

    assert report["source_gate_passed"] is True
    assert report["date_captured_classification"] == "capture_metadata_not_ignorable"
    assert report["date_captured_ignored_for_captured_at"] is False
    assert report["exclusions_by_reason"]["timestamp_filename_metadata_mismatch"] == 2


def test_ijmond_coco_mask_presence_disagreement_is_row_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fireviewer_model_lab.training.pointing_ijmond_audit._assign_group_splits", _unit_split_receipt
    )
    archive = tmp_path / "presence-disagreement.tar.gz"
    rows = [
        (
            "kooks_1_20260101T100000",
            _image_bytes(0, visible_source=True),
            _mask_bytes("connected"),
            True,
        ),
        (
            "kooks_1_20260101T110000",
            _image_bytes(1),
            _mask_bytes("negative"),
            True,
        ),
        (
            "kooks_2_20260101T120000",
            _image_bytes(3),
            _mask_bytes("detached"),
            True,
        ),
    ]
    images, _ = _archive(archive, rows)
    signatures = [_perceptual_hash(Image.open(io.BytesIO(value))) for value in images]
    for name in ("detection", "pointing", "benchmark"):
        _index(
            tmp_path / name,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash(signatures, f"presence-{name}"),
                }
            ],
        )
    output = tmp_path / "presence-disagreement-output"

    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=3,
        positive_images=3,
        raw_polygons=3,
    )

    assert report["source_gate_passed"] is True
    assert report["gate_errors"] == []
    assert report["positive_raw_coco_images"] == 3
    assert report["positive_raw_masks"] == 2
    assert report["coco_mask_presence_disagreements"] == 1
    assert report["exclusions_by_reason"]["coco_mask_presence_disagreement"] == 1
    dispositions = [
        json.loads(line)
        for line in (output / "ijmond_automatic_dispositions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    disagreement = next(row for row in dispositions if row["captured_at"][11:19] == "11:00:00")
    assert disagreement["training_eligible"] is False


def test_ijmond_fails_closed_when_an_exclusion_index_lacks_phash(tmp_path: Path) -> None:
    archive = tmp_path / "ijmond.tar.gz"
    image = _image_bytes(0, visible_source=True)
    _archive(
        archive,
        [
            (
                "kooks_1_20260101T100000",
                image,
                _mask_bytes("connected"),
                True,
            )
        ],
    )
    signature = _perceptual_hash(Image.open(io.BytesIO(image)))
    for name in ("detection", "benchmark"):
        _index(
            tmp_path / name,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash([signature], name),
                }
            ],
        )
    _index(
        tmp_path / "pointing",
        [{"sample_id": "pointing:incomplete", "image_sha256": "a" * 64}],
    )

    output = tmp_path / "output"
    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=1,
        positive_images=1,
    )
    validated = (output / "ijmond_strict_validated_manifest.jsonl").read_text(encoding="utf-8")

    assert report["source_gate_passed"] is False
    assert any(
        error.startswith("pointing_exclusion_index_incomplete_rows")
        for error in report["gate_errors"]
    )
    assert report["strict_defensible_point_ceiling"] == 0
    assert validated == ""


def test_ijmond_fails_closed_when_an_exclusion_receipt_drifts(tmp_path: Path) -> None:
    archive = tmp_path / "ijmond.tar.gz"
    image = _image_bytes(0, visible_source=True)
    _archive(
        archive,
        [("kooks_1_20260101T100000", image, _mask_bytes("connected"), True)],
    )
    signature = _perceptual_hash(Image.open(io.BytesIO(image)))
    contracts = {
        name: _index(
            tmp_path / name,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash([signature], name),
                }
            ],
        )
        for name in ("detection", "pointing", "benchmark")
    }
    receipt_path = tmp_path / "pointing" / "receipt.json"
    receipt_path.write_text(receipt_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    output = tmp_path / "output"
    report = _audit(
        archive,
        output,
        tmp_path / "detection",
        tmp_path / "pointing",
        tmp_path / "benchmark",
        raw_images=1,
        positive_images=1,
        expected_index_receipts=contracts,
    )

    assert report["source_gate_passed"] is False
    assert "pointing_exclusion_receipt_sha256_mismatch" in report["gate_errors"]
    assert (output / "ijmond_strict_validated_manifest.jsonl").read_text(encoding="utf-8") == ""


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bucket": "bucket",
        "role": "role",
        "image": "image",
        "detection_index_prefix": "pointing/exclusion-index/detection",
        "pointing_index_prefix": "pointing/exclusion-index/pointing",
        "benchmark_index_prefix": "pointing/exclusion-index/benchmark",
        "detection_index_receipt_sha256": "a" * 64,
        "pointing_index_receipt_sha256": "b" * 64,
        "benchmark_index_receipt_sha256": "c" * 64,
        "detection_index_rows": 102_257,
        "pointing_index_rows": 393,
        "benchmark_index_rows": 200,
        "code_prefix": "pointing-corpus-v2/code/ijmond-audit",
        "output_prefix": "pointing-corpus-v2/reports/ijmond-audit",
        "instance_type": "ml.m5.2xlarge",
        "volume_size": 30,
        "max_runtime": 7_200,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_ijmond_launcher_is_hash_only_closed_and_never_publishes() -> None:
    request = build_request(_args(), "ijmond-job")

    inputs = {item["InputName"]: item for item in request["ProcessingInputs"]}
    assert set(inputs) == {
        "code",
        "detection-exclusion-index",
        "pointing-exclusion-index",
        "benchmark-exclusion-index",
    }
    assert all("strict-payload" not in json.dumps(value) for value in inputs.values())
    tags = {item["Key"]: item["Value"] for item in request["Tags"]}
    assert tags["fireviewer:publication-allowed"] == "false"
    assert tags["fireviewer:reviews-admitted"] == "false"
    assert tags["fireviewer:benchmark-access"] == "hash-only-exclusion"
    assert request["Environment"]["FIREVIEWER_PUBLICATION_ALLOWED"] == "false"


def test_ijmond_launcher_rejects_raw_or_publication_paths() -> None:
    with pytest.raises(ValueError, match="hash-only"):
        build_request(_args(benchmark_index_prefix="benchdata/raw"), "ijmond-job")
    with pytest.raises(ValueError, match="publication"):
        build_request(_args(output_prefix="pointing/publish/huggingface"), "ijmond-job")


def _materialization_launcher_args(**overrides: object) -> argparse.Namespace:
    values = vars(_args()).copy()
    values.update(
        {
            "audit_prefix": "pointing-corpus-v2/reports/ijmond-audit/successful-job",
            "audit_summary_sha256": "d" * 64,
            "audit_strict_manifest_sha256": "e" * 64,
            "audit_point_manifest_sha256": "f" * 64,
            "code_prefix": "pointing-corpus-v2/code/ijmond-materialize/run",
            "output_prefix": "pointing-corpus-v2/materialized/ijmond",
            "point_cap": 750,
            "instance_type": "ml.t3.xlarge",
        }
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_ijmond_materialization_launcher_is_pinned_and_isolated() -> None:
    request = build_materialization_request(
        _materialization_launcher_args(), "ijmond-materialize-job"
    )

    inputs = {item["InputName"]: item for item in request["ProcessingInputs"]}
    assert set(inputs) == {
        "code",
        "successful-audit",
        "detection-exclusion-index",
        "pointing-exclusion-index",
        "benchmark-exclusion-index",
    }
    arguments = request["AppSpecification"]["ContainerArguments"]
    assert "--download-pinned-archive" in arguments
    assert "--expected-audit-summary-sha256" in arguments
    assert "--expected-audit-strict-manifest-sha256" in arguments
    assert "--expected-audit-point-manifest-sha256" in arguments
    tags = {item["Key"]: item["Value"] for item in request["Tags"]}
    assert tags["fireviewer:labels-generated"] == "false"
    assert tags["fireviewer:bbox-conversion"] == "false"
    assert tags["fireviewer:publication-allowed"] == "false"


def test_ijmond_materialization_launcher_fails_closed() -> None:
    with pytest.raises(ValueError, match="publication"):
        build_materialization_request(
            _materialization_launcher_args(output_prefix="pointing/publish/hf-upload"),
            "ijmond-materialize-job",
        )
    with pytest.raises(ValueError, match="source cap"):
        build_materialization_request(
            _materialization_launcher_args(point_cap=751), "ijmond-materialize-job"
        )
    with pytest.raises(ValueError, match="SHA-256"):
        build_materialization_request(
            _materialization_launcher_args(audit_summary_sha256="unpinned"),
            "ijmond-materialize-job",
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialization_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path], dict]:
    archive = tmp_path / "ijmond-materialize.tar.gz"
    rows = [
        (
            "kooks_1_20260101T120000",
            _image_bytes(0, visible_source=True),
            _mask_bytes("connected"),
            True,
        ),
        (
            "kooks_1_20260101T121000",
            _image_bytes(1),
            _mask_bytes("detached"),
            True,
        ),
        (
            "kooks_2_20260101T140000",
            _image_bytes(3),
            _mask_bytes("negative"),
            False,
        ),
    ]
    images, _ = _archive(archive, rows)
    signatures = [_perceptual_hash(Image.open(io.BytesIO(value))) for value in images]
    roots = {
        name: tmp_path / "materialize-indexes" / name
        for name in ("detection", "pointing", "benchmark")
    }
    receipts = {
        name: _index(
            root,
            [
                {
                    "sample_id": f"{name}:seed",
                    "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "phash": _far_phash(signatures, f"materialize-{name}"),
                }
            ],
        )
        for name, root in roots.items()
    }
    audit_output = tmp_path / "materialize-audit"
    report = _audit(
        archive,
        audit_output,
        roots["detection"],
        roots["pointing"],
        roots["benchmark"],
        raw_images=3,
        positive_images=2,
        expected_index_receipts=receipts,
    )
    dispositions = [
        json.loads(line)
        for line in (audit_output / "ijmond_automatic_dispositions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    dispositions.sort(key=lambda row: str(row["captured_at"]))
    for index, row in enumerate(dispositions):
        split = ("train", "validation", "test")[index]
        row.update(
            {
                "split": split,
                "final_split": split,
                "source_group": f"ijmond:test:episode-{index + 1:04d}",
                "split_group": f"ijmond:test:episode-{index + 1:04d}",
                "strict_keep": True,
                "training_eligible": True,
                "sample_validation_status": "strict_automated_validated_auxiliary",
                "corpus_role": "segmentation_presence_abstention_auxiliary",
            }
        )
    point = dispositions[0]
    diagnostics = point["point_gate_diagnostics"]
    point["point_supervised"] = True
    point["anchor_points"] = [
        {
            "kind": "smoke_column_base",
            "x": round(diagnostics["base_x_px"] / (point["width"] - 1), 8),
            "y": round(diagnostics["base_y_px"] / (point["height"] - 1), 8),
            "origin": "human_revised_mask_connected_visible_source_gate_v1",
        }
    ]
    point["point_derivation"] = "human_revised_mask_connected_visible_source_gate_v1"
    point["point_gate_errors"] = []
    point["visual_abstention_reason"] = None
    point["point_cap_status"] = "selected"
    point["sample_validation_status"] = "strict_automated_validated_point"
    point["corpus_role"] = "pointing_segmentation_presence_abstention"
    strict_path = audit_output / "ijmond_strict_validated_manifest.jsonl"
    point_path = audit_output / "ijmond_strict_point_manifest.jsonl"
    strict_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in dispositions),
        encoding="utf-8",
    )
    point_path.write_text(json.dumps(point, sort_keys=True) + "\n", encoding="utf-8")
    report.update(
        {
            "source_gate_passed": True,
            "gate_errors": [],
            "strict_automated_validated_rows": len(dispositions),
            "smoke_column_base_points": 1,
            "validated_split_counts": {"train": 1, "validation": 1, "test": 1},
            "camera_episode_split_leakage": [],
        }
    )
    (audit_output / "ijmond_audit_summary.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )
    return archive, audit_output, roots, receipts


def _materialize(
    archive: Path,
    audit_output: Path,
    roots: dict[str, Path],
    receipts: dict,
    output: Path,
    **overrides: object,
) -> dict:
    summary = audit_output / "ijmond_audit_summary.json"
    strict = audit_output / "ijmond_strict_validated_manifest.jsonl"
    points = audit_output / "ijmond_strict_point_manifest.jsonl"
    values = {
        "archive_path": archive,
        "audit_summary_path": summary,
        "audit_strict_manifest_path": strict,
        "audit_point_manifest_path": points,
        "expected_archive_sha256": _sha256(archive),
        "expected_audit_summary_sha256": _sha256(summary),
        "expected_audit_strict_manifest_sha256": _sha256(strict),
        "expected_audit_point_manifest_sha256": _sha256(points),
        "detection_index_root": roots["detection"],
        "pointing_index_root": roots["pointing"],
        "benchmark_index_root": roots["benchmark"],
        "expected_index_receipts": receipts,
        "output_dir": output,
    }
    values.update(overrides)
    return materialize_ijmond_archive(**values)


def test_ijmond_materialization_binds_audit_exclusions_and_payloads(tmp_path: Path) -> None:
    archive, audit_output, roots, receipts = _materialization_fixture(tmp_path)
    output = tmp_path / "materialized"

    receipt = _materialize(archive, audit_output, roots, receipts, output)

    assert receipt["materialization_gate_passed"] is True
    assert receipt["materialized_rows"] == 3
    assert receipt["materialized_point_rows"] == 1
    assert receipt["materialized_auxiliary_rows"] == 2
    assert receipt["materialized_payload_files"] == 6
    assert receipt["split_group_leaks"] == 0
    assert receipt["labels_generated_during_materialization"] == 0
    assert receipt["audited_native_geometry_derived_points"] == 1
    assert receipt["box_derived_labels"] == 0
    materialized = [
        json.loads(line)
        for line in (output / "ijmond_materialized_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    strict_points = [
        json.loads(line)
        for line in (output / "ijmond_strict_validated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(strict_points) == 1
    assert strict_points[0]["sample_validation_status"] == "strict_automated_validated"
    assert strict_points[0]["anchor_points"][0]["origin"] == (
        "human_revised_mask_connected_visible_source_gate_v1"
    )
    for row in materialized:
        image = output / row["image_relpath"]
        mask = output / row["mask_relpath"]
        assert _sha256(image) == row["source_image_sha256"]
        assert _sha256(mask) == row["mask_sha256"]
        assert row["media_license"] == SOURCE_LICENSE
        assert row["mask_license"] == SOURCE_LICENSE


def test_ijmond_materialization_refuses_failed_audit(tmp_path: Path) -> None:
    archive, audit_output, roots, receipts = _materialization_fixture(tmp_path)
    summary = audit_output / "ijmond_audit_summary.json"
    value = json.loads(summary.read_text(encoding="utf-8"))
    value["source_gate_passed"] = False
    value["gate_errors"] = ["forced_test_failure"]
    summary.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="audit source gate did not pass"):
        _materialize(archive, audit_output, roots, receipts, output)

    assert not output.exists()


def test_ijmond_materialization_refuses_unpinned_audit_or_excessive_cap(
    tmp_path: Path,
) -> None:
    archive, audit_output, roots, receipts = _materialization_fixture(tmp_path)
    summary = audit_output / "ijmond_audit_summary.json"

    with pytest.raises(ValueError, match="audit summary SHA-256 mismatch"):
        _materialize(
            archive,
            audit_output,
            roots,
            receipts,
            tmp_path / "bad-summary",
            expected_audit_summary_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="point_cap must be between"):
        _materialize(
            archive,
            audit_output,
            roots,
            receipts,
            tmp_path / "bad-cap",
            point_cap=751,
        )
    assert summary.is_file()


def test_ijmond_materialization_rechecks_all_exclusions_after_audit(tmp_path: Path) -> None:
    archive, audit_output, roots, receipts = _materialization_fixture(tmp_path)
    strict_row = json.loads(
        (audit_output / "ijmond_strict_validated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    replacement_root = tmp_path / "replacement-indexes" / "detection"
    replacement_receipt = _index(
        replacement_root,
        [
            {
                "sample_id": "detection:new-overlap",
                "image_sha256": strict_row["source_image_sha256"],
                "phash": strict_row["phash"],
            }
        ],
    )
    roots["detection"] = replacement_root
    receipts["detection"] = replacement_receipt
    summary_path = audit_output / "ijmond_audit_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["external_exclusion_indexes"]["detection"].update(
        {
            "receipt_sha256": replacement_receipt["receipt_sha256"],
            "indexed_rows": replacement_receipt["rows"],
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "overlap-must-not-materialize"

    with pytest.raises(ValueError, match="newly detected detection corpus overlap"):
        _materialize(archive, audit_output, roots, receipts, output)

    assert not output.exists()
