from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fireviewer_model_lab.tools.prepare_detector_coco_manifests import prepare_manifests


def _record(source: str, split: str, index: int) -> dict[str, object]:
    digest = hashlib.sha256(f"source-{source}-{split}-{index}".encode()).hexdigest()
    return {
        "annotations": [],
        "candidate_classes": [],
        "consent_basis": {"kind": "source_license", "reference": "test"},
        "corpus_role": "detector_training",
        "event_id": f"event-{source}",
        "height": 8,
        "image_relpath": f"images/{digest[:2]}/{digest}.jpg",
        "license": "test",
        "location": None,
        "near_duplicate_of": None,
        "negative_tags": ["no_target_visible"],
        "phash": f"phash-{index}",
        "sample_id": f"{source}-{split}-{index}",
        "sample_validation_status": "source_provided",
        "sequence_id": f"sequence-{source}-{index}",
        "sha256": digest,
        "source_id": source,
        "source_record_id": f"record-{index}",
        "split": split,
        "split_group": f"group-{source}-{index}",
        "visual_fingerprint": f"fingerprint-{index}",
        "width": 8,
    }


def _write_split(
    root: Path,
    parquet_split: str,
    coco_split: str,
    records: list[dict[str, object]],
    *,
    duplicate_last_image: bool = False,
) -> None:
    parquet_dir = root / "data" / parquet_split
    coco_dir = root / "_rfdetr_coco" / coco_split
    parquet_dir.mkdir(parents=True, exist_ok=True)
    coco_dir.mkdir(parents=True, exist_ok=True)
    parquet_rows = []
    images = []
    source_shard = f"data/{parquet_split}/part-00000.parquet"
    for index, record in enumerate(records):
        identity = f"{coco_split}|{source_shard}|{index}|row-{index}".encode()
        suffix = hashlib.sha1(identity, usedforsecurity=False).hexdigest()[:12]
        file_name = f"{index:09d}_row-{index}_{suffix}.jpg"
        image_payload = (
            f"jpeg-{parquet_split}-0"
            if duplicate_last_image and index
            else f"jpeg-{parquet_split}-{index}"
        ).encode()
        (coco_dir / file_name).write_bytes(image_payload)
        images.append({"id": index + 1, "file_name": file_name, "width": 9, "height": 7})
        parquet_rows.append(
            {
                "source_name": str(record["source_id"]),
                "sha256": str(record["sha256"]),
                "width": 8,
                "height": 8,
                "original_record_json": json.dumps(record),
            }
        )
    pq.write_table(pa.Table.from_pylist(parquet_rows), parquet_dir / "part-00000.parquet")
    (coco_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": [], "categories": []}), encoding="utf-8"
    )


def _fixture(root: Path) -> None:
    _write_split(
        root,
        "train",
        "train",
        [_record("fasdd", "train", 0), _record("pyro-sdis", "train", 1)],
    )
    _write_split(root, "validation", "valid", [_record("alarmod", "validation", 2)])
    _write_split(root, "test", "test", [_record("boreal", "test", 3)])
    (root / "_rfdetr_coco" / "_conversion_complete.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )


def test_prepare_manifests_uses_materialized_digest_and_preserves_source(tmp_path: Path) -> None:
    _fixture(tmp_path)

    report = prepare_manifests(tmp_path, tmp_path / "_rfdetr_coco")

    assert report["rows"] == 4
    assert report["source_counts"] == {
        "alarmod": 1,
        "boreal": 1,
        "fasdd": 1,
        "pyro-sdis": 1,
    }
    row = json.loads(
        (tmp_path / "_rfdetr_coco" / "manifest.fasdd.jsonl").read_text(encoding="utf-8")
    )
    image_path = tmp_path / "_rfdetr_coco" / str(row["image_relpath"])
    assert row["sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert row["source_sha256"] != row["sha256"]
    assert row["source_image_relpath"].startswith("images/")


def test_prepare_manifests_fails_on_coco_row_count_drift(tmp_path: Path) -> None:
    _fixture(tmp_path)
    annotations = tmp_path / "_rfdetr_coco" / "train" / "_annotations.coco.json"
    value = json.loads(annotations.read_text(encoding="utf-8"))
    value["images"].pop()
    annotations.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        prepare_manifests(tmp_path, tmp_path / "_rfdetr_coco")


def test_prepare_manifests_deduplicates_materialized_jpegs(tmp_path: Path) -> None:
    _fixture(tmp_path)
    _write_split(
        tmp_path,
        "train",
        "train",
        [_record("fasdd", "train", 0), _record("fasdd", "train", 1)],
        duplicate_last_image=True,
    )

    report = prepare_manifests(tmp_path, tmp_path / "_rfdetr_coco")

    assert report["input_rows"] == 4
    assert report["rows"] == 3
    assert report["duplicate_materialized_images_removed"] == 1
    assert report["duplicate_counts"] == {"fasdd": 1}
