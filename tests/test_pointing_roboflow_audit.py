from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from fireviewer_model_lab.tools.launch_sagemaker_pointing_roboflow import build_request
from fireviewer_model_lab.training.pointing_ijmond_audit import _perceptual_hash
from fireviewer_model_lab.training.pointing_roboflow_audit import audit_roboflow_manifest


def _image(seed: int) -> bytes:
    rng = random.Random(seed)  # noqa: S311 - deterministic visual fixture only
    image = Image.new("RGB", (100, 100))
    image.putdata(
        [
            (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(10_000)
        ]
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _phash(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        return _perceptual_hash(image)


def _same_pixels_different_payload(payload: bytes) -> bytes:
    with Image.open(io.BytesIO(payload)) as opened:
        image = opened.copy()
    metadata = PngInfo()
    metadata.add_text("immutable-test-variant", "different encoded bytes")
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


def _far_phash(signatures: list[str], seed: str) -> str:
    for nonce in range(10_000):
        candidate = hashlib.sha256(f"{seed}:{nonce}".encode()).hexdigest()[:16]
        if all((int(candidate, 16) ^ int(value, 16)).bit_count() > 20 for value in signatures):
            return candidate
    raise AssertionError("could not find a distant pHash")


def _index(root: Path, name: str, row: dict) -> dict[str, object]:
    root.mkdir(parents=True)
    index = root / "hash-index.jsonl"
    normalized = {**row, "index_partition": name}
    if name == "benchmark":
        normalized.setdefault("phash64_flipped", normalized["phash"])
    index.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "partition": name,
        "index_filename": index.name,
        "index_rows": 1,
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
        "rows": 1,
    }


def _record(
    number: int,
    *,
    label: str = "fire",
    status: str = "original",
    preprocessing: object = None,
    polygons: object = None,
    annotation_extra: dict | None = None,
    event_id: str | None = None,
) -> dict:
    source_id = f"raw-{number}"
    source = {
        "id": source_id,
        "owner": "owner",
        "name": f"image-{number}.png",
        "split": "test" if number % 2 else "train",
        "width": 100,
        "height": 100,
        "status": status,
        "preprocessing": preprocessing,
    }
    if event_id:
        source["event_id"] = event_id
    annotation = {
        "boxes": [{"x": 50, "y": 50, "width": 40, "height": 75}],
        "polygons": polygons
        if polygons is not None
        else [[30, 10], [70, 10], [60, 60], [55, 85], [45, 85], [40, 60]],
        "label": label,
        "width": 100,
        "height": 100,
        **(annotation_extra or {}),
    }
    return {
        "page_url": f"https://universe.roboflow.com/project/dataset/images/{source_id}",
        "source": source,
        "annotation": annotation,
        "original_url": f"https://source.roboflow.com/owner/{source_id}/original.png",
    }


def _manifest(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit(
    tmp_path: Path,
    rows: list[dict],
    payloads: dict[str, bytes],
    *,
    index_rows: dict[str, dict] | None = None,
) -> tuple[dict, Path]:
    manifest = tmp_path / "roboflow.jsonl"
    manifest_sha = _manifest(manifest, rows)
    signatures = [_phash(payload) for payload in payloads.values()]
    contracts = {}
    for name in ("detection", "pointing", "benchmark"):
        row = (index_rows or {}).get(name) if index_rows else None
        contracts[name] = _index(
            tmp_path / name,
            name,
            row
            or {
                "sample_id": f"{name}:seed",
                "image_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "phash": _far_phash(signatures, name),
            },
        )
    output = tmp_path / "output"
    report = audit_roboflow_manifest(
        input_manifest=manifest,
        expected_manifest_sha256=manifest_sha,
        detection_index_root=tmp_path / "detection",
        pointing_index_root=tmp_path / "pointing",
        benchmark_index_root=tmp_path / "benchmark",
        expected_index_receipts=contracts,
        output_dir=output,
        source_id="roboflow-fire-smoke-v1",
        source_family="Roboflow Fire and Smoke Segmentation",
        source_revision="universe-v1-manifest-sha256:" + manifest_sha,
        source_license="CC BY 4.0",
        fetcher=lambda url: payloads[url],
    )
    return report, output


def test_native_polygons_are_downloaded_hashed_grouped_and_resplit(tmp_path: Path) -> None:
    rows = [
        _record(1, event_id="event-a"),
        _record(2, event_id="event-a"),
        _record(3),
        _record(4),
        _record(5),
    ]
    payloads = {row["original_url"]: _image(index) for index, row in enumerate(rows, 1)}
    report, output = _audit(tmp_path, rows, payloads)

    strict = [
        json.loads(line)
        for line in (output / "roboflow_strict_validated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_source = {row["source_record_id"]: row for row in strict}
    assert report["source_gate_passed"] is True
    assert report["strict_point_rows"] == 5
    assert report["points_by_kind"] == {"fire_base": 5}
    assert report["boxes_consumed"] == 0
    assert by_source["raw-1"]["split_group"] == by_source["raw-2"]["split_group"]
    assert by_source["raw-1"]["split"] == by_source["raw-2"]["split"]
    assert {row["split"] for row in strict} == {"train", "validation", "test"}
    assert all(row["upstream_split_ignored"] != "validation" for row in strict)
    assert all(Path(output / row["image_path"]).is_file() for row in strict)
    assert all(Path(output / row["mask_path"]).is_file() for row in strict)
    assert (output / "ROBoflow_NATIVE_POLYGON_AUDIT_RECEIPT.json").is_file()


def test_augmented_pseudo_and_polygonless_rows_fail_closed_without_download(
    tmp_path: Path,
) -> None:
    valid = [_record(number) for number in (1, 2, 3)]
    invalid = [
        _record(4, status="augmented"),
        _record(5, preprocessing={"resize": "640x640"}),
        _record(6, polygons=[], annotation_extra={"provenance": "bbox-derived SAM pseudo"}),
        _record(
            7,
            polygons=[
                {
                    "label": "smoke",
                    "points": [[30, 10], [70, 10], [55, 85], [45, 85]],
                }
            ],
        ),
    ]
    payloads = {row["original_url"]: _image(index) for index, row in enumerate(valid, 1)}
    report, output = _audit(tmp_path, valid + invalid, payloads)

    dispositions = [
        json.loads(line)
        for line in (output / "roboflow_automatic_dispositions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rejected = [row for row in dispositions if row["sample_id"] is None]
    assert report["source_gate_passed"] is True
    assert report["rejected_before_payload_admission"] == 4
    assert report["strict_point_rows"] == 3
    assert {row["exclusion_reasons"][0] for row in rejected} == {
        "source_view_not_original",
        "source_preprocessing_forbidden",
        "forbidden_annotation_provenance",
        "polygon_label_disagrees_with_annotation_label",
    }
    assert report["pseudo_labels_admitted"] == 0


def test_uncertain_polygon_abstains_and_external_overlap_is_excluded(tmp_path: Path) -> None:
    rows = [
        _record(1),
        _record(2),
        _record(3),
        _record(
            4,
            label="smoke",
            polygons=[[15, 10], [85, 10], [85, 85], [15, 85]],
        ),
    ]
    payloads = {row["original_url"]: _image(index) for index, row in enumerate(rows, 1)}
    overlap_payload = payloads[rows[0]["original_url"]]
    report, output = _audit(
        tmp_path,
        rows,
        payloads,
        index_rows={
            "detection": {
                "sample_id": "detection:overlap",
                "image_sha256": hashlib.sha256(overlap_payload).hexdigest(),
                "phash": _phash(overlap_payload),
            }
        },
    )

    abstentions = [
        json.loads(line)
        for line in (output / "roboflow_segmentation_abstentions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert report["source_gate_passed"] is True
    assert report["strict_point_rows"] == 2
    assert len(abstentions) == 1
    assert abstentions[0]["label"] == "smoke"
    assert abstentions[0]["anchor_points"] == []
    assert abstentions[0]["visual_abstention_reason"] == "physical_base_not_provable"
    assert report["exclusions_by_reason"]["detection_exact_sha_overlap"] == 1


def test_equal_phash_payload_variants_are_kept_in_one_split_group(tmp_path: Path) -> None:
    rows = [_record(number) for number in (1, 2, 3, 4)]
    first = _image(1)
    second = _same_pixels_different_payload(first)
    payloads = {
        rows[0]["original_url"]: first,
        rows[1]["original_url"]: second,
        rows[2]["original_url"]: _image(3),
        rows[3]["original_url"]: _image(4),
    }

    report, output = _audit(tmp_path, rows, payloads)
    strict = [
        json.loads(line)
        for line in (output / "roboflow_strict_validated_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_source = {row["source_record_id"]: row for row in strict}

    assert hashlib.sha256(first).hexdigest() != hashlib.sha256(second).hexdigest()
    assert _phash(first) == _phash(second)
    assert report["deduplication"]["exact_rows_excluded"] == 0
    assert by_source["raw-1"]["split_group"] == by_source["raw-2"]["split_group"]
    assert by_source["raw-1"]["split"] == by_source["raw-2"]["split"]
    assert report["split"]["split_group_leaks"] == []


def test_manifest_hash_and_output_immutability_are_enforced(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    row = _record(1)
    _manifest(manifest, [row])
    called = False

    def fetcher(_: str) -> bytes:
        nonlocal called
        called = True
        return _image(1)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_roboflow_manifest(
            input_manifest=manifest,
            expected_manifest_sha256="a" * 64,
            detection_index_root=tmp_path / "missing-detection",
            pointing_index_root=tmp_path / "missing-pointing",
            benchmark_index_root=tmp_path / "missing-benchmark",
            expected_index_receipts={
                name: {"receipt_sha256": "b" * 64, "rows": 1}
                for name in ("detection", "pointing", "benchmark")
            },
            output_dir=tmp_path / "output",
            source_id="source",
            source_family="family",
            source_revision="revision",
            source_license="CC BY 4.0",
            fetcher=fetcher,
        )
    assert called is False


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bucket": "bucket",
        "role": "role",
        "image": "image",
        "manifest_prefix": "pointing/manifests/roboflow-v1",
        "manifest_filename": "roboflow.jsonl",
        "manifest_sha256": "d" * 64,
        "detection_index_prefix": "pointing/exclusion-index/detection",
        "pointing_index_prefix": "pointing/exclusion-index/pointing",
        "benchmark_index_prefix": "pointing/exclusion-index/benchmark",
        "detection_index_receipt_sha256": "a" * 64,
        "pointing_index_receipt_sha256": "b" * 64,
        "benchmark_index_receipt_sha256": "c" * 64,
        "detection_index_rows": 102_257,
        "pointing_index_rows": 393,
        "benchmark_index_rows": 200,
        "source_id": "roboflow-v1",
        "source_family": "Roboflow native fire smoke polygons",
        "source_revision": "universe-v1",
        "source_license": "CC BY 4.0",
        "code_prefix": "pointing/code/roboflow-audit",
        "output_prefix": "pointing/reports/roboflow-audit",
        "instance_type": "ml.t3.xlarge",
        "volume_size": 30,
        "max_runtime": 7_200,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_launcher_is_hash_only_original_only_and_never_publishes() -> None:
    request = build_request(_args(), "job")

    inputs = {item["InputName"]: item for item in request["ProcessingInputs"]}
    assert set(inputs) == {
        "code",
        "roboflow-jsonl-manifest",
        "detection-exclusion-index",
        "pointing-exclusion-index",
        "benchmark-exclusion-index",
    }
    assert request["Environment"]["FIREVIEWER_PUBLICATION_ALLOWED"] == "false"
    tags = {item["Key"]: item["Value"] for item in request["Tags"]}
    assert tags["fireviewer:pseudo-labels"] == "false"
    assert tags["fireviewer:bbox-conversion"] == "false"
    assert tags["fireviewer:benchmark-access"] == "hash-only-exclusion"


def test_launcher_rejects_publication_and_media_index_paths() -> None:
    with pytest.raises(ValueError, match="publication"):
        build_request(_args(output_prefix="pointing/publish/huggingface"), "job")
    with pytest.raises(ValueError, match="must not reference media"):
        build_request(
            _args(benchmark_index_prefix="pointing/exclusion-index/benchmark/images"), "job"
        )
