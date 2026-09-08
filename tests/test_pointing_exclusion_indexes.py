from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError

import imagehash
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from fireviewer_model_lab.tools.launch_sagemaker_pointing_exclusion_indexes import (
    DEFAULT_BUCKET,
    INSTANCE_TYPE,
    build_request,
)
from fireviewer_model_lab.training import pointing_exclusion_indexes as exclusion


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jsonl_bytes(rows))


def _png(seed: int) -> bytes:
    image = Image.new("RGB", (16, 16), (seed * 19 % 255, seed * 37 % 255, seed * 71 % 255))
    for coordinate in range(seed % 8 + 1):
        image.putpixel((coordinate, (coordinate * 3 + seed) % 16), (255, 255, 255))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _parquet_bytes(
    *, sample_id: str, digest: str, phash: str, image_payload: bytes | None = None
) -> bytes:
    sink = io.BytesIO()
    pq.write_table(
        pa.table(
            {
                "image": [image_payload or b"embedded-image-column-must-not-be-projected"],
                "sample_id": [sample_id],
                "sha256": [digest],
                "phash": [phash],
                "unrelated_metadata": ["ignored"],
            }
        ),
        sink,
    )
    return sink.getvalue()


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes], versions: dict[str, str]) -> None:
        self.objects = objects
        self.versions = versions
        self.head_counts: Counter[str] = Counter()
        self.change_on_second_head: set[str] = set()

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        assert Bucket == exclusion.PUBLICATION_STAGING_BUCKET
        self.head_counts[Key] += 1
        version = self.versions[Key]
        if Key in self.change_on_second_head and self.head_counts[Key] >= 2:
            version = f"changed-{version}"
        return {
            "VersionId": version,
            "ContentLength": len(self.objects[Key]),
            "ServerSideEncryption": "aws:kms",
        }


class FakeParquetFilesystem:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.opened: list[str] = []

    def open_input_file(self, path: str) -> pa.BufferReader:
        bucket, key = path.split("/", 1)
        assert bucket == exclusion.PUBLICATION_STAGING_BUCKET
        self.opened.append(path)
        return pa.BufferReader(self.objects[key])


def _make_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invalid_detection_row: bool = False,
    incomplete_benchmark_row: bool = False,
    parquet_sha_divergence: bool = False,
    invalid_parquet_phash: bool = False,
    manifest_phash_divergence: bool = False,
) -> dict:
    revision = "a" * 40
    repository = "fireviewer/synthetic-detection"
    validation_profile = "detection-strict-v1"
    validation_run_id = "20260904T173254Z"
    manifest_paths = {
        "alarmod": "manifests/alarmod/manifest.jsonl",
        "boreal": "manifests/boreal/manifest.jsonl",
        "fasdd": "manifests/fasdd/manifest.jsonl",
        "pyro-sdis": "manifests/pyro-sdis/manifest.jsonl",
    }
    network_payloads: dict[str, bytes] = {}
    detection_receipts: list[dict] = []
    detection_metadata: list[dict] = []
    splits = ("train", "validation", "test", "train")
    for index, (name, relative) in enumerate(sorted(manifest_paths.items())):
        sample_id = f"detection-{name}"
        digest = hashlib.sha256(sample_id.encode()).hexdigest()
        phash = f"{index + 1:016x}"
        image_relative = f"images/{name}.png"
        if invalid_parquet_phash and index == 0:
            image_payload = _png(77)
            digest = _sha_bytes(image_payload)
        row = {
            "sample_id": sample_id,
            "sha256": digest,
            "image_relpath": image_relative,
            "sample_validation_status": "strict_automated_validated",
            "validation_profile": validation_profile,
            "split": splits[index],
            "split_group": f"incident-{index}",
            "license": "CC-BY-4.0",
            "consent_basis": {"basis": "synthetic-test"},
        }
        if index:
            row["phash64"] = phash
        if manifest_phash_divergence and index == 1:
            row["phash64"] = "f" * 16
        if invalid_detection_row and index == 1:
            row.pop("sample_id")
        manifest_payload = _jsonl_bytes([row])
        manifest_url = exclusion._hf_url(repository, revision, relative)
        network_payloads[manifest_url] = manifest_payload
        detection_receipts.append(
            {
                "name": name,
                "repository_path": relative,
                "revision": revision,
                "bytes": len(manifest_payload),
                "sha256": _sha_bytes(manifest_payload),
            }
        )
        parquet_digest = digest
        if parquet_sha_divergence and index == 0:
            parquet_digest = hashlib.sha256(b"different-identity").hexdigest()
        parquet_phash = "invalid" if invalid_parquet_phash and index == 0 else phash
        detection_metadata.append(
            {
                "sample_id": sample_id,
                "sha256": parquet_digest,
                "phash": parquet_phash,
                "split": splits[index],
            }
        )

    source_contracts = {
        "boreal": {
            "source_id": "boreal-source",
            "source_revision": "boreal-revision",
            "profile": "pointing-strict-v1",
            "manifest": "boreal_strict_validated_manifest.jsonl",
            "report": "boreal_audit_summary.json",
        },
        "camp-swift": {
            "source_id": "camp-source",
            "source_revision": "camp-revision",
            "profile": "pointing-strict-v1",
            "manifest": "camp_swift_strict_validated_manifest.jsonl",
            "report": "camp_swift_audit_summary.json",
        },
        "kit": {
            "source_id": "kit-source",
            "source_revision": "kit-revision",
            "profile": "pointing-strict-v2",
            "manifest": "kit_strict_validated_manifest.jsonl",
            "report": "kit_strict_audit_summary.json",
        },
    }
    pointing_roots: dict[str, Path] = {}
    overlays: list[dict] = []
    overlay_receipts: list[dict] = []
    for index, (name, source) in enumerate(source_contracts.items(), 20):
        root = tmp_path / "pointing" / name
        image_relative = "payload/images/sample.png"
        image_payload = _png(index)
        image_path = root / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_payload)
        manifest_path = root / source["manifest"]
        _write_jsonl(
            manifest_path,
            [
                {
                    "sample_id": f"pointing-{name}",
                    "image_sha256": _sha_bytes(image_payload),
                    "image_relpath": image_relative,
                    "sample_validation_status": "strict_automated_validated",
                    "strict_keep": True,
                    "training_eligible": True,
                    "reviews_admitted": False,
                    "source_id": source["source_id"],
                    "source_revision": source["source_revision"],
                    "validation_profile": source["profile"],
                }
            ],
        )
        report_path = root / source["report"]
        _write_json(
            report_path,
            {
                "strict_automated_validated_rows": 1,
                "source_gate_passed": True,
                "gate_errors": [],
                "reviews_admitted": False,
                "publication_allowed": False,
                "decode_or_payload_errors": [],
                "split_group_leakage": [],
            },
        )
        manifest_sha = exclusion._sha256(manifest_path)
        report_sha = exclusion._sha256(report_path)
        overlays.append(
            {
                "name": name,
                "source_id": source["source_id"],
                "source_revision": source["source_revision"],
                "manifest_filename": source["manifest"],
                "report_filename": source["report"],
                "strict_rows_min": 1,
                "strict_rows_max": 1,
                "manifest_sha256": manifest_sha,
                "report_sha256": report_sha,
                "allowed_validation_profiles": [source["profile"]],
            }
        )
        overlay_receipts.append(
            {
                "name": name,
                "rows": 1,
                "manifest_sha256": manifest_sha,
                "report_sha256": report_sha,
                "source_gate_passed": True,
            }
        )
        pointing_roots[name] = root

    split_counts = dict(Counter(splits))
    registry = {
        "schema_version": 1,
        "campaign_id": "synthetic-composition-v1",
        "detection_base": {
            "repository": repository,
            "revision": revision,
            "validation_profile": validation_profile,
            "validation_run_id": validation_run_id,
            "rows": 4,
            "split_counts": split_counts,
            "manifest_paths": manifest_paths,
        },
        "overlay_sources": overlays,
    }
    registry_path = tmp_path / "code" / "dinov3-multitask-composition-v1.json"
    _write_json(registry_path, registry)
    registry_sha = exclusion._sha256(registry_path)
    composition_root = tmp_path / "composition"
    integrity = {
        "schema_version": 1,
        "campaign_id": registry["campaign_id"],
        "composition_registry_sha256": registry_sha,
        "detection_revision": revision,
        "composition_rows": 7,
        "detection_manifest_sha256": {
            item["name"]: item["sha256"] for item in detection_receipts
        },
        "overlay_manifest_sha256": {
            item["name"]: item["manifest_sha256"] for item in overlay_receipts
        },
        "integrity_gates_passed": True,
        "publication_allowed": False,
    }
    integrity_path = composition_root / "composition-integrity-receipt.json"
    _write_json(integrity_path, integrity)
    report = {
        "schema_version": 1,
        "campaign_id": registry["campaign_id"],
        "composition_registry_sha256": registry_sha,
        "detection_repository": repository,
        "detection_revision": revision,
        "detection_rows": 4,
        "detection_split_counts": split_counts,
        "composition_rows": 7,
        "detection_manifest_receipts": detection_receipts,
        "overlay_receipts": overlay_receipts,
        "composition_integrity_receipt": integrity_path.name,
        "composition_integrity_receipt_sha256": exclusion._sha256(integrity_path),
        "integrity_gates_passed": True,
        "reviews_admitted": False,
        "publication_allowed": False,
    }
    report_path = composition_root / "composition_report.json"
    _write_json(report_path, report)

    objects: dict[str, bytes] = {}
    versions: dict[str, str] = {}
    shards: list[dict] = []
    split_shards: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    for metadata in detection_metadata:
        split = metadata["split"]
        index = split_shards[split]
        split_shards[split] += 1
        split_rows[split] += 1
        path = f"data/{split}/{split}-{index:05d}.parquet"
        key = (
            f"synthetic-detection/staging/strict-clean/runs/"
            f"{validation_run_id}/{path}"
        )
        payload = _parquet_bytes(
            sample_id=metadata["sample_id"],
            digest=metadata["sha256"],
            phash=metadata["phash"],
            image_payload=(
                _png(77)
                if invalid_parquet_phash and metadata["sample_id"] == "detection-alarmod"
                else None
            ),
        )
        version_id = f"version-{split}-{index}"
        objects[key] = payload
        versions[key] = version_id
        shards.append(
            {
                "index": index,
                "parquet_bytes": len(payload),
                "path": path,
                "rows": 1,
                "sha256": _sha_bytes(payload),
                "split": split,
                "staged_s3_key": key,
                "staged_s3_version_id": version_id,
            }
        )
    publication = {
        "archives_included": False,
        "format": exclusion.PUBLICATION_FORMAT,
        "release_kind": exclusion.PUBLICATION_RELEASE_KIND,
        "repo_id": repository,
        "schema_version": 2,
        "metadata": [
            {
                "path": receipt["repository_path"],
                "bytes": receipt["bytes"],
                "sha256": receipt["sha256"],
            }
            for receipt in detection_receipts
        ],
        "shards": shards,
        "splits": {
            split: {"rows": split_rows[split], "shards": split_shards[split]}
            for split in sorted(exclusion.VALID_SPLITS)
        },
        "validation_profile": validation_profile,
        "validation_run_id": validation_run_id,
    }
    publication_manifest_path = tmp_path / exclusion.PUBLICATION_MANIFEST_FILENAME
    _write_json(publication_manifest_path, publication)
    s3_client = FakeS3Client(objects, versions)
    parquet_filesystem = FakeParquetFilesystem(objects)

    benchmark_rows = [
        {
            "sample_id": f"benchmark-{index}",
            "corpus_id": exclusion.BENCHMARK_CORPUS_ID,
            "sha256": hashlib.sha256(f"benchmark-{index}".encode()).hexdigest(),
            "phash64": f"{100 + index:016x}",
            "phash64_flipped": f"{200 + index:016x}",
        }
        for index in range(2)
    ]
    if incomplete_benchmark_row:
        benchmark_rows[0].pop("phash64_flipped")
    benchmark_root = tmp_path / "benchmark"
    guard_path = benchmark_root / exclusion.BENCHMARK_GUARD_FILENAME
    _write_jsonl(guard_path, benchmark_rows)

    def fake_download_to_path(url: str, destination: Path, maximum_bytes: int) -> dict:
        payload = network_payloads[url]
        assert len(payload) <= maximum_bytes
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return {"bytes": len(payload), "sha256": _sha_bytes(payload), "url": url}

    projected_columns: list[tuple[str, ...]] = []
    real_parquet_file = exclusion.pq.ParquetFile

    class RecordingParquetFile:
        def __init__(self, source: object) -> None:
            self.inner = real_parquet_file(source)

        @property
        def schema_arrow(self) -> pa.Schema:
            return self.inner.schema_arrow

        @property
        def metadata(self) -> object:
            return self.inner.metadata

        def read_row_group(self, row_group: int, **kwargs: object) -> object:
            projected_columns.append(tuple(kwargs["columns"]))
            return self.inner.read_row_group(row_group, **kwargs)

    monkeypatch.setattr(exclusion, "_http_download_to_path", fake_download_to_path)
    monkeypatch.setattr(exclusion.pq, "ParquetFile", RecordingParquetFile)
    return {
        "registry_path": registry_path,
        "composition_root": composition_root,
        "composition_report_sha256": exclusion._sha256(report_path),
        "publication_manifest_path": publication_manifest_path,
        "publication_manifest_sha256": exclusion._sha256(publication_manifest_path),
        "pointing_roots": pointing_roots,
        "benchmark_root": benchmark_root,
        "benchmark_sha256": exclusion._sha256(guard_path),
        "work_dir": tmp_path / "work",
        "output_dir": tmp_path / "output",
        "network_payloads": network_payloads,
        "s3_client": s3_client,
        "parquet_filesystem": parquet_filesystem,
        "projected_columns": projected_columns,
        "shard_keys": [item["staged_s3_key"] for item in shards],
        "first_metadata_phash": detection_metadata[0]["phash"],
    }


def _build(case: dict, **overrides: object) -> dict:
    arguments: dict[str, object] = {
        "registry_path": case["registry_path"],
        "composition_root": case["composition_root"],
        "expected_composition_report_sha256": case["composition_report_sha256"],
        "publication_manifest_path": case["publication_manifest_path"],
        "pointing_roots": case["pointing_roots"],
        "benchmark_guard_root": case["benchmark_root"],
        "work_dir": case["work_dir"],
        "output_dir": case["output_dir"],
        "s3_client": case["s3_client"],
        "parquet_filesystem": case["parquet_filesystem"],
        "expected_publication_manifest_sha256": case["publication_manifest_sha256"],
        "expected_publication_rows": 4,
        "expected_publication_shards": 4,
        "expected_benchmark_rows": 2,
        "expected_benchmark_sha256": case["benchmark_sha256"],
    }
    arguments.update(overrides)
    return exclusion.build_exclusion_indexes(**arguments)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_builds_three_indexes_from_projected_versioned_parquet_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch)

    result = _build(case)

    assert set(result["partitions"]) == {"detection", "pointing", "benchmark"}
    assert result["total_index_rows"] == 9
    detection_rows = _read_jsonl(case["output_dir"] / "detection" / "hash-index.jsonl")
    projected_row = next(
        row for row in detection_rows if row["phash_origin"].startswith("versioned_s3")
    )
    assert projected_row["phash"] == case["first_metadata_phash"]
    assert set(projected_row) >= {"sample_id", "image_sha256", "phash"}
    assert case["projected_columns"] == [exclusion.PARQUET_PROJECTED_COLUMNS] * 4
    assert len(case["parquet_filesystem"].opened) == 4
    assert case["s3_client"].head_counts == Counter(
        {key: 2 for key in case["shard_keys"]}
    )
    detection_receipt = json.loads(
        (case["output_dir"] / "detection" / "receipt.json").read_text()
    )
    assert detection_receipt["http_media_download_rows"] == 0
    assert detection_receipt["publication_metadata_receipt"]["image_column_read"] is False
    assert all("/manifests/" in url for url in case["network_payloads"])
    assert not hasattr(exclusion, "_http_download_bytes")
    assert not any(
        path.suffix.casefold() in exclusion.IMAGE_SUFFIXES
        for path in case["output_dir"].rglob("*")
        if path.is_file()
    )
    assert result["publication_allowed"] is False


def test_recovers_missing_parquet_phash_from_versioned_parquet_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch, invalid_parquet_phash=True)

    _build(case)

    receipt = json.loads(
        (case["output_dir"] / "detection" / "receipt.json").read_text()
    )
    rows = _read_jsonl(case["output_dir"] / "detection" / "hash-index.jsonl")
    recovered = next(row for row in rows if row["sample_id"] == "detection-alarmod")
    assert len(recovered["phash"]) == 16
    assert receipt["http_media_download_rows"] == 0
    assert receipt["publication_metadata_receipt"]["missing_phash_rows"] == 1
    assert receipt["publication_metadata_receipt"]["recovered_phash_rows"] == 1
    assert receipt["publication_metadata_receipt"]["fallback_image_row_groups"] == 1
    assert receipt["publication_metadata_receipt"]["image_column_read"] is True


@pytest.mark.parametrize("seed", [1, 7, 31, 93])
def test_internal_pointing_phash_is_compatible_with_imagehash(seed: int) -> None:
    with Image.open(io.BytesIO(_png(seed))) as image:
        assert exclusion._phash64(image) == str(imagehash.phash(image, hash_size=8))


@pytest.mark.parametrize(
    "case_option,error_match",
    [
        ({"parquet_sha_divergence": True}, "identity diverged"),
        ({"manifest_phash_divergence": True}, "pHash diverged"),
        ({"invalid_detection_row": True}, "invalid strict detection row"),
        ({"incomplete_benchmark_row": True}, "invalid or duplicate benchmark guard row"),
    ],
)
def test_incomplete_or_divergent_metadata_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case_option: dict,
    error_match: str,
) -> None:
    case = _make_case(tmp_path, monkeypatch, **case_option)

    with pytest.raises(ValueError, match=error_match):
        _build(case)

    assert not list(case["output_dir"].rglob("hash-index.jsonl"))
    receipt = json.loads(
        (case["output_dir"] / "exclusion_indexes_failure_receipt.json").read_text()
    )
    assert receipt["source_gate_passed"] is False
    assert receipt["indexes_emitted"] == 0


def test_s3_current_version_change_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch)
    case["s3_client"].change_on_second_head.add(case["shard_keys"][0])

    with pytest.raises(ValueError, match="current S3 object does not match publication pin"):
        _build(case)

    assert not list(case["output_dir"].rglob("hash-index.jsonl"))


def test_s3_current_content_length_must_match_publication_pin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch)
    key = case["shard_keys"][0]
    case["s3_client"].objects[key] += b"changed"

    with pytest.raises(ValueError, match="current S3 object does not match publication pin"):
        _build(case)

    assert not list(case["output_dir"].rglob("hash-index.jsonl"))


def test_publication_manifest_hash_is_exact_and_cli_pin_cannot_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="publication-manifest SHA-256 mismatch"):
        _build(case, expected_publication_manifest_sha256="0" * 64)

    destinations = {action.dest for action in exclusion.build_parser()._actions}
    assert "expected_publication_manifest_sha256" not in destinations
    assert "expected_publication_rows" not in destinations
    assert "expected_publication_shards" not in destinations
    assert "publication_manifest" in destinations
    assert exclusion.PUBLICATION_MANIFEST_SHA256 == (
        "155fd807ed14934fa5d22a7c5c081cde912f4bd79aa0764af784753916cfc155"
    )
    assert exclusion.PUBLICATION_ROWS == 102_257
    assert exclusion.PUBLICATION_SHARDS == 32


def test_downloaded_manifest_receipt_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch)
    manifest_url = next(iter(case["network_payloads"]))
    case["network_payloads"][manifest_url] += b"\n"

    with pytest.raises(ValueError, match="manifest receipt mismatch"):
        _build(case)

    assert not list(case["output_dir"].rglob("hash-index.jsonl"))


def test_mounted_pointing_image_is_rehashed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch)
    image_path = case["pointing_roots"]["kit"] / "payload/images/sample.png"
    image_path.write_bytes(_png(111))

    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        _build(case)

    assert not list(case["output_dir"].rglob("hash-index.jsonl"))


def test_composition_report_hash_is_mandatory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _make_case(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="composition_report SHA-256 mismatch"):
        _build(case, expected_composition_report_sha256="0" * 64)


def test_production_benchmark_pin_and_exact_row_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    assert exclusion.BENCHMARK_ROWS == 200
    assert exclusion.BENCHMARK_GUARD_SHA256 == (
        "685c8a345d8df8a9534676b492496c79a8d96483b06b233e9517fa299cdbbccc"
    )
    case = _make_case(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="benchmark guard row-count mismatch"):
        _build(case, expected_benchmark_rows=3)


def test_http_manifest_retry_honours_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"bounded-manifest"
    requests: list[object] = []
    sleeps: list[float] = []

    class Response:
        def __init__(self) -> None:
            self.sent = False

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://huggingface.co/manifest.jsonl"

        def read(self, _size: int) -> bytes:
            if self.sent:
                return b""
            self.sent = True
            return payload

    def fake_urlopen(request: object, timeout: int) -> Response:
        assert timeout == 180
        requests.append(request)
        if len(requests) == 1:
            raise HTTPError(
                "https://huggingface.co/manifest.jsonl",
                429,
                "rate limited",
                {"Retry-After": "3"},
                None,
            )
        return Response()

    monkeypatch.setattr(exclusion, "_pace_http_request", lambda: None)
    monkeypatch.setattr(exclusion.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(exclusion.time, "sleep", sleeps.append)
    destination = tmp_path / "manifest.jsonl"

    receipt = exclusion._http_download_to_path(
        "https://huggingface.co/manifest.jsonl",
        destination,
        1024,
    )

    assert destination.read_bytes() == payload
    assert receipt["sha256"] == _sha_bytes(payload)
    assert sleeps == [3.0]
    assert requests[0].get_header("User-agent") == (
        "FireViewer-Pointing-Exclusion-Index/1.0"
    )


def _launcher_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bucket": DEFAULT_BUCKET,
        "role": "role",
        "image": "image",
        "code_prefix": "pointing/code/exclusion-indexes",
        "boreal_prefix": "pointing/runs/boreal",
        "camp_swift_prefix": "pointing/runs/camp-swift",
        "kit_prefix": "pointing/runs/kit",
        "benchmark_guard_prefix": "pointing/benchmark/external-guard",
        "composition_receipt_prefix": "pointing/composition/receipt",
        "composition_report_sha256": "b" * 64,
        "output_prefix": "pointing/exclusion-indexes/output",
        "volume_size": 40,
        "max_runtime": 21_600,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_launcher_passes_publication_manifest_from_code_root_without_new_input() -> None:
    request = build_request(_launcher_args(), "fireviewer-pointing-exclusions-test")

    inputs = {item["InputName"] for item in request["ProcessingInputs"]}
    assert inputs == {
        "code",
        "boreal",
        "camp-swift",
        "kit",
        "benchmark-guard",
        "composition-receipt",
    }
    cluster = request["ProcessingResources"]["ClusterConfig"]
    assert cluster == {"InstanceCount": 1, "InstanceType": INSTANCE_TYPE, "VolumeSizeInGB": 40}
    arguments = request["AppSpecification"]["ContainerArguments"]
    publication_index = arguments.index("--publication-manifest")
    assert arguments[publication_index + 1] == (
        "/opt/ml/processing/input/code/publication-manifest.json"
    )
    environment = request["Environment"]
    assert environment["FIREVIEWER_PARQUET_PROJECTED_COLUMNS"] == "sample_id,sha256,phash"
    assert environment["FIREVIEWER_PUBLICATION_STAGING_BUCKET"] == DEFAULT_BUCKET
    tags = {item["Key"]: item["Value"] for item in request["Tags"]}
    assert tags["fireviewer:publication-allowed"] == "false"
    assert tags["fireviewer:submitted"] == "false"


@pytest.mark.parametrize(
    "override",
    [
        {"output_prefix": "pointing/code/exclusion-indexes/output"},
        {"output_prefix": "pointing/exclusion-publish"},
        {"benchmark_guard_prefix": "pointing/guards/external"},
        {"bucket": "wrong-staging-bucket"},
    ],
)
def test_launcher_rejects_non_isolated_or_unpinned_paths(override: dict) -> None:
    with pytest.raises(ValueError):
        build_request(_launcher_args(**override), "fireviewer-pointing-exclusions-test")
