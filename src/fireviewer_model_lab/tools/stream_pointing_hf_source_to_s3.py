"""Stream one immutable HF pointing source to S3 without local corpus storage."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Any

SOURCE_FIELDS = {
    "image_relpath": "image_sha256",
    "mask_relpath": "mask_sha256",
    "valid_mask_relpath": "valid_mask_sha256",
}


def _safe_path(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe HF source path: {value}")
    return path.as_posix()


def select_source_rows(manifest_text: str, source_id: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest_text.splitlines() if line.strip()]
    selected = [row for row in rows if str(row.get("source_id") or "") == source_id]
    if not selected:
        raise ValueError(f"HF manifest contains no rows for source {source_id!r}")
    sample_ids = [str(row.get("sample_id") or "") for row in selected]
    if len(sample_ids) != len(set(sample_ids)) or not all(sample_ids):
        raise ValueError("selected HF source has missing or duplicate sample ids")
    return selected


def stream_source(
    *,
    repository: str,
    revision: str,
    source_id: str,
    bucket: str,
    prefix: str,
    region: str,
    workers: int,
) -> dict[str, Any]:
    import boto3
    from botocore.config import Config
    from huggingface_hub import HfFileSystem

    if workers <= 0:
        raise ValueError("workers must be positive")
    lowered = prefix.lower()
    if any(value in lowered for value in ("fire-smoke-detection-corpus-v1", "fireviewer_bench")):
        raise ValueError("HF source export escaped the isolated pointing namespace")
    prefix = prefix.strip("/")
    hf_root = f"datasets/{repository}@{revision}/"
    filesystem = HfFileSystem()
    manifest_text = filesystem.open(hf_root + "manifest.jsonl", "r").read()
    rows = select_source_rows(manifest_text, source_id)
    session = boto3.Session(region_name=region)
    s3 = session.client(
        "s3",
        config=Config(
            retries={"total_max_attempts": 4, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=60,
            max_pool_connections=max(10, workers * 2),
        ),
    )
    payloads: dict[str, str] = {}
    for row in rows:
        for path_field, sha_field in SOURCE_FIELDS.items():
            if not row.get(path_field):
                if path_field == "valid_mask_relpath":
                    continue
                raise ValueError(f"missing {path_field} for {row['sample_id']}")
            path = _safe_path(str(row[path_field]))
            expected = str(row.get(sha_field) or "")
            if len(expected) != 64:
                raise ValueError(f"invalid {sha_field} for {row['sample_id']}")
            prior = payloads.setdefault(path, expected)
            if prior != expected:
                raise ValueError(f"conflicting hashes for HF payload {path}")

    def transfer(item: tuple[str, str]) -> tuple[str, int]:
        relative_path, expected_sha = item
        with filesystem.open(hf_root + relative_path, "rb") as handle:
            payload = handle.read()
        observed_sha = hashlib.sha256(payload).hexdigest()
        if observed_sha != expected_sha:
            raise ValueError(f"HF payload SHA-256 mismatch: {relative_path}")
        key = f"{prefix}/source/{relative_path}"
        s3.upload_fileobj(
            io.BytesIO(payload),
            bucket,
            key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        return relative_path, len(payload)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        transferred = list(executor.map(transfer, sorted(payloads.items())))

    exported_rows: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        for path_field in SOURCE_FIELDS:
            if not row.get(path_field):
                continue
            original = _safe_path(str(row[path_field]))
            row[f"source_{path_field}"] = original
            row[path_field] = f"source/{original}"
            uri_field = path_field.removesuffix("_relpath") + "_s3_uri"
            row[uri_field] = f"s3://{bucket}/{prefix}/source/{original}"
        row.update(
            {
                "source_repository": repository,
                "source_repository_revision": revision,
                "corpus_disposition": "pending_strict_automated_validation",
                "training_eligible": False,
                "reviews_admitted": False,
            }
        )
        exported_rows.append(row)
    manifest_body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in exported_rows
    ).encode()
    s3.upload_fileobj(
        io.BytesIO(manifest_body),
        bucket,
        f"{prefix}/candidate_manifest.jsonl",
        ExtraArgs={
            "ContentType": "application/x-ndjson",
            "ServerSideEncryption": "AES256",
        },
    )
    report = {
        "schema_version": 1,
        "source_id": source_id,
        "source_repository": repository,
        "source_revision": revision,
        "rows": len(exported_rows),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "split_group_counts": dict(
            sorted(Counter(str(row["split_group"]) for row in rows).items())
        ),
        "payload_files": len(transferred),
        "payload_bytes": sum(size for _path, size in transferred),
        "candidate_manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
        "local_corpus_materialized": False,
        "reviews_admitted": False,
        "admission_status": "pending_strict_automated_validation",
    }
    report_body = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    s3.upload_fileobj(
        io.BytesIO(report_body),
        bucket,
        f"{prefix}/stream_report.json",
        ExtraArgs={
            "ContentType": "application/json",
            "ServerSideEncryption": "AES256",
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--region", default="eu-west-2")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = stream_source(
        repository=args.repository,
        revision=args.revision,
        source_id=args.source_id,
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        workers=args.workers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
