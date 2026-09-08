"""Build, but never submit, an isolated SageMaker IJmond audit request."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
DEFAULT_CODE_PREFIX = "pointing-corpus-v2/code/ijmond-audit"
DEFAULT_OUTPUT_PREFIX = "pointing-corpus-v2/reports/ijmond-audit"
_INDEX_SIGNAL = re.compile(r"(?:^|[-_/])(dedup|exclusion|hash|index)(?:[-_/]|$)", re.IGNORECASE)


def _safe_prefix(value: str, purpose: str) -> str:
    if not value or value.startswith(("/", "s3://")) or "\\" in value:
        raise ValueError(f"{purpose} must be a relative S3 key prefix")
    normalized = PurePosixPath(value)
    if ".." in normalized.parts or normalized.as_posix() in {"", "."}:
        raise ValueError(f"unsafe {purpose} prefix")
    return normalized.as_posix().rstrip("/")


def _hash_index_prefix(value: str, purpose: str) -> str:
    normalized = _safe_prefix(value, purpose)
    if _INDEX_SIGNAL.search(normalized) is None:
        raise ValueError(f"{purpose} must identify a hash-only exclusion index")
    if any(token in normalized.casefold() for token in ("strict-payload", "/images", "/masks")):
        raise ValueError(f"{purpose} must not reference media payloads")
    return normalized


def _input(name: str, uri: str, local_path: str) -> dict:
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": uri,
            "LocalPath": local_path,
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
            "S3CompressionType": "None",
        },
    }


def _require_disjoint_prefixes(prefixes: dict[str, str]) -> None:
    ordered = sorted(prefixes.items())
    for index, (left_name, left) in enumerate(ordered):
        for right_name, right in ordered[index + 1 :]:
            if left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/"):
                raise ValueError(
                    f"S3 prefixes must be disjoint: {left_name}={left} overlaps "
                    f"{right_name}={right}"
                )


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    code_prefix = _safe_prefix(args.code_prefix, "code")
    output_prefix = _safe_prefix(args.output_prefix, "output")
    detection_prefix = _hash_index_prefix(args.detection_index_prefix, "detection index")
    pointing_prefix = _hash_index_prefix(args.pointing_index_prefix, "pointing index")
    benchmark_prefix = _hash_index_prefix(args.benchmark_index_prefix, "benchmark index")
    if any(
        marker in value.casefold()
        for value in (code_prefix, output_prefix)
        for marker in ("fire-smoke-detection-corpus-v1", "benchdata", "fireviewer_bench")
    ):
        raise ValueError("IJmond code/output escaped the isolated pointing campaign")
    publication_markers = ("publish", "huggingface", "hf-upload")
    if any(marker in output_prefix.casefold() for marker in publication_markers):
        raise ValueError("IJmond audit output cannot be a publication target")
    _require_disjoint_prefixes(
        {
            "code": code_prefix,
            "output": output_prefix,
            "detection-index": detection_prefix,
            "pointing-index": pointing_prefix,
            "benchmark-index": benchmark_prefix,
        }
    )
    receipt_contracts = {
        "detection": (
            str(args.detection_index_receipt_sha256).casefold(),
            int(args.detection_index_rows),
        ),
        "pointing": (
            str(args.pointing_index_receipt_sha256).casefold(),
            int(args.pointing_index_rows),
        ),
        "benchmark": (
            str(args.benchmark_index_receipt_sha256).casefold(),
            int(args.benchmark_index_rows),
        ),
    }
    for name, (receipt_sha256, rows) in receipt_contracts.items():
        if not re.fullmatch(r"[0-9a-f]{64}", receipt_sha256):
            raise ValueError(f"{name} exclusion receipt SHA-256 is invalid")
        if rows <= 0:
            raise ValueError(f"{name} exclusion index row count must be positive")

    output_s3 = f"s3://{args.bucket}/{output_prefix}/{job_name}"
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                "/opt/ml/processing/input/code/pointing_ijmond_audit.py",
                "--detection-index-root",
                "/opt/ml/processing/input/exclusions/detection",
                "--pointing-index-root",
                "/opt/ml/processing/input/exclusions/pointing",
                "--benchmark-index-root",
                "/opt/ml/processing/input/exclusions/benchmark",
                "--detection-index-receipt-sha256",
                receipt_contracts["detection"][0],
                "--pointing-index-receipt-sha256",
                receipt_contracts["pointing"][0],
                "--benchmark-index-receipt-sha256",
                receipt_contracts["benchmark"][0],
                "--detection-index-rows",
                str(receipt_contracts["detection"][1]),
                "--pointing-index-rows",
                str(receipt_contracts["pointing"][1]),
                "--benchmark-index-rows",
                str(receipt_contracts["benchmark"][1]),
                "--output-dir",
                "/opt/ml/processing/output",
                "--work-dir",
                "/opt/ml/processing/work/ijmond",
                "--output-s3-prefix",
                output_s3,
            ],
        },
        "ProcessingInputs": [
            _input(
                "code",
                f"s3://{args.bucket}/{code_prefix}",
                "/opt/ml/processing/input/code",
            ),
            _input(
                "detection-exclusion-index",
                f"s3://{args.bucket}/{detection_prefix}",
                "/opt/ml/processing/input/exclusions/detection",
            ),
            _input(
                "pointing-exclusion-index",
                f"s3://{args.bucket}/{pointing_prefix}",
                "/opt/ml/processing/input/exclusions/pointing",
            ),
            _input(
                "benchmark-exclusion-index",
                f"s3://{args.bucket}/{benchmark_prefix}",
                "/opt/ml/processing/input/exclusions/benchmark",
            ),
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "ijmond-strict-audit-reports",
                    "S3Output": {
                        "S3Uri": output_s3,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": args.volume_size,
            }
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime},
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "FIREVIEWER_IJMOND_SOURCE_REVISION": "10.21942/uva.31847188.v3",
            "FIREVIEWER_IJMOND_ARCHIVE_MD5": "20eb0e868e1fb8575612b9a4b77367e1",
            "FIREVIEWER_IJMOND_ARCHIVE_SHA256": (
                "5a26dbad99b5e590608cdbb98f269f4a24266c020852401090f040f75bcb6343"
            ),
            "FIREVIEWER_PUBLICATION_ALLOWED": "false",
        },
        "Tags": [
            {"Key": "fireviewer:corpus", "Value": "pointing-v2"},
            {"Key": "fireviewer:stage", "Value": "ijmond-strict-isolated-audit"},
            {"Key": "fireviewer:source-revision", "Value": "figshare-v3"},
            {"Key": "fireviewer:reviews-admitted", "Value": "false"},
            {"Key": "fireviewer:publication-allowed", "Value": "false"},
            {"Key": "fireviewer:benchmark-access", "Value": "hash-only-exclusion"},
            {"Key": "fireviewer:rgb-only", "Value": "true"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--detection-index-prefix", required=True)
    parser.add_argument("--pointing-index-prefix", required=True)
    parser.add_argument("--benchmark-index-prefix", required=True)
    parser.add_argument("--detection-index-receipt-sha256", required=True)
    parser.add_argument("--pointing-index-receipt-sha256", required=True)
    parser.add_argument("--benchmark-index-receipt-sha256", required=True)
    parser.add_argument("--detection-index-rows", type=int, required=True)
    parser.add_argument("--pointing-index-rows", type=int, required=True)
    parser.add_argument("--benchmark-index-rows", type=int, required=True)
    parser.add_argument("--code-prefix", default=DEFAULT_CODE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--instance-type", default="ml.m5.2xlarge")
    parser.add_argument("--volume-size", type=int, default=30)
    parser.add_argument("--max-runtime", type=int, default=7_200)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()  # noqa: UP017
    job_name = args.job_name or f"fireviewer-pointing-ijmond-audit-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "job_name": job_name,
                "request": str(args.emit_request),
                "submitted": False,
                "publication_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
