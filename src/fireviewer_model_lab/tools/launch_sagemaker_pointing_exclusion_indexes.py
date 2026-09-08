"""Build, but never submit, the SageMaker pointing exclusion-index request."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = (
    "764974769150.dkr.ecr.eu-west-2.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-cpu-py3"
)
# This account currently has a non-zero Processing quota only for the T3 family.
# Four vCPUs are sufficient for the projected three-column Parquet metadata scan.
INSTANCE_TYPE = "ml.t3.xlarge"
DEFAULT_COMPOSITION_REPORT_SHA256 = (
    "590953e21f422d5ce5bb2dd25af60596de5cc7076e501128ef72756c26dfc0f0"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUCKET = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
UTC_COMPAT = timezone.utc  # noqa: UP017 - SageMaker image currently uses Python 3.10


def _safe_prefix(value: str, purpose: str) -> str:
    if not value or value.startswith(("/", "s3://")) or "\\" in value:
        raise ValueError(f"{purpose} must be a relative S3 key prefix")
    normalized = PurePosixPath(value)
    if ".." in normalized.parts or normalized.as_posix() in {"", "."}:
        raise ValueError(f"unsafe {purpose} prefix")
    return normalized.as_posix().rstrip("/")


def _prefixes_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


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


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    if not _BUCKET.fullmatch(args.bucket):
        raise ValueError("invalid S3 bucket name")
    if args.bucket != DEFAULT_BUCKET:
        raise ValueError("exclusion metadata projection is pinned to the staging bucket")
    if not _SHA256.fullmatch(str(args.composition_report_sha256).casefold()):
        raise ValueError("composition_report SHA-256 must be 64 hexadecimal characters")
    if int(args.volume_size) < 30:
        raise ValueError("SageMaker volume must be at least 30 GiB")
    if not 300 <= int(args.max_runtime) <= 86_400:
        raise ValueError("SageMaker maximum runtime must be between 300 and 86400 seconds")
    prefixes = {
        "code": _safe_prefix(args.code_prefix, "code"),
        "boreal": _safe_prefix(args.boreal_prefix, "Boreal"),
        "camp-swift": _safe_prefix(args.camp_swift_prefix, "Camp Swift"),
        "kit": _safe_prefix(args.kit_prefix, "KIT"),
        "benchmark-guard": _safe_prefix(args.benchmark_guard_prefix, "benchmark guard"),
        "composition-receipt": _safe_prefix(
            args.composition_receipt_prefix,
            "composition receipt",
        ),
    }
    output_prefix = _safe_prefix(args.output_prefix, "output")
    if "benchmark" not in prefixes["benchmark-guard"].casefold():
        raise ValueError("benchmark guard input must identify a benchmark prefix")
    if not any(marker in output_prefix.casefold() for marker in ("exclusion", "hash-index")):
        raise ValueError("output must identify the isolated exclusion-index campaign")
    if any(
        marker in output_prefix.casefold()
        for marker in ("publish", "huggingface", "hf-upload", "strict-payload")
    ):
        raise ValueError("exclusion-index output cannot be a publication or media target")
    if any(_prefixes_overlap(output_prefix, prefix) for prefix in prefixes.values()):
        raise ValueError("output prefix must be isolated from every input prefix")
    if len(set(prefixes.values())) != len(prefixes):
        raise ValueError("SageMaker inputs must use distinct S3 prefixes")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", job_name):
        raise ValueError("invalid SageMaker ProcessingJobName")

    output_s3 = f"s3://{args.bucket}/{output_prefix}/{job_name}"
    input_root = "/opt/ml/processing/input"
    code_root = f"{input_root}/code"
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                f"{code_root}/pointing_exclusion_indexes.py",
                "--registry",
                f"{code_root}/dinov3-multitask-composition-v1.json",
                "--composition-root",
                f"{input_root}/composition-receipt",
                "--composition-report-sha256",
                str(args.composition_report_sha256).casefold(),
                "--publication-manifest",
                f"{code_root}/publication-manifest.json",
                "--boreal-root",
                f"{input_root}/boreal",
                "--camp-swift-root",
                f"{input_root}/camp-swift",
                "--kit-root",
                f"{input_root}/kit",
                "--benchmark-guard-root",
                f"{input_root}/benchmark-guard",
                "--work-dir",
                "/opt/ml/processing/work/exclusion-indexes",
                "--output-dir",
                "/opt/ml/processing/output",
            ],
        },
        "ProcessingInputs": [
            _input("code", f"s3://{args.bucket}/{prefixes['code']}", code_root),
            _input(
                "boreal",
                f"s3://{args.bucket}/{prefixes['boreal']}",
                f"{input_root}/boreal",
            ),
            _input(
                "camp-swift",
                f"s3://{args.bucket}/{prefixes['camp-swift']}",
                f"{input_root}/camp-swift",
            ),
            _input(
                "kit",
                f"s3://{args.bucket}/{prefixes['kit']}",
                f"{input_root}/kit",
            ),
            _input(
                "benchmark-guard",
                f"s3://{args.bucket}/{prefixes['benchmark-guard']}",
                f"{input_root}/benchmark-guard",
            ),
            _input(
                "composition-receipt",
                f"s3://{args.bucket}/{prefixes['composition-receipt']}",
                f"{input_root}/composition-receipt",
            ),
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "pointing-exclusion-indexes-candidate",
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
                "InstanceType": INSTANCE_TYPE,
                "VolumeSizeInGB": args.volume_size,
            }
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime},
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "FIREVIEWER_HASH_ONLY": "true",
            "FIREVIEWER_PARQUET_PROJECTED_COLUMNS": "sample_id,sha256,phash",
            "FIREVIEWER_PUBLICATION_STAGING_BUCKET": args.bucket,
            "FIREVIEWER_PUBLICATION_ALLOWED": "false",
        },
        "Tags": [
            {"Key": "fireviewer:corpus", "Value": "pointing-v2"},
            {"Key": "fireviewer:stage", "Value": "exclusion-index-candidate"},
            {"Key": "fireviewer:hash-only", "Value": "true"},
            {"Key": "fireviewer:reviews-admitted", "Value": "false"},
            {"Key": "fireviewer:publication-allowed", "Value": "false"},
            {"Key": "fireviewer:submitted", "Value": "false"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--code-prefix", required=True)
    parser.add_argument("--boreal-prefix", required=True)
    parser.add_argument("--camp-swift-prefix", required=True)
    parser.add_argument("--kit-prefix", required=True)
    parser.add_argument("--benchmark-guard-prefix", required=True)
    parser.add_argument("--composition-receipt-prefix", required=True)
    parser.add_argument(
        "--composition-report-sha256",
        default=DEFAULT_COMPOSITION_REPORT_SHA256,
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--volume-size", type=int, default=40)
    parser.add_argument("--max-runtime", type=int, default=21_600)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC_COMPAT).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-pointing-exclusions-{stamp}"
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
                "instance_type": INSTANCE_TYPE,
                "submitted": False,
                "publication_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
