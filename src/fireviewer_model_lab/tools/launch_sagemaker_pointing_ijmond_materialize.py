"""Build, but never submit, a strict IJmond SageMaker materialization request."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = (
    "764974769150.dkr.ecr.eu-west-2.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-cpu-py3"
)
DEFAULT_CODE_PREFIX = "pointing-corpus-v2/code/ijmond-materialize"
DEFAULT_OUTPUT_PREFIX = "pointing-corpus-v2/materialized/ijmond"
EXPECTED_ARCHIVE_SHA256 = (
    "5a26dbad99b5e590608cdbb98f269f4a24266c020852401090f040f75bcb6343"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
INDEX_SIGNAL = re.compile(r"(?:^|[-_/])(dedup|exclusion|hash|index)(?:[-_/]|$)", re.I)


def _safe_prefix(value: str, purpose: str) -> str:
    if not value or value.startswith(("/", "s3://")) or "\\" in value:
        raise ValueError(f"{purpose} must be a relative S3 key prefix")
    normalized = PurePosixPath(value)
    if ".." in normalized.parts or normalized.as_posix() in {"", "."}:
        raise ValueError(f"unsafe {purpose} prefix")
    return normalized.as_posix().rstrip("/")


def _hash_index_prefix(value: str, purpose: str) -> str:
    normalized = _safe_prefix(value, purpose)
    if INDEX_SIGNAL.search(normalized) is None:
        raise ValueError(f"{purpose} must identify a hash-only exclusion index")
    if any(marker in normalized.casefold() for marker in ("strict-payload", "/images", "/masks")):
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


def _require_disjoint(prefixes: dict[str, str]) -> None:
    entries = sorted(prefixes.items())
    for index, (left_name, left) in enumerate(entries):
        for right_name, right in entries[index + 1 :]:
            if left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/"):
                raise ValueError(
                    f"S3 prefixes must be disjoint: {left_name}={left} overlaps "
                    f"{right_name}={right}"
                )


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    code_prefix = _safe_prefix(args.code_prefix, "code")
    audit_prefix = _safe_prefix(args.audit_prefix, "audit")
    output_prefix = _safe_prefix(args.output_prefix, "output")
    indexes = {
        name: _hash_index_prefix(getattr(args, f"{name}_index_prefix"), f"{name} index")
        for name in ("detection", "pointing", "benchmark")
    }
    _require_disjoint(
        {
            "code": code_prefix,
            "audit": audit_prefix,
            "output": output_prefix,
            **{f"{name}-index": value for name, value in indexes.items()},
        }
    )
    if any(
        marker in output_prefix.casefold()
        for marker in ("publish", "huggingface", "hf-upload")
    ):
        raise ValueError("IJmond materialization output cannot be a publication target")
    if any(
        marker in value.casefold()
        for value in (code_prefix, audit_prefix, output_prefix)
        for marker in ("fire-smoke-detection-corpus-v1", "benchdata", "fireviewer_bench")
    ):
        raise ValueError("IJmond materialization escaped the isolated pointing campaign")
    audit_hashes = {
        "summary": str(args.audit_summary_sha256).casefold(),
        "strict": str(args.audit_strict_manifest_sha256).casefold(),
        "point": str(args.audit_point_manifest_sha256).casefold(),
    }
    if any(HEX64.fullmatch(value) is None for value in audit_hashes.values()):
        raise ValueError("all audit artifacts require valid SHA-256 contracts")
    index_contracts: dict[str, tuple[str, int]] = {}
    for name in indexes:
        receipt_sha = str(getattr(args, f"{name}_index_receipt_sha256")).casefold()
        rows = int(getattr(args, f"{name}_index_rows"))
        if HEX64.fullmatch(receipt_sha) is None or rows <= 0:
            raise ValueError(f"{name} exclusion receipt contract is invalid")
        index_contracts[name] = (receipt_sha, rows)
    if not 1 <= args.point_cap <= 750:
        raise ValueError("point cap must stay between 1 and the approved source cap of 750")

    output_s3 = f"s3://{args.bucket}/{output_prefix}/{job_name}"
    arguments = [
        "/opt/ml/processing/input/code/pointing_ijmond_materialize.py",
        "--download-pinned-archive",
        "--work-dir",
        "/opt/ml/processing/work/ijmond",
        "--audit-summary",
        "/opt/ml/processing/input/audit/ijmond_audit_summary.json",
        "--audit-strict-manifest",
        "/opt/ml/processing/input/audit/ijmond_strict_validated_manifest.jsonl",
        "--audit-point-manifest",
        "/opt/ml/processing/input/audit/ijmond_strict_point_manifest.jsonl",
        "--expected-archive-sha256",
        EXPECTED_ARCHIVE_SHA256,
        "--expected-audit-summary-sha256",
        audit_hashes["summary"],
        "--expected-audit-strict-manifest-sha256",
        audit_hashes["strict"],
        "--expected-audit-point-manifest-sha256",
        audit_hashes["point"],
        "--output-dir",
        "/opt/ml/processing/output",
        "--point-cap",
        str(args.point_cap),
        "--output-s3-prefix",
        output_s3,
    ]
    for name in indexes:
        arguments.extend(
            [
                f"--{name}-index-root",
                f"/opt/ml/processing/input/exclusions/{name}",
                f"--{name}-index-receipt-sha256",
                index_contracts[name][0],
                f"--{name}-index-rows",
                str(index_contracts[name][1]),
            ]
        )
    inputs = [
        _input("code", f"s3://{args.bucket}/{code_prefix}", "/opt/ml/processing/input/code"),
        _input(
            "successful-audit",
            f"s3://{args.bucket}/{audit_prefix}",
            "/opt/ml/processing/input/audit",
        ),
    ]
    inputs.extend(
        _input(
            f"{name}-exclusion-index",
            f"s3://{args.bucket}/{prefix}",
            f"/opt/ml/processing/input/exclusions/{name}",
        )
        for name, prefix in indexes.items()
    )
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": arguments,
        },
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "ijmond-strict-materialized-payload",
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
            "FIREVIEWER_IJMOND_ARCHIVE_SHA256": EXPECTED_ARCHIVE_SHA256,
            "FIREVIEWER_PUBLICATION_ALLOWED": "false",
        },
        "Tags": [
            {"Key": "fireviewer:corpus", "Value": "pointing-v2"},
            {"Key": "fireviewer:stage", "Value": "ijmond-strict-materialization"},
            {"Key": "fireviewer:source-revision", "Value": "figshare-v3"},
            {"Key": "fireviewer:labels-generated", "Value": "false"},
            {"Key": "fireviewer:bbox-conversion", "Value": "false"},
            {"Key": "fireviewer:publication-allowed", "Value": "false"},
            {"Key": "fireviewer:benchmark-access", "Value": "hash-only-exclusion"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--audit-prefix", required=True)
    parser.add_argument("--audit-summary-sha256", required=True)
    parser.add_argument("--audit-strict-manifest-sha256", required=True)
    parser.add_argument("--audit-point-manifest-sha256", required=True)
    for name in ("detection", "pointing", "benchmark"):
        parser.add_argument(f"--{name}-index-prefix", required=True)
        parser.add_argument(f"--{name}-index-receipt-sha256", required=True)
        parser.add_argument(f"--{name}-index-rows", type=int, required=True)
    parser.add_argument("--code-prefix", default=DEFAULT_CODE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--point-cap", type=int, default=750)
    parser.add_argument("--instance-type", default="ml.t3.xlarge")
    parser.add_argument("--volume-size", type=int, default=30)
    parser.add_argument("--max-runtime", type=int, default=7_200)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-pointing-ijmond-materialize-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
