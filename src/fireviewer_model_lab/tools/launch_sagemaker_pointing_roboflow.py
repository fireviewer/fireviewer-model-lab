"""Build, but never submit, a strict Roboflow native-polygon SageMaker audit."""

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
DEFAULT_CODE_PREFIX = "pointing-corpus-v2/code/roboflow-native-polygon-audit"
DEFAULT_OUTPUT_PREFIX = "pointing-corpus-v2/reports/roboflow-native-polygon-audit"
INDEX_SIGNAL = re.compile(r"(?:^|[-_/])(dedup|exclusion|hash|index)(?:[-_/]|$)", re.I)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _safe_prefix(value: str, purpose: str) -> str:
    if not value or value.startswith(("/", "s3://")) or "\\" in value:
        raise ValueError(f"{purpose} must be a relative S3 key prefix")
    normalized = PurePosixPath(value)
    if ".." in normalized.parts or normalized.as_posix() in {"", "."}:
        raise ValueError(f"unsafe {purpose} prefix")
    return normalized.as_posix().rstrip("/")


def _safe_filename(value: str) -> str:
    path = PurePosixPath(value)
    if len(path.parts) != 1 or path.suffix.casefold() != ".jsonl":
        raise ValueError("manifest filename must be one JSONL basename")
    return path.name


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
    manifest_prefix = _safe_prefix(args.manifest_prefix, "manifest")
    output_prefix = _safe_prefix(args.output_prefix, "output")
    manifest_filename = _safe_filename(args.manifest_filename)
    indexes = {
        name: _hash_index_prefix(getattr(args, f"{name}_index_prefix"), f"{name} index")
        for name in ("detection", "pointing", "benchmark")
    }
    forbidden = ("fire-smoke-detection-corpus-v1", "benchdata", "fireviewer_bench")
    if any(
        marker in value.casefold()
        for value in (code_prefix, manifest_prefix, output_prefix)
        for marker in forbidden
    ):
        raise ValueError("Roboflow audit escaped the isolated pointing campaign")
    if any(
        marker in output_prefix.casefold() for marker in ("publish", "huggingface", "hf-upload")
    ):
        raise ValueError("Roboflow audit output cannot be a publication target")
    if args.source_license not in {"CC BY 4.0", "CC0 1.0", "Public Domain"}:
        raise ValueError("source licence is not allowlisted")
    manifest_sha = str(args.manifest_sha256).casefold()
    if not HEX64.fullmatch(manifest_sha):
        raise ValueError("manifest SHA-256 is invalid")
    contracts: dict[str, tuple[str, int]] = {}
    for name in indexes:
        receipt_sha = str(getattr(args, f"{name}_index_receipt_sha256")).casefold()
        rows = int(getattr(args, f"{name}_index_rows"))
        if not HEX64.fullmatch(receipt_sha) or rows <= 0:
            raise ValueError(f"{name} exclusion receipt contract is invalid")
        contracts[name] = (receipt_sha, rows)
    _require_disjoint(
        {
            "code": code_prefix,
            "manifest": manifest_prefix,
            "output": output_prefix,
            **{f"{name}-index": value for name, value in indexes.items()},
        }
    )
    output_s3 = f"s3://{args.bucket}/{output_prefix}/{job_name}"
    arguments = [
        "/opt/ml/processing/input/code/pointing_roboflow_audit.py",
        "--input-manifest",
        f"/opt/ml/processing/input/manifest/{manifest_filename}",
        "--expected-manifest-sha256",
        manifest_sha,
        "--source-id",
        args.source_id,
        "--source-family",
        args.source_family,
        "--source-revision",
        args.source_revision,
        "--source-license",
        args.source_license,
        "--output-dir",
        "/opt/ml/processing/output",
        "--output-s3-prefix",
        output_s3,
    ]
    for name in indexes:
        arguments.extend(
            [
                f"--{name}-index-root",
                f"/opt/ml/processing/input/exclusions/{name}",
                f"--{name}-index-receipt-sha256",
                contracts[name][0],
                f"--{name}-index-rows",
                str(contracts[name][1]),
            ]
        )
    inputs = [
        _input("code", f"s3://{args.bucket}/{code_prefix}", "/opt/ml/processing/input/code"),
        _input(
            "roboflow-jsonl-manifest",
            f"s3://{args.bucket}/{manifest_prefix}",
            "/opt/ml/processing/input/manifest",
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
                    "OutputName": "roboflow-native-polygon-strict-audit",
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
            "FIREVIEWER_ROBOFLOW_MANIFEST_SHA256": manifest_sha,
            "FIREVIEWER_PUBLICATION_ALLOWED": "false",
        },
        "Tags": [
            {"Key": "fireviewer:corpus", "Value": "pointing-v2"},
            {"Key": "fireviewer:stage", "Value": "roboflow-native-polygon-audit"},
            {"Key": "fireviewer:source-revision", "Value": args.source_revision[:256]},
            {"Key": "fireviewer:reviews-admitted", "Value": "false"},
            {"Key": "fireviewer:pseudo-labels", "Value": "false"},
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
    parser.add_argument("--manifest-prefix", required=True)
    parser.add_argument("--manifest-filename", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    for name in ("detection", "pointing", "benchmark"):
        parser.add_argument(f"--{name}-index-prefix", required=True)
        parser.add_argument(f"--{name}-index-receipt-sha256", required=True)
        parser.add_argument(f"--{name}-index-rows", type=int, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-family", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--code-prefix", default=DEFAULT_CODE_PREFIX)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--instance-type", default="ml.t3.xlarge")
    parser.add_argument("--volume-size", type=int, default=30)
    parser.add_argument("--max-runtime", type=int, default=7_200)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-pointing-roboflow-audit-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"job_name": job_name, "submitted": False}, indent=2))


if __name__ == "__main__":
    main()
