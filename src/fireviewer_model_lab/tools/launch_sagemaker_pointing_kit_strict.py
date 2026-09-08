"""Build the SageMaker request for strict KIT multi-task auxiliary admission."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"


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
    prefixes = (
        args.inventory_prefix,
        args.baseline_prefix,
        args.code_prefix,
        args.output_prefix,
    )
    forbidden = ("fire-smoke-detection-corpus-v1", "benchdata", "fireviewer_bench")
    if any(marker in value.lower() for value in prefixes for marker in forbidden):
        raise ValueError("KIT audit escaped the isolated pointing source campaign")
    output_s3 = f"s3://{args.bucket}/{args.output_prefix.rstrip('/')}/{job_name}"
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                "/opt/ml/processing/input/code/pointing_kit_strict_audit.py",
                "--inventory-root",
                "/opt/ml/processing/input/inventory",
                "--baseline-root",
                "/opt/ml/processing/input/baseline",
                "--output-dir",
                "/opt/ml/processing/output",
                "--output-s3-prefix",
                output_s3,
            ],
        },
        "ProcessingInputs": [
            _input(
                "code",
                f"s3://{args.bucket}/{args.code_prefix.rstrip('/')}",
                "/opt/ml/processing/input/code",
            ),
            _input(
                "kit-inventory",
                f"s3://{args.bucket}/{args.inventory_prefix.rstrip('/')}",
                "/opt/ml/processing/input/inventory",
            ),
            _input(
                "strict-baseline",
                f"s3://{args.bucket}/{args.baseline_prefix.rstrip('/')}",
                "/opt/ml/processing/input/baseline",
            ),
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "kit-strict-auxiliary",
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
        "Environment": {"PYTHONUNBUFFERED": "1"},
        "Tags": [
            {"Key": "fireviewer:corpus", "Value": "pointing-v2"},
            {"Key": "fireviewer:stage", "Value": "kit-strict-multitask-auxiliary"},
            {"Key": "fireviewer:reviews-admitted", "Value": "false"},
            {"Key": "fireviewer:publication-allowed", "Value": "false"},
            {"Key": "fireviewer:point-supervision", "Value": "false"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--inventory-prefix", required=True)
    parser.add_argument("--baseline-prefix", required=True)
    parser.add_argument("--code-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--instance-type", default="ml.t3.large")
    parser.add_argument("--volume-size", type=int, default=20)
    parser.add_argument("--max-runtime", type=int, default=1800)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-pointing-kit-strict-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"job_name": job_name, "request": str(args.emit_request)}, indent=2))


if __name__ == "__main__":
    main()
