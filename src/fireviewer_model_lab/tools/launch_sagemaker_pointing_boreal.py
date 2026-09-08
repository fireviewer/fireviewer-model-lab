"""Build the SageMaker request for strict Boreal pointing-source admission."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    prefixes = (
        args.stage2_baseline_prefix,
        args.flame2_baseline_prefix,
        args.code_prefix,
        args.output_prefix,
    )
    forbidden = ("fire-smoke-detection-corpus-v1", "benchdata", "fireviewer_bench")
    if any(marker in value.lower() for value in prefixes for marker in forbidden):
        raise ValueError("Boreal audit escaped the isolated pointing campaign")
    bucket = args.bucket
    output_s3 = f"s3://{bucket}/{args.output_prefix.rstrip('/')}/{job_name}"
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                "/opt/ml/processing/input/code/pointing_boreal_hf_audit.py",
                "--baseline-root",
                "/opt/ml/processing/input/baseline",
                "--output-dir",
                "/opt/ml/processing/output",
                "--work-dir",
                "/opt/ml/processing/work/fireviewer-boreal",
                "--output-s3-prefix",
                output_s3,
            ],
        },
        "ProcessingInputs": [
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{args.code_prefix.rstrip('/')}",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "stage2-baseline",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{args.stage2_baseline_prefix.rstrip('/')}",
                    "LocalPath": "/opt/ml/processing/input/baseline/stage2",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "flame2-baseline",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{args.flame2_baseline_prefix.rstrip('/')}",
                    "LocalPath": "/opt/ml/processing/input/baseline/flame2",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "boreal-strict",
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
            {"Key": "fireviewer:stage", "Value": "boreal-strict-admission"},
            {"Key": "fireviewer:reviews-admitted", "Value": "false"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--stage2-baseline-prefix", required=True)
    parser.add_argument("--flame2-baseline-prefix", required=True)
    parser.add_argument("--code-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--instance-type", default="ml.t3.large")
    parser.add_argument("--volume-size", type=int, default=20)
    parser.add_argument("--max-runtime", type=int, default=3600)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-pointing-boreal-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"job_name": job_name, "request": str(args.emit_request)}, indent=2))


if __name__ == "__main__":
    main()
