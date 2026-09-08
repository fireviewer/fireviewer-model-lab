"""Launch the isolated pointing-corpus Stage 1 SageMaker Processing job."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_REGION = "eu-west-2"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = (
    "764974769150.dkr.ecr.eu-west-2.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-cpu-py3"
)
FORBIDDEN = (
    "fire-smoke-detection-corpus-v1",
    "benchdata",
    "fireviewer_bench",
    "independent-benchmark",
)


def _assert_isolated(*values: str) -> None:
    for value in values:
        lowered = value.lower()
        if any(marker in lowered for marker in FORBIDDEN):
            raise ValueError(f"non-pointing corpus reference rejected: {value}")


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    _assert_isolated(
        args.hf_metadata_prefix,
        args.v8_metadata_prefix,
        args.code_prefix,
        args.output_prefix,
        args.hf_image_prefix,
        args.v8_image_prefix,
    )
    bucket = args.bucket
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                "/opt/ml/processing/input/code/pointing_corpus_campaign.py",
                "stage1",
                "--registry",
                "/opt/ml/processing/input/registry/pointing-corpus-v2.json",
                "--hf-root",
                "/opt/ml/processing/input/hf",
                "--v8-root",
                "/opt/ml/processing/input/v8",
                "--output",
                "/opt/ml/processing/output",
                "--hf-s3-prefix",
                f"s3://{bucket}/{args.hf_image_prefix.rstrip('/')}",
                "--v8-s3-prefix",
                f"s3://{bucket}/{args.v8_image_prefix.rstrip('/')}",
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
                "InputName": "registry",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{args.registry_prefix.rstrip('/')}",
                    "LocalPath": "/opt/ml/processing/input/registry",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "hf-point-seed-metadata",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{args.hf_metadata_prefix.rstrip('/')}",
                    "LocalPath": "/opt/ml/processing/input/hf",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "pointing-v8-metadata",
                "S3Input": {
                    "S3Uri": f"s3://{bucket}/{args.v8_metadata_prefix.rstrip('/')}",
                    "LocalPath": "/opt/ml/processing/input/v8",
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
                    "OutputName": "stage1",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{args.output_prefix.rstrip('/')}/{job_name}",
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
            {"Key": "fireviewer:stage", "Value": "metadata-audit"},
            {"Key": "fireviewer:detection-input", "Value": "forbidden"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--instance-type", default="ml.t3.large")
    parser.add_argument("--volume-size", type=int, default=20)
    parser.add_argument("--max-runtime", type=int, default=3600)
    parser.add_argument("--hf-metadata-prefix", default="pointing-corpus-v2/inputs/hf-v1")
    parser.add_argument("--v8-metadata-prefix", default="pointing-v8-reservoir-v1/metadata")
    parser.add_argument("--code-prefix", default="pointing-corpus-v2/code/stage1")
    parser.add_argument("--registry-prefix", default="pointing-corpus-v2/config")
    parser.add_argument("--output-prefix", default="pointing-corpus-v2/reports/stage1")
    parser.add_argument("--hf-image-prefix", default="pointing-ground-v1/raw")
    parser.add_argument("--v8-image-prefix", default="pointing-v8-reservoir-v1/raw")
    parser.add_argument("--job-name")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--emit-request",
        type=Path,
        help="Write the CreateProcessingJob request for an authenticated AWS CLI",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-pointing-v2-stage1-{stamp}"
    request = build_request(args, job_name)
    if args.emit_request is not None:
        args.emit_request.parent.mkdir(parents=True, exist_ok=True)
        args.emit_request.write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"job_name": job_name, "request": str(args.emit_request)}, indent=2))
        return

    import boto3

    client = boto3.client("sagemaker", region_name=args.region)
    response = client.create_processing_job(**request)
    print(json.dumps({"job_name": job_name, "arn": response["ProcessingJobArn"]}, indent=2))
    if not args.wait:
        return
    while True:
        description = client.describe_processing_job(ProcessingJobName=job_name)
        status = description["ProcessingJobStatus"]
        print(json.dumps({"job_name": job_name, "status": status}))
        if status in {"Completed", "Failed", "Stopped"}:
            if status != "Completed":
                raise RuntimeError(description.get("FailureReason") or status)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
