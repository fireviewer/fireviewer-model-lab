"""Build the SageMaker Processing request for the pinned FLAME2 acquisition."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = (
    "764974769150.dkr.ecr.eu-west-2.amazonaws.com/"
    "sagemaker-scikit-learn:1.4-2-cpu-py3"
)


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    if any(
        marker in value.lower()
        for value in (args.code_prefix, args.output_prefix)
        for marker in ("fire-smoke-detection-corpus-v1", "benchdata", "fireviewer_bench")
    ):
        raise ValueError("FLAME2 acquisition escaped the pointing campaign")
    bucket = args.bucket
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                "/opt/ml/processing/input/code/pointing_new_source_flame2.py",
                "--output-dir",
                "/opt/ml/processing/output",
                "--work-dir",
                "/opt/ml/processing/work",
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
            }
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "flame2-candidate",
                    "S3Output": {
                        "S3Uri": f"s3://{bucket}/{args.output_prefix.rstrip('/')}",
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
            {"Key": "fireviewer:stage", "Value": "new-source-acquisition"},
            {"Key": "fireviewer:source", "Value": "robofirefusenet-flame2"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--code-prefix", default="pointing-corpus-v2/code/new-source-flame2")
    parser.add_argument(
        "--output-prefix",
        default=(
            "pointing-corpus-v2/new-sources/robofirefusenet-flame2/"
            "0d8ec502da0bafea7c388a989650aa53d1ecf278"
        ),
    )
    parser.add_argument("--instance-type", default="ml.t3.large")
    parser.add_argument("--volume-size", type=int, default=20)
    parser.add_argument("--max-runtime", type=int, default=3600)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-pointing-flame2-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"job_name": job_name, "request": str(args.emit_request)}, indent=2))


if __name__ == "__main__":
    main()
