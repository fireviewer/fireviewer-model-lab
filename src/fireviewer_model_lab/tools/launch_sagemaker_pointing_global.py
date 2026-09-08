"""Build the SageMaker request for the strict pointing global quality gate."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
SOURCE_INPUT_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


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


def _additional_source_inputs(values: list[str], bucket: str) -> list[dict]:
    inputs: list[dict] = []
    seen = {"code", "registry", "flame2-strict", "boreal-strict", "camp-swift-strict"}
    for value in values:
        name, separator, prefix = value.partition("=")
        if not separator or not prefix or not SOURCE_INPUT_NAME.fullmatch(name):
            raise ValueError("additional source must use name=s3-prefix with a safe unique name")
        input_name = f"source-{name}"
        if input_name in seen:
            raise ValueError(f"duplicate additional source name: {name}")
        seen.add(input_name)
        inputs.append(
            _input(
                input_name,
                f"s3://{bucket}/{prefix.rstrip('/')}",
                f"/opt/ml/processing/input/sources/{name}",
            )
        )
    return inputs


def build_request(args: argparse.Namespace, job_name: str) -> dict:
    prefixes = [
        args.flame2_prefix,
        args.boreal_prefix,
        args.camp_swift_prefix,
        args.code_prefix,
        args.registry_prefix,
        args.output_prefix,
    ]
    additional_values = list(args.additional_source_prefix or [])
    prefixes.extend(value.partition("=")[2] for value in additional_values)
    forbidden = ("fire-smoke-detection-corpus-v1", "benchdata", "fireviewer_bench")
    if any(marker in value.lower() for value in prefixes for marker in forbidden):
        raise ValueError("global pointing gate escaped the isolated campaign")
    bucket = args.bucket
    output_s3 = f"s3://{bucket}/{args.output_prefix.rstrip('/')}/{job_name}"
    source_inputs = [
        _input(
            "flame2-strict",
            f"s3://{bucket}/{args.flame2_prefix.rstrip('/')}",
            "/opt/ml/processing/input/sources/flame2",
        ),
        _input(
            "boreal-strict",
            f"s3://{bucket}/{args.boreal_prefix.rstrip('/')}",
            "/opt/ml/processing/input/sources/boreal",
        ),
        _input(
            "camp-swift-strict",
            f"s3://{bucket}/{args.camp_swift_prefix.rstrip('/')}",
            "/opt/ml/processing/input/sources/camp-swift",
        ),
        *_additional_source_inputs(additional_values, bucket),
    ]
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                "/opt/ml/processing/input/code/pointing_global_assemble.py",
                "--input-root",
                "/opt/ml/processing/input/sources",
                "--registry",
                "/opt/ml/processing/input/registry/pointing-corpus-v2.json",
                "--output-dir",
                "/opt/ml/processing/output",
            ],
        },
        "ProcessingInputs": [
            _input(
                "code",
                f"s3://{bucket}/{args.code_prefix.rstrip('/')}",
                "/opt/ml/processing/input/code",
            ),
            _input(
                "registry",
                f"s3://{bucket}/{args.registry_prefix.rstrip('/')}",
                "/opt/ml/processing/input/registry",
            ),
            *source_inputs,
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "strict-global",
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
            {"Key": "fireviewer:stage", "Value": "strict-global-gate"},
            {"Key": "fireviewer:reviews-admitted", "Value": "false"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--flame2-prefix", required=True)
    parser.add_argument("--boreal-prefix", required=True)
    parser.add_argument("--camp-swift-prefix", required=True)
    parser.add_argument(
        "--additional-source-prefix",
        action="append",
        default=[],
        metavar="NAME=S3_PREFIX",
        help="Add a registered strict source without changing this launcher.",
    )
    parser.add_argument("--code-prefix", required=True)
    parser.add_argument("--registry-prefix", required=True)
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
    job_name = args.job_name or f"fireviewer-pointing-global-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"job_name": job_name, "request": str(args.emit_request)}, indent=2))


if __name__ == "__main__":
    main()
