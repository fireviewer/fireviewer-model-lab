"""Build a bounded SageMaker Processing request for DINOv3 dataset QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
DEFAULT_ROLE = "arn:aws:iam::640538430954:role/FireViewerPointingSageMakerRole"
DEFAULT_IMAGE = "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
SCRIPT_RELATIVE = "training/dinov3_dataset_quality_profile.py"
UTC_COMPAT = timezone.utc
VERIFIER = (
    "import hashlib,pathlib,runpy,sys;"
    "p=pathlib.Path(sys.argv[1]);"
    "p.is_file()and hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]"
    "or sys.exit('dataset QA script integrity failure');"
    "sys.argv=[str(p),*sys.argv[3:]];"
    "runpy.run_path(str(p),run_name='__main__')"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    script = (Path(args.code_bundle_root).resolve() / SCRIPT_RELATIVE).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)
    script_sha256 = _sha256(script)
    if not args.manifest_sha256 or len(args.manifest_sha256) != 64:
        raise ValueError("a pinned 64-character manifest SHA-256 is required")
    for value in (args.input_prefix, args.code_prefix, args.output_prefix):
        if any(marker in value.casefold() for marker in ("benchdata", "fireviewer_bench", "independent-benchmark")):
            raise ValueError("independent benchmark paths are forbidden")
    code_root = "/opt/ml/processing/input/code"
    arguments = [
        "-c",
        VERIFIER,
        f"{code_root}/{SCRIPT_RELATIVE}",
        script_sha256,
        "--input-dir",
        "/opt/ml/processing/input/composition",
        "--output-dir",
        "/opt/ml/processing/output",
        "--expected-manifest-sha256",
        args.manifest_sha256.casefold(),
    ]
    return {
        "ProcessingJobName": job_name,
        "RoleArn": args.role,
        "AppSpecification": {
            "ImageUri": args.image,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": arguments,
        },
        "ProcessingInputs": [
            _input("code", f"s3://{args.bucket}/{args.code_prefix.rstrip('/')}", code_root),
            _input(
                "composition",
                f"s3://{args.bucket}/{args.input_prefix.rstrip('/')}",
                "/opt/ml/processing/input/composition",
            ),
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "dataset-quality-profile",
                    "S3Output": {
                        "S3Uri": f"s3://{args.bucket}/{args.output_prefix.rstrip('/')}/{job_name}",
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
            "FIREVIEWER_QA_SCRIPT_SHA256": script_sha256,
            "FIREVIEWER_COMPOSITION_MANIFEST_SHA256": args.manifest_sha256.casefold(),
        },
        "Tags": [
            {"Key": "fireviewer:corpus", "Value": "dinov3-multitask-v4"},
            {"Key": "fireviewer:stage", "Value": "dataset-quality-profile"},
            {"Key": "fireviewer:publication-allowed", "Value": "false"},
            {"Key": "fireviewer:training-ready", "Value": "false"},
            {"Key": "fireviewer:cost-class", "Value": "bounded-cpu"},
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--role", default=DEFAULT_ROLE)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--code-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--code-bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--instance-type", default="ml.t3.large")
    parser.add_argument("--volume-size", type=int, default=30)
    parser.add_argument("--max-runtime", type=int, default=1800)
    parser.add_argument("--job-name")
    parser.add_argument("--emit-request", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stamp = datetime.now(UTC_COMPAT).strftime("%Y%m%dT%H%M%SZ").lower()
    job_name = args.job_name or f"fireviewer-dinov3-dataset-qa-{stamp}"
    request = build_request(args, job_name)
    args.emit_request.parent.mkdir(parents=True, exist_ok=True)
    args.emit_request.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"job_name": job_name, "request": str(args.emit_request)}, indent=2))


if __name__ == "__main__":
    main()
