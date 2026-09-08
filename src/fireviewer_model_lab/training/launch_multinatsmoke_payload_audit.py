"""Launch the bounded CPU-only D-Fire payload audit on SageMaker Processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3

BUCKET = "fireviewer-dataset-qa-640538430954-eu-west-2-an"
REGION = "eu-west-2"
ROLE = "arn:aws:iam::640538430954:role/FireViewerSageMakerProcessingRole"
IMAGE = "764974769150.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-scikit-learn:1.4-2-cpu-py3"
SOURCE = "https://huggingface.co/datasets/hongjinzhao0615/MultiNatSmoke/resolve/5b7a4e0f7a094d7e27381a5154ad71be99f58137/MultiNatSmokeDataset.zip"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_request(*, run_id: str, script_key: str) -> dict[str, Any]:
    job_name = re.sub(r"[^a-z0-9-]", "-", f"fireviewer-dfire-audit-{run_id.lower()}")[:63].rstrip("-")
    return {
        "ProcessingJobName": job_name,
        "RoleArn": ROLE,
        "AppSpecification": {
            "ImageUri": IMAGE,
            "ContainerEntrypoint": ["python3"],
            "ContainerArguments": [
                "/opt/ml/processing/input/code/multinatsmoke_payload_audit.py",
                "--source", SOURCE,
                "--output-dir", "/opt/ml/processing/output",
                "--allowed-source", "D-Fire",
            ],
        },
        "ProcessingInputs": [{
            "InputName": "code",
            "S3Input": {
                "S3Uri": f"s3://{BUCKET}/{script_key}",
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
                "S3CompressionType": "None",
            },
        }],
        "ProcessingOutputConfig": {"Outputs": [{
            "OutputName": "audit",
            "S3Output": {
                "S3Uri": f"s3://{BUCKET}/pointing-corpus-v2/payload-audits/{run_id}/dfire/",
                "LocalPath": "/opt/ml/processing/output",
                "S3UploadMode": "EndOfJob",
            },
        }]},
        "ProcessingResources": {"ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.t3.large",
            "VolumeSizeInGB": 30,
        }},
        "StoppingCondition": {"MaxRuntimeInSeconds": 14400},
        "Environment": {
            "PYTHONUNBUFFERED": "1",
            "FIREVIEWER_TRAINING_AUTHORIZED": "false",
            "FIREVIEWER_ANNOTATION_MODE": "published_masks_only",
        },
        "Tags": [
            {"Key": "Project", "Value": "FireViewer"},
            {"Key": "Purpose", "Value": "PointingCorpusPayloadAudit"},
            {"Key": "TrainingAuthorized", "Value": "false"},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.script.is_file():
        raise FileNotFoundError(args.script)
    if not re.fullmatch(r"[0-9A-Za-z-]+", args.run_id):
        raise ValueError("run-id must contain only letters, digits, and hyphens")

    digest = _sha256(args.script)
    script_key = f"pointing-corpus-v2/payload-audits/{args.run_id}/code/multinatsmoke_payload_audit.py"
    request = build_request(run_id=args.run_id, script_key=script_key)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "script_sha256": digest,
        "script_s3_uri": f"s3://{BUCKET}/{script_key}",
        "source_revision_pinned": True,
        "request": request,
    }
    if args.dry_run:
        receipt["launch_status"] = "dry_run_not_submitted"
    else:
        session = boto3.session.Session(region_name=REGION)
        receipt["identity"] = session.client("sts").get_caller_identity()
        s3 = session.client("s3")
        s3.put_object(
            Bucket=BUCKET,
            Key=script_key,
            Body=args.script.read_bytes(),
            ContentType="text/x-python",
            Metadata={"sha256": digest, "training-authorized": "false"},
        )
        head = s3.head_object(Bucket=BUCKET, Key=script_key)
        if head.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError("uploaded script hash metadata mismatch")
        response = session.client("sagemaker").create_processing_job(**request)
        receipt["launch_status"] = "submitted"
        receipt["processing_job_arn"] = response["ProcessingJobArn"]
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
