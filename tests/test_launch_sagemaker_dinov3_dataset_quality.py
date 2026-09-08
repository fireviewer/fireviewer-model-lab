from __future__ import annotations

import argparse
from pathlib import Path

from fireviewer_model_lab.tools.launch_sagemaker_dinov3_dataset_quality import build_request


def test_quality_request_is_bounded_cpu_and_fail_closed() -> None:
    root = Path(__import__("fireviewer_model_lab", fromlist=["_"]).__file__).parent
    args = argparse.Namespace(
        bucket="bucket",
        role="arn:aws:iam::123456789012:role/role",
        image="image",
        input_prefix="pointing/composition",
        code_prefix="pointing/code/qa",
        output_prefix="pointing/reports/qa",
        manifest_sha256="a" * 64,
        code_bundle_root=root,
        instance_type="ml.t3.large",
        volume_size=30,
        max_runtime=1800,
    )

    request = build_request(args, "fireviewer-dinov3-dataset-qa-test")

    cluster = request["ProcessingResources"]["ClusterConfig"]
    assert cluster == {"InstanceCount": 1, "InstanceType": "ml.t3.large", "VolumeSizeInGB": 30}
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 1800}
    assert {tag["Key"]: tag["Value"] for tag in request["Tags"]}["fireviewer:training-ready"] == "false"
    assert request["Environment"]["FIREVIEWER_COMPOSITION_MANIFEST_SHA256"] == "a" * 64
