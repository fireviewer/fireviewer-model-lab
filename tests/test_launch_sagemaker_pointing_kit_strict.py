from __future__ import annotations

import argparse

import pytest
from fireviewer_model_lab.tools.launch_sagemaker_pointing_kit_strict import build_request


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bucket": "bucket",
        "role": "role",
        "image": "image",
        "inventory_prefix": "pointing/kit-inventory",
        "baseline_prefix": "pointing/baseline",
        "code_prefix": "pointing/code",
        "output_prefix": "pointing/output",
        "instance_type": "ml.t3.large",
        "volume_size": 20,
        "max_runtime": 1800,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_kit_strict_launcher_reuses_cloud_archive_and_keeps_publication_closed() -> None:
    request = build_request(_args(), "job")

    inventory = next(
        item for item in request["ProcessingInputs"] if item["InputName"] == "kit-inventory"
    )
    assert inventory["S3Input"]["S3Uri"] == "s3://bucket/pointing/kit-inventory"
    tags = {item["Key"]: item["Value"] for item in request["Tags"]}
    assert tags["fireviewer:publication-allowed"] == "false"
    assert tags["fireviewer:point-supervision"] == "false"


def test_kit_strict_launcher_rejects_detection_or_benchmark_paths() -> None:
    with pytest.raises(ValueError, match="isolated"):
        build_request(_args(baseline_prefix="benchdata/kit"), "job")
