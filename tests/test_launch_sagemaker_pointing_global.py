from __future__ import annotations

import argparse

import pytest
from fireviewer_model_lab.tools.launch_sagemaker_pointing_global import build_request


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bucket": "bucket",
        "role": "role",
        "image": "image",
        "flame2_prefix": "pointing/flame2",
        "boreal_prefix": "pointing/boreal",
        "camp_swift_prefix": "pointing/camp-swift",
        "additional_source_prefix": [],
        "code_prefix": "pointing/code",
        "registry_prefix": "pointing/registry",
        "output_prefix": "pointing/output",
        "instance_type": "ml.t3.large",
        "volume_size": 20,
        "max_runtime": 1800,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_global_launcher_adds_registered_source_input() -> None:
    request = build_request(
        _args(additional_source_prefix=["activefire=pointing/activefire/run"]),
        "job",
    )

    source = next(
        item for item in request["ProcessingInputs"] if item["InputName"] == "source-activefire"
    )
    assert source["S3Input"]["S3Uri"] == "s3://bucket/pointing/activefire/run"
    assert source["S3Input"]["LocalPath"].endswith("/sources/activefire")


def test_global_launcher_rejects_unsafe_additional_source_name() -> None:
    with pytest.raises(ValueError, match="safe unique name"):
        build_request(
            _args(additional_source_prefix=["../activefire=pointing/activefire/run"]),
            "job",
        )
