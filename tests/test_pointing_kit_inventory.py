from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import pytest
from fireviewer_model_lab.tools.launch_sagemaker_pointing_kit_inventory import build_request
from fireviewer_model_lab.training.pointing_kit_inventory import inventory_tar, validate_tar_members


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_kit_tar_validation_rejects_path_escape(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        _add_bytes(archive, "../escape.png", b"unsafe")

    with tarfile.open(archive_path) as archive, pytest.raises(
        ValueError,
        match="unsafe tar member path",
    ):
        validate_tar_members(archive)
    assert not (tmp_path / "escape.png").exists()


def test_kit_tar_validation_rejects_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("dataset/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        archive.addfile(member)

    with tarfile.open(archive_path) as archive, pytest.raises(
        ValueError,
        match="unsupported tar member type",
    ):
        validate_tar_members(archive)


def test_kit_inventory_identifies_exact_relabelled_200_subset(tmp_path: Path) -> None:
    archive_path = tmp_path / "kit.tar"
    with tarfile.open(archive_path, "w") as archive:
        _add_bytes(
            archive,
            "KIT/README.md",
            b"The 200 manually relabelled masks are provided under CC BY 4.0.",
        )
        _add_bytes(archive, "KIT/LICENSE.txt", b"Creative Commons Attribution 4.0")
        _add_bytes(archive, "KIT/images/frame_0001.png", b"image")
        for index in range(200):
            _add_bytes(
                archive,
                f"KIT/relabelled_masks/frame_{index:04d}_mask.png",
                b"mask",
            )

    summary = inventory_tar(
        archive_path,
        inventory_path=tmp_path / "output" / "kit_inventory.jsonl",
        metadata_output_dir=tmp_path / "output" / "source_metadata",
    )

    assert summary["classification_counts"]["mask_candidate"] == 200
    assert summary["classification_counts"]["relabelled_mask_candidate"] == 200
    assert summary["classification_counts"]["source_image_candidate"] == 1
    assert summary["embedded_cc_by_4_signal"] is True
    assert summary["relabelled_subset"]["identified"] is True
    assert summary["relabelled_subset"]["selected_candidate_count"] == 200
    assert summary["relabelled_subset"]["identification_strategy"] == (
        "exact_path_marked_relabelled_mask_count"
    )
    assert (tmp_path / "output" / "source_metadata" / "KIT" / "README.md").is_file()


def test_kit_sagemaker_request_is_inventory_only() -> None:
    args = argparse.Namespace(
        bucket="example-bucket",
        role="arn:aws:iam::123456789012:role/PointingRole",
        image="example.invalid/sagemaker:cpu",
        code_prefix="pointing-corpus-v2/code/kit-inventory",
        output_prefix="pointing-corpus-v2/reports/kit-inventory",
        instance_type="ml.t3.large",
        volume_size=20,
        max_runtime=3600,
    )

    request = build_request(args, "fireviewer-pointing-kit-inventory-test")
    arguments = request["AppSpecification"]["ContainerArguments"]
    tags = {item["Key"]: item["Value"] for item in request["Tags"]}

    assert "pointing_kit_inventory.py" in arguments[0]
    assert "383521792" in arguments
    assert request["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "ml.t3.large"
    assert request["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"] == (
        "s3://example-bucket/pointing-corpus-v2/reports/kit-inventory/"
        "fireviewer-pointing-kit-inventory-test"
    )
    assert tags["fireviewer:reviews-admitted"] == "false"
    assert tags["fireviewer:publication-allowed"] == "false"
    assert tags["fireviewer:strict-rows-admitted"] == "0"


def test_kit_sagemaker_request_rejects_detection_prefix() -> None:
    args = argparse.Namespace(
        bucket="example-bucket",
        role="arn:aws:iam::123456789012:role/PointingRole",
        image="example.invalid/sagemaker:cpu",
        code_prefix="fire-smoke-detection-corpus-v1/code",
        output_prefix="pointing-corpus-v2/reports/kit-inventory",
        instance_type="ml.t3.large",
        volume_size=20,
        max_runtime=3600,
    )

    with pytest.raises(ValueError, match="isolated pointing campaign"):
        build_request(args, "fireviewer-pointing-kit-inventory-test")
