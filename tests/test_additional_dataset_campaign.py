from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from fireviewer_model_lab.training.additional_dataset_campaign import (
    REGISTRY_PATH,
    acquire_source,
    build_alarmod_manifest,
    build_tar_inventory,
    find_batch,
    load_registry,
    plan,
    remote_only_workspace,
    safe_extract_zip,
    write_reference_batch,
)


def test_registry_has_four_isolated_batches_and_pinned_mirrors() -> None:
    registry = load_registry(REGISTRY_PATH)

    assert [batch["batch_id"] for batch in registry["batches"]] == [
        "detection_uav_pointing_v1",
        "satellite_burnscar_multisensor_v1",
        "triage_reference_quarantine_v1",
        "crossview_benchmark_external_v1",
    ]
    mirrors = [
        source
        for batch in registry["batches"]
        for source in batch["sources"]
        if source["payload_policy"] == "mirror_normalized"
    ]
    assert {source["source_id"] for source in mirrors} == {
        "alarmod_forest_fire",
        "eo4wildfires",
    }
    assert all(len(source["revision"]) == 40 for source in mirrors)


def test_plan_never_promotes_reference_payloads() -> None:
    registry = load_registry(REGISTRY_PATH)
    campaign = plan(registry)
    policies = {
        source["source_id"]: source["payload_policy"]
        for batch in campaign["batches"]
        for source in batch["sources"]
    }

    assert policies["datacluster_fire_and_smoke"] == "blocked_no_redistribution"
    assert policies["aerialmegadepth"] == "direct_benchmark_only"


def test_payload_acquisition_refuses_reference_only_source(tmp_path: Path) -> None:
    registry = load_registry(REGISTRY_PATH)
    batch = find_batch(registry, "triage_reference_quarantine_v1")

    with pytest.raises(ValueError, match="Payload acquisition forbidden"):
        acquire_source(batch["sources"][0], tmp_path / "payload")


def test_workspace_guard_allows_temporary_local_payload_after_explicit_policy(
    tmp_path: Path,
) -> None:
    registry = load_registry(REGISTRY_PATH)

    assert remote_only_workspace(registry, tmp_path) == tmp_path.resolve()


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "forbidden")

    with pytest.raises(ValueError, match="escapes destination"):
        safe_extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_inventory_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"forbidden"
        member = tarfile.TarInfo("../escaped.bin")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="escapes destination"):
        build_tar_inventory(archive, tmp_path / "inventory.jsonl")


def test_reference_batch_contains_no_payload(tmp_path: Path) -> None:
    registry = load_registry(REGISTRY_PATH)
    batch = find_batch(registry, "triage_reference_quarantine_v1")
    output = tmp_path / "reference"

    write_reference_batch(registry, batch, output)
    payload = json.loads((output / "reference-registry.json").read_text(encoding="utf-8"))

    assert payload["payload_included"] is False
    assert {item["payload_policy"] for item in payload["sources"]} == {
        "reference_only",
        "blocked_no_redistribution",
    }


def test_alarmod_manifest_keeps_temporal_groups_and_points(tmp_path: Path) -> None:
    from PIL import Image

    for split, stem, color in (
        ("train", "image_1_0_0", (16, 32, 64)),
        ("validation", "image_2_0_0", (64, 32, 16)),
    ):
        images = tmp_path / split / "images"
        labels = tmp_path / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (1280, 720), color=color).save(images / f"{stem}.jpg")
        (labels / f"{stem}.txt").write_text(
            "0 0.5 0.6 0.2 0.1\n" if split == "train" else "",
            encoding="utf-8",
        )
    source = {
        "repository": "alarmod/forest_fire",
        "revision": "a" * 40,
        "declared_license": "GPL-3.0",
        "declared_rows": {"train": 1, "validation": 1},
    }

    report = build_alarmod_manifest(tmp_path, source)
    records = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["positive_counts"] == {"train": 1}
    assert report["negative_counts"] == {"train": 0, "validation": 1}
    assert records[0]["annotations"][0]["point_xy_normalized"] == [0.5, 0.6]
    assert {record["split_group"] for record in records} == {
        "alarmod_frame:1",
        "alarmod_frame:2",
    }
