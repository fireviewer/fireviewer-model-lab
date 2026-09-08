from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest
from PIL import Image
from fireviewer_model_lab.tools.launch_sagemaker_pointing_pyro_negatives import build_request
from fireviewer_model_lab.training.pointing_pyro_negatives import (
    HF_REVISION,
    _difference_hash,
    apply_strict_deduplication,
    assign_group_splits,
    fetch_exact_empty_rows,
    select_balanced_strict_rows,
    validate_viewer_row,
)


def test_difference_hash_uses_stable_pillow_bytes_api() -> None:
    image = Image.new("RGB", (16, 16), (42, 84, 126))

    signature = _difference_hash(image)

    assert signature == "0" * 16


def _viewer_entry(*, annotations: str = "", row_index: int = 7) -> dict:
    return {
        "row_idx": row_index,
        "row": {
            "image": {
                "src": (
                    "https://datasets-server.huggingface.co/cached-assets/"
                    f"pyronear/pyro-sdis/--/{HF_REVISION}/--/default/train/7/image/image.jpg"
                ),
                "height": 720,
                "width": 1280,
            },
            "annotations": annotations,
            "image_name": "sdis-07_brison-200_2024-01-15T14-32-36.jpg",
            "partner": "sdis-07",
            "camera": "brison-200",
            "date": "2024-01-15T14-32-36",
        },
        "truncated_cells": [],
    }


def _candidate(sample_id: str, signature: str, digest: str, group: str, split: str) -> dict:
    return {
        "sample_id": sample_id,
        "dhash": signature,
        "image_sha256": digest,
        "split_group": group,
        "split": split,
        "exclusion_reasons": [],
    }


def test_viewer_row_requires_exact_empty_annotation_and_pinned_asset() -> None:
    row = validate_viewer_row(_viewer_entry(), "train")

    assert row["source_annotations"] == ""
    assert row["annotation_strength"] == "negative"
    assert row["visual_abstention_reason"] is None
    assert row["mask_quality"] == "source_explicit_empty_annotation_rasterized"
    assert row["reviews_admitted"] is False
    assert row["split_group"] == "pyro-sdis-camera:sdis-07:brison-200"
    with pytest.raises(ValueError, match="non-empty annotation"):
        validate_viewer_row(_viewer_entry(annotations=" "), "train")
    entry = _viewer_entry()
    entry["row"]["image"]["src"] = entry["row"]["image"]["src"].replace(
        HF_REVISION, "0" * 40
    )
    with pytest.raises(ValueError, match="pinned revision"):
        validate_viewer_row(entry, "train")


def test_full_rows_enumeration_filters_exact_empty_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fireviewer_model_lab.training.pointing_pyro_negatives.EXPECTED_SOURCE_ROWS", 4)
    monkeypatch.setattr("fireviewer_model_lab.training.pointing_pyro_negatives.EXPECTED_EMPTY_ROWS", 2)

    def fetch(url: str) -> dict:
        split = "train" if "split=train" in url else "val"
        empty = _viewer_entry(row_index=0)
        empty["row"]["image"]["src"] = empty["row"]["image"]["src"].replace(
            "/train/7/", f"/{split}/0/"
        )
        positive = _viewer_entry(annotations="1 0.5 0.5 0.1 0.1", row_index=1)
        positive["row"]["image"]["src"] = positive["row"]["image"]["src"].replace(
            "/train/7/", f"/{split}/1/"
        )
        return {"partial": False, "num_rows_total": 2, "rows": [empty, positive]}

    rows = fetch_exact_empty_rows(fetch)

    assert len(rows) == 2
    assert {row["source_split"] for row in rows} == {"train", "val"}


def test_camera_groups_are_never_split_across_partitions() -> None:
    rows = [
        {"sample_id": f"sample-{index}", "split_group": f"camera-{index % 8}"}
        for index in range(80)
    ]
    assignments = assign_group_splits(rows)

    assert set(assignments.values()) == {"train", "validation", "test"}
    assert len(assignments) == 8
    for group in assignments:
        observed = {
            assignments[row["split_group"]]
            for row in rows
            if row["split_group"] == group
        }
        assert len(observed) == 1


def test_strict_deduplication_rejects_baseline_and_internal_neighbours() -> None:
    rows = [
        _candidate("a", "0000000000000000", "a" * 64, "camera-a", "train"),
        _candidate("b", "0000000000000001", "b" * 64, "camera-b", "validation"),
        _candidate("c", "ffffffffffffffff", "c" * 64, "camera-c", "test"),
    ]
    report = apply_strict_deduplication(
        rows,
        baseline_sha={"c" * 64: "baseline-c"},
        baseline_dhash=[("ffffffffffffffff", "baseline-c")],
    )

    by_id = {row["sample_id"]: row for row in rows}
    first, second = sorted(("a", "b"), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    assert by_id[first]["exclusion_reasons"] == []
    assert "within_source_perceptual_duplicate" in by_id[second]["exclusion_reasons"]
    assert "baseline_exact_sha_overlap" in by_id["c"]["exclusion_reasons"]
    assert report["baseline_exact_overlaps"] == 1


def test_balanced_selection_marks_non_selected_rows() -> None:
    rows = [
        _candidate(
            f"{split}-{index}",
            f"{index + offset:016x}",
            f"{index + offset:064x}",
            f"{split}-camera-{index % 3}",
            split,
        )
        for split, offset in (("train", 0), ("validation", 100), ("test", 200))
        for index in range(10)
    ]
    selected = select_balanced_strict_rows(rows, max_validated=10)

    assert len(selected) == 10
    assert sum(row["split"] == "train" for row in selected) == 7
    assert sum(row["split"] == "validation" for row in selected) == 2
    assert sum(row["split"] == "test" for row in selected) == 1
    assert sum("strict_source_balance_quota" in row["exclusion_reasons"] for row in rows) == 20


def _launcher_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bucket": "bucket",
        "role": "role",
        "image": "image",
        "code_prefix": "pointing/code/pyro",
        "baseline_prefix": "pointing/baseline/global",
        "output_prefix": "pointing/output/pyro",
        "instance_type": "ml.t3.large",
        "volume_size": 20,
        "max_runtime": 7200,
        "workers": 12,
        "max_validated": 600,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_sagemaker_request_is_cloud_only_and_isolated(tmp_path: Path) -> None:
    request = build_request(_launcher_args(), "job")

    assert request["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "ml.t3.large"
    assert request["ProcessingInputs"][1]["S3Input"]["S3Uri"] == "s3://bucket/pointing/baseline/global"
    assert request["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"] == (
        "s3://bucket/pointing/output/pyro/job"
    )
    container_arguments = request["AppSpecification"]["ContainerArguments"]
    assert container_arguments[:2] == [
        "/opt/ml/processing/input/code/pointing_pyro_negatives_bootstrap.py",
        "/opt/ml/processing/input/code/pointing_pyro_negatives.py",
    ]
    assert "--max-validated" in container_arguments
    assert max(map(len, container_arguments)) <= 256
    assert not (tmp_path / "corpus").exists()
    with pytest.raises(ValueError, match="isolated pointing"):
        build_request(
            _launcher_args(baseline_prefix="fire-smoke-detection-corpus-v1/run"),
            "job",
        )
