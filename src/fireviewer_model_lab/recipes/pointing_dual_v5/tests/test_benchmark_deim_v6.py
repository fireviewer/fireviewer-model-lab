from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_deim_v6 import cluster_bootstrap_indices, markdown_report, split_accounting, validate_corpus_counts  # noqa: E402
from benchmark_shared_v5 import counts  # noqa: E402


def test_counts_reports_operating_and_localization_metrics() -> None:
    targets = [
        {
            "boxes": torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32),
            "labels": torch.tensor([0, 1], dtype=torch.int64),
        }
    ]
    predictions = [
        {
            "boxes": torch.tensor(
                [[0, 0, 10, 10], [50, 50, 60, 60], [21, 21, 29, 29]], dtype=torch.float32
            ),
            "labels": torch.tensor([0, 0, 1], dtype=torch.int64),
            "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
        }
    ]

    at_50 = counts(targets, predictions, threshold=0.5, iou_threshold=0.5)
    assert at_50["tp"] == 2
    assert at_50["fp"] == 1
    assert at_50["fn"] == 0
    assert at_50["precision"] == pytest.approx(2 / 3)
    assert at_50["recall"] == 1.0
    assert at_50["localization"]["matched"] == 2
    assert at_50["localization"]["mean_iou"] == pytest.approx(0.82)

    at_75 = counts(targets, predictions, threshold=0.5, iou_threshold=0.75)
    assert at_75["tp"] == 1
    assert at_75["fp"] == 2
    assert at_75["fn"] == 1


def test_cluster_bootstrap_resamples_whole_source_groups_deterministically() -> None:
    images = [
        {"id": 1, "fireviewer_source_group_id": "group-a"},
        {"id": 2, "fireviewer_source_group_id": "group-a"},
        {"id": 3, "fireviewer_source_group_id": "group-b"},
    ]
    first, groups = cluster_bootstrap_indices(images, np.random.default_rng(42))
    second, _ = cluster_bootstrap_indices(images, np.random.default_rng(42))
    assert groups == 2
    assert first == second
    assert first.count(0) == first.count(1)


def test_split_accounting_keeps_negatives_sources_and_group_counts() -> None:
    coco = {
        "images": [
            {
                "id": 1,
                "fireviewer_source_dataset": "a",
                "fireviewer_scene_bin": "fire_small",
                "fireviewer_source_group_id": "group-a",
            },
            {
                "id": 2,
                "fireviewer_source_dataset": "a",
                "fireviewer_scene_bin": "fire_small",
                "fireviewer_source_group_id": "group-a",
            },
            {
                "id": 3,
                "fireviewer_source_dataset": "b",
                "fireviewer_scene_bin": "smoke_only",
                "fireviewer_source_group_id": "group-b",
            },
        ],
        "annotations": [
            {"image_id": 1, "category_id": 0},
            {"image_id": 3, "category_id": 1},
        ],
    }
    result = split_accounting(coco)
    assert result["images"] == 3
    assert result["negative_images"] == 1
    assert result["class_annotations"] == {"fire": 1, "smoke": 1}
    assert result["source_groups"] == 2
    assert result["multi_image_source_groups"] == 1
    assert result["max_source_group_images"] == 2
    expected = {"images": 3, "annotations": 2, "negative_images": 1, "class_counts": {"fire": 1, "smoke": 1}}
    validate_corpus_counts(coco, expected, "test")
    with pytest.raises(RuntimeError, match="annotations mismatch"):
        validate_corpus_counts(coco, expected | {"annotations": 876}, "test")


def test_markdown_report_names_the_actual_candidate_and_baseline() -> None:
    report = {
        "protocol": {
            "report_title": "V7 frozen-panel benchmark",
            "candidate_model": "v7",
            "baseline_model": "v6",
        },
        "models": [],
        "comparison": {"point_estimate": {}, "intervals": {}},
        "corpus": {
            "test": {
                "images": 1,
                "annotations": 0,
                "class_annotations": {"fire": 0, "smoke": 0},
                "source_groups": 1,
                "negative_images": 1,
            }
        },
        "limitations": [],
    }
    report["comparison"] = None
    rendered = markdown_report(report)
    assert rendered.startswith("# V7 frozen-panel benchmark")
