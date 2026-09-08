from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from fireviewer_model_lab.tools.export_rfdetr_hf import prepare_release


def test_prepare_small_release_strips_training_state_and_records_test_metrics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pth"
    torch.save(
        {
            "model": {"backbone.weight": torch.ones(2, 2)},
            "args": {
                "class_names": ["flame_visible", "smoke_visible"],
                "num_classes": 2,
                "num_queries": 300,
                "group_detr": 13,
                "pretrain_weights": r"C:\private\rf-detr-small.pth",
                "dataset_dir": r"C:\private\dataset",
            },
            "model_name": "RFDETRSmall",
            "rfdetr_version": "1.8.3",
            "optimizer_states": [{"private": "training-only"}],
        },
        checkpoint,
    )
    training_config = tmp_path / "training_config.json"
    training_config.write_text(
        json.dumps(
            {
                "train_config": {
                    "notes": {"dataset": "fireviewer/fire-smoke-ground-elite-rfdetr-small-v1"}
                },
                "model_config": {
                    "model_name": "RFDETRSmall",
                    "num_queries": 300,
                    "group_detr": 13,
                    "num_classes": 2,
                    "resolution": 512,
                },
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.csv"
    fieldnames = [
        "epoch",
        "step",
        "train/loss",
        "val/mAP_50",
        "val/mAP_50_95",
        "val/ema_mAP_50",
        "val/ema_mAP_50_95",
        "val/F1",
        "val/precision",
        "val/recall",
        "val/AP/flame_visible",
        "val/AP/smoke_visible",
        "test/mAP_50",
        "test/mAP_50_95",
        "test/mAP_75",
        "test/mAR",
        "test/F1",
        "test/precision",
        "test/recall",
        "test/AP/flame_visible",
        "test/AP/smoke_visible",
    ]
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 0,
                "step": 10,
                "train/loss": 1.0,
                "val/mAP_50": 0.7,
                "val/mAP_50_95": 0.45,
                "val/ema_mAP_50": 0.71,
                "val/ema_mAP_50_95": 0.46,
                "val/F1": 0.66,
                "val/precision": 0.68,
                "val/recall": 0.64,
                "val/AP/flame_visible": 0.38,
                "val/AP/smoke_visible": 0.52,
            }
        )
        writer.writerow(
            {
                "epoch": 1,
                "step": 11,
                "test/mAP_50": 0.69,
                "test/mAP_50_95": 0.43,
                "test/mAP_75": 0.45,
                "test/mAR": 0.67,
                "test/F1": 0.67,
                "test/precision": 0.69,
                "test/recall": 0.65,
                "test/AP/flame_visible": 0.37,
                "test/AP/smoke_visible": 0.50,
            }
        )

    output = tmp_path / "release"
    result = prepare_release(
        checkpoint=checkpoint,
        training_config=training_config,
        metrics_csv=metrics,
        output=output,
    )

    exported = torch.load(output / "checkpoint_best_total.pth", map_location="cpu")
    assert exported["model_name"] == "RFDETRSmall"
    assert exported["args"]["pretrain_weights"] == "rf-detr-small.pth"
    assert "optimizer_states" not in exported
    assert "private" not in json.dumps(exported["args"]).lower()
    hub_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert hub_config["architectures"] == ["RFDETRSmall"]
    assert hub_config["dataset"].endswith("ground-elite-rfdetr-small-v1")
    published_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert published_metrics["epochs_completed"] == 1
    assert published_metrics["test"]["map_50_95"] == 0.43
    assert result["merge_required"] is False
