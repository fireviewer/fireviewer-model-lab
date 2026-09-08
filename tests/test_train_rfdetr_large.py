from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest
from fireviewer_model_lab.training.train_rfdetr_large import (
    DATASET_PROFILES,
    HISTORICAL_ENCODER_LEARNING_RATE,
    HISTORICAL_LEARNING_RATE,
    HISTORICAL_RESOLUTION,
    _build_plan,
    _check_pretrain_weights,
    _resolve_variant_defaults,
)


def _args(tmp_path: Path, variant: str = "large") -> argparse.Namespace:
    return argparse.Namespace(
        variant=variant,
        epochs=None,
        batch_size=None,
        grad_accum_steps=None,
        pretrain_weights=None,
        rf_home=tmp_path,
        learning_rate=HISTORICAL_LEARNING_RATE,
        encoder_learning_rate=HISTORICAL_ENCODER_LEARNING_RATE,
        weight_decay=1e-4,
        resolution=HISTORICAL_RESOLUTION,
        seed=420,
    )


def test_large_defaults_reapply_historical_training_profile(tmp_path: Path) -> None:
    args = _args(tmp_path)

    _resolve_variant_defaults(args)

    assert args.epochs == 3
    assert args.batch_size == 4
    assert args.grad_accum_steps == 16
    assert args.pretrain_weights == tmp_path / "rf-detr-large-2026.pth"


def test_small_defaults_use_accelerated_premium_profile(tmp_path: Path) -> None:
    args = _args(tmp_path, variant="small")

    _resolve_variant_defaults(args)

    assert args.epochs == 12
    assert args.batch_size == 8
    assert args.grad_accum_steps == 4
    assert args.pretrain_weights == tmp_path / "rf-detr-small.pth"


def test_pretrain_weight_gate_rejects_digest_drift(tmp_path: Path) -> None:
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"pinned-rfdetr")
    expected = hashlib.md5(b"pinned-rfdetr", usedforsecurity=False).hexdigest()

    report = _check_pretrain_weights(weights, expected)

    assert report["md5"] == expected
    with pytest.raises(ValueError, match="MD5 drift"):
        _check_pretrain_weights(weights, "0" * 32)


def test_ground_premium_profile_is_pinned_to_selected_corpus() -> None:
    profile = DATASET_PROFILES["ground-premium"]

    assert profile["dataset_id"] == "fireviewer/fire-smoke-ground-premium-rfdetr-small-v1"
    assert profile["splits"]["train"] == {"images": 21224, "annotations": 24053}


def test_ground_elite_profile_is_pinned_to_low_ram_corpus() -> None:
    profile = DATASET_PROFILES["ground-elite"]

    assert profile["dataset_id"] == "fireviewer/fire-smoke-ground-elite-rfdetr-small-v1"
    assert profile["splits"]["train"] == {"images": 5813, "annotations": 6792}


def test_plan_records_complete_historical_methodology(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _resolve_variant_defaults(args)
    report = {
        "conversion": {"prepared_coco_dir": str(tmp_path / "coco")},
        "pretrain_weights": {"path": str(args.pretrain_weights), "md5": "abc"},
    }

    plan = _build_plan(args, report)

    assert plan["model_class"] == "RFDETRLarge"
    assert plan["hyperparameters"] == {
        "epochs": 3,
        "batch_size": 4,
        "grad_accum_steps": 16,
        "learning_rate": 1e-4,
        "encoder_learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "resolution": 512,
        "gradient_checkpointing": True,
        "freeze_encoder": False,
        "amp_dtype": "bf16",
        "use_ema": True,
        "checkpoint_interval": 1,
        "num_workers": 0,
        "seed": 420,
    }
    assert plan["methodology"]["training_profile"] == "historical_pushed_adapted_to_expanded_corpus"
