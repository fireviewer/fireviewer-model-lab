#!/usr/bin/env python3
"""Optimized local RT-DETR trial for the reviewed FireViewer pointing V5 corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import albumentations as A
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import EvalPrediction


HERE = Path(__file__).resolve().parent
V4_DIR = HERE.parent / "pointing_rtdetr_v4"
V2_DIR = HERE.parent / "pointing_rtdetr_v2"
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(V2_DIR))

import train_rtdetr_pointing_v4 as v4  # noqa: E402

sys.path.insert(0, str(HERE))
from .v5_contract import (  # noqa: E402
    V5ContractError,
    V5DatasetContract,
    load_v5_contract,
    validate_materialized_v5,
)


SEED = 20260824
MODEL_ID = v4.MODEL_ID
MODEL_REVISION = v4.MODEL_REVISION
IMAGE_SIZE = 800
SMOKE_PLAN = ((8, 4), (6, 6), (4, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--gpu-profile", default="rtx5070ti")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2400)
    parser.add_argument("--training-minutes", type=int, default=90)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-6)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def refuse(message: str) -> None:
    raise v4.TrainingRefused(message)


def _within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def validate_arguments(args: argparse.Namespace) -> None:
    fixed = {
        "gpu_profile": "rtx5070ti",
        "seed": SEED,
        "eval_batch_size": 8,
        "workers": 6,
        "prefetch_factor": 2,
        "max_steps": 2400,
        "training_minutes": 90,
        "eval_steps": 150,
        "learning_rate": 7.5e-5,
        "backbone_learning_rate": 3e-6,
        "score_threshold": 0.05,
        "iou_threshold": 0.50,
    }
    for name, expected in fixed.items():
        if getattr(args, name) != expected:
            refuse(f"Optimized V5 trial fixes --{name.replace('_', '-')} at {expected}")
    if args.preflight_only and args.smoke_only:
        refuse("--preflight-only and --smoke-only are mutually exclusive")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    resume_checkpoint = (
        args.resume_from_checkpoint.expanduser().resolve()
        if args.resume_from_checkpoint is not None
        else None
    )
    expected_batch = (6, 6) if resume_checkpoint is not None else (8, 4)
    requested_batch = (args.train_batch_size, args.gradient_accumulation_steps)
    if requested_batch != expected_batch:
        refuse(
            "V5 fresh runs require batch/accumulation 8/4; safe OOM recovery "
            "requires 6/6"
        )
    if resume_checkpoint is not None:
        if resume_checkpoint.parent != output_dir:
            refuse("Resume checkpoint must be a direct child of the immutable run output")
        required_checkpoint_files = (
            "model.safetensors",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
        )
        missing = [
            name
            for name in required_checkpoint_files
            if not (resume_checkpoint / name).is_file()
        ]
        if missing:
            refuse(f"Resume checkpoint is incomplete; missing {missing}")
    if any(
        (
            _within(output_dir, dataset_root),
            _within(dataset_root, output_dir),
            _within(cache_root, dataset_root),
            _within(dataset_root, cache_root),
            _within(output_dir, cache_root),
            _within(cache_root, output_dir),
        )
    ):
        refuse("Dataset, output and cache roots must remain mutually isolated")


def load_v5(
    root: Path,
    cache_root: Path,
    workers: int,
    contract: V5DatasetContract,
) -> DatasetDict:
    dataset = load_dataset(
        "imagefolder",
        data_dir=str(root / "data"),
        cache_dir=str(cache_root / "datasets-arrow-v5"),
    )
    if set(dataset) != set(contract.split_counts):
        refuse(
            f"Loaded V5 splits differ from report: {sorted(dataset)} != "
            f"{sorted(contract.split_counts)}"
        )
    for split, expected in contract.split_counts.items():
        v4.validate_v4_split(dataset[split], expected, split)
        dataset[split] = v4.sanitize_v4_dataset(dataset[split], workers, split)
        if len(dataset[split]) != expected:
            refuse(f"V5 {split} changed cardinality during sanitization")
    return dataset


def build_train_augmentation(seed: int) -> A.Compose:
    """Preserve tiny smoke while retaining modest viewpoint/lighting diversity."""

    return A.Compose(
        [
            A.Affine(
                scale=(0.85, 1.20),
                translate_percent=(-0.06, 0.06),
                rotate=(-3, 3),
                shear=(-1.5, 1.5),
                p=0.35,
            ),
            A.Compose(
                [
                    A.SmallestMaxSize(max_size=900, p=1.0),
                    A.RandomSizedBBoxSafeCrop(
                        height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0
                    ),
                ],
                p=0.18,
                seed=seed,
            ),
            A.OneOf(
                [A.Blur(blur_limit=3, p=0.5), A.MotionBlur(blur_limit=3, p=0.5)],
                p=0.07,
            ),
            A.Perspective(scale=(0.005, 0.025), p=0.04),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.35),
            A.HueSaturationValue(p=0.08),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category"],
            clip=True,
            min_area=4,
            min_visibility=0.15,
        ),
        seed=seed,
    )


def compute_metrics_v5(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    score_threshold: float,
    iou_threshold: float,
) -> Mapping[str, float]:
    result = dict(
        v4.compute_metrics_v4(
            evaluation_results,
            image_processor=image_processor,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
    )
    overall = float(result.get("map", 0.0))
    small = float(result.get("map_small", 0.0))
    if not math.isfinite(overall) or not math.isfinite(small):
        refuse("Validation mAP or small-object AP is non-finite")
    result["pointing_score"] = round(0.70 * overall + 0.30 * small, 6)
    return result


def adaptive_training_smoke(
    model: torch.nn.Module,
    train_dataset: Dataset,
    smoke_plan: tuple[tuple[int, int], ...],
) -> Mapping[str, Any]:
    attempts: list[dict[str, Any]] = []
    for batch_size, accumulation_steps in smoke_plan:
        try:
            metrics = v4._smoke_attempt(model, train_dataset, batch_size)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as error:
            attempts.append(
                {
                    "batch_size": batch_size,
                    "gradient_accumulation_steps": accumulation_steps,
                    "effective_batch_size": batch_size * accumulation_steps,
                    "status": "cuda_oom",
                    "error_type": type(error).__name__,
                }
            )
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        attempts.append(
            {
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps,
                "status": "passed",
                **metrics,
            }
        )
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {
            "schema": "fireviewer.rtdetr-v5-small800-adaptive-smoke.v1",
            "status": "passed",
            "includes_backward": True,
            "attempts": attempts,
            "selected_batch_size": batch_size,
            "selected_gradient_accumulation_steps": accumulation_steps,
            "selected_effective_batch_size": batch_size * accumulation_steps,
        }
    refuse(f"V5 800 px smoke exhausted safe plan {smoke_plan} due to CUDA OOM")
    raise AssertionError("unreachable")


def training_arguments(
    args: argparse.Namespace,
    physical_batch_size: int,
    accumulation_steps: int,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(args.output_dir.expanduser().resolve()),
        run_name="fireviewer-rtdetr-v2-r50-pointing-v5-small800-local-5070ti",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=physical_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        auto_find_batch_size=False,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=1e-4,
        max_grad_norm=1.0,
        max_steps=args.max_steps,
        num_train_epochs=60,
        lr_scheduler_type="cosine",
        warmup_steps=120,
        optim="adamw_torch_fused",
        bf16=True,
        bf16_full_eval=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_pointing_score",
        greater_is_better=True,
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        dataloader_num_workers=args.workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        push_to_hub=False,
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.time()
    validate_arguments(args)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    resume_checkpoint = (
        args.resume_from_checkpoint.expanduser().resolve()
        if args.resume_from_checkpoint is not None
        else None
    )
    if resume_checkpoint is None and output_dir.exists() and any(output_dir.iterdir()):
        refuse(f"Output directory is not empty: {output_dir}")
    if resume_checkpoint is not None and not output_dir.is_dir():
        refuse(f"Resume output directory is missing: {output_dir}")
    resume_tag = (
        f"resume-{resume_checkpoint.name}-b{args.train_batch_size}"
        if resume_checkpoint is not None
        else "fresh"
    )
    smoke_plan = (
        ((6, 6), (4, 8)) if resume_checkpoint is not None else SMOKE_PLAN
    )

    try:
        contract = load_v5_contract(dataset_root)
        materialized = validate_materialized_v5(contract)
    except V5ContractError as error:
        refuse(str(error))

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    preflight: dict[str, Any] = {
        "schema": "fireviewer.rtdetr-v5-small800-preflight.v1",
        "status": "passed",
        "dataset_root": str(dataset_root),
        "dataset_contract": {
            "report_schema": contract.report.get("schema"),
            "report_sha256": contract.report_sha256,
            "selection_manifest_sha256": contract.report.get(
                "selection_manifest_sha256"
            ),
            "selected_count": contract.selected_count,
            "split_counts": contract.split_counts,
            "source_counts": contract.source_counts,
            "extension_selected_count": contract.report.get(
                "extension_selected_count"
            ),
            "extension_visibility_counts": contract.report.get(
                "extension_visibility_counts"
            ),
        },
        "materialized_validation": materialized,
        "runtime_environment": v4.runtime_environment_receipt(),
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "optimization_trial": {
            "image_size": IMAGE_SIZE,
            "selection_metric": "0.70*mAP + 0.30*AP_small",
            "smoke_plan": [
                {
                    "batch_size": batch,
                    "gradient_accumulation_steps": accumulation,
                    "effective_batch_size": batch * accumulation,
                }
                for batch, accumulation in smoke_plan
            ],
            "eval_batch_size": args.eval_batch_size,
            "workers": args.workers,
            "prefetch_factor": args.prefetch_factor,
            "max_steps": args.max_steps,
            "training_minutes": args.training_minutes,
            "eval_steps": args.eval_steps,
            "learning_rate": args.learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "warmup_steps": 120,
            "early_stopping_patience": 5,
            "early_stopping_threshold": 0.0015,
            "resume_from_checkpoint": (
                str(resume_checkpoint) if resume_checkpoint is not None else None
            ),
        },
        "automatic_hub_push": False,
        "automatic_cleanup": False,
    }
    dataset = load_v5(dataset_root, cache_root, args.workers, contract)
    preflight["loaded_dataset_validation"] = {
        "status": "passed",
        "split_counts": {split: len(dataset[split]) for split in dataset},
        "verified_negative_counts": {
            split: sum(len(objects["bbox"]) == 0 for objects in dataset[split]["objects"])
            for split in dataset
        },
    }
    preflight_path = (
        output_dir / f"{resume_tag}-preflight.json"
        if resume_checkpoint is not None
        else output_dir / "preflight.json"
    )
    v4.atomic_json(preflight_path, preflight)
    if args.preflight_only:
        return preflight

    set_seed(args.seed)
    cuda = v4.validate_local_cuda(args.gpu_profile, args.workers)
    common: dict[str, Any] = {
        "revision": MODEL_REVISION,
        "cache_dir": str(cache_root / "models"),
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        common["token"] = token
    labels = {0: "fire", 1: "smoke"}
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        id2label=labels,
        label2id={value: key for key, value in labels.items()},
        **common,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
        **common,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        do_resize=True,
        size={"max_height": IMAGE_SIZE, "max_width": IMAGE_SIZE},
        do_pad=True,
        pad_size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        use_fast=True,
        **common,
    )
    initial_head_sha = v4.model_detection_head_sha256(model)

    fallback = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )
    train_transform = partial(
        v4.transform_v4_batch,
        transform=build_train_augmentation(args.seed),
        fallback=fallback,
        image_processor=image_processor,
    )
    eval_transform = partial(
        v4.transform_v4_batch,
        transform=fallback,
        fallback=fallback,
        image_processor=image_processor,
    )

    dataset["train"] = dataset["train"].shuffle(seed=args.seed)
    raw_test_sample = dataset["test"][0]
    expected_test_sources = materialized["split_source_counts"]["test"]
    partitions, source_filter_receipt = v4.verify_source_filters(
        dataset["test"], expected_test_sources
    )
    source_partition_path = (
        output_dir / f"{resume_tag}-source-test-partitions.json"
        if resume_checkpoint is not None
        else output_dir / "source_test_partitions.json"
    )
    v4.atomic_json(source_partition_path, source_filter_receipt)
    train_dataset = dataset["train"].with_transform(train_transform)
    validation_dataset = dataset["validation"].with_transform(eval_transform)
    test_dataset = dataset["test"].with_transform(eval_transform)

    model = model.cuda().train()
    smoke = adaptive_training_smoke(model, train_dataset, smoke_plan)
    smoke_path = (
        output_dir / f"{resume_tag}-training-smoke.json"
        if resume_checkpoint is not None
        else output_dir / "training_smoke.json"
    )
    v4.atomic_json(smoke_path, smoke)
    if args.smoke_only:
        return smoke

    physical_batch_size = int(smoke["selected_batch_size"])
    accumulation_steps = int(smoke["selected_gradient_accumulation_steps"])
    train_args = training_arguments(args, physical_batch_size, accumulation_steps)
    run_manifest: dict[str, Any] = {
        "schema": "fireviewer.rtdetr-v5-small800-training-run.v1",
        "status": "starting",
        "started_at_unix": started,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "runtime_environment": preflight["runtime_environment"],
        "cuda": cuda,
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "initialization": "immutable COCO base; no V4 checkpoint reuse",
        "resume_from_checkpoint": (
            str(resume_checkpoint) if resume_checkpoint is not None else None
        ),
        "dataset": preflight["dataset_contract"],
        "optimization_trial": preflight["optimization_trial"],
        "arguments": {
            **vars(args),
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "cache_root": str(cache_root),
            "resume_from_checkpoint": (
                str(resume_checkpoint) if resume_checkpoint is not None else None
            ),
        },
        "adaptive_smoke": smoke,
        "selected_physical_batch_size": physical_batch_size,
        "selected_gradient_accumulation_steps": accumulation_steps,
        "selected_effective_batch_size": physical_batch_size * accumulation_steps,
        "initial_detection_head_sha256": initial_head_sha,
        "test_protocol": (
            "threshold selected on validation; exactly one combined held-out test "
            "inference pass; by-source metrics sliced from that pass"
        ),
    }
    run_manifest_path = (
        output_dir / f"{resume_tag}-run-manifest.json"
        if resume_checkpoint is not None
        else output_dir / "run_manifest.json"
    )
    v4.atomic_json(run_manifest_path, run_manifest)

    validation_metric = partial(
        compute_metrics_v5,
        image_processor=image_processor,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
    )
    trainer = v4.DifferentialLRTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        data_collator=v4.collate_fn,
        compute_metrics=validation_metric,
        backbone_learning_rate=args.backbone_learning_rate,
        callbacks=[
            v4.WallClockBudgetCallback(args.training_minutes * 60),
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.0015,
            ),
        ],
    )
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )
    if trainer.state.global_step <= 0:
        refuse("Trainer completed without an optimization step")
    final_head_sha = v4.model_detection_head_sha256(trainer.model)
    if final_head_sha == initial_head_sha:
        refuse("Detection head hash is unchanged after training")
    trainer.save_model(str(output_dir))
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    calibration = trainer.evaluate(validation_dataset, metric_key_prefix="calibration")
    threshold_key = "calibration_operating_best_threshold"
    if threshold_key not in calibration:
        refuse(f"Validation calibration did not produce {threshold_key}")
    selected_threshold = float(calibration[threshold_key])
    if not 0.0 < selected_threshold < 1.0:
        refuse(f"Invalid validation-selected threshold: {selected_threshold}")
    calibration_receipt = {
        "schema": "fireviewer.rtdetr-v5-small800-threshold-calibration.v1",
        "selection_corpus": "validation",
        "selected_threshold": selected_threshold,
        "metrics": calibration,
    }
    v4.atomic_json(output_dir / "threshold_calibration.json", calibration_receipt)

    test_metric = partial(
        compute_metrics_v5,
        image_processor=image_processor,
        score_threshold=selected_threshold,
        iou_threshold=args.iou_threshold,
    )
    trainer.compute_metrics = test_metric
    test_output = trainer.predict(test_dataset, metric_key_prefix="test_combined")
    prediction_digest = v4._prediction_sha256(test_output.predictions)
    held_out_negatives = v4.negative_image_false_positive_metrics(
        EvalPrediction(
            predictions=test_output.predictions,
            label_ids=test_output.label_ids,
        ),
        image_processor=image_processor,
        score_threshold=selected_threshold,
    )
    by_source: dict[str, Any] = {}
    for source, indices in sorted(partitions.items()):
        subset = v4._slice_eval_prediction(
            test_output.predictions,
            test_output.label_ids,
            indices,
        )
        source_metrics = test_metric(subset)
        by_source[source] = {
            f"test_{source}_{key}": value for key, value in source_metrics.items()
        }
    tests = {
        "schema": "fireviewer.rtdetr-v5-small800-held-out-test.v1",
        "selection_corpus": "validation",
        "selected_threshold": selected_threshold,
        "test_inference_passes": 1,
        "test_prediction_sha256": prediction_digest,
        "counts": {"images": len(test_dataset), "by_source": expected_test_sources},
        "combined": dict(test_output.metrics),
        "by_source": by_source,
        "held_out_negatives": held_out_negatives,
    }
    v4.atomic_json(output_dir / "test_metrics.json", tests)

    trainer.model.to("cpu")
    torch.cuda.empty_cache()
    reload_receipt = v4.reload_validation(output_dir, image_processor, raw_test_sample)
    run_manifest.update(
        {
            "status": "completed",
            "completed_at_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "global_step": trainer.state.global_step,
            "final_detection_head_sha256": final_head_sha,
            "train_metrics": train_result.metrics,
            "threshold_calibration": calibration_receipt,
            "test_metrics": tests,
            "reload_validation": reload_receipt,
            "pushed_to_hub": False,
            "automatic_cleanup": False,
        }
    )
    v4.atomic_json(run_manifest_path, run_manifest)
    v4.artifact_manifest(output_dir)
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "selected_threshold": selected_threshold,
        "test_metrics": tests,
    }


def main() -> int:
    try:
        result = run(parse_args())
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except (v4.TrainingRefused, V5ContractError, v4.ContractError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
