#!/usr/bin/env python3
"""Receipt-bound held-out benchmark for reloadable DEIM-D-FINE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.pointing_dataset_v7.split_registry import load_registry, verify_coco

from .benchmark_shared_v5 import (
    CLASS_NAMES,
    THRESHOLDS,
    atomic_json,
    confidence_interval,
    counts,
    evaluate_model,
    load_split,
    map_metrics,
    sha256_file,
)


PREDICTION_RECEIPT_SCHEMA = "fireviewer.deim-dfine-pointing-v6-heldout-predictions.v1"
REPORT_SCHEMA = "fireviewer.deim-dfine-pointing-v6-heldout-benchmark.v1"
_BOOTSTRAP_STATE: dict[str, Any] | None = None


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_coco(coco_root: Path, split: str) -> dict[str, Any]:
    return json.loads((coco_root / split / "_annotations.coco.json").read_text(encoding="utf-8"))


def split_accounting(coco: dict[str, Any]) -> dict[str, Any]:
    annotated_images = {int(row["image_id"]) for row in coco["annotations"]}
    class_counts = {
        CLASS_NAMES[class_id]: sum(int(row["category_id"]) == class_id for row in coco["annotations"])
        for class_id in CLASS_NAMES
    }
    source_counts: dict[str, int] = defaultdict(int)
    scene_counts: dict[str, int] = defaultdict(int)
    group_counts: dict[str, int] = defaultdict(int)
    for image in coco["images"]:
        source_counts[str(image.get("fireviewer_source_dataset") or "unknown")] += 1
        scene_counts[str(image.get("fireviewer_scene_bin") or "unknown")] += 1
        group = str(image.get("fireviewer_source_group_id") or f"image:{image['id']}")
        group_counts[group] += 1
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "negative_images": len(coco["images"]) - len(annotated_images),
        "class_annotations": dict(sorted(class_counts.items())),
        "source_images": dict(sorted(source_counts.items())),
        "scene_bin_images": dict(sorted(scene_counts.items())),
        "source_groups": len(group_counts),
        "multi_image_source_groups": sum(size > 1 for size in group_counts.values()),
        "max_source_group_images": max(group_counts.values(), default=0),
    }


def validate_prediction_receipt(
    *,
    model: str,
    split: str,
    prediction_path: Path,
    annotation_sha256: str,
    expected_images: int,
    receipt_schema: str = PREDICTION_RECEIPT_SCHEMA,
) -> dict[str, Any]:
    receipt_path = prediction_path.with_suffix(".receipt.json")
    if not prediction_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(prediction_path if not prediction_path.is_file() else receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema": receipt_schema,
        "status": "completed",
        "model": model,
        "split": split,
        "annotation_sha256": annotation_sha256,
        "images": expected_images,
        "split_images": expected_images,
        "complete_split": True,
        "candidate_threshold": 0.001,
        "max_detections_per_image": 100,
    }
    mismatches = {
        key: {"expected": expected, "observed": receipt.get(key)}
        for key, expected in required.items()
        if receipt.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Prediction receipt mismatch for {model}/{split}: {mismatches}")
    if receipt.get("prediction_sha256") != sha256_file(prediction_path):
        raise RuntimeError(f"Prediction hash mismatch for {model}/{split}")
    checkpoint = Path(str(receipt.get("checkpoint", "")))
    if not checkpoint.is_file() or receipt.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError(f"Checkpoint identity mismatch for {model}/{split}")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    unexpected = sorted({int(row["category_id"]) for row in predictions} - set(CLASS_NAMES))
    if unexpected:
        raise RuntimeError(f"Non-comparable prediction categories for {model}/{split}: {unexpected}")
    return receipt


def validate_corpus_counts(coco: dict, expected: dict, split: str) -> None:
    actual = split_accounting(coco)
    for field, actual_field in (("images", "images"), ("annotations", "annotations"),
                                ("negative_images", "negative_images"), ("class_counts", "class_annotations")):
        if actual[actual_field] != expected[field]:
            raise RuntimeError(f"Dataset {field} mismatch for {split}")


def metric_vector(mapped: dict[str, Any], operating: dict[str, Any]) -> dict[str, float]:
    vector = {
        key: float(mapped[key])
        for key in (
            "map",
            "map_50",
            "map_75",
            "map_small",
            "map_medium",
            "map_large",
            "mar_100",
            "pointing_score",
            "map_fire",
            "map_smoke",
            "mar_100_fire",
            "mar_100_smoke",
        )
    }
    for key in ("precision", "recall", "f1"):
        vector[key] = float(operating[key])
        for class_name in CLASS_NAMES.values():
            vector[f"{key}_{class_name}"] = float(operating["per_class"][class_name][key])
    return vector


def grouped_breakdown(
    image_rows: list[dict[str, Any]],
    targets: list[dict[str, torch.Tensor]],
    predictions: list[dict[str, torch.Tensor]],
    threshold: float,
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(image_rows):
        grouped[str(image.get(field) or "unknown")].append(index)
    result: dict[str, Any] = {}
    for name, indices in sorted(grouped.items()):
        subset_targets = [targets[index] for index in indices]
        subset_predictions = [predictions[index] for index in indices]
        result[name] = {
            "images": len(indices),
            "map": map_metrics(subset_targets, subset_predictions),
            "operating_iou_50": counts(subset_targets, subset_predictions, threshold, iou_threshold=0.5),
            "operating_iou_75": counts(subset_targets, subset_predictions, threshold, iou_threshold=0.75),
        }
    return result


def cluster_bootstrap_indices(image_rows: list[dict[str, Any]], rng: np.random.Generator) -> tuple[list[int], int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(image_rows):
        group = str(image.get("fireviewer_source_group_id") or f"image:{image['id']}")
        groups[group].append(index)
    names = sorted(groups)
    sampled_names = rng.choice(names, size=len(names), replace=True).tolist()
    indices = [index for name in sampled_names for index in groups[str(name)]]
    return indices, len(names)


def initialize_bootstrap_worker(
    image_rows: list[dict[str, Any]],
    targets: list[dict[str, torch.Tensor]],
    predictions: dict[str, list[dict[str, torch.Tensor]]],
    thresholds: dict[str, float],
) -> None:
    global _BOOTSTRAP_STATE
    torch.set_num_threads(1)
    _BOOTSTRAP_STATE = {
        "image_rows": image_rows,
        "targets": targets,
        "predictions": predictions,
        "thresholds": thresholds,
    }


def bootstrap_worker(seed: int) -> dict[str, dict[str, float]]:
    if _BOOTSTRAP_STATE is None:
        raise RuntimeError("Bootstrap worker was not initialized")
    rng = np.random.default_rng(seed)
    indices, _ = cluster_bootstrap_indices(_BOOTSTRAP_STATE["image_rows"], rng)
    sampled_targets = [_BOOTSTRAP_STATE["targets"][index] for index in indices]
    sampled_vectors: dict[str, dict[str, float]] = {}
    for model, model_predictions in _BOOTSTRAP_STATE["predictions"].items():
        sampled_predictions = [model_predictions[index] for index in indices]
        mapped = map_metrics(sampled_targets, sampled_predictions)
        operating = counts(
            sampled_targets,
            sampled_predictions,
            _BOOTSTRAP_STATE["thresholds"][model],
            iou_threshold=0.5,
        )
        sampled_vectors[model] = metric_vector(mapped, operating)
    return sampled_vectors


def joint_cluster_bootstrap(
    *,
    image_rows: list[dict[str, Any]],
    targets: list[dict[str, torch.Tensor]],
    predictions: dict[str, list[dict[str, torch.Tensor]]],
    thresholds: dict[str, float],
    candidate_model: str,
    baseline_model: str | None,
    full_vectors: dict[str, dict[str, float]],
    resamples: int,
    seed: int,
    workers: int,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    distributions: dict[str, dict[str, list[float]]] = {
        model: defaultdict(list) for model in predictions
    }
    delta_distributions: dict[str, list[float]] = defaultdict(list)
    _, cluster_count = cluster_bootstrap_indices(image_rows, np.random.default_rng(seed))
    progress_interval = max(1, resamples // 20)

    def record(sampled_vectors: dict[str, dict[str, float]]) -> None:
        for model, vector in sampled_vectors.items():
            for key, value in vector.items():
                distributions[model][key].append(value)
        if baseline_model is not None:
            for key, value in sampled_vectors[candidate_model].items():
                delta_distributions[key].append(value - sampled_vectors[baseline_model][key])

    child_sequences = np.random.SeedSequence(seed).spawn(resamples)
    child_seeds = [int(sequence.generate_state(1, dtype=np.uint64)[0]) for sequence in child_sequences]
    workers = max(1, min(workers, resamples))
    if workers == 1:
        initialize_bootstrap_worker(image_rows, targets, predictions, thresholds)
        for iteration, child_seed in enumerate(child_seeds, start=1):
            record(bootstrap_worker(child_seed))
            if iteration % progress_interval == 0 or iteration == resamples:
                print(f"cluster bootstrap {iteration}/{resamples}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initialize_bootstrap_worker,
            initargs=(image_rows, targets, predictions, thresholds),
        ) as executor:
            futures = [executor.submit(bootstrap_worker, child_seed) for child_seed in child_seeds]
            for iteration, future in enumerate(as_completed(futures), start=1):
                record(future.result())
                if iteration % progress_interval == 0 or iteration == resamples:
                    print(f"cluster bootstrap {iteration}/{resamples}", flush=True)

    result: dict[str, Any] = {
        "method": "paired source-group cluster percentile bootstrap with replacement",
        "cluster_field": "fireviewer_source_group_id",
        "clusters": cluster_count,
        "seed": seed,
        "resamples": resamples,
        "workers": workers,
        "models": {
            model: {key: confidence_interval(values) for key, values in metrics.items()}
            for model, metrics in distributions.items()
        },
    }
    if baseline_model is not None:
        point_estimate = {
            key: full_vectors[candidate_model][key] - full_vectors[baseline_model][key]
            for key in full_vectors[candidate_model]
        }
        result["paired_delta"] = {
            "candidate": candidate_model,
            "baseline": baseline_model,
            "definition": "candidate minus baseline on identical held-out images",
            "point_estimate": point_estimate,
            "intervals": {key: confidence_interval(values) for key, values in delta_distributions.items()},
        }
    return result


def write_threshold_sweep(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "fire_tp",
        "fire_fp",
        "fire_fn",
        "fire_precision",
        "fire_recall",
        "fire_f1",
        "smoke_tp",
        "smoke_fp",
        "smoke_fn",
        "smoke_precision",
        "smoke_recall",
        "smoke_f1",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in results:
                for threshold in THRESHOLDS:
                    item = result["validation"]["threshold_sweep"][str(threshold)]
                    row: dict[str, Any] = {"model": result["model"], "threshold": threshold}
                    for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                        row[key] = item[key]
                        row[f"fire_{key}"] = item["per_class"]["fire"][key]
                        row[f"smoke_{key}"] = item["per_class"]["smoke"][key]
                    writer.writerow(row)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_prediction_audit(
    path: Path,
    loaded: dict[str, tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]],
    thresholds: dict[str, float],
) -> None:
    lines: list[str] = []
    for model, (images, targets, predictions) in loaded.items():
        for image, target, prediction in zip(images, targets, predictions, strict=True):
            threshold = thresholds[model]
            keep = prediction["scores"] >= threshold
            record = {
                "model": model,
                "image_id": int(image["id"]),
                "file_name": image["file_name"],
                "image_sha256": image.get("fireviewer_sha256"),
                "source_dataset": image.get("fireviewer_source_dataset"),
                "source_group_id": image.get("fireviewer_source_group_id"),
                "scene_bin": image.get("fireviewer_scene_bin"),
                "ground_truth_boxes": len(target["labels"]),
                "candidate_predictions": len(prediction["scores"]),
                "selected_threshold": threshold,
                "predictions_at_threshold": int(keep.sum().item()),
                "operating_iou_50": counts([target], [prediction], threshold, iou_threshold=0.5),
                "operating_iou_75": counts([target], [prediction], threshold, iou_threshold=0.75),
            }
            lines.append(json.dumps(record, sort_keys=True))
    atomic_text(path, "\n".join(lines) + "\n")


def markdown_report(report: dict[str, Any]) -> str:
    candidate = report["protocol"]["candidate_model"]
    baseline = report["protocol"].get("baseline_model")
    lines = [
        f"# {report['protocol']['report_title']}",
        "",
        "## Results",
        "",
        "| Model | Threshold | mAP | AP50 | AP75 | AP small | AP fire | AP smoke | Precision | Recall | F1 | TP | FP | FN | FPS | P95 ms | VRAM GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["models"]:
        mapped = result["test"]["map"]
        operating = result["test"]["operating"]
        performance = result["performance"]
        lines.append(
            f"| {result['model']} | {result['selected_threshold']:.3f} | {mapped['map']:.4f} | "
            f"{mapped['map_50']:.4f} | {mapped['map_75']:.4f} | {mapped['map_small']:.4f} | "
            f"{mapped['map_fire']:.4f} | {mapped['map_smoke']:.4f} | {operating['precision']:.4f} | "
            f"{operating['recall']:.4f} | {operating['f1']:.4f} | {operating['tp']} | {operating['fp']} | "
            f"{operating['fn']} | {performance['fps_batch1']:.2f} | "
            f"{performance['latency_batch1_ms']['p95']:.2f} | {performance['peak_vram_gib']:.2f} |"
        )
    comparison = report.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                f"## Paired {candidate} versus {baseline}",
                "",
                "The intervals are paired source-group cluster bootstrap intervals on the identical held-out test split.",
                "",
                f"| Metric | Delta {candidate} - {baseline} | 95% lower | 95% upper |",
                "|---|---:|---:|---:|",
            ]
        )
        for key in ("map", "map_50", "map_75", "map_small", "map_fire", "map_smoke", "precision", "recall", "f1"):
            interval = comparison["intervals"][key]
            lines.append(
                f"| {key} | {comparison['point_estimate'][key]:.4f} | "
                f"{interval['lower_95']:.4f} | {interval['upper_95']:.4f} |"
            )
    coverage = report["corpus"]["test"]
    lines.extend(
        [
            "",
            "## Corpus accounting",
            "",
            f"- Test images: {coverage['images']}",
            f"- Test annotations: {coverage['annotations']}",
            f"- Fire annotations: {coverage['class_annotations']['fire']}",
            f"- Smoke annotations: {coverage['class_annotations']['smoke']}",
            f"- Source groups: {coverage['source_groups']}",
            f"- Negative test images: {coverage['negative_images']}",
            "",
            "## Declared limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "VALID_PREDICTIONS", "TEST_PREDICTIONS"),
        required=True,
    )
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-model")
    parser.add_argument("--split-registry", type=Path)
    parser.add_argument("--prediction-receipt-schema", default=PREDICTION_RECEIPT_SCHEMA)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-workers", type=int, default=8)
    parser.add_argument(
        "--report-title",
        default="FireViewer pointing V6 — held-out D-FINE benchmark",
    )
    args = parser.parse_args()

    coco_root = args.coco_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_specs = {
        name: (Path(validation).resolve(), Path(test).resolve())
        for name, validation, test in args.model
    }
    if len(model_specs) != len(args.model):
        raise RuntimeError("Duplicate benchmark model name")
    if args.candidate_model not in model_specs:
        raise RuntimeError("Candidate model is missing from --model specifications")
    if args.baseline_model is not None and args.baseline_model not in model_specs:
        raise RuntimeError("Baseline model is missing from --model specifications")

    dataset_receipt_path = coco_root / "coco_view_receipt.json"
    dataset_receipt = json.loads(dataset_receipt_path.read_text(encoding="utf-8"))
    if dataset_receipt.get("status") != "ready":
        raise RuntimeError("COCO receipt is not ready")
    if "-v7-" in dataset_receipt.get("schema", "") and not args.split_registry:
        raise RuntimeError("V7 benchmark requires a persistent split registry")
    split_integrity = verify_coco(coco_root, load_registry(args.split_registry)) if args.split_registry else None
    if split_integrity and split_integrity["status"] != "passed":
        raise ValueError("Unresolved source group conflicts: refusing held-out benchmark")
    coco = {split: load_coco(coco_root, split) for split in ("valid", "test")}
    annotation_paths = {
        split: coco_root / split / "_annotations.coco.json" for split in ("valid", "test")
    }
    annotation_hashes = {split: sha256_file(path) for split, path in annotation_paths.items()}
    for split in ("valid", "test"):
        receipt_key = "validation" if split == "valid" else split
        expected = dataset_receipt["splits"][receipt_key]
        if expected["annotation_sha256"] != annotation_hashes[split]:
            raise RuntimeError(f"Dataset annotation hash mismatch for {split}")
        validate_corpus_counts(coco[split], expected, split)

    validated_receipts: dict[str, dict[str, Any]] = {}
    for model, (validation_path, test_path) in model_specs.items():
        validated_receipts[model] = {
            "validation": validate_prediction_receipt(
                model=model,
                split="valid",
                prediction_path=validation_path,
                annotation_sha256=annotation_hashes["valid"],
                expected_images=len(coco["valid"]["images"]),
                receipt_schema=args.prediction_receipt_schema,
            ),
            "test": validate_prediction_receipt(
                model=model,
                split="test",
                prediction_path=test_path,
                annotation_sha256=annotation_hashes["test"],
                expected_images=len(coco["test"]["images"]),
                receipt_schema=args.prediction_receipt_schema,
            ),
        }
        if validated_receipts[model]["validation"]["checkpoint_sha256"] != validated_receipts[model]["test"]["checkpoint_sha256"]:
            raise RuntimeError(f"Validation and test checkpoints differ for {model}")

    results: list[dict[str, Any]] = []
    loaded: dict[str, tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]] = {}
    thresholds: dict[str, float] = {}
    full_vectors: dict[str, dict[str, float]] = {}
    for model, (validation_path, test_path) in model_specs.items():
        result = evaluate_model(coco_root, model, validation_path, test_path, 0)
        images, targets, predictions = load_split(coco_root, "test", test_path)
        threshold = float(result["selected_threshold"])
        result["test"]["operating_iou_75"] = counts(targets, predictions, threshold, iou_threshold=0.75)
        result["test"]["by_scene_bin"] = grouped_breakdown(
            images, targets, predictions, threshold, "fireviewer_scene_bin"
        )
        loaded[model] = (images, targets, predictions)
        thresholds[model] = threshold
        full_vectors[model] = metric_vector(result["test"]["map"], result["test"]["operating"])
        results.append(result)

    reference_images, reference_targets, _ = loaded[args.candidate_model]
    for model, (images, targets, _) in loaded.items():
        if [int(row["id"]) for row in images] != [int(row["id"]) for row in reference_images]:
            raise RuntimeError(f"Held-out image ordering differs for {model}")
        if len(targets) != len(reference_targets):
            raise RuntimeError(f"Held-out target count differs for {model}")
    bootstrap = joint_cluster_bootstrap(
        image_rows=reference_images,
        targets=reference_targets,
        predictions={model: item[2] for model, item in loaded.items()},
        thresholds=thresholds,
        candidate_model=args.candidate_model,
        baseline_model=args.baseline_model,
        full_vectors=full_vectors,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
        workers=args.bootstrap_workers,
    )
    for result in results:
        result["test"]["confidence_intervals"] = bootstrap["models"][result["model"]]

    report = {
        "schema": "fireviewer.deim-dfine-pointing-heldout-benchmark.v2" if args.split_registry else REPORT_SCHEMA,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "report_title": args.report_title,
            "candidate_model": args.candidate_model,
            "baseline_model": args.baseline_model,
            "checkpoint_reloaded": True,
            "same_validation_and_test_corpus_for_all_models": True,
            "candidate_threshold": 0.001,
            "max_detections_per_image": 100,
            "threshold_selected_on_validation_only": True,
            "threshold_objective": "maximum class-aware micro F1 at IoU 0.50; recall then lower threshold tie-break",
            "held_out_test_passes_per_model": 1,
            "coco_map_iou_range": "0.50:0.95",
            "operating_iou_thresholds": [0.5, 0.75],
            "bootstrap": {key: value for key, value in bootstrap.items() if key not in {"models", "paired_delta"}},
            "dataset_receipt": str(dataset_receipt_path),
            "dataset_receipt_sha256": sha256_file(dataset_receipt_path),
            "dataset_report_sha256": dataset_receipt["dataset_report_sha256"],
            "selection_manifest_sha256": dataset_receipt["selection_manifest_sha256"],
            "annotation_sha256": annotation_hashes,
            "split_integrity": split_integrity,
            "split_registry_sha256": sha256_file(args.split_registry) if args.split_registry else None,
        },
        "corpus": {split: split_accounting(coco[split]) for split in ("valid", "test")},
        "models": results,
        "comparison": bootstrap.get("paired_delta"),
        "prediction_receipts": validated_receipts,
        "limitations": [
            "This is an internal FireViewer test split, not a newly acquired external corpus.",
            "Exact-byte and declared-group independence does not establish independence of all physical fire events.",
            "Negative-image false-positive rates are reported only on the negatives actually present in each split.",
            "Threshold calibration uses validation only; the test split is evaluated once at the selected threshold.",
            "Throughput is local RTX 5070 Ti batch-1 end-to-end timing and is hardware-specific.",
        ],
    }
    report_path = output_dir / "benchmark_report.json"
    markdown_path = output_dir / "benchmark_report.md"
    sweep_path = output_dir / "threshold_sweep.csv"
    audit_path = output_dir / "test_prediction_audit.jsonl"
    validation_path = output_dir / "benchmark_validation.json"
    atomic_json(report_path, report)
    atomic_text(markdown_path, markdown_report(report))
    write_threshold_sweep(sweep_path, results)
    write_prediction_audit(audit_path, loaded, thresholds)

    finite_metrics = all(
        math.isfinite(float(result["test"]["map"][key]))
        for result in results
        for key in ("map", "map_50", "map_75", "map_small", "map_fire", "map_smoke")
    )
    gates = {
        "report_completed": report["status"] == "completed",
        "validation_images_complete": report["corpus"]["valid"]["images"] == dataset_receipt["splits"]["validation"]["images"],
        "test_images_complete": report["corpus"]["test"]["images"] == dataset_receipt["splits"]["test"]["images"],
        "test_annotations_complete": report["corpus"]["test"]["annotations"] == dataset_receipt["splits"]["test"]["annotations"],
        "persistent_splits_verified": split_integrity is not None if args.split_registry else True,
        "prediction_receipts_validated": len(validated_receipts) == len(model_specs),
        "metrics_finite": finite_metrics,
        "bootstrap_resamples_complete": bootstrap["resamples"] == args.bootstrap_resamples,
        "prediction_audit_rows_complete": sum(1 for _ in audit_path.open("r", encoding="utf-8"))
        == len(coco["test"]["images"]) * len(model_specs),
    }
    validation = {
        "schema": "fireviewer.deim-dfine-pointing-benchmark-validation.v2",
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        "automatic_cleanup": False,
    }
    atomic_json(validation_path, validation)
    if validation["status"] != "passed":
        raise RuntimeError(f"Benchmark artifact validation failed: {gates}")

    artifact_paths = [report_path, markdown_path, sweep_path, audit_path, validation_path]
    for validation_predictions, test_predictions in model_specs.values():
        artifact_paths.extend(
            [
                validation_predictions,
                validation_predictions.with_suffix(".receipt.json"),
                test_predictions,
                test_predictions.with_suffix(".receipt.json"),
            ]
        )
    artifact_paths.extend([dataset_receipt_path, *annotation_paths.values(), Path(__file__).resolve()])
    if args.split_registry:
        artifact_paths.append(args.split_registry.resolve())
    manifest = {
        "schema": "fireviewer.deim-dfine-pointing-benchmark-artifacts.v2",
        "status": "completed",
        "automatic_cleanup": False,
        "retain_until_user_review": True,
        "artifacts": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifact_paths
        ],
    }
    atomic_json(output_dir / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "models": {
                    result["model"]: {
                        "selected_threshold": result["selected_threshold"],
                        "test_map": result["test"]["map"],
                        "test_operating": result["test"]["operating"],
                    }
                    for result in results
                },
                "comparison": report["comparison"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
