#!/usr/bin/env python3
"""One architecture-neutral benchmark for RF-DETR and DEIM-D-FINE on V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


THRESHOLDS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CLASS_NAMES = {0: "fire", 1: "smoke"}
SEED = 20260824


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def tensor_boxes(rows: list[list[float]]) -> torch.Tensor:
    if not rows:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)


def load_split(coco_root: Path, split: str, prediction_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    coco = json.loads((coco_root / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    by_image_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_image_pred: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image_gt[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        image_id = int(prediction["image_id"])
        category = int(prediction["category_id"])
        if category not in CLASS_NAMES:
            raise RuntimeError(f"Prediction has non-comparable category {category}")
        by_image_pred[image_id].append(prediction)

    image_rows = sorted(coco["images"], key=lambda row: int(row["id"]))
    targets: list[dict[str, torch.Tensor]] = []
    preds: list[dict[str, torch.Tensor]] = []
    known_ids = {int(row["id"]) for row in image_rows}
    if set(by_image_pred) - known_ids:
        raise RuntimeError("Predictions contain image ids outside the benchmark split")
    for image in image_rows:
        image_id = int(image["id"])
        gt_rows = by_image_gt[image_id]
        pred_rows = sorted(by_image_pred[image_id], key=lambda row: float(row["score"]), reverse=True)[:100]
        targets.append(
            {
                "boxes": tensor_boxes([xywh_to_xyxy(row["bbox"]) for row in gt_rows]),
                "labels": torch.tensor([int(row["category_id"]) for row in gt_rows], dtype=torch.int64),
            }
        )
        preds.append(
            {
                "boxes": tensor_boxes([xywh_to_xyxy(row["bbox"]) for row in pred_rows]),
                "labels": torch.tensor([int(row["category_id"]) for row in pred_rows], dtype=torch.int64),
                "scores": torch.tensor([float(row["score"]) for row in pred_rows], dtype=torch.float32),
            }
        )
    return image_rows, targets, preds


def map_metrics(targets: list[dict[str, torch.Tensor]], predictions: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True, max_detection_thresholds=[1, 10, 100])
    metric.update(predictions, targets)
    raw = metric.compute()
    result = {
        key: float(raw[key].item())
        for key in ("map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100", "mar_small", "mar_medium", "mar_large")
    }
    classes = raw["classes"].tolist()
    for class_id, value in zip(classes, raw["map_per_class"].tolist(), strict=True):
        result[f"map_{CLASS_NAMES[int(class_id)]}"] = float(value)
    for class_id, value in zip(classes, raw["mar_100_per_class"].tolist(), strict=True):
        result[f"mar_100_{CLASS_NAMES[int(class_id)]}"] = float(value)
    result["pointing_score"] = 0.70 * result["map"] + 0.30 * max(0.0, result["map_small"])
    return result


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    box_area = (box[2:] - box[:2]).clamp(min=0).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (box_area + areas - intersection).clamp(min=1e-9)


def counts(
    targets: list[dict[str, torch.Tensor]],
    predictions: list[dict[str, torch.Tensor]],
    threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    total = {"tp": 0, "fp": 0, "fn": 0}
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES.values()}
    matched_ious: list[float] = []
    matched_ious_per_class: dict[str, list[float]] = {name: [] for name in CLASS_NAMES.values()}
    negative_images = 0
    negative_with_prediction = 0
    negative_prediction_count = 0
    for target, prediction in zip(targets, predictions, strict=True):
        keep = prediction["scores"] >= threshold
        boxes = prediction["boxes"][keep]
        labels = prediction["labels"][keep]
        scores = prediction["scores"][keep]
        order = torch.argsort(scores, descending=True)
        boxes, labels = boxes[order], labels[order]
        if len(target["labels"]) == 0:
            negative_images += 1
            negative_prediction_count += len(labels)
            negative_with_prediction += int(len(labels) > 0)
        for class_id, class_name in CLASS_NAMES.items():
            gt_boxes = target["boxes"][target["labels"] == class_id]
            pred_boxes = boxes[labels == class_id]
            matched: set[int] = set()
            tp = 0
            fp = 0
            for predicted_box in pred_boxes:
                ious = box_iou(predicted_box, gt_boxes)
                if ious.numel() == 0:
                    fp += 1
                    continue
                values, indices = torch.sort(ious, descending=True)
                match = next(
                    (
                        (int(index), float(value))
                        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
                        if value >= iou_threshold and int(index) not in matched
                    ),
                    None,
                )
                if match is None:
                    fp += 1
                else:
                    matched.add(match[0])
                    matched_ious.append(match[1])
                    matched_ious_per_class[class_name].append(match[1])
                    tp += 1
            fn = len(gt_boxes) - tp
            per_class[class_name]["tp"] += tp
            per_class[class_name]["fp"] += fp
            per_class[class_name]["fn"] += fn
            total["tp"] += tp
            total["fp"] += fp
            total["fn"] += fn

    def rates(item: dict[str, int]) -> dict[str, float | int]:
        precision = item["tp"] / (item["tp"] + item["fp"]) if item["tp"] + item["fp"] else 0.0
        recall = item["tp"] / (item["tp"] + item["fn"]) if item["tp"] + item["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**item, "precision": precision, "recall": recall, "f1": f1}

    def localization(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "matched": 0,
                "mean_iou": None,
                "median_iou": None,
                "p10_iou": None,
                "p25_iou": None,
                "p75_iou": None,
                "p90_iou": None,
                "fraction_iou_ge_0_75": None,
            }
        array = np.asarray(values, dtype=np.float64)
        return {
            "matched": len(values),
            "mean_iou": float(array.mean()),
            "median_iou": float(np.quantile(array, 0.50)),
            "p10_iou": float(np.quantile(array, 0.10)),
            "p25_iou": float(np.quantile(array, 0.25)),
            "p75_iou": float(np.quantile(array, 0.75)),
            "p90_iou": float(np.quantile(array, 0.90)),
            "fraction_iou_ge_0_75": float((array >= 0.75).mean()),
        }

    return {
        **rates(total),
        "iou_threshold": iou_threshold,
        "localization": localization(matched_ious),
        "per_class": {
            name: {**rates(value), "localization": localization(matched_ious_per_class[name])}
            for name, value in per_class.items()
        },
        "negative_images": negative_images,
        "negative_images_with_prediction": negative_with_prediction,
        "negative_image_false_positive_rate": negative_with_prediction / negative_images if negative_images else None,
        "false_positive_detections_on_negatives": negative_prediction_count,
    }


def confidence_interval(values: list[float]) -> dict[str, float]:
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def bootstrap(targets: list[dict[str, torch.Tensor]], predictions: list[dict[str, torch.Tensor]], threshold: float, resamples: int) -> dict[str, Any]:
    if resamples <= 0:
        return {
            "method": "image-level bootstrap with replacement",
            "seed": SEED,
            "resamples": 0,
            "intervals": {},
        }
    rng = np.random.default_rng(SEED)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        indices = rng.integers(0, len(targets), size=len(targets)).tolist()
        sampled_targets = [targets[index] for index in indices]
        sampled_predictions = [predictions[index] for index in indices]
        mapped = map_metrics(sampled_targets, sampled_predictions)
        operating = counts(sampled_targets, sampled_predictions, threshold)
        for key in ("map", "map_50", "map_75", "map_small", "pointing_score"):
            distributions[key].append(mapped[key])
        for key in ("precision", "recall", "f1"):
            distributions[key].append(float(operating[key]))
    return {
        "method": "image-level bootstrap with replacement",
        "seed": SEED,
        "resamples": resamples,
        "intervals": {key: confidence_interval(values) for key, values in distributions.items()},
    }


def evaluate_model(coco_root: Path, model: str, validation_path: Path, test_path: Path, resamples: int) -> dict[str, Any]:
    val_images, val_targets, val_predictions = load_split(coco_root, "valid", validation_path)
    sweep = {str(threshold): counts(val_targets, val_predictions, threshold) for threshold in THRESHOLDS}
    selected_threshold = max(THRESHOLDS, key=lambda threshold: (sweep[str(threshold)]["f1"], sweep[str(threshold)]["recall"], -threshold))
    test_images, test_targets, test_predictions = load_split(coco_root, "test", test_path)
    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, image in enumerate(test_images):
        source_indices[str(image.get("fireviewer_source_dataset") or "unknown")].append(index)
    test_map = map_metrics(test_targets, test_predictions)
    test_operating = counts(test_targets, test_predictions, selected_threshold)
    by_source = {}
    for source, indices in sorted(source_indices.items()):
        source_targets = [test_targets[index] for index in indices]
        source_predictions = [test_predictions[index] for index in indices]
        by_source[source] = {
            "images": len(indices),
            "map": map_metrics(source_targets, source_predictions),
            "operating": counts(source_targets, source_predictions, selected_threshold),
        }
    validation_receipt = json.loads(validation_path.with_suffix(".receipt.json").read_text(encoding="utf-8"))
    test_receipt = json.loads(test_path.with_suffix(".receipt.json").read_text(encoding="utf-8"))
    if validation_receipt["prediction_sha256"] != sha256_file(validation_path) or test_receipt["prediction_sha256"] != sha256_file(test_path):
        raise RuntimeError(f"Prediction receipt hash mismatch for {model}")
    return {
        "model": model,
        "selection_corpus": "validation",
        "selected_threshold": selected_threshold,
        "validation": {
            "images": len(val_images),
            "map": map_metrics(val_targets, val_predictions),
            "threshold_sweep": sweep,
        },
        "test": {
            "images": len(test_images),
            "map": test_map,
            "operating": test_operating,
            "by_source": by_source,
            "confidence_intervals": bootstrap(test_targets, test_predictions, selected_threshold, resamples),
        },
        "performance": test_receipt,
        "prediction_receipts": {"validation": validation_receipt, "test": test_receipt},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    args = parser.parse_args()
    coco_root = args.coco_root.resolve()
    predictions_root = args.predictions_root.resolve()
    output_dir = args.output_dir.resolve()
    models = {
        "rf-detr-large": (predictions_root / "rf-detr-large-valid.json", predictions_root / "rf-detr-large-test.json"),
        "deim-dfine-large": (predictions_root / "deim-dfine-large-valid.json", predictions_root / "deim-dfine-large-test.json"),
    }
    for paths in models.values():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    receipt = json.loads((coco_root / "coco_view_receipt.json").read_text(encoding="utf-8"))
    results = [evaluate_model(coco_root, model, *paths, args.bootstrap_resamples) for model, paths in models.items()]
    report = {
        "schema": "fireviewer.pointing-v5-shared-detector-benchmark.v1",
        "status": "completed",
        "protocol": {
            "same_corpus": True,
            "same_splits": True,
            "same_candidate_threshold": 0.001,
            "same_max_detections_per_image": 100,
            "same_iou_threshold": 0.5,
            "threshold_selected_on_validation_only": True,
            "held_out_test_passes_per_model": 1,
            "dataset_report_sha256": receipt["dataset_report_sha256"],
            "selection_manifest_sha256": receipt["selection_manifest_sha256"],
        },
        "models": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "benchmark_report.json", report)
    lines = [
        "# FireViewer pointing V5 — shared detector benchmark",
        "",
        "| Model | mAP | AP50 | AP75 | AP small | Precision | Recall | F1 | FPS batch 1 | Peak VRAM GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        mapped = result["test"]["map"]
        operating = result["test"]["operating"]
        performance = result["performance"]
        lines.append(
            f"| {result['model']} | {mapped['map']:.4f} | {mapped['map_50']:.4f} | {mapped['map_75']:.4f} | "
            f"{mapped['map_small']:.4f} | {operating['precision']:.4f} | {operating['recall']:.4f} | "
            f"{operating['f1']:.4f} | {performance['fps_batch1']:.2f} | {performance['peak_vram_gib']:.2f} |"
        )
    (output_dir / "benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
