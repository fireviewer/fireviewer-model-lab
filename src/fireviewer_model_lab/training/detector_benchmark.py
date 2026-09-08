from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.corpus_pipeline import CLASS_NAMES
from fireviewer_model_lab.training.train_rtdetr import TRAINING_PROFILES, LoadedRecord, load_records

SCHEMA_VERSION = 1
PYRONEAR_MODEL = "pyronear/yolo11s_quick-quokka_v8.0.0"
PYRONEAR_MODEL_REVISION = "086799292e7e84a6ff8bd6394a5011575a6400ea"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _selection_digest(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(entries)).hexdigest()


def _present_classes(record: dict[str, Any], allowed_class_ids: frozenset[int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(annotation["class_id"])
                for annotation in record["annotations"]
                if int(annotation["class_id"]) in allowed_class_ids
            }
        )
    )


def freeze_selection(
    records: list[LoadedRecord],
    *,
    allowed_class_ids: frozenset[int],
    limit: int,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("benchmark selection limit must be positive")
    groups: dict[tuple[str, tuple[int, ...]], list[LoadedRecord]] = defaultdict(list)
    for loaded in records:
        record = loaded.record
        role = str(record.get("corpus_role"))
        split = str(record.get("split"))
        if role == "detector_critical_test":
            benchmark_split = "critical_test"
        elif role == "detector_training" and split in {"validation", "test"}:
            benchmark_split = split
        else:
            continue
        groups[(benchmark_split, _present_classes(record, allowed_class_ids))].append(loaded)
    if not groups:
        raise ValueError("No validation, test or critical-test detector records are available")

    ordered_groups: list[list[LoadedRecord]] = []
    for key in sorted(groups):
        ordered_groups.append(
            sorted(
                groups[key],
                key=lambda loaded: (
                    str(loaded.record["sha256"]),
                    str(loaded.record["sample_id"]),
                ),
            )
        )
    selected: list[LoadedRecord] = []
    index = 0
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if index >= len(group):
                continue
            selected.append(group[index])
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
        index += 1

    present_classes = {
        class_id
        for loaded in selected
        for class_id in _present_classes(loaded.record, allowed_class_ids)
    }
    missing_classes = sorted(allowed_class_ids - present_classes)
    if missing_classes:
        raise ValueError(f"Frozen benchmark is missing classes: {missing_classes}")
    if not any(not loaded.record["annotations"] for loaded in selected):
        raise ValueError("Frozen benchmark must include at least one negative image")

    entries = [
        {
            "sample_id": str(loaded.record["sample_id"]),
            "sha256": str(loaded.record["sha256"]),
            "split": (
                "critical_test"
                if str(loaded.record["corpus_role"]) == "detector_critical_test"
                else str(loaded.record["split"])
            ),
            "width": int(loaded.record["width"]),
            "height": int(loaded.record["height"]),
            "annotations": [
                {
                    "class_id": int(annotation["class_id"]),
                    "class_name": str(annotation["class_name"]),
                    "bbox_xywh": [float(value) for value in annotation["bbox_xywh"]],
                }
                for annotation in loaded.record["annotations"]
                if int(annotation["class_id"]) in allowed_class_ids
            ],
        }
        for loaded in selected
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "class_ids": sorted(allowed_class_ids),
        "selection_sha256": _selection_digest(entries),
        "entries": entries,
    }


def validate_selection(
    selection: dict[str, Any],
    records: list[LoadedRecord],
) -> list[LoadedRecord]:
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported detector benchmark selection schema")
    entries = selection.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Detector benchmark selection is empty")
    class_ids = selection.get("class_ids")
    if (
        not isinstance(class_ids, list)
        or not class_ids
        or any(not isinstance(class_id, int) for class_id in class_ids)
    ):
        raise ValueError("Detector benchmark selection class_ids are missing or invalid")
    allowed_class_ids = frozenset(class_ids)
    if not allowed_class_ids.issubset(CLASS_NAMES):
        raise ValueError("Detector benchmark selection contains unsupported class_ids")
    if selection.get("selection_sha256") != _selection_digest(entries):
        raise ValueError("Detector benchmark selection digest mismatch")
    by_digest = {str(loaded.record["sha256"]): loaded for loaded in records}
    resolved: list[LoadedRecord] = []
    for entry in entries:
        digest = str(entry["sha256"])
        loaded = by_digest.get(digest)
        if loaded is None:
            raise FileNotFoundError(f"Benchmark image is absent from supplied manifests: {digest}")
        if str(loaded.record["sample_id"]) != str(entry["sample_id"]):
            raise ValueError(f"Benchmark sample identity drift for {digest}")
        expected_annotations = [
            {
                "class_id": int(annotation["class_id"]),
                "class_name": str(annotation["class_name"]),
                "bbox_xywh": [float(value) for value in annotation["bbox_xywh"]],
            }
            for annotation in loaded.record["annotations"]
            if int(annotation["class_id"]) in allowed_class_ids
        ]
        if expected_annotations != entry["annotations"]:
            raise ValueError(f"Benchmark annotation drift for {digest}")
        resolved.append(loaded)
    return resolved


def validate_prediction_payload(
    payload: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported detector prediction schema")
    if payload.get("selection_sha256") != selection["selection_sha256"]:
        raise ValueError("Candidate predictions do not target the frozen benchmark")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Candidate predictions must be a list")
    expected = [str(entry["sample_id"]) for entry in selection["entries"]]
    actual = [str(entry["sample_id"]) for entry in predictions]
    if actual != expected:
        raise ValueError("Candidate predictions are missing, reordered or contain extra images")
    for entry in predictions:
        boxes = entry.get("boxes_xyxy")
        labels = entry.get("labels")
        scores = entry.get("scores")
        if (
            not isinstance(boxes, list)
            or not isinstance(labels, list)
            or not isinstance(scores, list)
        ):
            raise ValueError("Each prediction row must contain boxes_xyxy, labels and scores")
        if len(boxes) != len(labels) or len(boxes) != len(scores):
            raise ValueError("Prediction boxes, labels and scores must have identical lengths")
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Candidate predictions must include inference timing")
    for field in ("total_seconds", "images_per_second", "milliseconds_per_image"):
        value = timing.get(field)
        if not isinstance(value, int | float) or not float(value) > 0:
            raise ValueError(f"Candidate inference timing is invalid: {field}")


def _box_iou(left: list[float], right: list[float]) -> float:
    x_min = max(left[0], right[0])
    y_min = max(left[1], right[1])
    x_max = min(left[2], right[2])
    y_max = min(left[3], right[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _operational_detection_metrics(
    selection: dict[str, Any],
    payload: dict[str, Any],
    *,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    targets_by_class: dict[int, int] = defaultdict(int)
    matched_by_class: dict[int, int] = defaultdict(int)
    unmatched_predictions = 0
    negative_images = 0
    negative_images_with_detections = 0
    detections_on_negative_images = 0

    for target_entry, prediction_entry in zip(
        selection["entries"],
        payload["predictions"],
        strict=True,
    ):
        targets = []
        for annotation in target_entry["annotations"]:
            x, y, width, height = annotation["bbox_xywh"]
            class_id = int(annotation["class_id"])
            targets_by_class[class_id] += 1
            targets.append(
                {
                    "class_id": class_id,
                    "box": [x, y, x + width, y + height],
                    "matched": False,
                }
            )

        predictions = sorted(
            (
                {
                    "box": [float(value) for value in box],
                    "class_id": int(class_id),
                    "score": float(score),
                }
                for box, class_id, score in zip(
                    prediction_entry["boxes_xyxy"],
                    prediction_entry["labels"],
                    prediction_entry["scores"],
                    strict=True,
                )
                if float(score) >= score_threshold
            ),
            key=lambda row: row["score"],
            reverse=True,
        )
        if not targets:
            negative_images += 1
            detections_on_negative_images += len(predictions)
            negative_images_with_detections += int(bool(predictions))

        for prediction in predictions:
            candidates = [
                (index, _box_iou(prediction["box"], target["box"]))
                for index, target in enumerate(targets)
                if not target["matched"] and target["class_id"] == prediction["class_id"]
            ]
            if not candidates:
                unmatched_predictions += 1
                continue
            best_index, best_iou = max(candidates, key=lambda item: item[1])
            if best_iou < iou_threshold:
                unmatched_predictions += 1
                continue
            targets[best_index]["matched"] = True
            matched_by_class[prediction["class_id"]] += 1

    recall = {
        CLASS_NAMES[class_id]: round(
            matched_by_class[class_id] / targets_by_class[class_id],
            6,
        )
        if targets_by_class[class_id]
        else None
        for class_id in selection["class_ids"]
    }
    return {
        "thresholds": {
            "score": score_threshold,
            "iou": iou_threshold,
        },
        "recall_by_class": recall,
        "false_positives": {
            "unmatched_detection_count": unmatched_predictions,
            "detections_on_negative_images": detections_on_negative_images,
            "negative_images_with_detections": negative_images_with_detections,
            "negative_image_count": negative_images,
            "negative_image_rate": round(
                negative_images_with_detections / negative_images,
                6,
            )
            if negative_images
            else None,
        },
    }


def score_predictions(
    selection: dict[str, Any],
    candidate_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    targets = []
    for entry in selection["entries"]:
        boxes = []
        labels = []
        for annotation in entry["annotations"]:
            x, y, width, height = annotation["bbox_xywh"]
            boxes.append([x, y, x + width, y + height])
            labels.append(int(annotation["class_id"]))
        targets.append(
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
        )

    candidates: dict[str, Any] = {}
    for payload in candidate_payloads:
        validate_prediction_payload(payload, selection)
        predictions = [
            {
                "boxes": torch.tensor(row["boxes_xyxy"], dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(row["labels"], dtype=torch.int64),
                "scores": torch.tensor(row["scores"], dtype=torch.float32),
            }
            for row in payload["predictions"]
        ]
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        metric.update(predictions, targets)
        raw = metric.compute()
        candidates[str(payload["candidate_id"])] = {
            "map": {
                key: round(float(value.item()), 6)
                for key, value in raw.items()
                if getattr(value, "ndim", 1) == 0
            },
            "operational": _operational_detection_metrics(selection, payload),
            "timing": payload["timing"],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_sha256": selection["selection_sha256"],
        "image_count": len(selection["entries"]),
        "candidates": candidates,
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze and score one detector benchmark shared by all candidates"
    )
    parser.add_argument("command", choices=("freeze", "validate", "score"))
    parser.add_argument("--profile", choices=tuple(TRAINING_PROFILES), default="media_filter_v1")
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--prediction", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    records = load_records(args.manifest, verify_files=args.verify_files)
    if args.command == "freeze":
        selection = freeze_selection(
            records,
            allowed_class_ids=frozenset(TRAINING_PROFILES[args.profile]),
            limit=args.limit,
        )
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        args.selection.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(selection, ensure_ascii=False, sort_keys=True))
        return

    selection = _load_json(args.selection)
    validate_selection(selection, records)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "selection_sha256": selection["selection_sha256"],
                    "image_count": len(selection["entries"]),
                },
                sort_keys=True,
            )
        )
        return

    if not args.prediction or args.output is None:
        raise ValueError("score requires at least one --prediction and --output")
    result = score_predictions(
        selection,
        [_load_json(path) for path in args.prediction],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
