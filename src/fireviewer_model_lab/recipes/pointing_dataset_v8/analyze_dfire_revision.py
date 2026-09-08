"""Rank a D-Fire review queue using an existing detector as a triage signal."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8.prepare_dfire_revision import render_packets, write_json, write_rows


def iou_xywh(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = left
    rx1, ry1, rw, rh = right
    lx2, ly2, rx2, ry2 = lx1 + lw, ly1 + lh, rx1 + rw, ry1 + rh
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def analyze(rows: list[dict], coco: dict, predictions: list[dict], support_score: float = 0.3,
            extra_score: float = 0.5, match_iou: float = 0.3) -> list[dict]:
    images = {int(image["id"]): image for image in coco["images"]}
    sha_to_row = {row["sha256"]: row for row in rows}
    targets, predicted = defaultdict(list), defaultdict(list)
    for annotation in coco["annotations"]:
        targets[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        predicted[int(prediction["image_id"])].append(prediction)
    result = []
    for image_id in sorted(images):
        image = images[image_id]
        row = sha_to_row[image["fireviewer_sha256"]]
        gt = targets[image_id]
        candidates = predicted[image_id]
        unsupported = []
        for target in gt:
            best = max((prediction["score"] for prediction in candidates
                        if prediction["category_id"] == target["category_id"]
                        and iou_xywh(prediction["bbox"], target["bbox"]) >= match_iou), default=0.0)
            if best < support_score:
                unsupported.append({"category_id": target["category_id"], "bbox": target["bbox"], "best_score": best})
        extras = []
        for prediction in candidates:
            if prediction["score"] < extra_score:
                continue
            best = max((iou_xywh(prediction["bbox"], target["bbox"]) for target in gt
                        if target["category_id"] == prediction["category_id"]), default=0.0)
            if best < match_iou:
                extras.append(prediction)
        smoke_misses = sum(item["category_id"] == 1 for item in unsupported)
        score = len(unsupported) / max(1, len(gt)) + smoke_misses * 0.5 + min(len(extras), 4) * 0.25
        result.append({
            "revision_review_index": row["revision_review_index"],
            "sha256": row["sha256"],
            "source_record_id": row["source_record_id"],
            "source_group_id": row["source_group_id"],
            "ground_truth_boxes": len(gt),
            "unsupported_annotations": unsupported,
            "unmatched_high_confidence_predictions": extras,
            "model_disagreement_score": score,
            "screening_only": True,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue, coco_path, predictions_path, output = (
        args.queue.resolve(), args.coco.resolve(), args.predictions.resolve(), args.output.resolve()
    )
    rows = read_rows(queue)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    triage = analyze(rows, coco, predictions)
    by_index = {row["revision_review_index"]: row for row in rows}
    ranked = sorted(triage, key=lambda row: (-row["model_disagreement_score"], row["revision_review_index"]))
    review_rows = [by_index[row["revision_review_index"]] for row in ranked]
    packets = render_packets(review_rows, output / "triage")
    write_rows(output / "model_triage.jsonl", ranked)
    write_json(output / "triage" / "packets.json", packets)
    report = {
        "schema": "fireviewer.pointing-v8p4-model-assisted-triage.v1",
        "status": "screening_complete_visual_review_still_required",
        "queue_sha256": digest(queue),
        "coco_sha256": digest(coco_path),
        "predictions_sha256": digest(predictions_path),
        "images": len(rows),
        "images_with_unsupported_annotations": sum(bool(row["unsupported_annotations"]) for row in triage),
        "unsupported_annotations": sum(len(row["unsupported_annotations"]) for row in triage),
        "images_with_unmatched_high_confidence_predictions": sum(bool(row["unmatched_high_confidence_predictions"]) for row in triage),
        "unmatched_high_confidence_predictions": sum(len(row["unmatched_high_confidence_predictions"]) for row in triage),
        "review_pages": len(packets),
        "thresholds": {"support_score": 0.3, "extra_score": 0.5, "match_iou": 0.3},
        "qualification_limit": "Detector agreement is not ground truth and never admits an image automatically.",
    }
    write_json(output / "model_triage_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
