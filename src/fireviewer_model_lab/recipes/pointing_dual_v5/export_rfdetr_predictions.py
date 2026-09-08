#!/usr/bin/env python3
"""Export low-threshold RF-DETR predictions for the shared V5 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRLarge


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=100)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    split_root = args.coco_root.resolve() / args.split
    annotations = json.loads((split_root / "_annotations.coco.json").read_text(encoding="utf-8"))
    model = RFDETRLarge.from_checkpoint(checkpoint, device="cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    first = split_root / annotations["images"][0]["file_name"]
    for _ in range(5):
        model.predict(Image.open(first).convert("RGB"), threshold=args.candidate_threshold, include_source_image=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    for image_row in annotations["images"]:
        image = Image.open(split_root / image_row["file_name"]).convert("RGB")
        detections = model.predict(
            image,
            threshold=args.candidate_threshold,
            include_source_image=False,
        )
        boxes = np.asarray(detections.xyxy, dtype=np.float64)
        labels = np.asarray(detections.class_id, dtype=np.int64)
        scores = np.asarray(detections.confidence, dtype=np.float64)
        order = np.argsort(-scores)[: args.max_detections]
        for index in order:
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            predictions.append(
                {
                    "image_id": int(image_row["id"]),
                    "category_id": int(labels[index]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(scores[index]),
                }
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = args.output.resolve()
    atomic_json(output, predictions)
    receipt = {
        "schema": "fireviewer.shared-v5-prediction-export.v1",
        "model": "rf-detr-large",
        "split": args.split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "candidate_threshold": args.candidate_threshold,
        "max_detections_per_image": args.max_detections,
        "images": len(annotations["images"]),
        "predictions": len(predictions),
        "elapsed_seconds": elapsed,
        "fps_batch1": len(annotations["images"]) / elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "prediction_sha256": sha256_file(output),
    }
    atomic_json(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
