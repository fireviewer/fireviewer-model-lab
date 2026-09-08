#!/usr/bin/env python3
"""Export low-threshold DEIM-D-FINE predictions with a reloadable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


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
    parser.add_argument("--deim-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--model-name", default="deim-dfine-large")
    parser.add_argument("--receipt-schema", default="fireviewer.shared-v5-prediction-export.v1")
    parser.add_argument("--resolution", type=int, default=704)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--split-registry", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for measured D-FINE prediction export")
    if args.candidate_threshold < 0 or args.candidate_threshold > 1:
        raise ValueError("candidate threshold must be between 0 and 1")
    if args.max_detections <= 0 or args.resolution <= 0 or args.warmup < 0:
        raise ValueError("max detections and resolution must be positive; warmup cannot be negative")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max images must be positive when provided")

    started_at = datetime.now(timezone.utc)
    deim_root = args.deim_root.resolve()
    config = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    coco_root = args.coco_root.resolve()
    output = args.output.resolve()
    for required in (deim_root, config, checkpoint, coco_root):
        if not required.exists():
            raise FileNotFoundError(required)
    split_integrity = None
    if args.split_registry:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from training.pointing_dataset_v7.split_registry import load_registry, verify_coco
        split_integrity = verify_coco(coco_root, load_registry(args.split_registry))
        if split_integrity["status"] != "passed":
            raise ValueError("Unresolved source group conflicts: refusing held-out inference")
    sys.path.insert(0, str(deim_root))
    from engine.core import YAMLConfig  # noqa: E402

    checkpoint_sha256 = sha256_file(checkpoint)
    cfg = YAMLConfig(str(config), resume=str(checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if "ema" in state else state["model"]
    cfg.model.load_state_dict(weights)

    class DeployModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images: torch.Tensor, sizes: torch.Tensor):
            return self.postprocessor(self.model(images), sizes)

    device = torch.device("cuda:0")
    model = DeployModel().to(device).eval()
    transform = T.Compose([T.Resize((args.resolution, args.resolution)), T.ToTensor()])
    split_root = coco_root / args.split
    annotation_path = split_root / "_annotations.coco.json"
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in annotations["categories"]}
    if categories != {0: "fire", 1: "smoke"}:
        raise RuntimeError(f"Unexpected benchmark categories: {categories}")
    all_image_rows = list(annotations["images"])
    image_rows = all_image_rows[: args.max_images] if args.max_images is not None else all_image_rows
    missing_images = [row["file_name"] for row in image_rows if not (split_root / row["file_name"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} benchmark images; first={missing_images[0]}")

    def infer(image_path: Path):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tensor = transform(image).unsqueeze(0).to(device)
        sizes = torch.tensor([[width, height]], device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(tensor, sizes)

    first = split_root / image_rows[0]["file_name"]
    for _ in range(args.warmup):
        infer(first)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    predictions: list[dict[str, Any]] = []
    latencies_seconds: list[float] = []
    prediction_class_counts = {name: 0 for name in categories.values()}
    for image_row in image_rows:
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        labels, boxes, scores = infer(split_root / image_row["file_name"])
        torch.cuda.synchronize()
        latencies_seconds.append(time.perf_counter() - inference_started)
        labels = labels[0].detach().cpu()
        boxes = boxes[0].detach().float().cpu()
        scores = scores[0].detach().float().cpu()
        keep = torch.nonzero(scores >= args.candidate_threshold, as_tuple=False).flatten()
        keep = keep[torch.argsort(scores[keep], descending=True)[: args.max_detections]]
        for index in keep.tolist():
            category_id = int(labels[index])
            if category_id not in categories:
                raise RuntimeError(f"Model emitted non-comparable category {category_id}")
            x1, y1, x2, y2 = (float(value) for value in boxes[index])
            predictions.append(
                {
                    "image_id": int(image_row["id"]),
                    "category_id": category_id,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(scores[index]),
                }
            )
            prediction_class_counts[categories[category_id]] += 1
    elapsed = float(sum(latencies_seconds))
    latency_ms = np.asarray(latencies_seconds, dtype=np.float64) * 1000.0
    atomic_json(output, predictions)
    receipt = {
        "schema": args.receipt_schema,
        "status": "completed",
        "model": args.model_name,
        "split": args.split,
        "deim_revision": "09d35d53d39ee3145a1e61e3a989b28b9468d1dd",
        "config": str(config),
        "config_sha256": sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_state": "ema.module" if "ema" in state else "model",
        "split_integrity": split_integrity,
        "split_registry_sha256": sha256_file(args.split_registry) if args.split_registry else None,
        "coco_root": str(coco_root),
        "annotation_file": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "candidate_threshold": args.candidate_threshold,
        "max_detections_per_image": args.max_detections,
        "split_images": len(all_image_rows),
        "images": len(image_rows),
        "complete_split": len(image_rows) == len(all_image_rows),
        "ground_truth_annotations": len(annotations["annotations"]),
        "predictions": len(predictions),
        "prediction_class_counts": prediction_class_counts,
        "elapsed_seconds": elapsed,
        "fps_batch1": len(image_rows) / elapsed,
        "latency_batch1_ms": {
            "mean": float(latency_ms.mean()),
            "p50": float(np.quantile(latency_ms, 0.50)),
            "p90": float(np.quantile(latency_ms, 0.90)),
            "p95": float(np.quantile(latency_ms, 0.95)),
            "p99": float(np.quantile(latency_ms, 0.99)),
        },
        "timing_scope": "sequential batch-1 end-to-end image decode, resize, host-to-device, inference, and postprocess",
        "warmup_images": args.warmup,
        "resolution": [args.resolution, args.resolution],
        "amp_dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "prediction_sha256": sha256_file(output),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
    }
    atomic_json(output.with_suffix(".receipt.json"), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
