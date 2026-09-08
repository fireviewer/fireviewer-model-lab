from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from fireviewer_model_lab.training.detector_benchmark import PYRONEAR_MODEL_REVISION, validate_selection
from fireviewer_model_lab.training.train_rtdetr import load_records


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _firewarning_class_id(name: str) -> int:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if "smoke" in normalized or "fumee" in normalized or "fumée" in normalized:
        return 0
    if "fire" in normalized or "flame" in normalized or "feu" in normalized:
        return 1
    raise ValueError(f"Detector label cannot be mapped to fire or smoke: {name!r}")


def _image_paths(records: list[Any]) -> list[Path]:
    return [
        (loaded.corpus_root / str(loaded.record["image_relpath"])).resolve() for loaded in records
    ]


def infer_transformers(
    *,
    checkpoint: str,
    revision: str | None,
    candidate_id: str,
    selection: dict[str, Any],
    records: list[Any],
    batch_size: int,
    cache_dir: Path,
) -> dict[str, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    pretrained: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "trust_remote_code": False,
    }
    if revision:
        pretrained["revision"] = revision
    processor = AutoImageProcessor.from_pretrained(checkpoint, **pretrained)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        **pretrained,
    ).to(device)
    model.eval()
    id2label = {
        int(identifier): str(label)
        for identifier, label in getattr(model.config, "id2label", {}).items()
    }
    mapped_labels = {
        identifier: _firewarning_class_id(label) for identifier, label in id2label.items()
    }
    paths = _image_paths(records)
    predictions: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    for offset in range(0, len(paths), batch_size):
        batch_paths = paths[offset : offset + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(device)
            if not torch.is_floating_point(value) or dtype is None
            else value.to(device=device, dtype=dtype)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor(
            [[image.height, image.width] for image in images],
            device=device,
        )
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.001,
            target_sizes=target_sizes,
        )
        for entry, result in zip(
            selection["entries"][offset : offset + len(results)],
            results,
            strict=True,
        ):
            labels = [mapped_labels[int(value)] for value in result["labels"].cpu().tolist()]
            predictions.append(
                {
                    "sample_id": entry["sample_id"],
                    "boxes_xyxy": result["boxes"].float().cpu().tolist(),
                    "labels": labels,
                    "scores": result["scores"].float().cpu().tolist(),
                }
            )
        for image in images:
            image.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "checkpoint": checkpoint,
        "revision": revision,
        "selection_sha256": selection["selection_sha256"],
        "timing": {
            "scope": "preprocess_inference_postprocess_excluding_model_load",
            "total_seconds": inference_seconds,
            "images_per_second": len(paths) / inference_seconds,
            "milliseconds_per_image": inference_seconds * 1000 / len(paths),
        },
        "predictions": predictions,
    }


def infer_yolo(
    *,
    checkpoint: Path,
    candidate_id: str,
    selection: dict[str, Any],
    records: list[Any],
    batch_size: int,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "The Pyronear adapter requires the optional ultralytics dependency"
        ) from exc
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = YOLO(str(checkpoint))
    predictions: list[dict[str, Any]] = []
    paths = _image_paths(records)
    try:
        import torch
    except ImportError:  # pragma: no cover - ultralytics already requires torch
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in paths],
        stream=True,
        batch=batch_size,
        verbose=False,
    )
    for entry, result in zip(selection["entries"], results, strict=True):
        names = result.names
        labels = [
            _firewarning_class_id(str(names[int(identifier)]))
            for identifier in result.boxes.cls.cpu().tolist()
        ]
        predictions.append(
            {
                "sample_id": entry["sample_id"],
                "boxes_xyxy": result.boxes.xyxy.float().cpu().tolist(),
                "labels": labels,
                "scores": result.boxes.conf.float().cpu().tolist(),
            }
        )
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "checkpoint": str(checkpoint.resolve()),
        "revision": PYRONEAR_MODEL_REVISION,
        "selection_sha256": selection["selection_sha256"],
        "timing": {
            "scope": "preprocess_inference_postprocess_excluding_model_load",
            "total_seconds": inference_seconds,
            "images_per_second": len(paths) / inference_seconds,
            "milliseconds_per_image": inference_seconds * 1000 / len(paths),
        },
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one detector candidate on the immutable FireWarning benchmark"
    )
    parser.add_argument("adapter", choices=("transformers", "pyronear-yolo"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/huggingface-cache"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verify-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    records = load_records(args.manifest, verify_files=args.verify_files)
    selection = _load_json(args.selection)
    selected_records = validate_selection(selection, records)
    if args.adapter == "transformers":
        payload = infer_transformers(
            checkpoint=args.checkpoint,
            revision=args.revision,
            candidate_id=args.candidate_id,
            selection=selection,
            records=selected_records,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
        )
    else:
        payload = infer_yolo(
            checkpoint=Path(args.checkpoint),
            candidate_id=args.candidate_id,
            selection=selection,
            records=selected_records,
            batch_size=args.batch_size,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_id": payload["candidate_id"],
                "selection_sha256": payload["selection_sha256"],
                "predictions": len(payload["predictions"]),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
