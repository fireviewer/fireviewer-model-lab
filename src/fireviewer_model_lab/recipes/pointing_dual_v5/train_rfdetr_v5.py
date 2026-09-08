#!/usr/bin/env python3
"""Train RF-DETR Large on the immutable FireViewer pointing V5 COCO view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from rfdetr import RFDETRLarge


SEED = 20260824
RESOLUTION = 704
MICRO_BATCH_SIZE = 3
GRAD_ACCUM_STEPS = 6


def configure_utf8_output() -> None:
    """Keep Rich metric tables writable when stdout is redirected on Windows."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


configure_utf8_output()


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
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    coco_root = args.coco_root.resolve()
    output_dir = args.output_dir.resolve()
    weights = args.pretrain_weights.resolve()
    receipt_path = output_dir / "run_manifest.json"
    if not (coco_root / "coco_view_receipt.json").is_file():
        raise RuntimeError("Missing validated V5 COCO receipt")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_receipt = json.loads((coco_root / "coco_view_receipt.json").read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "schema": "fireviewer.rf-detr-large-pointing-v5-train.v1",
        "status": "initializing",
        "started_at_unix": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset": {
            "coco_root": str(coco_root),
            "dataset_report_sha256": coco_receipt["dataset_report_sha256"],
            "selection_manifest_sha256": coco_receipt["selection_manifest_sha256"],
            "split_counts": {key: value["images"] for key, value in coco_receipt["splits"].items()},
        },
        "base_weights": {"path": str(weights), "sha256": sha256_file(weights)},
        "model": "RFDETRLarge",
        "rfdetr_version": "1.8.3",
        "resolution": RESOLUTION,
        "epochs": args.epochs,
        "physical_batch_size": MICRO_BATCH_SIZE,
        "grad_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS,
        "batch_selection": {
            "method": "measured_rfdetr_auto_batch_probe",
            "worst_case_resolution": 832,
            "ema_headroom": 0.65,
            "selected_physical_batch_size": MICRO_BATCH_SIZE,
        },
        "seed": SEED,
        "automatic_cleanup": False,
    }
    atomic_json(receipt_path, manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        model = RFDETRLarge(
            pretrain_weights=str(weights),
            num_classes=2,
            resolution=RESOLUTION,
            gradient_checkpointing=True,
            freeze_encoder=False,
            device="cuda:0",
        )
        manifest["status"] = "preflight_passed"
        manifest["initialized_at_unix"] = time.time()
        atomic_json(receipt_path, manifest)
        if args.preflight_only:
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0

        manifest["status"] = "training"
        atomic_json(receipt_path, manifest)
        model.train(
            dataset_dir=str(coco_root),
            output_dir=str(output_dir),
            class_names=["fire", "smoke"],
            epochs=args.epochs,
            batch_size=MICRO_BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            lr=1e-4,
            lr_encoder=1e-5,
            weight_decay=1e-4,
            lr_scheduler="cosine",
            lr_min_factor=0.05,
            warmup_epochs=2.0,
            num_workers=6,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
            amp_dtype="bf16",
            use_ema=True,
            ema_update_interval=1,
            checkpoint_interval=10,
            early_stopping=True,
            early_stopping_patience=8,
            early_stopping_min_delta=0.001,
            early_stopping_use_ema=True,
            skip_best_epochs=2,
            eval_interval=1,
            eval_max_dets=100,
            log_per_class_metrics=True,
            multi_scale=True,
            expanded_scales=False,
            augmentation_backend="cpu",
            tensorboard=True,
            wandb=False,
            mlflow=False,
            run_test=False,
            progress_bar="tqdm",
            seed=SEED,
            notes={
                "project": "FireViewer",
                "task": "pointing-v5",
                "dataset_report_sha256": coco_receipt["dataset_report_sha256"],
                "selection_manifest_sha256": coco_receipt["selection_manifest_sha256"],
                "benchmark": "shared-v5-test-v1",
            },
        )
        checkpoints = sorted(output_dir.glob("*.pth"))
        if not checkpoints:
            raise RuntimeError("RF-DETR returned without a checkpoint")
        manifest.update(
            {
                "status": "completed",
                "completed_at_unix": time.time(),
                "elapsed_seconds": time.time() - manifest["started_at_unix"],
                "checkpoints": [
                    {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in checkpoints
                ],
            }
        )
        atomic_json(receipt_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        atomic_json(receipt_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
