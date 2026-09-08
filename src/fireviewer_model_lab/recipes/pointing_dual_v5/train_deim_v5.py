#!/usr/bin/env python3
"""Run the pinned official DEIM-D-FINE Large trainer with immutable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


DEIM_REVISION = "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"


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
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-schema", default="fireviewer.deim-dfine-large-pointing-v5-train.v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--checkpoint-frequency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    if args.micro_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Micro-batch size and gradient accumulation must be positive")
    deim_root = args.deim_root.resolve()
    config = args.config.resolve()
    coco_root = args.coco_root.resolve()
    weights = args.weights.resolve()
    resume_checkpoint = args.resume_checkpoint.resolve() if args.resume_checkpoint else None
    output_dir = args.output_dir.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing a silent CPU training run")
    sys.path.insert(0, str(deim_root))
    from engine.core.yaml_utils import load_config
    resolved_config = load_config(str(config), {})
    spatial_size = resolved_config["eval_spatial_size"]
    if len(spatial_size) != 2 or spatial_size[0] != spatial_size[1]:
        raise ValueError("This launcher requires an explicit square evaluation resolution")
    resize_ops = [op for op in resolved_config["train_dataloader"]["dataset"]["transforms"]["ops"] if op["type"] == "Resize"]
    if len(resize_ops) != 1 or resize_ops[0]["size"] != spatial_size:
        raise ValueError("Training/evaluation base resolution mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_receipt = None
    if resume_checkpoint is not None:
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        resume_state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        required_resume_keys = {"model", "ema", "optimizer", "last_epoch"}
        missing_resume_keys = sorted(required_resume_keys - set(resume_state))
        if missing_resume_keys:
            raise RuntimeError(f"Resume checkpoint is incomplete: {missing_resume_keys}")
        resume_receipt = {
            "path": str(resume_checkpoint),
            "sha256": sha256_file(resume_checkpoint),
            "last_epoch": int(resume_state["last_epoch"]),
        }
        del resume_state
    coco_receipt = json.loads((coco_root / "coco_view_receipt.json").read_text(encoding="utf-8"))
    patched_sources = [
        deim_root / "engine" / "core" / "_config.py",
        deim_root / "engine" / "solver" / "det_engine.py",
        deim_root / "engine" / "solver" / "det_solver.py",
        deim_root / "engine" / "data" / "transforms" / "_transforms.py",
        deim_root / "engine" / "data" / "transforms" / "lighting.py",
        deim_root / "engine" / "data" / "transforms" / "__init__.py",
    ]
    manifest: dict[str, Any] = {
        "schema": args.manifest_schema,
        "status": "starting",
        "started_at_unix": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deim_revision": DEIM_REVISION,
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "local_compatibility_patch": [
            {"path": str(path.relative_to(deim_root)), "sha256": sha256_file(path)} for path in patched_sources
        ],
        "base_weights": {"path": str(weights), "sha256": sha256_file(weights)},
        "dataset": {
            "coco_root": str(coco_root),
            "dataset_report_sha256": coco_receipt["dataset_report_sha256"],
            "selection_manifest_sha256": coco_receipt["selection_manifest_sha256"],
            "split_counts": {key: value["images"] for key, value in coco_receipt["splits"].items()},
            "coco_view_schema": coco_receipt.get("schema"),
            "coco_view_receipt_sha256": sha256_file(coco_root / "coco_view_receipt.json"),
            "coverage_qualified": coco_receipt.get("coverage_qualified"),
            "local_training_authorization": coco_receipt.get("local_training_authorization"),
            "acknowledged_limitations": coco_receipt.get("acknowledged_limitations", []),
            "synthetic_images": coco_receipt.get("synthetic_images", 0),
            "prelaunch_excluded_real_images": coco_receipt.get("prelaunch_excluded_real_images", 0),
        },
        "training": {
            "epochs": args.epochs,
            "resolution": int(spatial_size[0]),
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
            "amp_dtype": "bf16",
            "seed": args.seed,
            "checkpoint_frequency": args.checkpoint_frequency,
            "resolved_train_transforms": resolved_config["train_dataloader"]["dataset"]["transforms"],
            "resolved_train_collate": resolved_config["train_dataloader"]["collate_fn"],
        },
        "automatic_cleanup": False,
    }
    if resume_receipt is not None:
        manifest["resume_checkpoint"] = resume_receipt
    manifest_path = output_dir / "run_manifest.json"
    atomic_json(manifest_path, manifest)
    command = [
        sys.executable,
        "train.py",
        "-c",
        str(config),
        "-r" if resume_checkpoint is not None else "-t",
        str(resume_checkpoint if resume_checkpoint is not None else weights),
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed),
        "--use-amp",
        "--output-dir",
        str(output_dir),
        "-u",
        f"epoches={args.epochs}",
        f"checkpoint_freq={args.checkpoint_frequency}",
        f"grad_accum_steps={args.gradient_accumulation_steps}",
        f"train_dataloader.total_batch_size={args.micro_batch_size}",
        f"train_dataloader.dataset.img_folder={coco_root / 'train'}",
        f"train_dataloader.dataset.ann_file={coco_root / 'train' / '_annotations.coco.json'}",
        f"val_dataloader.dataset.img_folder={coco_root / 'valid'}",
        f"val_dataloader.dataset.ann_file={coco_root / 'valid' / '_annotations.coco.json'}",
    ]
    try:
        manifest["status"] = "training"
        manifest["command"] = command
        atomic_json(manifest_path, manifest)
        completed = subprocess.run(command, cwd=deim_root, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"DEIM trainer exited with code {completed.returncode}")
        checkpoints = sorted(output_dir.glob("*.pth"))
        if not checkpoints:
            raise RuntimeError("DEIM returned without a checkpoint")
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
        atomic_json(manifest_path, manifest)
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
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
