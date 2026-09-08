#!/usr/bin/env python3
"""Exercise one real DEIM-D-FINE forward/backward batch before the long run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-schema", default="fireviewer.deim-dfine-large-pointing-v5-preflight.v1")
    parser.add_argument("--stress-max-targets", action="store_true")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--check-eval", action="store_true")
    args = parser.parse_args()
    deim_root = args.deim_root.resolve()
    sys.path.insert(0, str(deim_root))
    from engine.core import YAMLConfig  # noqa: E402
    from engine.solver import TASKS  # noqa: E402

    coco_root = args.coco_root.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    overrides = {
        "tuning": str(weights),
        "device": "cuda:0",
        "use_amp": True,
        "output_dir": str(output.parent / "deim-preflight-runtime"),
        "train_dataloader": {
            "dataset": {
                "img_folder": str(coco_root / "train"),
                "ann_file": str(coco_root / "train" / "_annotations.coco.json"),
            }
        },
        "val_dataloader": {
            "dataset": {
                "img_folder": str(coco_root / "valid"),
                "ann_file": str(coco_root / "valid" / "_annotations.coco.json"),
            }
        },
    }
    cfg = YAMLConfig(str(args.config.resolve()), **overrides)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    started = time.time()
    try:
        solver.train()
        solver.train_dataloader.set_epoch(0)
        if args.stress_max_targets:
            dataset = solver.train_dataloader.dataset
            indices = sorted(range(len(dataset)), key=lambda i: len(dataset.coco.imgToAnns[dataset.ids[i]]), reverse=True)
            samples, targets = solver.train_dataloader.collate_fn([dataset[i] for i in indices[:solver.train_dataloader.batch_size]])
        else:
            samples, targets = next(iter(solver.train_dataloader))
        samples = samples.to(solver.device)
        targets = [{key: value.to(solver.device) for key, value in target.items()} for target in targets]
        solver.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = solver.model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses = solver.criterion(
                predictions,
                targets,
                epoch=0,
                step=0,
                global_step=0,
                epoch_step=len(solver.train_dataloader),
            )
        loss = sum(losses.values())
        if not math.isfinite(float(loss.detach().cpu())):
            raise RuntimeError(f"Non-finite preflight loss: {float(loss.detach().cpu())}")
        (loss / int(cfg.grad_accum_steps)).backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(solver.model.parameters(), cfg.clip_max_norm).detach().cpu())
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient norm: {gradient_norm}")
        if args.optimizer_step:
            solver.optimizer.step()
            solver.optimizer.zero_grad(set_to_none=True)
        if args.check_eval:
            solver.model.eval()
            with torch.no_grad():
                eval_predictions = solver.model(samples)
                if not torch.isfinite(eval_predictions["pred_boxes"]).all():
                    raise RuntimeError("Non-finite evaluation boxes")
                sizes = torch.stack([t["orig_size"] for t in targets])
                processed = solver.postprocessor(eval_predictions, sizes)
                if len(processed) != samples.shape[0]:
                    raise RuntimeError("Evaluation postprocessing batch mismatch")
                del eval_predictions, processed
        torch.cuda.synchronize()
        receipt = {
            "schema": args.receipt_schema,
            "status": "passed",
            "deim_revision": "09d35d53d39ee3145a1e61e3a989b28b9468d1dd",
            "base_weights_sha256": sha256_file(weights),
            "config_sha256": sha256_file(args.config.resolve()),
            "coco_view_receipt_sha256": sha256_file(coco_root / "coco_view_receipt.json"),
            "resolution": cfg.yaml_cfg["eval_spatial_size"],
            "micro_batch_size": int(samples.shape[0]),
            "gradient_accumulation_steps": int(cfg.grad_accum_steps),
            "effective_batch_size": int(samples.shape[0]) * int(cfg.grad_accum_steps),
            "loss": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
            "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / (1024**3),
            "stress_max_targets": args.stress_max_targets,
            "targets_per_image": [len(t["labels"]) for t in targets],
            "optimizer_step_passed": args.optimizer_step,
            "fp32_eval_forward_and_postprocess_passed": args.check_eval,
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
