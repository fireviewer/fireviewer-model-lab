"""SegFormer-B2 baseline trained on the exact shared FireViewer manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from fireviewer_model_lab.training.dinov3_adapter import IMAGE_MEAN, IMAGE_STD


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class FireViewerSegmentationDataset:
    def __init__(self, manifest: Path, data_root: Path, split: str, image_size: int) -> None:
        self.data_root = data_root.resolve()
        self.image_size = image_size
        self.rows = [
            row
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for row in (json.loads(line),)
            if row.get("split") == split
        ]
        if not self.rows:
            raise ValueError(f"SegFormer manifest has no {split} rows")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> Path:
        path = (self.data_root / value).resolve()
        if path != self.data_root and self.data_root not in path.parents:
            raise ValueError(f"manifest path escapes data root: {value}")
        return path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(self._path(str(row["image_relpath"]))) as opened:
            image = opened.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        with Image.open(self._path(str(row["mask_relpath"]))) as opened:
            mask = opened.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST
            )
        valid_mask_relpath = row.get("valid_mask_relpath")
        if valid_mask_relpath:
            with Image.open(self._path(str(valid_mask_relpath))) as opened:
                valid_mask = opened.convert("L").resize(
                    (self.image_size, self.image_size), Image.Resampling.NEAREST
                )
            valid_mask_tensor = torch.from_numpy(
                (np.asarray(valid_mask) > 0).astype(np.float32).copy()
            )[None]
        else:
            valid_mask_tensor = torch.ones(
                (1, self.image_size, self.image_size), dtype=torch.float32
            )
        image_tensor = (
            torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        )
        return {
            "image": (image_tensor - IMAGE_MEAN) / IMAGE_STD,
            "mask": torch.from_numpy((np.asarray(mask) > 0).astype(np.float32).copy())[None],
            "valid_mask": valid_mask_tensor,
            "sample_id": str(row["sample_id"]),
        }


class FireViewerSegFormer(nn.Module):
    def __init__(self, model_id: str, revision: str) -> None:
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        local = Path(model_id).exists()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id,
            revision=None if local else revision,
            local_files_only=local,
            token=not local,
            num_labels=1,
            id2label={0: "fire_or_smoke"},
            label2id={"fire_or_smoke": 0},
            ignore_mismatched_sizes=True,
        )

    def forward(self, images: Any) -> Any:
        logits = self.model(pixel_values=images).logits
        return nn.functional.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )


def _loss(logits: Any, mask: Any, valid_mask: Any) -> Any:
    probability = logits.sigmoid()
    intersection = (probability * mask * valid_mask).sum(dim=(1, 2, 3))
    denominator = (probability * valid_mask).sum(dim=(1, 2, 3)) + (mask * valid_mask).sum(
        dim=(1, 2, 3)
    )
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    binary = nn.functional.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    return (binary * valid_mask).sum() / valid_mask.sum().clamp_min(1.0) + dice


def finite_loss_probe(
    *,
    manifest: Path,
    data_root: Path,
    model_id: str,
    model_revision: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SegFormer finite-loss probe")
    dataset = FireViewerSegmentationDataset(manifest, data_root, "train", image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = FireViewerSegFormer(model_id, model_revision).to(device)
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(images)
        loss = _loss(logits, mask, valid_mask)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    all_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    if not all_finite or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("SegFormer finite-loss probe produced non-finite values")
    return {
        "passed": True,
        "device": str(device),
        "batch_size": images.shape[0],
        "image_size": image_size,
        "sample_ids": list(batch["sample_id"]),
        "loss": float(loss.detach().cpu()),
        "gradient_tensors": len(gradients),
        "all_gradients_finite": all_finite,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, loader: Any, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = intersections = unions = dice_num = dice_den = 0.0
    rows = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss = _loss(logits, mask, valid_mask)
        valid = valid_mask >= 0.5
        prediction = (logits.sigmoid() >= 0.5) & valid
        target = (mask >= 0.5) & valid
        intersections += float((prediction & target).sum())
        unions += float((prediction | target).sum())
        dice_num += float((prediction & target).sum()) * 2.0
        dice_den += float(prediction.sum() + target.sum())
        total_loss += float(loss)
        rows += images.shape[0]
    return {
        "loss": total_loss / max(1, len(loader)),
        "iou": intersections / max(1.0, unions),
        "dice": dice_num / max(1.0, dice_den),
        "rows": float(rows),
    }


def _atomic_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run_training(
    *,
    manifest: Path,
    data_root: Path,
    output: Path,
    model_id: str,
    model_revision: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
    image_size: int,
    num_workers: int,
    early_stopping_patience: int = 8,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the SegFormer baseline train")
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: FireViewerSegmentationDataset(manifest, data_root, split, image_size)
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
        )
        for split, dataset in datasets.items()
    }
    model = FireViewerSegFormer(model_id, model_revision).to(device)
    backbone = list(model.model.segformer.parameters())
    head = list(model.model.decode_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": learning_rate},
            {"params": head, "lr": learning_rate * 5.0},
        ],
        weight_decay=0.05,
    )
    updates_per_epoch = math.ceil(len(loaders["train"]) / gradient_accumulation_steps)
    total_updates = max(1, epochs * updates_per_epoch)
    warmup = max(1, round(total_updates * 0.1))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_path = output / "checkpoints" / "best.pt"
    last_path = output / "checkpoints" / "last.pt"
    start_epoch, best_epoch, stale, global_step = 1, 0, 0, 0
    best_loss = math.inf
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_loss = float(state["best_loss"])
        stale = int(state["stale_epochs"])
        global_step = int(state["global_step"])
    metrics_path = output / "metrics.csv"
    append = start_epoch > 1 and metrics_path.is_file()
    with metrics_path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "train_loss",
                "validation_loss",
                "validation_iou",
                "validation_dice",
                "learning_rate",
            ),
        )
        if not append:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for step, batch in enumerate(loaders["train"], 1):
                images = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                valid_mask = batch["valid_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = _loss(model(images), mask, valid_mask)
                    scaled = loss / gradient_accumulation_steps
                if not bool(torch.isfinite(scaled).item()):
                    raise RuntimeError(f"non-finite SegFormer loss at epoch {epoch}, step {step}")
                scaled.backward()
                if step % gradient_accumulation_steps == 0 or step == len(loaders["train"]):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                train_loss += float(loss.detach())
            validation = _evaluate(model, loaders["validation"], device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / len(loaders["train"]),
                "validation_loss": validation["loss"],
                "validation_iou": validation["iou"],
                "validation_dice": validation["dice"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            handle.flush()
            if validation["loss"] < best_loss:
                best_loss, best_epoch, stale = validation["loss"], epoch, 0
                _atomic_save(
                    {
                        "schema_version": 1,
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "validation": validation,
                        "model_revision": model_revision,
                        "manifest_sha256": _sha256(manifest),
                    },
                    best_path,
                )
            else:
                stale += 1
            _atomic_save(
                {
                    "schema_version": 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale,
                },
                last_path,
            )
            if stale >= early_stopping_patience:
                break
    if not best_path.is_file():
        raise RuntimeError("SegFormer train ended without a best checkpoint")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    model.to(device)
    test_metrics = _evaluate(model, loaders["test"], device)
    return {
        "schema_version": 1,
        "model_id": model_id,
        "model_revision": model_revision,
        "device": str(device),
        "train_rows": len(datasets["train"]),
        "validation_rows": len(datasets["validation"]),
        "test_rows": len(datasets["test"]),
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256(best_path),
        "metrics": str(metrics_path),
    }
