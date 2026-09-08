"""Materialize fire/smoke exclusion masks for every cross-view image."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from fireviewer_model_lab.training.dinov3_adapter import (
    IMAGE_MEAN,
    IMAGE_STD,
    DinoV3MultiTaskModel,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relpath: str) -> Path:
    path = (root / relpath).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"manifest path escapes data root: {relpath}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class _ImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: Path, assets: list[dict[str, str]], image_size: int) -> None:
        self.root = root
        self.assets = assets
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.assets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        asset = self.assets[index]
        with Image.open(_safe_path(self.root, asset["relpath"])) as opened:
            width, height = opened.size
            image = opened.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return {
            "image": (tensor - IMAGE_MEAN) / IMAGE_STD,
            "sha256": asset["sha256"],
            "relpath": asset["relpath"],
            "width": width,
            "height": height,
        }


def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([row["image"] for row in rows]),
        "sha256": [row["sha256"] for row in rows],
        "relpath": [row["relpath"] for row in rows],
        "width": [row["width"] for row in rows],
        "height": [row["height"] for row in rows],
    }


def _collect_assets(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_hash: dict[str, str] = {}
    for row in rows:
        for view in (row["source_view"], row["map_view"]):
            digest = str(view["sha256"])
            relpath = str(view["image_relpath"])
            existing = by_hash.setdefault(digest, relpath)
            if existing != relpath:
                # Identical bytes may occur under different paths; one canonical mask is sufficient.
                continue
    return [{"sha256": digest, "relpath": relpath} for digest, relpath in sorted(by_hash.items())]


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialize cross-view transient masks")
    if args.dilation_kernel < 1 or args.dilation_kernel % 2 == 0:
        raise ValueError("dilation kernel must be a positive odd integer")
    data_root = args.data_root.resolve()
    source_manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir != data_root and data_root not in output_dir.parents:
        raise ValueError("output directory must be inside data root")
    rows = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("cross-view manifest is empty")
    assets = _collect_assets(rows)
    for asset in assets:
        image = _safe_path(data_root, asset["relpath"])
        if _sha256(image) != asset["sha256"]:
            raise RuntimeError(f"source image SHA-256 mismatch: {asset['relpath']}")

    checkpoint_sha = _sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if str(payload.get("model_revision")) != args.model_revision:
        raise RuntimeError("checkpoint base-model revision mismatch")
    model = DinoV3MultiTaskModel(
        str(args.model_path.resolve()), args.model_revision, args.image_size
    ).network
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    loader = DataLoader(
        _ImageDataset(data_root, assets, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=_collate,
    )
    mask_by_hash: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    processed = 0
    progress_bucket = -1
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)["segmentation_logits"]
            masks = (logits.sigmoid() >= args.threshold).float()
            if args.dilation_kernel > 1:
                masks = F.max_pool2d(
                    masks,
                    kernel_size=args.dilation_kernel,
                    stride=1,
                    padding=args.dilation_kernel // 2,
                )
            for index, digest in enumerate(batch["sha256"]):
                mask_small = (masks[index, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                resized = Image.fromarray(mask_small, mode="L").resize(
                    (int(batch["width"][index]), int(batch["height"][index])),
                    Image.Resampling.NEAREST,
                )
                mask_path = output_dir / "masks" / digest[:2] / f"{digest}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                resized.save(mask_path, optimize=True)
                ratio = float((np.asarray(resized) > 0).mean())
                ratios.append(ratio)
                mask_by_hash[digest] = {
                    "relpath": mask_path.relative_to(data_root).as_posix(),
                    "sha256": _sha256(mask_path),
                    "positive_pixel_ratio": ratio,
                }
            processed += len(batch["sha256"])
            bucket = processed // 256
            if bucket > progress_bucket or processed == len(assets):
                progress_bucket = bucket
                print(
                    f"transient_masks={processed}/{len(assets)} "
                    f"({100.0 * processed / len(assets):.1f}%)",
                    flush=True,
                )

    updated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        source_mask = mask_by_hash[str(row["source_view"]["sha256"])]
        map_mask = mask_by_hash[str(row["map_view"]["sha256"])]
        row.update(
            {
                "source_transient_mask_relpath": source_mask["relpath"],
                "source_transient_mask_sha256": source_mask["sha256"],
                "map_transient_mask_relpath": map_mask["relpath"],
                "map_transient_mask_sha256": map_mask["sha256"],
                "transient_mask_status": "fireviewer_dinov3_multitask_v3_inference",
            }
        )
        updated.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / "manifest.jsonl"
    output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in updated),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "dataset_family": "fireviewer-cross-view-v3-transient-masked",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": _sha256(source_manifest),
        "manifest": str(output_manifest),
        "manifest_sha256": _sha256(output_manifest),
        "rows": len(updated),
        "unique_images": len(assets),
        "mask_files": len(mask_by_hash),
        "nonempty_masks": sum(ratio > 0 for ratio in ratios),
        "mean_positive_pixel_ratio": statistics.fmean(ratios),
        "maximum_positive_pixel_ratio": max(ratios),
        "model_revision": args.model_revision,
        "checkpoint_sha256": checkpoint_sha,
        "threshold": args.threshold,
        "dilation_kernel": args.dilation_kernel,
        "gpu_peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "training_ready": len(mask_by_hash) == len(assets),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dilation-kernel", type=int, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(materialize(parse_args()), ensure_ascii=False, indent=2))
