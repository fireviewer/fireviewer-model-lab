"""Prepare, smoke-test and train DINOv3 on real FireViewer cross-view pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fireviewer_model_lab.training.dinov3_cross_view_adapter import (
    CrossViewPairDataset,
    DinoV3CrossViewModel,
    cross_view_loss,
    finite_loss_probe,
)

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
MODEL_LICENSE = "DINOv3 License"
MODEL_WEIGHTS_SHA256 = "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b"
DEFAULT_BUNDLE_ROOT = Path(
    os.environ.get(
        "FIREVIEWER_CROSS_VIEW_BUNDLE_ROOT",
        "data/datasets/cross-view-registration-v1",
    )
)
DEFAULT_MANIFEST_RELPATH = Path("corpus/cross-view-registration-v0.1.0/manifest.jsonl")
DEFAULT_MODEL_PATH = Path("data/models/dinov3-vitb16-pretrain-lvd1689m")
DEFAULT_OUTPUT = Path("data/training/dinov3-cross-view-retrieval-v1")
REQUIRED_SPLITS = {"train", "validation", "test"}
EXPECTED_MANIFEST_SHA256 = "532d3b545242a08f76322a8ef9edbb8ee554b53d0935087f9babb2cce0ab2965"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON line {line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        rows.append(row)
    return rows


def _safe_bundle_path(bundle_root: Path, relpath: str) -> Path:
    path = (bundle_root / relpath).resolve()
    if path != bundle_root and bundle_root not in path.parents:
        raise ValueError(f"manifest path escapes bundle root: {relpath}")
    return path


def _model_revision_from_cache(model_path: Path) -> str | None:
    metadata = model_path / ".cache/huggingface/download/config.json.metadata"
    if not metadata.is_file():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else None


def build_preflight_report(
    bundle_root: Path,
    model_path: Path,
    *,
    manifest_relpath: Path = DEFAULT_MANIFEST_RELPATH,
    verify_file_hashes: bool = False,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256,
    verify_model_hash: bool = False,
) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    model_path = model_path.resolve()
    manifest = bundle_root / manifest_relpath
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    if not manifest.is_file():
        errors.append(f"manifest_missing:{manifest}")
    else:
        rows = _load_rows(manifest)

    split_counts = Counter(str(row.get("split")) for row in rows)
    source_counts = Counter(str(row.get("source_id")) for row in rows)
    licenses = Counter(str(row.get("license")) for row in rows)
    split_groups: dict[str, set[str]] = defaultdict(set)
    source_hash_splits: dict[str, set[str]] = defaultdict(set)
    map_hash_splits: dict[str, set[str]] = defaultdict(set)
    verified_paths: set[Path] = set()
    transient_rows = 0
    for index, row in enumerate(rows, 1):
        split = str(row.get("split"))
        group = str(row.get("split_group"))
        split_groups[split].add(group)
        if row.get("family") != "cross_view_registration":
            errors.append(f"family_invalid:{index}")
        if row.get("operational_incident") is not False:
            errors.append(f"operational_incident_forbidden:{index}")
        if not row.get("license") or not row.get("consent_basis"):
            errors.append(f"license_or_consent_missing:{index}")
        try:
            source = row["source_view"]
            map_view = row["map_view"]
            point_target_valid = bool(row.get("point_target_valid", True))
            target = map_view.get("optical_axis_ground_pixel_normalized")
            if point_target_valid and (
                not isinstance(target, list)
                or len(target) != 2
                or not all(0.0 <= float(value) <= 1.0 for value in target)
            ):
                errors.append(f"target_xy_invalid:{index}")
            source_hash_splits[str(source["sha256"])].add(split)
            map_hash_splits[str(map_view["sha256"])].add(split)
            for asset in (source, map_view):
                path = _safe_bundle_path(bundle_root, str(asset["image_relpath"]))
                if not path.is_file():
                    errors.append(f"image_missing:{index}:{asset['image_relpath']}")
                elif verify_file_hashes and path not in verified_paths:
                    if _sha256(path) != str(asset["sha256"]):
                        errors.append(f"image_hash_mismatch:{index}:{asset['image_relpath']}")
                    verified_paths.add(path)
            source_transient_relpath = row.get("source_transient_mask_relpath") or row.get(
                "transient_mask_relpath"
            )
            map_transient_relpath = row.get("map_transient_mask_relpath")
            if source_transient_relpath and map_transient_relpath:
                transient_rows += 1
                for view, transient_relpath in (
                    ("source", source_transient_relpath),
                    ("map", map_transient_relpath),
                ):
                    if not _safe_bundle_path(bundle_root, str(transient_relpath)).is_file():
                        errors.append(f"transient_mask_missing:{index}:{view}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"schema_invalid:{index}:{exc}")

    if set(split_counts) != REQUIRED_SPLITS:
        errors.append(f"required_splits_missing:{sorted(REQUIRED_SPLITS - set(split_counts))}")
    group_owners: dict[str, set[str]] = defaultdict(set)
    for split, groups in split_groups.items():
        for group in groups:
            group_owners[group].add(split)
    leaking_groups = sorted(group for group, splits in group_owners.items() if len(splits) > 1)
    if leaking_groups:
        errors.append(f"split_group_leakage:{leaking_groups}")
    leaking_sources = sorted(
        value for value, splits in source_hash_splits.items() if len(splits) > 1
    )
    leaking_maps = sorted(value for value, splits in map_hash_splits.items() if len(splits) > 1)
    if leaking_sources:
        errors.append(f"source_asset_split_leakage:{len(leaking_sources)}")
    if leaking_maps:
        errors.append(f"map_asset_split_leakage:{len(leaking_maps)}")
    if transient_rows == 0:
        warnings.append("no_transient_masks_in_geometry_bundle")

    required_model_files = ("config.json", "model.safetensors", "preprocessor_config.json")
    for filename in required_model_files:
        if not (model_path / filename).is_file():
            errors.append(f"model_asset_missing:{filename}")
    cached_revision = _model_revision_from_cache(model_path)
    if cached_revision != MODEL_REVISION:
        errors.append(f"model_revision_mismatch:{cached_revision}")

    manifest_sha = _sha256(manifest) if manifest.is_file() else None
    if manifest_sha and expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        errors.append(f"manifest_sha_mismatch:{manifest_sha}:expected:{expected_manifest_sha256}")
    model_weights_sha = None
    weights_path = model_path / "model.safetensors"
    if verify_model_hash and weights_path.is_file():
        model_weights_sha = _sha256(weights_path)
        if model_weights_sha != MODEL_WEIGHTS_SHA256:
            errors.append(f"model_weights_sha_mismatch:{model_weights_sha}")
    return {
        "schema_version": 1,
        "training_kind": "full_dinov3_cross_view_retrieval_and_pointing",
        "bundle_root": str(bundle_root),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "license_counts": dict(sorted(licenses.items())),
        "split_group_counts": {
            split: len(groups) for split, groups in sorted(split_groups.items())
        },
        "unique_map_counts": {
            split: len({value for value, splits in map_hash_splits.items() if split in splits})
            for split in sorted(split_counts)
        },
        "verified_image_files": len(verified_paths) if verify_file_hashes else None,
        "file_hash_verification": verify_file_hashes,
        "transient_mask_rows": transient_rows,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path),
        "cached_model_revision": cached_revision,
        "model_weights_sha256": model_weights_sha,
        "fine_tuning_mode": "full_model_all_parameters_trainable_shared_encoder",
        "errors": errors,
        "warnings": warnings,
        "schema_ready": not errors,
        "training_ready": not errors and bool(rows) and transient_rows == len(rows),
        "promotion_ready": False,
        "promotion_blockers": [
            "independent_double_validated_geographic_test_missing",
            "downstream_roma_pycolmap_benchmark_missing",
            "fire_smoke_transient_mask_benchmark_missing",
        ],
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_image": torch.stack([row["source_image"] for row in batch]),
        "map_image": torch.stack([row["map_image"] for row in batch]),
        "map_label": torch.stack([row["map_label"] for row in batch]),
        "target_xy": torch.stack([row["target_xy"] for row in batch]),
        "point_valid": torch.stack([row["point_valid"] for row in batch]),
        "sample_id": [row["sample_id"] for row in batch],
        "map_sha256": [row["map_sha256"] for row in batch],
    }


@torch.no_grad()
def _evaluate(
    model: DinoV3CrossViewModel,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    source_embeddings: list[torch.Tensor] = []
    map_by_hash: dict[str, torch.Tensor] = {}
    target_hashes: list[str] = []
    point_errors: list[torch.Tensor] = []
    for batch in loader:
        source = batch["source_image"].to(device)
        maps = batch["map_image"].to(device)
        targets = batch["target_xy"].to(device)
        point_valid = batch["point_valid"].to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs = model(source, maps)
        source_embeddings.append(outputs["source_embeddings"].float().cpu())
        if bool(point_valid.any()):
            point_errors.append(
                torch.linalg.vector_norm(
                    outputs["target_xy"].float()[point_valid] - targets[point_valid], dim=1
                ).cpu()
            )
        for map_hash, embedding in zip(
            batch["map_sha256"], outputs["map_embeddings"].float().cpu(), strict=True
        ):
            map_by_hash.setdefault(map_hash, embedding)
        target_hashes.extend(batch["map_sha256"])
    source_matrix = torch.cat(source_embeddings)
    map_hashes = sorted(map_by_hash)
    map_matrix = torch.stack([map_by_hash[value] for value in map_hashes])
    similarities = source_matrix @ map_matrix.transpose(0, 1)
    target_indices = torch.tensor([map_hashes.index(value) for value in target_hashes])
    order = similarities.argsort(dim=1, descending=True)
    ranks = (order == target_indices[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    errors = torch.cat(point_errors) if point_errors else torch.tensor([math.inf])
    return {
        "recall_at_1": float((ranks <= 1).float().mean()),
        "recall_at_5": float((ranks <= min(5, len(map_hashes))).float().mean()),
        "median_rank": float(ranks.float().median()),
        "mean_rank": float(ranks.float().mean()),
        "point_error_normalized_mean": float(errors.mean()),
        "point_error_normalized_median": float(errors.median()),
        "rows": float(len(target_hashes)),
        "candidate_maps": float(len(map_hashes)),
    }


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _checkpoint(
    model: DinoV3CrossViewModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    epoch: int,
    global_step: int,
    best: dict[str, Any],
    model_weights_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model_weights_sha256": model_weights_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best": best,
    }


def run_training(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required for the full DINOv3 cross-view train")
    manifest = Path(report["manifest"])
    rows = _load_rows(manifest)
    by_split = {split: [row for row in rows if row["split"] == split] for split in REQUIRED_SPLITS}
    datasets = {
        split: CrossViewPairDataset(
            manifest, args.bundle_root, split, args.image_size, rows=by_split[split]
        )
        for split in REQUIRED_SPLITS
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            collate_fn=_collate,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    model = DinoV3CrossViewModel(args.model_path, args.projection_dimension).to(device)
    backbone_parameters = list(model.backbone.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr_ratio + 0.5 * (1 - args.min_lr_ratio) * (
            1 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    args.output.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    global_step = 0
    best: dict[str, Any] = {
        "epoch": 0,
        "recall_at_1": -1.0,
        "point_error_normalized_median": math.inf,
    }
    resume_path = (
        args.output / "checkpoints/last.pt"
        if args.resume_from == "auto"
        else (Path(args.resume_from).resolve() if args.resume_from else None)
    )
    if resume_path is not None and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best = dict(checkpoint["best"])
    elif args.resume_from and args.resume_from != "auto":
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    metrics_path = args.output / "metrics.csv"
    new_metrics = start_epoch == 1 or not metrics_path.is_file()
    metrics_handle = metrics_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics_handle,
        fieldnames=(
            "epoch",
            "train_loss",
            "retrieval_loss",
            "point_loss",
            "recall_at_1",
            "recall_at_5",
            "median_rank",
            "point_error_normalized_median",
            "elapsed_seconds",
        ),
    )
    if new_metrics:
        writer.writeheader()
    started = time.monotonic()
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = Counter()
            batches = 0
            for batch_index, batch in enumerate(loaders["train"], 1):
                source = batch["source_image"].to(device, non_blocking=True)
                maps = batch["map_image"].to(device, non_blocking=True)
                labels = batch["map_label"].to(device, non_blocking=True)
                targets = batch["target_xy"].to(device, non_blocking=True)
                point_valid = batch["point_valid"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    losses = cross_view_loss(model(source, maps), labels, targets, point_valid)
                if not all(bool(torch.isfinite(value).item()) for value in losses.values()):
                    raise RuntimeError(f"non-finite loss at epoch {epoch} batch {batch_index}")
                (losses["loss"] / args.gradient_accumulation_steps).backward()
                if batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(
                    loaders["train"]
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                for name, value in losses.items():
                    totals[name] += float(value.detach().cpu())
                batches += 1
            validation = _evaluate(model, loaders["validation"], device)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": totals["loss"] / batches,
                "retrieval_loss": totals["retrieval_loss"] / batches,
                "point_loss": totals["point_loss"] / batches,
                **validation,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            history.append(epoch_metrics)
            writer.writerow({name: epoch_metrics[name] for name in writer.fieldnames})
            metrics_handle.flush()
            score = (validation["recall_at_1"], -validation["point_error_normalized_median"])
            best_score = (best["recall_at_1"], -best["point_error_normalized_median"])
            improved = score > best_score
            if improved:
                best = {"epoch": epoch, **validation}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            state = _checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                global_step=global_step,
                best=best,
                model_weights_sha256=report["model_weights_sha256"],
            )
            _atomic_torch_save(state, args.output / "checkpoints/last.pt")
            if improved:
                _atomic_torch_save(state, args.output / "checkpoints/best.pt")
            print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
            if (
                args.early_stop_patience
                and epoch >= args.min_epochs
                and epochs_without_improvement >= args.early_stop_patience
            ):
                break
    finally:
        metrics_handle.close()

    best_checkpoint = torch.load(
        args.output / "checkpoints/best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = _evaluate(model, loaders["test"], device)
    final_model = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "training_kind": report["training_kind"],
        "model": model.state_dict(),
        "projection_dimension": args.projection_dimension,
        "image_size": args.image_size,
    }
    _atomic_torch_save(final_model, args.output / "final/model.pt")
    result = {
        "training_complete": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "fine_tuning_mode": report["fine_tuning_mode"],
        "best_validation": best,
        "held_out_test": test_metrics,
        "epochs_completed": history[-1]["epoch"] if history else start_epoch - 1,
        "global_steps": global_step,
        "final_model": str((args.output / "final/model.pt").resolve()),
        "promotion_ready": False,
        "promotion_blockers": report["promotion_blockers"],
    }
    _write_json(args.output / "training-result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "plan", "smoke", "train"))
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument(
        "--manifest-relpath",
        type=Path,
        default=DEFAULT_MANIFEST_RELPATH,
        help="Manifest path relative to --bundle-root.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-file-hashes", action="store_true")
    parser.add_argument("--verify-model-hash", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="Do not apply early stopping before this epoch.",
    )
    parser.add_argument("--projection-dimension", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--expected-manifest-sha256",
        help="Optional immutable manifest SHA-256. Omit for a newly prepared corpus.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "projection_dimension",
        "image_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.min_epochs < 0 or args.min_epochs > args.epochs:
        raise ValueError("min-epochs must be between zero and epochs")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    report = build_preflight_report(
        args.bundle_root,
        args.model_path,
        manifest_relpath=args.manifest_relpath,
        verify_file_hashes=args.verify_file_hashes,
        expected_manifest_sha256=args.expected_manifest_sha256,
        verify_model_hash=args.verify_model_hash,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "preflight-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["schema_ready"]:
        raise SystemExit(2)
    if args.command == "train" and not report["training_ready"]:
        raise RuntimeError(
            "full cross-view training requires a transient fire/smoke mask for every row"
        )
    plan = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": report["model_weights_sha256"],
        "model_license": MODEL_LICENSE,
        "training_kind": report["training_kind"],
        "fine_tuning_mode": report["fine_tuning_mode"],
        "dataset": {
            "manifest": report["manifest"],
            "manifest_sha256": report["manifest_sha256"],
            "split_counts": report["split_counts"],
            "source_counts": report["source_counts"],
        },
        "hyperparameters": {
            name: getattr(args, name)
            for name in (
                "epochs",
                "batch_size",
                "gradient_accumulation_steps",
                "learning_rate",
                "head_learning_rate",
                "weight_decay",
                "warmup_ratio",
                "min_lr_ratio",
                "max_grad_norm",
                "early_stop_patience",
                "min_epochs",
                "projection_dimension",
                "image_size",
                "seed",
            )
        },
        "selection_metric": "validation recall_at_1 then normalized point error",
        "held_out_test_policy": "evaluated_once_after_best_checkpoint_selection",
        "transient_policy": "mask dynamic fire/smoke pixels; never use them as geometry landmarks",
        "resume_policy": "explicit checkpoint or --resume-from auto",
    }
    _write_json(args.output / "training-plan.json", plan)
    if args.command in {"preflight", "plan"}:
        return
    if args.command == "smoke":
        rows = _load_rows(Path(report["manifest"]))
        train_rows = [row for row in rows if row["split"] == "train"]
        smoke_rows: list[dict[str, Any]] = []
        smoke_map_hashes: set[str] = set()
        for row in train_rows:
            map_hash = str(row["map_view"]["sha256"])
            if map_hash in smoke_map_hashes:
                continue
            smoke_rows.append(row)
            smoke_map_hashes.add(map_hash)
            if len(smoke_rows) == max(2, args.batch_size):
                break
        if len(smoke_rows) < 2:
            raise RuntimeError("smoke requires at least two distinct map targets")
        dataset = CrossViewPairDataset(
            Path(report["manifest"]), args.bundle_root, "train", args.image_size, rows=smoke_rows
        )
        loader = DataLoader(
            dataset, batch_size=min(args.batch_size, len(dataset)), collate_fn=_collate
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda" and not args.allow_cpu:
            raise RuntimeError("CUDA is required for the DINOv3 smoke")
        model = DinoV3CrossViewModel(args.model_path, args.projection_dimension).to(device)
        losses = finite_loss_probe(model, next(iter(loader)), device)
        smoke = {
            "passed": True,
            "device": str(device),
            "rows": len(dataset),
            "losses": losses,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        }
        _write_json(args.output / "smoke-report.json", smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return
    result = run_training(args, report)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
