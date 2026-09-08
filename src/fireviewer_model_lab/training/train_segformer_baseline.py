"""FireViewer SegFormer-B2 offline baseline entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.train_dinov3_multitask import build_preflight_report

DEFAULT_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
DEFAULT_REVISION = "de01bae28967510f9ddd496c60a969357195400c"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FireViewer SegFormer-B2 baseline")
    parser.add_argument("command", choices=("preflight", "plan", "smoke", "train"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=Path("data/training/segformer-b2-v2"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        min(
            args.epochs,
            args.batch_size,
            args.gradient_accumulation_steps,
            args.image_size,
        )
        <= 0
    ):
        raise ValueError("epochs, batch size, accumulation and image size must be positive")
    report = build_preflight_report(
        pointing_root=args.manifest.resolve().parent,
        multitask_manifest=args.manifest,
        data_root=args.data_root,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    report["model_family"] = "SegFormer-B2 baseline"
    report["fine_tuning_mode"] = "full_model_binary_segmentation"
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "preflight-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "preflight":
        if not report["training_ready"]:
            raise SystemExit(2)
        return
    plan = {
        "schema_version": 1,
        "model_family": report["model_family"],
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": report["multitask_manifest_sha256"],
        "data_root": str(args.data_root.resolve()),
        "shares_manifest_with": "DINOv3 multi-task",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
        },
        "training_ready": report["training_ready"],
    }
    _write_json(args.output / "training-plan.json", plan)
    if args.command == "plan":
        return
    if not report["training_ready"]:
        raise RuntimeError("SegFormer training gate failed")
    if args.command == "smoke":
        from fireviewer_model_lab.training.segformer_adapter import finite_loss_probe

        smoke = finite_loss_probe(
            manifest=args.manifest.resolve(),
            data_root=args.data_root.resolve(),
            model_id=args.model_id,
            model_revision=args.model_revision,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        _write_json(args.output / "smoke-report.json", smoke)
        print(json.dumps(smoke, ensure_ascii=False, indent=2, sort_keys=True))
        return
    from fireviewer_model_lab.training.segformer_adapter import run_training

    result = run_training(
        manifest=args.manifest.resolve(),
        data_root=args.data_root.resolve(),
        output=args.output.resolve(),
        model_id=args.model_id,
        model_revision=args.model_revision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        image_size=args.image_size,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
    )
    _write_json(args.output / "training-result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
