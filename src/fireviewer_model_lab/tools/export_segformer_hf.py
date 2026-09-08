"""Export and publish the trained FireViewer SegFormer baseline."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from fireviewer_model_lab.training.segformer_adapter import FireViewerSegFormer

DEFAULT_REPO = "fireviewer/segformer-b2-fire-smoke-baseline-v1"


def _best_metrics(metrics: Path) -> dict[str, float | int]:
    with metrics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"metrics file is empty: {metrics}")
    best = min(rows, key=lambda row: float(row["validation_loss"]))
    return {
        "epoch": int(best["epoch"]),
        "train_loss": float(best["train_loss"]),
        "validation_loss": float(best["validation_loss"]),
        "validation_iou": float(best["validation_iou"]),
        "validation_dice": float(best["validation_dice"]),
    }


def export_release(
    *,
    checkpoint: Path,
    base_model: Path,
    metrics: Path,
    training_result: Path,
    output: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    base_model = base_model.resolve()
    metrics = metrics.resolve()
    training_result = training_result.resolve()
    output = output.resolve()
    for required in (checkpoint, metrics, training_result):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not base_model.is_dir():
        raise FileNotFoundError(base_model)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_revision = str(payload["model_revision"])
    model = FireViewerSegFormer(str(base_model), model_revision)
    model.load_state_dict(payload["model"], strict=True)
    output.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output, safe_serialization=True)

    from transformers import SegformerImageProcessor

    processor = SegformerImageProcessor.from_pretrained(str(base_model), local_files_only=True)
    processor.save_pretrained(output)

    result = json.loads(training_result.read_text(encoding="utf-8"))
    best = _best_metrics(metrics)
    release = {
        "schema_version": 1,
        "architecture": "SegformerForSemanticSegmentation",
        "base_model": "nvidia/segformer-b2-finetuned-ade-512-512",
        "base_model_revision": model_revision,
        "fine_tuning_mode": "full_model_all_parameters_trainable",
        "label": "fire_or_smoke",
        "selection": "minimum_validation_loss",
        "best_validation": best,
        "held_out_test": result["test_metrics"],
        "train_rows": int(result["train_rows"]),
        "validation_rows": int(result["validation_rows"]),
        "test_rows": int(result["test_rows"]),
    }
    (output / "training-result.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(metrics, output / "metrics.csv")
    (output / "README.md").write_text(
        f"""---
library_name: transformers
pipeline_tag: image-segmentation
tags:
- fireviewer
- segformer
- wildfire
- fire
- smoke
---

# FireViewer SegFormer-B2 fire and smoke baseline

This repository contains the FireViewer SegFormer-B2 baseline trained with full
parameter fine-tuning for binary fire-or-smoke segmentation.

## Held-out test

- IoU: `{float(result["test_metrics"]["iou"]):.6f}`
- Dice: `{float(result["test_metrics"]["dice"]):.6f}`
- Loss: `{float(result["test_metrics"]["loss"]):.6f}`
- Images: `{int(result["test_rows"])}`

The selected checkpoint is epoch `{best["epoch"]}`, chosen by minimum validation
loss. The release uses standard Transformers files and can be loaded directly
with `SegformerForSemanticSegmentation.from_pretrained(...)` and
`SegformerImageProcessor.from_pretrained(...)`.

This model is an offline baseline for comparison with the FireViewer DINOv3
multi-task model. Production promotion requires the separate FireViewer review
and shadow-benchmark gates.
""",
        encoding="utf-8",
    )
    return release


def push_release(*, output: Path, repo_id: str, token_file: Path, private: bool) -> str:
    from huggingface_hub import HfApi

    token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"empty Hugging Face token file: {token_file}")
    api = HfApi(token=token)
    repo = api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output.resolve()),
        commit_message="Publish FireViewer SegFormer-B2 baseline",
    )
    return str(repo)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--base-model", type=Path, required=True)
    export.add_argument("--metrics", type=Path, required=True)
    export.add_argument("--training-result", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    push = subparsers.add_parser("push")
    push.add_argument("--output", type=Path, required=True)
    push.add_argument("--repo-id", default=DEFAULT_REPO)
    push.add_argument("--token-file", type=Path, required=True)
    push.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.command == "export":
        result = export_release(
            checkpoint=args.checkpoint,
            base_model=args.base_model,
            metrics=args.metrics,
            training_result=args.training_result,
            output=args.output,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(
        push_release(
            output=args.output,
            repo_id=args.repo_id,
            token_file=args.token_file,
            private=args.private,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
