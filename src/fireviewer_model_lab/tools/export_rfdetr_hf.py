#!/usr/bin/env python3
"""Prepare, publish, and verify a self-contained FireViewer RF-DETR release."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

DEFAULT_REPO = "fireviewer/rf-detr-large-ground-fire-smoke-v2"
CHECKPOINT_NAME = "checkpoint_best_total.pth"
MODEL_RELEASES = {
    "RFDETRLarge": {
        "pretrain_weights": "rf-detr-large-2026.pth",
        "title": "FireViewer RF-DETR Large Ground Fire/Smoke v2",
        "onnx": "rfdetr-large.onnx",
    },
    "RFDETRSmall": {
        "pretrain_weights": "rf-detr-small.pth",
        "title": "FireViewer RF-DETR Small Ground Elite Fire/Smoke v1",
        "onnx": "rfdetr-small.onnx",
    },
}
REQUIRED_REMOTE_FILES = {
    "README.md",
    "config.json",
    CHECKPOINT_NAME,
    "metrics.json",
    "requirements.txt",
    "inference_onnx.py",
}
MODEL_CONFIG_KEYS = {
    "amp",
    "backbone_lora",
    "bbox_reparam",
    "ca_nheads",
    "cls_loss_coef",
    "compile",
    "dec_layers",
    "dec_n_points",
    "dual_projector",
    "dual_projector_kp_only",
    "encoder",
    "freeze_encoder",
    "fused_optimizer",
    "gradient_checkpointing",
    "group_detr",
    "grouppose_keypoint_dim_downscale",
    "hidden_dim",
    "ia_bce_loss",
    "inter_instance_kp_attn",
    "keypoint_cross_attn",
    "layer_norm",
    "license",
    "lite_refpoint_refine",
    "mask_downsample_ratio",
    "model_name",
    "num_channels",
    "num_classes",
    "num_decoder_registers",
    "num_keypoints_per_class",
    "num_queries",
    "num_select",
    "num_windows",
    "out_feature_indexes",
    "patch_size",
    "positional_encoding_size",
    "postprocess_trace_alpha",
    "projector_scale",
    "resolution",
    "sa_nheads",
    "segmentation_head",
    "two_stage",
    "use_grouppose_keypoints",
}
ARG_KEYS = {
    "class_names",
    "group_detr",
    "num_classes",
    "num_queries",
    "pretrain_weights",
}
PRIVATE_PATH = re.compile(r"(?:^[A-Za-z]:[\\/]|(?:^|[\\/])Users[\\/])")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_token(path: Path) -> str:
    value = path.read_text(encoding="utf-8-sig").strip()
    if not value:
        raise ValueError(f"Token file is empty: {path}")
    return value


def _last_metric(rows: list[dict[str, str]], name: str) -> float:
    for row in reversed(rows):
        value = row.get(name, "").strip()
        if value:
            return float(value)
    raise ValueError(f"Metric is absent from metrics.csv: {name}")


def _metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metrics.csv is empty")
    validation_rows = [row for row in rows if row.get("val/mAP_50_95", "").strip()]
    if not validation_rows:
        raise ValueError("metrics.csv has no completed validation epoch")
    final_epoch = max(int(float(row["epoch"])) for row in validation_rows) + 1
    final_step = max(int(float(row["step"])) for row in rows)
    metrics = {
        "training_complete": True,
        "epochs_completed": final_epoch,
        "final_step": final_step,
        "validation": {
            "map_50": _last_metric(rows, "val/mAP_50"),
            "map_50_95": _last_metric(rows, "val/mAP_50_95"),
            "ema_map_50": _last_metric(rows, "val/ema_mAP_50"),
            "ema_map_50_95": _last_metric(rows, "val/ema_mAP_50_95"),
            "f1": _last_metric(rows, "val/F1"),
            "precision": _last_metric(rows, "val/precision"),
            "recall": _last_metric(rows, "val/recall"),
            "ap_per_class": {
                "flame_visible": _last_metric(rows, "val/AP/flame_visible"),
                "smoke_visible": _last_metric(rows, "val/AP/smoke_visible"),
            },
        },
        "final_train_loss": _last_metric(rows, "train/loss"),
    }
    if any(row.get("test/mAP_50_95", "").strip() for row in rows):
        metrics["test"] = {
            "map_50": _last_metric(rows, "test/mAP_50"),
            "map_50_95": _last_metric(rows, "test/mAP_50_95"),
            "map_75": _last_metric(rows, "test/mAP_75"),
            "mar": _last_metric(rows, "test/mAR"),
            "f1": _last_metric(rows, "test/F1"),
            "precision": _last_metric(rows, "test/precision"),
            "recall": _last_metric(rows, "test/recall"),
            "ap_per_class": {
                "flame_visible": _last_metric(rows, "test/AP/flame_visible"),
                "smoke_visible": _last_metric(rows, "test/AP/smoke_visible"),
            },
        }
    return metrics


def _assert_no_private_paths(value: Any, context: str = "release") -> None:
    if isinstance(value, str):
        if PRIVATE_PATH.search(value):
            raise ValueError(f"Private filesystem path found in {context}")
    elif isinstance(value, dict):
        for key, nested in value.items():
            _assert_no_private_paths(nested, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_no_private_paths(nested, f"{context}[{index}]")


def _test_markdown(metrics: dict[str, Any]) -> str:
    test = metrics.get("test")
    if not isinstance(test, dict):
        return "## Held-out test\n\nNo held-out test metrics were recorded for this release."
    return f"""## Held-out test

- mAP@50: {test["map_50"]:.8f}
- mAP@50:95: {test["map_50_95"]:.8f}
- mAP@75: {test["map_75"]:.8f}
- mAR: {test["mar"]:.8f}
- F1: {test["f1"]:.8f}
- Precision: {test["precision"]:.8f}
- Recall: {test["recall"]:.8f}
- Flame AP: {test["ap_per_class"]["flame_visible"]:.8f}
- Smoke AP: {test["ap_per_class"]["smoke_visible"]:.8f}"""


def prepare_release(
    *,
    checkpoint: Path,
    training_config: Path,
    metrics_csv: Path,
    output: Path,
    repo_id: str = DEFAULT_REPO,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Release directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import torch

    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = source.get("model")
    args = source.get("args")
    if not isinstance(model, dict) or not model:
        raise ValueError("RF-DETR checkpoint has no model state")
    if not isinstance(args, dict):
        raise ValueError("RF-DETR checkpoint args must be a dictionary")
    model_name = source.get("model_name")
    if model_name not in MODEL_RELEASES:
        raise ValueError(f"Unsupported RF-DETR checkpoint model: {model_name!r}")
    release = MODEL_RELEASES[model_name]

    saved_config = _read_json(training_config)
    raw_model_config = saved_config.get("model_config")
    if not isinstance(raw_model_config, dict):
        raise ValueError("training_config.json has no model_config")
    model_config = {
        key: value for key, value in raw_model_config.items() if key in MODEL_CONFIG_KEYS
    }
    model_config["device"] = "cpu"
    clean_args = {key: args[key] for key in ARG_KEYS if key in args}
    clean_args["pretrain_weights"] = release["pretrain_weights"]
    clean_args["class_names"] = ["flame_visible", "smoke_visible"]
    clean_args["num_classes"] = 2
    clean_args["num_queries"] = int(model_config["num_queries"])
    clean_args["group_detr"] = int(model_config["group_detr"])

    clean_checkpoint = {
        "model": model,
        "args": clean_args,
        "model_config": model_config,
        "model_name": model_name,
        "rfdetr_version": str(source.get("rfdetr_version", "1.8.3")),
    }
    _assert_no_private_paths(clean_checkpoint)
    checkpoint_output = output / CHECKPOINT_NAME
    torch.save(clean_checkpoint, checkpoint_output)

    metrics = _metrics(metrics_csv)
    train_config = saved_config.get("train_config")
    notes = train_config.get("notes") if isinstance(train_config, dict) else None
    dataset_id = notes.get("dataset") if isinstance(notes, dict) else None
    if not isinstance(dataset_id, str) or not dataset_id.startswith("fireviewer/"):
        raise ValueError("training_config.json has no publishable FireViewer dataset id")
    metrics["checkpoint"] = CHECKPOINT_NAME
    metrics["checkpoint_selection"] = "best_total"
    metrics["dataset"] = dataset_id
    metrics["classes"] = clean_args["class_names"]
    metrics["rfdetr_version"] = clean_checkpoint["rfdetr_version"]
    _assert_no_private_paths(metrics)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    hub_config = {
        "architectures": [model_name],
        "library_name": "rfdetr",
        "rfdetr_version": clean_checkpoint["rfdetr_version"],
        "checkpoint": CHECKPOINT_NAME,
        "classes": clean_args["class_names"],
        "num_classes": 2,
        "image_size": 512,
        "dataset": metrics["dataset"],
        "checkpoint_is_full_model": True,
        "merge_required": False,
        "onnx": {
            "file": release["onnx"],
            "opset": 17,
            "dynamic_batch": True,
            "input_shape": ["batch", 3, 512, 512],
        },
    }
    (output / "config.json").write_text(
        json.dumps(hub_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "requirements.txt").write_text(
        f"rfdetr[onnx]=={clean_checkpoint['rfdetr_version']}\n", encoding="utf-8"
    )
    inference_source = Path(__file__).with_name("rfdetr_onnx_inference.py")
    if not inference_source.is_file():
        raise FileNotFoundError(inference_source)
    shutil.copy2(inference_source, output / "inference_onnx.py")

    validation = metrics["validation"]
    (output / "README.md").write_text(
        f"""---
library_name: rfdetr
license: apache-2.0
pipeline_tag: object-detection
tags:
- fireviewer
- rf-detr
- wildfire
datasets:
- {dataset_id}
---

# {release["title"]}

{model_name} trained to detect visible flames and visible smoke in
ground-view imagery. The release contains an ONNX model ready for inference
and the selected full PyTorch checkpoint. It is not an adapter and does not
require a merge step.

## Classes

1. `flame_visible`
2. `smoke_visible`

## Validation

- EMA mAP@50: {validation["ema_map_50"]:.8f}
- EMA mAP@50:95: {validation["ema_map_50_95"]:.8f}
- mAP@50: {validation["map_50"]:.8f}
- mAP@50:95: {validation["map_50_95"]:.8f}
- F1: {validation["f1"]:.8f}
- Precision: {validation["precision"]:.8f}
- Recall: {validation["recall"]:.8f}

{_test_markdown(metrics)}

The run completed {metrics["epochs_completed"]} epochs and {metrics["final_step"]}
optimizer steps. Detailed metrics are available in `metrics.json`.

## ONNX inference

```bash
pip install -r requirements.txt
python inference_onnx.py --model {release["onnx"]} --image image.jpg --threshold 0.30
```

The ONNX graph uses opset 17, a dynamic batch dimension, and a fixed spatial
input of `512x512`. It returns normalized boxes and class logits; the companion
script performs the matching preprocessing and postprocessing.

## PyTorch loading

```python
from huggingface_hub import hf_hub_download
from rfdetr import RFDETR

checkpoint = hf_hub_download(
    repo_id="{repo_id}",
    filename="{CHECKPOINT_NAME}",
)
model = RFDETR.from_checkpoint(checkpoint, device="cuda")
```

Use `device="cpu"` on a machine without CUDA. Only load PyTorch checkpoints
from trusted sources.
""",
        encoding="utf-8",
    )
    return {
        "checkpoint": str(checkpoint_output),
        "checkpoint_bytes": checkpoint_output.stat().st_size,
        "model_state_entries": len(model),
        "metrics": metrics,
        "merge_required": False,
        "repo_id": repo_id,
    }


def confirm(repo_id: str, token: str) -> dict[str, Any]:
    info = HfApi(token=token).model_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    files = {item.rfilename for item in siblings}
    missing = sorted(REQUIRED_REMOTE_FILES.difference(files))
    if missing:
        raise RuntimeError(f"Remote model is missing files: {missing}")
    if not any(name.endswith(".onnx") for name in files):
        raise RuntimeError("Remote model is missing an ONNX inference graph")
    return {
        "repo_id": info.id,
        "private": info.private,
        "revision": info.sha,
        "files": len(files),
        "bytes": sum((item.size or 0) for item in siblings),
        "required_files_present": True,
    }


def push(output: Path, repo_id: str, token_file: Path, private: bool) -> dict[str, Any]:
    token = _read_token(token_file)
    api = HfApi(token=token)
    url = api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output,
        commit_message="Publish FireViewer RF-DETR ground-view release",
    )
    result = confirm(repo_id, token)
    result["url"] = str(url)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--training-config", type=Path, required=True)
    prepare.add_argument("--metrics", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--repo-id", default=DEFAULT_REPO)
    publish = sub.add_parser("push")
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--repo-id", default=DEFAULT_REPO)
    publish.add_argument("--token-file", type=Path, required=True)
    publish.add_argument("--private", action="store_true")
    verify = sub.add_parser("confirm")
    verify.add_argument("--repo-id", default=DEFAULT_REPO)
    verify.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        result = prepare_release(
            checkpoint=args.checkpoint,
            training_config=args.training_config,
            metrics_csv=args.metrics,
            output=args.output,
            repo_id=args.repo_id,
        )
    elif args.command == "push":
        result = push(args.output, args.repo_id, args.token_file, args.private)
    else:
        result = confirm(args.repo_id, _read_token(args.token_file))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
