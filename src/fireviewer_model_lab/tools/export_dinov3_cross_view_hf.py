#!/usr/bin/env python3
"""Prepare, publish, and confirm the FireViewer DINOv3 Cross-View release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

DEFAULT_REPO = "fireviewer/dinov3-vitb16-cross-view-fireviewer-v1"
REQUIRED_REMOTE_FILES = {
    "README.md",
    "config.json",
    "dinov3_cross_view_adapter.py",
    "fireviewer_dinov3_cross_view.pt",
    "training-result.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def prepare_release(
    *, model: Path, training_result: Path, adapter: Path, output: Path
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Release directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    payload = _read_json(training_result)
    if payload.get("training_complete") is not True:
        raise ValueError("Training result is not marked complete")

    import torch

    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    required = {
        "schema_version",
        "model_id",
        "model_revision",
        "training_kind",
        "model",
        "projection_dimension",
        "image_size",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"Final model is missing fields: {missing}")
    if not checkpoint["model"]:
        raise ValueError("Final model state dict is empty")

    weights = output / "fireviewer_dinov3_cross_view.pt"
    os.link(model.resolve(), weights)
    shutil.copy2(adapter, output / "dinov3_cross_view_adapter.py")
    sanitized = dict(payload)
    sanitized["final_model"] = weights.name
    (output / "training-result.json").write_text(
        json.dumps(sanitized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    config = {
        "architectures": ["DinoV3CrossViewModel"],
        "base_model": checkpoint["model_id"],
        "base_model_revision": checkpoint["model_revision"],
        "image_size": checkpoint["image_size"],
        "projection_dimension": checkpoint["projection_dimension"],
        "torch_checkpoint": weights.name,
        "training_kind": checkpoint["training_kind"],
        "fine_tuning_mode": payload["fine_tuning_mode"],
        "model_state_keys": len(checkpoint["model"]),
        "custom_adapter": "dinov3_cross_view_adapter.py",
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    test = payload["held_out_test"]
    best = payload["best_validation"]
    (output / "README.md").write_text(
        f"""---
library_name: transformers
tags:
- fireviewer
- dinov3
- cross-view
- image-retrieval
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license
---

# FireViewer DINOv3 ViT-B/16 Cross-View v1

Full-parameter DINOv3 ViT-B/16 model trained for shared-encoder cross-view
retrieval and image-plane point regression. The published weight file contains
the complete backbone and FireViewer heads; it is not a LoRA/PEFT adapter and
does not require a merge step.

The training corpus combines Gaussians on Fire and synchronized independent
Camp Swift cameras. Wildfire3Data is not included in this release.

## Evaluation

- Best validation epoch: {best["epoch"]}
- Validation Recall@1: {best["recall_at_1"]:.8f}
- Validation Recall@5: {best["recall_at_5"]:.8f}
- Held-out test Recall@1: {test["recall_at_1"]:.8f}
- Held-out test Recall@5: {test["recall_at_5"]:.8f}
- Held-out median rank: {test["median_rank"]:.2f}
- Held-out median normalized point error: {test["point_error_normalized_median"]:.8f}

This is a trained research checkpoint, not a production-promoted geolocation
model. The remaining promotion gates are recorded in `training-result.json`.

## Loading

Instantiate `DinoV3CrossViewModel` from `dinov3_cross_view_adapter.py` with the
pinned DINOv3 base revision in `config.json`, then load the `model` state dict
from `fireviewer_dinov3_cross_view.pt`.

The DINOv3-derived weights remain subject to the DINOv3 license.
""",
        encoding="utf-8",
    )
    return config


def _token(path: Path) -> str:
    value = path.read_text(encoding="utf-8-sig").strip()
    if not value:
        raise ValueError(f"Token file is empty: {path}")
    return value


def confirm(repo_id: str, token: str) -> dict[str, Any]:
    info = HfApi(token=token).model_info(repo_id, files_metadata=True)
    siblings = info.siblings or []
    files = {item.rfilename for item in siblings}
    missing = sorted(REQUIRED_REMOTE_FILES.difference(files))
    if missing:
        raise RuntimeError(f"Remote model is missing files: {missing}")
    return {
        "repo_id": info.id,
        "private": info.private,
        "sha": info.sha,
        "files": len(files),
        "bytes": sum((item.size or 0) for item in siblings),
        "required_files_present": True,
    }


def push(output: Path, repo_id: str, token_file: Path, private: bool) -> dict[str, Any]:
    token = _token(token_file)
    api = HfApi(token=token)
    url = api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output,
        commit_message="Publish FireViewer DINOv3 Cross-View v1",
    )
    result = confirm(repo_id, token)
    result["url"] = str(url)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--training-result", type=Path, required=True)
    prepare.add_argument("--adapter", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
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
            model=args.model,
            training_result=args.training_result,
            adapter=args.adapter,
            output=args.output,
        )
    elif args.command == "push":
        result = push(args.output, args.repo_id, args.token_file, args.private)
    else:
        result = confirm(args.repo_id, _token(args.token_file))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
