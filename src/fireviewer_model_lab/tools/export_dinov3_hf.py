"""Export, merge and publish the FireViewer DINOv3 multi-task checkpoint.

The trainer saves a complete PyTorch state dict (the DINOv3 backbone and the
three FireViewer heads).  There is no PEFT/LoRA adapter to merge: ``merge``
therefore creates a self-contained release directory and records immutable
provenance before the directory is uploaded to Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
DEFAULT_REPO = "fireviewer/dinov3-vitb16-multitask-fireviewer-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_best(metrics: Path) -> dict[str, Any]:
    import csv

    with metrics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"metrics file is empty: {metrics}")
    best = min(rows, key=lambda row: float(row["validation_loss"]))
    return {
        "epoch": int(best["epoch"]),
        "train_loss": float(best["train_loss"]),
        "validation_loss": float(best["validation_loss"]),
    }


def merge_checkpoint(
    *,
    checkpoint: Path,
    output: Path,
    metrics: Path,
    manifest: Path,
    model_revision: str,
    adapter_source: Path | None = None,
) -> dict[str, Any]:
    """Materialize a complete, uploadable release from the best checkpoint."""

    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not metrics.is_file() or not manifest.is_file():
        raise FileNotFoundError("metrics and manifest are required for provenance")
    output.mkdir(parents=True, exist_ok=True)
    final_weights = output / "fireviewer_dinov3_multitask.pt"
    shutil.copy2(checkpoint, final_weights)
    best = _load_best(metrics)
    metadata = {
        "architecture": "DinoV3MultiTaskModel",
        "base_model": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "base_model_revision": model_revision,
        "checkpoint_source": checkpoint.name,
        "checkpoint_sha256": _sha256(final_weights),
        "dataset_manifest_sha256": _sha256(manifest),
        "fine_tuning_mode": "full_model_all_parameters_trainable",
        "heads": ["segmentation", "anchor_heatmap", "visual_abstention"],
        "selection": "minimum_validation_loss",
        **best,
        "schema_version": 1,
    }
    (output / "best-checkpoint.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("training-plan.json", "training-result.json", "adaptation-report.json"):
        source = metrics.parent / name
        if name == "adaptation-report.json" and not source.is_file():
            source = manifest.parent / name
        if source.is_file():
            shutil.copy2(source, output / name)
    (output / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DinoV3MultiTaskModel"],
                "base_model": metadata["base_model"],
                "base_model_revision": model_revision,
                "fireviewer_heads": metadata["heads"],
                "torch_checkpoint": final_weights.name,
                "transformers_compatibility": "custom_fireviewer_model_lab.training.dinov3_adapter",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if adapter_source is not None:
        adapter_source = adapter_source.resolve()
        if not adapter_source.is_file():
            raise FileNotFoundError(adapter_source)
        shutil.copy2(adapter_source, output / "dinov3_adapter.py")
    return metadata


def write_readme(output: Path, metadata: dict[str, Any]) -> None:
    (output / "README.md").write_text(
        f"""---
library_name: transformers
tags:
- fireviewer
- dinov3
- segmentation
- visual-grounding
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license
---

# FireViewer DINOv3 ViT-B/16 multi-task

This repository contains the FireViewer full-parameter multi-task checkpoint
based on the gated DINOv3 ViT-B/16 pretraining revision
`{metadata["base_model_revision"]}`. It exposes segmentation, anchor heatmap
pointing, and explicit visual-abstention heads through the FireViewer adapter.

The DINOv3-derived weights remain subject to the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
The training manifest combines the FireViewer Boreal corpus, Camp Swift,
RxCADRE, and FireSentry. It contains strong, weak-teacher, sensor-derived, and
temporal-negative annotations. Source rights and the DINOv3 redistribution
terms must be reviewed independently before any downstream redistribution or
commercial use.

The release was selected by minimum validation loss: epoch
`{metadata["epoch"]}` with validation loss `{metadata["validation_loss"]:.10f}`.
The weight file is a complete PyTorch state dict; no LoRA/PEFT adapter remains
to be merged.

## Provenance

- Dataset manifest SHA-256: `{metadata["dataset_manifest_sha256"]}`
- Final checkpoint SHA-256: `{metadata["checkpoint_sha256"]}`
- Training mode: `{metadata["fine_tuning_mode"]}`
- Weak/strong annotation quality remains a promotion gate; this repository is
  a trained challenger, not an automatic production promotion.

## Loading

Load `fireviewer_dinov3_multitask.pt` with
`fireviewer_model_lab.training.dinov3_adapter.DinoV3MultiTaskModel` and the pinned base-model
revision. The companion JSON files contain the immutable release metadata.
""",
        encoding="utf-8",
    )


def push(output: Path, repo_id: str, token_file: Path, private: bool) -> str:
    from huggingface_hub import HfApi

    token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"empty Hugging Face token file: {token_file}")
    api = HfApi(token=token)
    repo = api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(output),
        commit_message="Publish FireViewer DINOv3 multi-task release",
    )
    return str(repo)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    merge = sub.add_parser("merge", help="materialize the full final checkpoint")
    merge.add_argument("--checkpoint", type=Path, required=True)
    merge.add_argument("--metrics", type=Path, required=True)
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--model-revision", default=MODEL_REVISION)
    merge.add_argument("--adapter-source", type=Path)
    publish = sub.add_parser("push", help="create/update the HF model repo")
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--repo-id", default=DEFAULT_REPO)
    publish.add_argument("--token-file", type=Path, required=True)
    publish.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.command == "merge":
        metadata = merge_checkpoint(
            checkpoint=args.checkpoint,
            output=args.output,
            metrics=args.metrics,
            manifest=args.manifest,
            model_revision=args.model_revision,
            adapter_source=args.adapter_source,
        )
        write_readme(args.output, metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(push(args.output, args.repo_id, args.token_file, args.private))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
