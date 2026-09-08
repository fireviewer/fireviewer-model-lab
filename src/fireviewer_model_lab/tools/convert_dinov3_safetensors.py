#!/usr/bin/env python3
"""Convert a FireViewer DINOv3 training checkpoint to safe inference weights."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MODEL_SCHEMA_VERSION = 4
PRESENCE_LABELS = ("flame_visible", "smoke_visible")
REQUIRED_STATE_PREFIXES = (
    "backbone.",
    "decoder.",
    "segmentation_head.",
    "point_head.",
    "abstention_head.",
    "presence_head.",
)


def _validate_v4_contract(checkpoint: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    if checkpoint.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError(f"DINOv3 checkpoint model schema must be {MODEL_SCHEMA_VERSION}")
    if tuple(checkpoint.get("presence_labels") or ()) != PRESENCE_LABELS:
        raise ValueError("DINOv3 checkpoint presence label contract mismatch")
    missing = [
        prefix
        for prefix in REQUIRED_STATE_PREFIXES
        if not any(isinstance(key, str) and key.startswith(prefix) for key in state)
    ]
    if missing:
        raise ValueError(f"DINOv3 checkpoint is incomplete; missing state prefixes: {missing}")


def _checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, str]:
    metadata = {
        "format": "pt",
        "architecture": "DinoV3MultiTaskModel",
        "state_dict": "complete_backbone_and_heads",
    }
    for key in (
        "schema_version",
        "model_schema_version",
        "epoch",
        "model_revision",
        "manifest_sha256",
    ):
        value = checkpoint.get(key)
        if value is not None:
            metadata[key] = str(value)
    metadata["presence_labels"] = json.dumps(PRESENCE_LABELS)
    validation = checkpoint.get("validation")
    if isinstance(validation, Mapping):
        metadata["validation"] = json.dumps(validation, sort_keys=True, allow_nan=False)
    return metadata


def convert_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file, save_file

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The DINOv3 checkpoint must be a mapping")
    state = checkpoint.get("model", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("The DINOv3 checkpoint has no model state dictionary")
    _validate_v4_contract(checkpoint, state)

    tensors: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Invalid state entry: {key!r} ({type(value).__name__})")
        tensors[key] = value.detach().cpu().contiguous()

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, destination, metadata=_checkpoint_metadata(checkpoint))
    restored = load_file(destination, device="cpu")
    if restored.keys() != tensors.keys():
        raise RuntimeError("Safetensors key set differs from the source checkpoint")

    mismatches = []
    for key, source_tensor in tensors.items():
        restored_tensor = restored[key]
        if (
            restored_tensor.shape != source_tensor.shape
            or restored_tensor.dtype != source_tensor.dtype
            or not torch.equal(restored_tensor, source_tensor)
        ):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(f"Safetensors validation failed for {len(mismatches)} tensors")

    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "tensor_count": len(tensors),
        "parameter_values": sum(tensor.numel() for tensor in tensors.values()),
        "validated_exact": True,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_revision": checkpoint.get("model_revision"),
        "model_schema_version": checkpoint.get("model_schema_version"),
        "presence_labels": list(PRESENCE_LABELS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert_checkpoint(args.input, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
