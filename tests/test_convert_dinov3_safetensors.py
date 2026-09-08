from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from fireviewer_model_lab.tools.convert_dinov3_safetensors import convert_checkpoint


def test_convert_checkpoint_preserves_complete_state(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.pt"
    destination = tmp_path / "model.safetensors"
    state = {
        "backbone.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "decoder.weight": torch.tensor([1.0]),
        "segmentation_head.bias": torch.tensor([1.0]),
        "point_head.bias": torch.tensor([-1.0]),
        "abstention_head.bias": torch.tensor([0.5]),
        "presence_head.bias": torch.tensor([1.0, -1.0]),
    }
    torch.save(
        {
            "schema_version": 2,
            "model_schema_version": 4,
            "presence_labels": ["flame_visible", "smoke_visible"],
            "model": state,
            "epoch": 5,
            "model_revision": "immutable-revision",
            "validation": {"loss": 0.25},
        },
        source,
    )

    report = convert_checkpoint(source, destination)

    restored = load_file(destination)
    assert report["validated_exact"] is True
    assert report["tensor_count"] == 6
    assert restored.keys() == state.keys()
    assert all(torch.equal(restored[key], value) for key, value in state.items())
    with safe_open(destination, framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata is not None
    assert metadata["architecture"] == "DinoV3MultiTaskModel"
    assert metadata["epoch"] == "5"
    assert metadata["model_schema_version"] == "4"
    assert json.loads(metadata["presence_labels"]) == ["flame_visible", "smoke_visible"]
