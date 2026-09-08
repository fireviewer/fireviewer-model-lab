from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from fireviewer_model_lab.training.segformer_adapter import FireViewerSegmentationDataset, _loss


def test_segformer_uses_shared_relative_manifest(tmp_path: Path) -> None:
    Image.new("RGB", (8, 6), "red").save(tmp_path / "image.jpg")
    Image.new("L", (8, 6), 255).save(tmp_path / "mask.png")
    Image.new("L", (8, 6), 0).save(tmp_path / "valid-mask.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "shared:1",
                "split": "train",
                "image_relpath": "image.jpg",
                "mask_relpath": "mask.png",
                "valid_mask_relpath": "valid-mask.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = FireViewerSegmentationDataset(manifest, tmp_path, "train", 32)
    sample = dataset[0]
    assert sample["sample_id"] == "shared:1"
    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert tuple(sample["mask"].shape) == (1, 32, 32)
    assert float(sample["mask"].mean()) == 1.0
    assert float(sample["valid_mask"].mean()) == 0.0


def test_segformer_loss_ignores_invalid_pixels() -> None:
    logits = torch.tensor([[[[30.0, -30.0]]]], requires_grad=True)
    mask = torch.tensor([[[[0.0, 1.0]]]])
    valid = torch.zeros_like(mask)
    loss = _loss(logits, mask, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))
