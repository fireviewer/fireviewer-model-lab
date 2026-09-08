"""Read-only loader for the published Boreal DINOv3 teacher checkpoint.

This module intentionally preserves the exact 224 px, unnormalised preprocessing
and simple patch heads used by the immutable v1 checkpoint.  It is only used to
produce explicitly weak pseudo-labels for new campaign sources; the v2 trainer
uses the DPT-style architecture in :mod:`fireviewer_model_lab.training.dinov3_adapter`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class LegacyDinoV3Teacher(nn.Module):
    """Exact inference graph of ``fireviewer-dinov3-...-boreal-v1``."""

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        super().__init__()
        from transformers import AutoModel

        local = Path(model_id).exists()
        self.backbone = AutoModel.from_pretrained(
            model_id,
            revision=None if local else revision,
            local_files_only=local,
            token=not local,
        )
        hidden = int(self.backbone.config.hidden_size)
        self.register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 4))
        self.patch_size = int(getattr(self.backbone.config, "patch_size", 16))
        self.segmentation_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.point_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.abstention_head = nn.Linear(hidden, 1)

    def _patch_map(self, outputs: Any, height: int, width: int) -> tuple[Any, Any]:
        tokens = outputs.last_hidden_state
        patch_tokens = tokens[:, 1 + self.register_tokens :]
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        expected = grid_h * grid_w
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"legacy DINOv3 patch grid mismatch: {patch_tokens.shape[1]} != {expected}"
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], grid_h, grid_w
        )
        return feature_map, tokens[:, 0]

    def forward(self, images: Any) -> dict[str, Any]:
        outputs = self.backbone(pixel_values=images)
        feature_map, class_token = self._patch_map(outputs, images.shape[-2], images.shape[-1])
        target_size = images.shape[-2:]
        return {
            "segmentation_logits": nn.functional.interpolate(
                self.segmentation_head(feature_map),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ),
            "point_logits": nn.functional.interpolate(
                self.point_head(feature_map),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ),
            "abstention_logits": self.abstention_head(class_token).squeeze(-1),
        }


def load_published_teacher(
    *, model_id: str, revision: str, checkpoint: Path, device: torch.device
) -> LegacyDinoV3Teacher:
    """Load the immutable v1 weights without changing their training graph."""

    model = LegacyDinoV3Teacher(model_id, revision)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"legacy teacher checkpoint has no model state: {checkpoint}")
    # The checkpoint predates the native Transformers DINOv3ViT integration.
    # Its encoder blocks lived under ``backbone.model.layer`` while the native
    # class exposes ``backbone.layer``.  Tensor names and shapes are otherwise
    # identical, so remap only that historical wrapper prefix and stay strict.
    state = {
        (
            "backbone." + key.removeprefix("backbone.model.")
            if key.startswith("backbone.model.")
            else key
        ): value
        for key, value in payload["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device)
