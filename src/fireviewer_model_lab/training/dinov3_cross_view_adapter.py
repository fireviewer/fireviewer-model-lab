"""Full-parameter DINOv3 cross-view retrieval adapter for FireViewer.

The adapter learns on real UAV/ground-to-orthophoto pairs.  Dynamic fire and smoke
pixels are deliberately not treated as geometric landmarks: a future manifest may
provide ``transient_mask_relpath`` and those pixels are neutralised before encoding.
The released THU Ninuo corpus remains quarantined until its licence and leakage-safe
event splits are available.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]


def _load_manifest_rows(manifest: Path, split: str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"cross-view manifest has no {split} rows")
    return selected


class CrossViewPairDataset(Dataset[dict[str, Any]]):
    """Lazy loader for one real source view and its orthophoto target crop."""

    def __init__(
        self,
        manifest: Path,
        data_root: Path,
        split: str,
        image_size: int = 224,
        *,
        rows: Iterable[dict[str, Any]] | None = None,
    ) -> None:
        self.data_root = data_root.resolve()
        self.image_size = image_size
        self.rows = list(rows) if rows is not None else _load_manifest_rows(manifest, split)
        if not self.rows:
            raise ValueError(f"cross-view manifest has no {split} rows")
        retrieval_groups = sorted(
            {str(row.get("retrieval_group") or row["map_view"]["sha256"]) for row in self.rows}
        )
        self.map_label_by_group = {value: index for index, value in enumerate(retrieval_groups)}

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> Path:
        path = (self.data_root / value).resolve()
        if path != self.data_root and self.data_root not in path.parents:
            raise ValueError(f"manifest path escapes data root: {value}")
        return path

    def _image_tensor(self, relpath: str, transient_mask_relpath: str | None = None) -> Any:
        image = Image.open(self._path(relpath)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        if transient_mask_relpath:
            mask = Image.open(self._path(transient_mask_relpath)).convert("L")
            mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            transient = np.asarray(mask, dtype=np.uint8) > 0
            # ImageNet mean becomes zero after normalisation, avoiding an artificial edge colour.
            array[transient] = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - IMAGE_MEAN) / IMAGE_STD

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        map_hash = str(row["map_view"]["sha256"])
        retrieval_group = str(row.get("retrieval_group") or map_hash)
        point_valid = bool(row.get("point_target_valid", True))
        target = row["map_view"].get("optical_axis_ground_pixel_normalized", [0.5, 0.5])
        return {
            "source_image": self._image_tensor(
                row["source_view"]["image_relpath"],
                row.get("source_transient_mask_relpath") or row.get("transient_mask_relpath"),
            ),
            "map_image": self._image_tensor(
                row["map_view"]["image_relpath"],
                row.get("map_transient_mask_relpath"),
            ),
            "map_label": torch.tensor(self.map_label_by_group[retrieval_group], dtype=torch.long),
            "target_xy": torch.tensor(target, dtype=torch.float32),
            "point_valid": torch.tensor(point_valid, dtype=torch.bool),
            "sample_id": str(row["sample_id"]),
            "map_sha256": map_hash,
            "split_group": str(row["split_group"]),
        }


class DinoV3CrossViewModel(nn.Module):
    """Shared, fully trainable DINOv3 encoder with retrieval and point heads."""

    def __init__(self, model_path: Path | str, projection_dimension: int = 256) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, token=False
        )
        hidden = int(self.backbone.config.hidden_size)
        self.source_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, projection_dimension),
        )
        self.map_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, projection_dimension),
        )
        paired_dimension = hidden * 4
        self.point_head = nn.Sequential(
            nn.LayerNorm(paired_dimension),
            nn.Linear(paired_dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def _class_tokens(self, images: Any) -> Any:
        return self.backbone(pixel_values=images).last_hidden_state[:, 0]

    def encode_source(self, images: Any) -> tuple[Any, Any]:
        features = self._class_tokens(images)
        return nn.functional.normalize(self.source_projection(features), dim=-1), features

    def encode_map(self, images: Any) -> tuple[Any, Any]:
        features = self._class_tokens(images)
        return nn.functional.normalize(self.map_projection(features), dim=-1), features

    def forward(self, source_images: Any, map_images: Any) -> dict[str, Any]:
        # Concatenating both domains keeps one shared-backbone pass and one BN-free graph.
        features = self._class_tokens(torch.cat((source_images, map_images), dim=0))
        source_features, map_features = features.chunk(2, dim=0)
        source_embeddings = nn.functional.normalize(self.source_projection(source_features), dim=-1)
        map_embeddings = nn.functional.normalize(self.map_projection(map_features), dim=-1)
        paired = torch.cat(
            (
                source_features,
                map_features,
                torch.abs(source_features - map_features),
                source_features * map_features,
            ),
            dim=-1,
        )
        return {
            "source_embeddings": source_embeddings,
            "map_embeddings": map_embeddings,
            "target_xy": self.point_head(paired),
            "logit_scale": self.logit_scale.exp().clamp(max=100.0),
        }


def multi_positive_contrastive_loss(
    source_embeddings: Any,
    map_embeddings: Any,
    map_labels: Any,
    logit_scale: Any,
) -> Any:
    """Symmetric InfoNCE where repeated crops are positives, never false negatives."""

    logits = source_embeddings @ map_embeddings.transpose(0, 1) * logit_scale
    positives = map_labels[:, None].eq(map_labels[None, :])
    if not bool(positives.diagonal().all().item()):  # pragma: no cover - defensive gate
        raise ValueError("each cross-view pair must be a positive match")

    def direction_loss(scores: Any, positive_mask: Any) -> Any:
        numerator = torch.logsumexp(scores.masked_fill(~positive_mask, -torch.inf), dim=1)
        denominator = torch.logsumexp(scores, dim=1)
        return (denominator - numerator).mean()

    return 0.5 * (
        direction_loss(logits, positives)
        + direction_loss(logits.transpose(0, 1), positives.transpose(0, 1))
    )


def cross_view_loss(
    outputs: dict[str, Any],
    map_labels: Any,
    target_xy: Any,
    point_valid: Any | None = None,
) -> dict[str, Any]:
    retrieval = multi_positive_contrastive_loss(
        outputs["source_embeddings"],
        outputs["map_embeddings"],
        map_labels,
        outputs["logit_scale"],
    )
    point_rows = nn.functional.smooth_l1_loss(
        outputs["target_xy"], target_xy, beta=0.05, reduction="none"
    ).mean(dim=1)
    if point_valid is None:
        point = point_rows.mean()
    else:
        valid = point_valid.to(dtype=torch.bool)
        point = point_rows[valid].mean() if bool(valid.any()) else point_rows.sum() * 0.0
    total = retrieval + 0.5 * point
    return {"loss": total, "retrieval_loss": retrieval, "point_loss": point}


def finite_loss_probe(
    model: DinoV3CrossViewModel,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Run the mandatory forward/backward finite-loss gate without an optimiser step."""

    model.train()
    model.zero_grad(set_to_none=True)
    source = batch["source_image"].to(device)
    maps = batch["map_image"].to(device)
    labels = batch["map_label"].to(device)
    targets = batch["target_xy"].to(device)
    point_valid = batch.get("point_valid")
    if point_valid is not None:
        point_valid = point_valid.to(device)
    amp_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        losses = cross_view_loss(model(source, maps), labels, targets, point_valid)
    if not all(bool(torch.isfinite(value).item()) for value in losses.values()):
        raise RuntimeError("DINOv3 cross-view finite-loss probe failed")
    losses["loss"].backward()
    finite_gradients = all(
        bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not finite_gradients:
        raise RuntimeError("DINOv3 cross-view finite-gradient probe failed")
    result = {name: float(value.detach().cpu()) for name, value in losses.items()}
    model.zero_grad(set_to_none=True)
    return result
