from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fireviewer_model_lab.training.dinov3_cross_view_adapter import (
    CrossViewPairDataset,
    cross_view_loss,
    multi_positive_contrastive_loss,
)
from fireviewer_model_lab.training.train_dinov3_cross_view import MODEL_REVISION, build_preflight_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_image(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((18, 24, 3), value, dtype=np.uint8)).save(path)
    return _sha256(path)


def _row(root: Path, split: str, value: int) -> dict[str, object]:
    source = root / f"images/{split}-source.jpg"
    map_image = root / f"images/{split}-map.jpg"
    source_sha = _write_image(source, value)
    map_sha = _write_image(map_image, value + 1)
    source_mask = root / f"masks/{split}-source.png"
    map_mask = root / f"masks/{split}-map.png"
    source_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((18, 24), dtype=np.uint8)).save(source_mask)
    Image.fromarray(np.zeros((18, 24), dtype=np.uint8)).save(map_mask)
    return {
        "sample_id": f"sample-{split}",
        "family": "cross_view_registration",
        "split": split,
        "split_group": f"group-{split}",
        "source_id": "fixture",
        "license": "CC0-1.0",
        "consent_basis": {"kind": "source_license", "reference": "fixture"},
        "operational_incident": False,
        "source_transient_mask_relpath": source_mask.relative_to(root).as_posix(),
        "map_transient_mask_relpath": map_mask.relative_to(root).as_posix(),
        "source_view": {
            "image_relpath": source.relative_to(root).as_posix(),
            "sha256": source_sha,
        },
        "map_view": {
            "image_relpath": map_image.relative_to(root).as_posix(),
            "sha256": map_sha,
            "optical_axis_ground_pixel_normalized": [0.25, 0.75],
        },
    }


def _write_model_fixture(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    for name in ("config.json", "model.safetensors", "preprocessor_config.json"):
        (model / name).write_bytes(b"fixture")
    metadata = model / ".cache/huggingface/download/config.json.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{MODEL_REVISION}\nfixture\n", encoding="utf-8")
    return model


def test_dataset_loads_pair_and_neutralises_transient_pixels(tmp_path: Path) -> None:
    row = _row(tmp_path, "train", 30)
    source_mask = tmp_path / str(row["source_transient_mask_relpath"])
    map_mask = tmp_path / str(row["map_transient_mask_relpath"])
    source_mask_array = np.zeros((18, 24), dtype=np.uint8)
    source_mask_array[:, :12] = 255
    Image.fromarray(source_mask_array).save(source_mask)
    map_mask_array = np.zeros((18, 24), dtype=np.uint8)
    map_mask_array[:, 12:] = 255
    Image.fromarray(map_mask_array).save(map_mask)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    sample = CrossViewPairDataset(manifest, tmp_path, "train", 32)[0]

    assert tuple(sample["source_image"].shape) == (3, 32, 32)
    assert tuple(sample["map_image"].shape) == (3, 32, 32)
    assert torch.allclose(sample["source_image"][:, 8, 4], torch.zeros(3), atol=1e-5)
    assert torch.allclose(sample["map_image"][:, 8, 28], torch.zeros(3), atol=1e-5)
    assert torch.equal(sample["target_xy"], torch.tensor([0.25, 0.75]))
    assert bool(sample["point_valid"])


def test_dataset_groups_synchronised_views_and_disables_unknown_point_target(
    tmp_path: Path,
) -> None:
    rows = [_row(tmp_path, "train", value) for value in (30, 60)]
    for index, row in enumerate(rows):
        row["sample_id"] = f"synchronised-{index}"
        row["retrieval_group"] = "camp-swift:block-1:t-12"
        row["point_target_valid"] = False
        row["map_view"].pop("optical_axis_ground_pixel_normalized")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    dataset = CrossViewPairDataset(manifest, tmp_path, "train", 32)
    first, second = dataset[0], dataset[1]

    assert first["map_label"].item() == second["map_label"].item()
    assert not bool(first["point_valid"])
    assert torch.equal(first["target_xy"], torch.tensor([0.5, 0.5]))


def test_cross_view_loss_ignores_point_head_when_target_is_unknown() -> None:
    outputs = {
        "source_embeddings": torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1
        ),
        "map_embeddings": torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1
        ),
        "target_xy": torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True),
        "logit_scale": torch.tensor(10.0),
    }

    losses = cross_view_loss(
        outputs,
        torch.tensor([0, 1]),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        torch.tensor([False, False]),
    )

    assert losses["point_loss"].item() == 0.0
    assert torch.isfinite(losses["loss"])


def test_multi_positive_loss_accepts_repeated_target_without_false_negative() -> None:
    source = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), dim=-1
    ).requires_grad_()
    maps = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), dim=-1
    ).requires_grad_()
    labels = torch.tensor([7, 7, 9])

    loss = multi_positive_contrastive_loss(source, maps, labels, torch.tensor(10.0))
    loss.backward()

    assert torch.isfinite(loss)
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert maps.grad is not None and torch.isfinite(maps.grad).all()


def test_preflight_accepts_isolated_splits_and_pinned_local_model(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus/cross-view-registration-v0.1.0/manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [
        _row(tmp_path, split, 20 + index * 20)
        for index, split in enumerate(("train", "validation", "test"))
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    model = _write_model_fixture(tmp_path)

    report = build_preflight_report(
        tmp_path,
        model,
        verify_file_hashes=True,
        expected_manifest_sha256=_sha256(manifest),
        verify_model_hash=False,
    )

    assert report["training_ready"] is True
    assert report["errors"] == []
    assert report["verified_image_files"] == 6
    assert report["split_counts"] == {"test": 1, "train": 1, "validation": 1}


def test_preflight_rejects_group_leakage_and_operational_media(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus/cross-view-registration-v0.1.0/manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = [
        _row(tmp_path, split, 20 + index * 20)
        for index, split in enumerate(("train", "validation", "test"))
    ]
    rows[1]["split_group"] = rows[0]["split_group"]
    rows[2]["operational_incident"] = True
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = build_preflight_report(
        tmp_path,
        _write_model_fixture(tmp_path),
        expected_manifest_sha256=_sha256(manifest),
        verify_model_hash=False,
    )

    assert report["training_ready"] is False
    assert any(error.startswith("split_group_leakage:") for error in report["errors"])
    assert "operational_incident_forbidden:3" in report["errors"]
