from __future__ import annotations

import json
from pathlib import Path

import pytest
from fireviewer_model_lab.training.spatial_train_qwen import (
    TrainingSetupError,
    _checkpoint_due_steps,
    prepare,
    train,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _media(tmp_path: Path, name: str) -> tuple[str, str]:
    path = tmp_path / "media" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = name.encode()
    path.write_bytes(payload)
    import hashlib

    return path.relative_to(tmp_path).as_posix(), hashlib.sha256(payload).hexdigest()


def _pointing_row(
    *, split: str, image_relpath: str, sha256: str, targets: int
) -> dict[str, object]:
    return {
        "pointing_status": "point_candidate",
        "proposed_split": split,
        "sample_id": f"pointing-{split}",
        "source": {
            "image_relpath": image_relpath,
            "license": "CC-BY-4.0",
            "sha256": sha256,
        },
        "split_group": f"pointing-group-{split}",
        "targets": [
            {
                "semantic_anchor": "fire_base" if index == 0 else "smoke_column_base",
                "source_pixel_normalized": [0.25 + index * 0.1, 0.75],
                "target_id": f"target-{index}",
            }
            for index in range(targets)
        ],
        "training_eligibility": "weak_supervision_only_until_point_validation",
    }


def test_prepare_exports_only_fire_pointing_annotations(tmp_path: Path) -> None:
    pointing_image, pointing_sha = _media(tmp_path, "pointing.jpg")
    _write_jsonl(
        tmp_path / "corpus/fire-pointing-v0.1.0/manifest.jsonl",
        [
            _pointing_row(
                split="train",
                image_relpath=pointing_image,
                sha256=pointing_sha,
                targets=2,
            ),
            _pointing_row(
                split="validation",
                image_relpath=pointing_image,
                sha256=pointing_sha,
                targets=1,
            ),
            {
                **_pointing_row(
                    split="test",
                    image_relpath=pointing_image,
                    sha256=pointing_sha,
                    targets=1,
                ),
                "sample_id": "pointing-critical-like-test",
            },
        ],
    )
    path = tmp_path / "corpus/fire-pointing-v0.1.0/build-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"gates":{"training_ready":true}}', encoding="utf-8")

    report = prepare(tmp_path, verify_hashes=False)

    assert report["files"]["fire-pointing-train"]["rows"] == 2
    assert report["files"]["fire-pointing-validation"]["rows"] == 1
    assert "cross-view-registration-train" not in report["files"]
    assert "cross_view_registration" not in report["datasets"]
    assert report["critical_lots_included"] is False
    assert report["datasets"]["fire_pointing"]["production_ready"] is False
    prepared = list(
        map(
            json.loads,
            (tmp_path / "training/qwen3-vl-4b-spatial/annotations/fire-pointing-train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    )
    assert all(Path(record["image"]).is_absolute() for record in prepared)
    assert {record["firewarning"]["split"] for record in prepared} == {"train"}


def test_checkpoints_start_at_half_and_include_completion() -> None:
    checkpoints = _checkpoint_due_steps(100)

    assert min(checkpoints) == 50
    assert checkpoints[50] == 50
    assert checkpoints[100] == 100
    assert len(checkpoints) == 6


def test_real_train_is_locked_without_confirmation(tmp_path: Path) -> None:
    with pytest.raises(TrainingSetupError, match="real training is locked"):
        train(
            tmp_path,
            profile_name="fire-pointing",
            confirm_training=False,
            seed=1,
        )
