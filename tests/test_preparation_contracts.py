from __future__ import annotations

import json
from pathlib import Path

from fireviewer_model_lab.training.benchmark_moge import build_preflight
from fireviewer_model_lab.training.prepare_inference_contracts import build_contract


def test_moge_preflight_requires_all_isolated_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "moge.jsonl"
    rows = [
        {
            "sample_id": f"sample-{split}",
            "split": split,
            "split_group": f"group-{split}",
            "image_relpath": f"images/{split}.jpg",
            "image_sha256": "a" * 64,
            "depth_relpath": f"depth/{split}.npy",
            "depth_kind": "sparse_sfm_points3D",
            "fov_ground_truth_deg": 60.0,
        }
        for split in ("train", "validation", "test")
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = build_preflight(manifest)

    assert report["benchmark_ready"] is True
    assert report["split_counts"] == {"test": 1, "train": 1, "validation": 1}


def test_inference_contract_forbids_coordinate_authority() -> None:
    contract = build_contract(
        ppocr_revision="a" * 40,
        ministral_revision="b" * 40,
    )

    models = {model["name"]: model for model in contract["models"]}
    assert "latitude" in models["PP-OCRv6 Small"]["forbidden_outputs"]
    assert "authoritative_coordinates" in models["Ministral 3 8B Instruct FP8"]["forbidden_outputs"]
    assert "human_gate" in contract["activation_gates"]
