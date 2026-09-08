from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from fireviewer_model_lab.training.pointing_flame2_audit import audit_flame2


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_flame2_source_masks_are_strictly_converted_without_review(tmp_path: Path) -> None:
    source = tmp_path / "source-input"
    output = tmp_path / "output"
    image_dir = source / "source" / "FLAME2" / "images"
    list_dir = source / "source" / "lists"
    image_dir.mkdir(parents=True)
    list_dir.mkdir(parents=True)
    rgb_path = image_dir / "img_rgb_(10).png"
    ir_path = image_dir / "img_ir_(10).png"
    mask_path = image_dir / "img_gt_(10).png"
    Image.new("RGB", (24, 16), (200, 30, 10)).save(rgb_path)
    Image.new("L", (24, 16), 40).save(ir_path)
    mask = Image.new("L", (24, 16), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((2, 2, 12, 8), fill=125)
    draw.rectangle((14, 6, 20, 14), fill=255)
    mask.save(mask_path)
    list_dir.joinpath("train_flm.txt").write_text(
        "FLAME2/images/img_XXX_(10).png\n", encoding="utf-8"
    )
    list_dir.joinpath("val_flm.txt").write_text("", encoding="utf-8")
    list_dir.joinpath("test_flm.txt").write_text("", encoding="utf-8")

    rows = []
    for kind, path in (("rgb", rgb_path), ("infrared", ir_path), ("mask", mask_path)):
        rows.append(
            {
                "asset_kind": kind,
                "relative_path": f"FLAME2/images/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_revision": "revision",
            }
        )
    _jsonl(source / "candidate_manifest.jsonl", rows)
    baseline = tmp_path / "baseline"
    _jsonl(
        baseline / "pixel_inventory.jsonl",
        [{"status": "ok", "sample_id": "seed:1", "sha256": "b" * 64, "dhash": "f" * 16}],
    )

    report = audit_flame2(
        source_root=source,
        baseline_root=baseline,
        output_dir=output,
        source_s3_prefix="s3://bucket/pointing-corpus-v2/flame2",
    )
    candidate_lines = (output / "flame2_automatic_dispositions.jsonl").read_text().splitlines()
    candidates = [json.loads(line) for line in candidate_lines]
    validated_lines = (output / "flame2_strict_validated_manifest.jsonl").read_text().splitlines()
    validated = [json.loads(line) for line in validated_lines]

    assert report["rgb_candidates"] == 1
    assert report["point_ground_truth_rows"] == 1
    assert report["training_eligible_rows"] == 1
    assert candidates[0]["provided_mask_role"] == "source_semantic_ground_truth"
    assert candidates[0]["mask_to_point_conversion"] == (
        "deterministic_class_mask_bottom_band_median"
    )
    assert candidates[0]["corpus_disposition"] == "eligible_genuinely_new_pool"
    assert {point["kind"] for point in validated[0]["anchor_points"]} == {
        "fire_base",
        "smoke_column_base",
    }
    assert validated[0]["sample_validation_status"] == "strict_automated_validated"
    assert validated[0]["split"] == "train"
    assert report["reviews_admitted"] is False
    assert candidates[0]["training_eligible"]
