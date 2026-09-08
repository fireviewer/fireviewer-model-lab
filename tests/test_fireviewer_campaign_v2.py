from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from fireviewer_model_lab.training.fireviewer_campaign_v2 import FIRESENTRY_SPLITS, build_segmentation_corpus


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_boreal(root: Path) -> None:
    (root / "payload").mkdir(parents=True)
    image = root / "payload" / "image.jpg"
    mask = root / "payload" / "mask.png"
    artifact = root / "payload" / "sample.json"
    Image.new("RGB", (8, 8), "red").save(image)
    Image.fromarray(np.pad(np.full((2, 2), 255, np.uint8), 3)).save(mask)
    artifact.write_text(
        json.dumps(
            {
                "image": {"path": "payload/image.jpg", "sha256": _sha(image)},
                "annotation": {"path": "payload/mask.png", "sha256": _sha(mask)},
                "annotation_strength": "strong",
            }
        ),
        encoding="utf-8",
    )
    (root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "boreal:1",
                "source_id": "boreal",
                "split": "train",
                "split_group": "boreal-group",
                "license": "CC-BY-4.0",
                "artifact": {"path": "payload/sample.json"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _make_firesentry(root: Path) -> None:
    for region in FIRESENTRY_SPLITS:
        region_root = root / f"Region {region}"
        visible = region_root / "Visible Light"
        videos = region_root / "Fire Mask Videos"
        visible.mkdir(parents=True)
        videos.mkdir(parents=True)
        Image.new("RGB", (12, 8), "black").save(visible / "00000.jpg")
        Image.new("RGB", (12, 8), "white").save(visible / "00001.jpg")
        (videos / "video_001.mp4").write_bytes(b"test-placeholder")


def test_build_shared_segmentation_manifest(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    boreal = campaign / "sources" / "boreal"
    firesentry = campaign / "sources" / "firesentry"
    output = campaign / "prepared" / "segmentation"
    _make_boreal(boreal)
    _make_firesentry(firesentry)

    def mask_loader(_: Path) -> np.ndarray:
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[1:3, 2:4] = 255
        return mask

    report = build_segmentation_corpus(
        campaign_root=campaign,
        boreal_root=boreal,
        firesentry_root=firesentry,
        output_root=output,
        frame_loader=mask_loader,
    )
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert report["rows"] == 6
    assert report["source_counts"] == {
        "FireSentry-Benchmark-Dataset": 5,
        "boreal-forest-fire-segmentation-v1": 1,
    }
    assert report["split_group_leakage"] == []
    assert all(not Path(row["image_relpath"]).is_absolute() for row in rows)
    fire_rows = [row for row in rows if row["source_id"] == "FireSentry-Benchmark-Dataset"]
    assert all(row["anchor_points"] for row in fire_rows)
    assert all(row["redistribution_allowed"] is False for row in fire_rows)
    assert len(list((output / "masks" / "firesentry").rglob("*.png"))) == 5
