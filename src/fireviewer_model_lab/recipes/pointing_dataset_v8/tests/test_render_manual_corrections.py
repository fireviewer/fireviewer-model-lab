import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from training.pointing_dataset_v7.split_registry import digest, read_rows
from training.pointing_dataset_v8 import render_manual_corrections


def test_target_crops_keep_native_pixels_and_never_admit_or_edit_original(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 80), "#345678").save(source)
    row = {"review_index": 1, "source_image": str(source), "sha256": digest(source),
           "width": 100, "height": 80, "objects": {"bbox": [[0, 0, 8, 6], [89, 70, 11, 10]], "category": [0, 1]}}
    manifest = tmp_path / "review_manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    before = digest(manifest)
    monkeypatch.setattr(sys, "argv", ["render", "--review-root", str(tmp_path), "--source-indices", "1", "--inspection-crops"])
    render_manual_corrections.main()
    receipt = read_rows(tmp_path / "source_overlays/receipts.jsonl")[0]
    assert receipt["admitted"] is False and digest(source) == row["sha256"] and digest(manifest) == before
    assert len(receipt["native_target_inspection_crops"]) == 2
    with Image.open(receipt["corrected_overlay"]) as overlay:
        for crop in receipt["native_target_inspection_crops"]:
            assert crop["scale"] == 1 and digest(Path(crop["path"])) == crop["sha256"]
            with Image.open(crop["path"]) as actual:
                expected = overlay.crop(crop["bounds_xyxy"])
                assert actual.size == expected.size and actual.tobytes() == expected.tobytes()
    assert not (tmp_path / "admitted_manifest.jsonl").exists()


def test_later_lot_preserves_receipts_and_refuses_changed_proposal(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (100, 80), "#345678").save(source)
    rows = [{"review_index": i, "source_image": str(source), "sha256": digest(source),
             "width": 100, "height": 80, "objects": {"bbox": [[10, 10, 20, 20]], "category": [1]}}
            for i in (1, 2)]
    manifest = tmp_path / "review_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for i in (1, 2):
        monkeypatch.setattr(sys, "argv", ["render", "--review-root", str(tmp_path), "--source-indices", str(i)])
        render_manual_corrections.main()
    receipt_path = tmp_path / "source_overlays/receipts.jsonl"
    receipts = read_rows(receipt_path)
    assert [r["review_index"] for r in receipts] == [1, 2]
    before = digest(receipt_path)
    overlay_before = digest(Path(receipts[0]["corrected_overlay"]))
    monkeypatch.setattr(sys, "argv", ["render", "--review-root", str(tmp_path), "--source-indices", "1"])
    render_manual_corrections.main()
    assert digest(receipt_path) == before
    rows[0]["objects"]["bbox"] = [[11, 10, 20, 20]]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="overwrite an existing annotation proposal"):
        render_manual_corrections.main()
    assert digest(receipt_path) == before
    assert digest(Path(receipts[0]["corrected_overlay"])) == overlay_before
