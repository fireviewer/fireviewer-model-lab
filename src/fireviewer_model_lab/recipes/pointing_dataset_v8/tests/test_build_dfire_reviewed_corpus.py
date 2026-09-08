import hashlib
import json
from pathlib import Path

from PIL import Image

from training.pointing_dataset_v8.build_dfire_reviewed_corpus import build, verified_decisions


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_preserves_holdouts_and_requires_bound_review(tmp_path):
    base = tmp_path / "base"
    categories = [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}]
    for split in ("train", "valid", "test"):
        image_path = base / split / "images" / f"{split}.jpg"
        image_path.parent.mkdir(parents=True)
        Image.new("RGB", (32, 24), "black").save(image_path)
        write_json(base / split / "_annotations.coco.json", {
            "images": [{"id": 1, "file_name": f"images/{split}.jpg", "width": 32, "height": 24}],
            "annotations": [], "categories": categories,
        })
    source = tmp_path / "candidate.jpg"
    Image.new("RGB", (40, 30), "red").save(source)
    page = tmp_path / "page.jpg"
    Image.new("RGB", (64, 64), "white").save(page)
    candidate = {
        "review_index": 7, "candidate_id": "candidate-7", "sha256": sha(source),
        "label_sha256": "label", "source_image": str(source), "width": 40, "height": 30,
        "source_group_id": "group-7", "objects": {"bbox": [[4, 3, 10, 8]],
        "category": [0], "area": [80]},
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    decision = {
        "review_index": 7, "candidate_id": "candidate-7", "sha256": sha(source),
        "label_sha256": "label", "decision": "accept", "reason": "complete annotation",
        "reviewed_at": "2026-08-29", "full_frame_review_complete": True,
        "all_annotations_review_complete": True, "review_page": str(page),
        "review_page_sha256": sha(page),
    }
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    valid_before = (base / "valid" / "_annotations.coco.json").read_bytes()
    test_before = (base / "test" / "_annotations.coco.json").read_bytes()

    report = build(base, manifest, decisions, tmp_path / "output")

    assert report["review"]["accepted"] == 1
    assert report["splits"]["train"]["images"] == 2
    assert (tmp_path / "output" / "valid" / "_annotations.coco.json").read_bytes() == valid_before
    assert (tmp_path / "output" / "test" / "_annotations.coco.json").read_bytes() == test_before


def test_review_binding_fails_closed(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    manifest.write_text(json.dumps({"review_index": 1, "candidate_id": "a", "sha256": "x",
                                    "label_sha256": "y"}) + "\n", encoding="utf-8")
    decisions.write_text("", encoding="utf-8")
    try:
        verified_decisions(manifest, decisions)
    except ValueError as exc:
        assert "Every candidate" in str(exc)
    else:
        raise AssertionError("missing review decision was accepted")
