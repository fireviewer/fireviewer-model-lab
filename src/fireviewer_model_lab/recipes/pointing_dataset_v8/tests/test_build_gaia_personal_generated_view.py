import json
from pathlib import Path

from PIL import Image

from training.pointing_dataset_v8.build_gaia_personal_generated_view import build


def _coco(root: Path, split: str, count: int = 1) -> bytes:
    folder = root / split / "images"
    folder.mkdir(parents=True)
    images, annotations = [], []
    for index in range(count):
        name = f"base-{split}-{index}.jpg"
        Image.new("RGB", (100, 100), "white").save(folder / name)
        images.append({"id": index + 1, "file_name": f"images/{name}", "width": 100, "height": 100})
        annotations.append({"id": index + 1, "image_id": index + 1, "category_id": 0,
                            "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0})
    value = {"images": images, "annotations": annotations,
             "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}]}
    payload = (json.dumps(value, indent=2) + "\n").encode()
    (root / split / "_annotations.coco.json").write_bytes(payload)
    return payload


def test_build_adds_reviewed_synthetic_draws_and_freezes_holdouts(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _coco(base, "train")
    valid = _coco(base, "valid")
    test = _coco(base, "test")
    synth = tmp_path / "synth"
    folder = synth / "coco/train"
    folder.mkdir(parents=True)
    import hashlib
    source = synth / "source.png"
    Image.new("RGB", (100, 100), "red").save(source)
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    target = folder / f"{sha}.png"
    target.write_bytes(source.read_bytes())
    coco = {
        "images": [{"id": 1, "file_name": target.name, "width": 100, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [2, 2, 5, 5], "area": 25, "iscrowd": 0},
        ],
        "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}],
    }
    (folder / "_annotations.coco.json").write_text(json.dumps(coco))
    review = {"sha256": sha, "whole_scene_and_all_targets_reviewed": True,
              "annotations_exploitable": True, "corrected_overlay_verified": True,
              "safe_ground_view_reviewed": True, "no_selfie_or_dominant_person_reviewed": True}
    (synth / "reviewed_synthetic_manifest.jsonl").write_text(json.dumps(review) + "\n")
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps({"training_authorized": True, "reviewed_synthetic_mixing_authorized": True}))
    output = tmp_path / "output"

    # Production expects 61 items. Replicate the one reviewed fixture with unique payloads.
    images, annotations, rows = [], [], []
    for index in range(61):
        item = folder / f"item-{index}.png"
        Image.new("RGB", (100, 100), (index, 0, 0)).save(item)
        item_sha = hashlib.sha256(item.read_bytes()).hexdigest()
        renamed = folder / f"{item_sha}.png"
        item.rename(renamed)
        images.append({"id": index + 1, "file_name": renamed.name, "width": 100, "height": 100})
        annotations.extend([
            {"id": index * 2 + 1, "image_id": index + 1, "category_id": 0,
             "bbox": [1, 1, 5, 5], "area": 25, "iscrowd": 0},
            {"id": index * 2 + 2, "image_id": index + 1, "category_id": 1,
             "bbox": [2, 2, 5, 5], "area": 25, "iscrowd": 0},
        ])
        rows.append(review | {"sha256": item_sha})
    coco.update(images=images, annotations=annotations)
    (folder / "_annotations.coco.json").write_text(json.dumps(coco))
    (synth / "reviewed_synthetic_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = build(base, synth, auth, output)
    assert report["stage_b"]["small_smoke_synthetic_images"] == 61
    assert report["stage_b"]["logical_synthetic_draws"] == 61 * 8
    assert report["split_counts"]["train"]["images"] == 1 + 61 * 8
    train_payload = (output / "train/_annotations.coco.json").read_bytes()
    assert train_payload.endswith(b"\n")
    assert b"\r\n" not in train_payload
    assert (output / "valid/_annotations.coco.json").read_bytes() == valid
    assert (output / "test/_annotations.coco.json").read_bytes() == test
