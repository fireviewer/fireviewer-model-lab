import json
from pathlib import Path

from PIL import Image

from training.pointing_dataset_v8.build_small_target_curriculum import build


def _write_split(root: Path, split: str, images: list[dict], annotations: list[dict]) -> bytes:
    directory = root / split / "images"
    directory.mkdir(parents=True)
    for image in images:
        Image.new("RGB", (image["width"], image["height"]), "white").save(
            directory / Path(image["file_name"]).name
        )
    value = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "fire"}, {"id": 1, "name": "smoke"}],
    }
    payload = (json.dumps(value, indent=2) + "\n").encode()
    (root / split / "_annotations.coco.json").write_bytes(payload)
    return payload


def test_build_repeats_small_targets_and_freezes_holdouts(tmp_path: Path) -> None:
    base = tmp_path / "base"
    images = [
        {"id": 1, "file_name": "images/a.jpg", "width": 64, "height": 64},
        {"id": 2, "file_name": "images/b.jpg", "width": 64, "height": 64},
        {"id": 3, "file_name": "images/c.jpg", "width": 64, "height": 64},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 8, 8], "area": 64, "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 0, "bbox": [2, 2, 10, 10], "area": 100, "iscrowd": 0},
        {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 40, 40], "area": 1600, "iscrowd": 0},
    ]
    _write_split(base, "train", images, annotations)
    valid_bytes = _write_split(base, "valid", images[:1], annotations[:1])
    test_bytes = _write_split(base, "test", images[1:2], [dict(annotations[1], id=1)])

    output = tmp_path / "curriculum"
    report = build(base, output, smoke_extra=2, fire_extra=1)
    train = json.loads((output / "train" / "_annotations.coco.json").read_text())

    assert report["train"]["logical_image_records"] == 6
    assert report["train"]["added_logical_records"] == 3
    assert len({image["file_name"] for image in train["images"]}) == 3
    assert sum("fireviewer_curriculum_repeat_of_image_id" in image for image in train["images"]) == 3
    assert (output / "valid" / "_annotations.coco.json").read_bytes() == valid_bytes
    assert (output / "test" / "_annotations.coco.json").read_bytes() == test_bytes
    assert (output / "train" / "images" / "a.jpg").stat().st_ino == (base / "train" / "images" / "a.jpg").stat().st_ino
