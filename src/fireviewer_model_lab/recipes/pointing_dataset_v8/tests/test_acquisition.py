import io

import numpy as np
import pytest
from PIL import Image

from training.pointing_dataset_v8.acquire_wsdataset import PREFIX, capture_key, check_image_pair, plan


def entry(path, size=300):
    return {"name": PREFIX + "/" + path, "bytes": size}


def test_unequal_encoded_sizes_still_plan_for_verified_full_frame_pairing():
    files = [entry("classification/train/smoke/smoke_03/frame_139.jpg", 1200),
             entry("train/images/frame_139.jpg", 200)]
    selected = plan(files)
    assert len(selected) == 1
    assert selected[0]["primary_bytes"] == 200


def test_capture_identity_joins_upstream_train_test_clips_and_excludes_train():
    a = "frame_camera_4_12_07_2022_15_03_00_3900.jpg"
    b = "frame_camera_4_12_07_2022_15_03_00_1554.jpg"
    assert capture_key(a, "smoke_40") == capture_key(b, "smoke_30")
    files = [entry("classification/train/smoke/smoke_40/" + a), entry("train/images/" + a),
             entry("classification/test/smoke/smoke_30/" + b)]
    assert plan(files) == []


def test_plan_caps_video_density_and_never_selects_source_test():
    files = [entry(f"classification/train/nonsmoke/bg/frame_{n}.jpg") for n in range(100)]
    files += [entry("classification/test/nonsmoke/other/frame_1.jpg")]
    selected = plan(files)
    assert len(selected) == 4
    assert all("/classification/train/" in r["name"] for r in selected)


def jpeg(image):
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=95)
    return out.getvalue()


def test_pair_validation_accepts_resize_not_different_content():
    rng = np.random.default_rng(9)
    original = Image.fromarray(rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)).resize((1280, 720))
    result = check_image_pair(jpeg(original), jpeg(original.resize((640, 640))))
    assert result["phash_distance"] <= 4
    assert isinstance(result["phash_distance"], int)
    with pytest.raises(ValueError, match="correspondence"):
        check_image_pair(jpeg(original), jpeg(Image.new("RGB", (640, 640), "white")))
