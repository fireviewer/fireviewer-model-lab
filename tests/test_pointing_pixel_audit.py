from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image
from fireviewer_model_lab.training.pointing_pixel_audit import (
    cross_split_visual_groups,
    difference_hash,
    inspect_row,
    parse_s3_uri,
)


def test_parse_s3_uri_enforces_pointing_boundary() -> None:
    assert parse_s3_uri("s3://bucket/pointing-v8-reservoir-v1/a.jpg") == (
        "bucket",
        "pointing-v8-reservoir-v1/a.jpg",
    )
    with pytest.raises(ValueError, match="forbidden non-pointing"):
        parse_s3_uri("s3://bucket/fire-smoke-detection-corpus-v1/a.jpg")


def test_difference_hash_is_deterministic() -> None:
    image = Image.new("RGB", (32, 32), (120, 80, 40))
    assert difference_hash(image) == difference_hash(image.copy())
    assert len(difference_hash(image)) == 16


def test_inspect_row_verifies_legacy_seed_sha_field() -> None:
    payload = io.BytesIO()
    Image.new("RGB", (16, 12), (120, 80, 40)).save(payload, format="PNG")
    image_bytes = payload.getvalue()

    class FakeBody:
        def read(self) -> bytes:
            return image_bytes

    class FakeS3:
        def get_object(self, **_kwargs: str) -> dict:
            return {"Body": FakeBody()}

    result = inspect_row(
        {
            "sample_id": "seed:1",
            "image_s3_uri": "s3://bucket/pointing-ground-v1/image.png",
            "split": "train",
            "source_family": "seed",
            "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "width": 16,
            "height": 12,
        },
        s3_client=FakeS3(),
    )

    assert result["status"] == "ok"
    assert result["sha256"] == hashlib.sha256(image_bytes).hexdigest()


def test_visual_signatures_are_automatically_rejected() -> None:
    rows = [
        {"sample_id": "a", "split": "train", "dhash": "1234"},
        {"sample_id": "b", "split": "test", "dhash": "1234"},
        {"sample_id": "c", "split": "test", "dhash": "5678"},
    ]

    groups = cross_split_visual_groups(rows)

    assert groups == [
        {
            "automatic_rejection": True,
            "dhash": "1234",
            "exclusion_reason": "cross_split_visual_signature",
            "samples": ["a", "b"],
            "splits": ["test", "train"],
        }
    ]
