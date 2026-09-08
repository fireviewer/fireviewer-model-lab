from __future__ import annotations

from fireviewer_model_lab.training.rfdetr_premium_ground import ELITE_SEQUENCE_CAPS, quality_score, select_rows


def _row(sample_id: str, *, phash: str, near: str | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "source_id": "fasdd_v9",
        "sequence_id": "sequence-a",
        "split": "train",
        "width": 1280,
        "height": 720,
        "sha256": sample_id.ljust(64, "0"),
        "visual_fingerprint": f"fp-{sample_id}",
        "phash": phash,
        "near_duplicate_of": near,
        "sample_validation_status": "source_provided",
        "annotations": [
            {
                "class_name": "smoke_visible",
                "bbox_xywh": [100.0, 100.0, 200.0, 150.0],
            }
        ],
    }


def test_quality_score_rejects_declared_near_duplicate() -> None:
    assert quality_score(_row("a", phash="0000000000000000", near="canonical")) is None


def test_select_rows_rejects_perceptually_near_duplicate() -> None:
    rows = [
        _row("a", phash="0000000000000000"),
        _row("b", phash="0000000000000001"),
        _row("c", phash="ffffffffffffffff"),
    ]
    selected = select_rows(rows, "train")
    assert [row["sample_id"] for row in selected] == ["a", "c"]


def test_quality_score_rejects_tiny_boxes_after_resize() -> None:
    row = _row("a", phash="0000000000000000")
    row["annotations"][0]["bbox_xywh"] = [100.0, 100.0, 4.0, 4.0]
    assert quality_score(row) is None


def test_elite_profile_caps_each_sequence_to_two_train_views() -> None:
    rows = [
        _row("a", phash="0000000000000000"),
        _row("b", phash="00000000000000ff"),
        _row("c", phash="000000000000ffff"),
    ]

    selected = select_rows(rows, "train", sequence_caps=ELITE_SEQUENCE_CAPS)

    assert len(selected) == 2
