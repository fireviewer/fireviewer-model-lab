import copy

import pytest

from training.pointing_dataset_v8.prepare_dfire_revision import (
    has_small, scene_kind, select_group_capped,
)


def row(name, group, categories, areas=None):
    areas = areas or [1000] * len(categories)
    return {
        "sha256": name.rjust(64, "0"), "source_record_id": name + ".jpg",
        "source_group_id": group, "source_family": "D-Fire", "split": "train",
        "width": 1000, "height": 1000, "objects": {
            "bbox": [[1, 1, 10, 10] for _ in categories],
            "category": categories, "area": areas,
        },
    }


def test_scene_kind_and_small_target_geometry():
    candidate = row("1", "g", [0, 1], [6000, 4000])
    assert scene_kind(candidate) == "fire_and_smoke"
    assert has_small(candidate, 0) is False
    assert has_small(candidate, 1) is True


def test_revision_quarantines_negatives_caps_groups_and_prioritizes_small_smoke():
    rows = [row("1", "g", []), row("2", "g", [0]), row("3", "g", [1], [4000]),
            row("4", "g", [0, 1], [1000, 3000]), row("5", "g", [1], [9000])]
    frozen = copy.deepcopy(rows)
    selected, excluded = select_group_capped(rows, 2)
    assert {item["source_record_id"] for item in selected} == {"3.jpg", "4.jpg"}
    assert max(sum(item["source_group_id"] == group for item in selected) for group in {"g"}) == 2
    assert {item["reason"] for item in excluded} == {
        "unreviewed_source_negative_quarantined", "sequence_diversity_cap"
    }
    assert rows == frozen
    assert all(item["review_status"] == "pending_v8p4_visual_and_annotation_review" for item in selected)
    assert not any(item["v8_training_admitted"] for item in selected)


def test_revision_rejects_non_dfire_or_non_train_inputs():
    with pytest.raises(ValueError, match="D-Fire train"):
        select_group_capped([row("1", "g", [0]) | {"split": "test"}], 4)
    with pytest.raises(ValueError, match="positive"):
        select_group_capped([], 0)
