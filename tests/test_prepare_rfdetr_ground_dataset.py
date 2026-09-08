from __future__ import annotations

from fireviewer_model_lab.tools.prepare_rfdetr_ground_dataset import CLASS_IDS, is_ground_row


def test_ground_selection_keeps_fasdd_cv_and_pyro_sdis_only() -> None:
    assert is_ground_row({"source_id": "fasdd_v9", "event_id": "fasdd-v9-cv"})
    assert is_ground_row({"source_id": "pyro_sdis_a1e553e", "event_id": "incident"})
    assert not is_ground_row({"source_id": "fasdd_v9", "event_id": "fasdd-v9-uav"})
    assert not is_ground_row({"source_id": "fasdd_v9", "event_id": "fasdd-v9-rs"})
    assert not is_ground_row({"source_id": "alarmod_forest_fire", "event_id": ""})
    assert not is_ground_row({"source_id": "boreal-forest-fire-detection-v1"})


def test_ground_coco_class_order_matches_trainer_contract() -> None:
    assert CLASS_IDS == {"flame_visible": 0, "smoke_visible": 1}
