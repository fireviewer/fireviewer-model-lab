from __future__ import annotations

import json
from pathlib import Path

from fireviewer_model_lab.training.a40_campaign import build_plan


def test_campaign_blocks_unimplemented_or_weakly_supervised_stages(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_paths = (
        tmp_path / "corpus" / "fasdd" / "manifest.jsonl",
        tmp_path / "corpus" / "pyro-sdis-v0.1.0" / "manifest.jsonl",
        tmp_path / "additional" / "alarmod-forest-fire" / "manifest.rtdetr.jsonl",
        tmp_path / "sources" / "boreal-forest-fire-detection-v1" / "manifest.jsonl",
        tmp_path / "corpus" / "hls-burn-scars-v1" / "manifest.jsonl",
        tmp_path / "additional" / "eo4wildfires" / "manifest.jsonl",
    )
    for manifest_path in manifest_paths:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("", encoding="utf-8")

    monkeypatch.setattr("fireviewer_model_lab.training.a40_campaign.load_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "fireviewer_model_lab.training.a40_campaign.build_preflight_report",
        lambda *_args, **_kwargs: {"training_ready": True, "deployment_ready": False, "errors": []},
    )
    monkeypatch.setattr(
        "fireviewer_model_lab.training.a40_campaign.build_burnscar_report",
        lambda *_args, **_kwargs: {
            "training_ready": False,
            "promotion_ready": False,
            "training_errors": ["independent_geographic_critical_test_missing"],
            "promotion_errors": ["trained_model_independent_evaluation_missing"],
        },
    )

    plan = build_plan(tmp_path)

    assert json.loads(json.dumps(plan))["hardware"]["vram_gib"] == 48
    assert plan["stages"][0]["training_ready"] is True
    assert plan["stages"][1]["training_ready"] is False
    assert plan["stages"][1]["promotion_ready"] is False
    assert "independent_geographic_critical_test_missing" in plan["stages"][1]["reason"]
    assert plan["stages"][2]["command"] is None
    assert "congeo_trainer_not_implemented" in plan["stages"][2]["reason"]
    assert plan["stages"][3]["training_ready"] is False
