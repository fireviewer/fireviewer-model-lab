from __future__ import annotations

import json
import zipfile
from pathlib import Path

from fireviewer_model_lab.training.orchestrator_gate_corpus import BUNDLE_ID, _build, _preflight


def test_orchestrator_gate_bundle_builds_and_passes_real_preflight(tmp_path: Path) -> None:
    report = _build(tmp_path)

    assert report["dataset_ready"] is True
    assert report["training_ready"] is False
    assert report["rows"]["split_leakage_groups"] == 0
    assert report["rows"]["task_counts"] == {
        "consensus_decision": 8,
        "stage_gate_decision": 121,
    }
    assert report["workflows"]["workflow_examples"] == 2

    preflight = _preflight(tmp_path / BUNDLE_ID)
    assert preflight["dataset_ready"] is True
    assert preflight["rows"] == 129
    assert preflight["contains_operational_incident"] is False

    with zipfile.ZipFile(tmp_path / f"{BUNDLE_ID}.zip") as archive:
        assert archive.testzip() is None
        assert all(name.startswith(f"{BUNDLE_ID}/") for name in archive.namelist())


def test_orchestrator_gate_bundle_uses_current_contract_digest(tmp_path: Path) -> None:
    _build(tmp_path)
    manifest = json.loads((tmp_path / BUNDLE_ID / "manifest.json").read_text(encoding="utf-8"))

    assert len(manifest["stage_contract_digest"]) == 64
    assert manifest["evaluation_excluded"] == ["operational-reference-a", "operational-reference-b"]
    assert manifest["training_ready"] is False
