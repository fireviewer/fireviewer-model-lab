from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from v5_contract import V5ContractError, load_v5_contract  # noqa: E402


def _write_minimal(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    report = {
        "schema": "fireviewer.pointing-dataset-v5-local-report.v1",
        "status": "ready_for_training",
        "training_ready": True,
        "selected_count": 2909,
        "base_selected_count": 2810,
        "extension_selected_count": 99,
        "extension_available_count": 261,
        "extension_recurrence_excluded_count": 162,
        "extension_recurrence_cap": 12,
        "extension_maximum_recurrence_group_size": 12,
        "extension_scene_count": 9,
        "extension_visibility_counts": {
            "small_le_1pct": 15,
            "tiny_le_0p5pct": 47,
            "ultra_tiny_le_0p1pct": 37,
        },
        "aerial_retained_count": 0,
        "retained_foreground_person_or_selfie_count": 0,
        "exact_base_extension_overlap_count": 0,
        "cross_split_group_overlap_count": 0,
        "all_boxes_geometry_valid": True,
        "all_image_sha256_reverified": True,
        "materialization": {"copies": 0, "hardlinks": 2909},
        "split_counts": {"train": 2323, "validation": 293, "test": 293},
        "source_counts": {"fixture": 2909},
    }
    manifest = root / "selection_manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    import hashlib

    report["selection_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    reload_receipt = {
        "schema": "fireviewer.pointing-dataset-v5-reload-validation.v1",
        "status": "passed",
        "all_images_decoded": True,
        "all_dimensions_match_metadata": True,
        "all_objects_reloadable": True,
        "decoded_total": 2909,
        "decoded_counts": report["split_counts"],
    }
    (root / "reload_validation.json").write_text(
        json.dumps(reload_receipt), encoding="utf-8"
    )


def test_load_contract_accepts_signed_shape(tmp_path: Path) -> None:
    _write_minimal(tmp_path)
    contract = load_v5_contract(tmp_path)
    assert contract.selected_count == 2909
    assert contract.split_counts["train"] == 2323


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_ready", False),
        ("aerial_retained_count", 1),
        ("retained_foreground_person_or_selfie_count", 1),
        ("cross_split_group_overlap_count", 1),
        ("all_image_sha256_reverified", False),
    ],
)
def test_load_contract_refuses_failed_gate(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_minimal(tmp_path)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(V5ContractError):
        load_v5_contract(tmp_path)


def test_load_contract_refuses_manifest_hash_change(tmp_path: Path) -> None:
    _write_minimal(tmp_path)
    (tmp_path / "selection_manifest.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(V5ContractError, match="manifest SHA-256"):
        load_v5_contract(tmp_path)

