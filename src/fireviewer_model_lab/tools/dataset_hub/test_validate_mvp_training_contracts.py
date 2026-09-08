from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from validate_mvp_training_contracts import (
    CONTRACT_REVISION,
    validate_bundle,
    validate_contract,
    validate_spec_directory,
)

SPEC_DIR = Path(__file__).parent / "specs"


def test_all_frozen_mvp_contracts_are_aligned() -> None:
    report = validate_spec_directory(SPEC_DIR)

    assert report["contract_revision"] == CONTRACT_REVISION
    assert len(report["contracts"]) == 4


def test_pointing_qwen_is_verifier_only() -> None:
    spec = json.loads((SPEC_DIR / "fire-pointing-lora-v1.json").read_text(encoding="utf-8"))

    report = validate_contract(spec)

    assert "allenai/MolmoPoint-8B" in report["model_ids"]
    assert report["executable_entrypoints"] == 0


def test_stale_public_bundle_contract_is_rejected(tmp_path: Path) -> None:
    spec = json.loads((SPEC_DIR / "fire-pointing-lora-v1.json").read_text(encoding="utf-8"))
    bundle = tmp_path / "fire-pointing-lora-v1.zip"
    embedded = {
        "schema_version": 1,
        "train_id": "fire-pointing-lora-v1",
        "entrypoints": [
            {
                "name": "old",
                "command": "python -m fireviewer_model_lab.training.spatial_train_qwen train",
            }
        ],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "fire-pointing-lora-v1/TRAIN_BUNDLE.json",
            json.dumps(embedded),
        )

    with pytest.raises(ValueError, match="contract revision"):
        validate_bundle(bundle, spec)
