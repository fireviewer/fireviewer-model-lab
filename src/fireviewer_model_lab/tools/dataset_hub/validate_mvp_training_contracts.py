from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

CONTRACT_REVISION = "mvp-a40-v2"
FORBIDDEN_EXECUTABLE_FRAGMENTS = (
    "fireviewer_model_lab.training.spatial_train_qwen",
    "fireviewer_model_lab.training.spatial_train_cross_view_localizer",
    "dinov2_coarse_cross_view",
)
REQUIRED_MODELS = {
    "fire-pointing-lora-v1": {
        "allenai/MolmoPoint-8B",
        "microsoft/Florence-2-large-ft",
        "Qwen/Qwen3.5-9B",
    },
    "cross-view-localization-v1": {
        "eceo-epfl/ConGeo",
        "1203ll/PLGeo",
    },
    "dfine-fire-smoke-v1": {
        "ustc-community/dfine-xlarge-obj365",
        "pyronear/yolo11s_quick-quokka_v8.0.0",
        "PekingU/rtdetr_v2_r50vd",
    },
    "burned-area-segmentation-v1": {
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars",
        "ibm-esa-geospatial/TerraMind-base-Fire",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _model_ids(value: Any) -> set[str]:
    discovered: set[str] = set()
    if isinstance(value, dict):
        model_id = value.get("model_id")
        if isinstance(model_id, str):
            discovered.add(model_id)
        for child in value.values():
            discovered.update(_model_ids(child))
    elif isinstance(value, list):
        for child in value:
            discovered.update(_model_ids(child))
    return discovered


def validate_contract(spec: dict[str, Any]) -> dict[str, Any]:
    train_id = str(spec.get("train_id", ""))
    required = REQUIRED_MODELS.get(train_id)
    if required is None:
        raise ValueError(f"Unsupported MVP training contract: {train_id!r}")
    if spec.get("contract_revision") != CONTRACT_REVISION:
        raise ValueError(f"{train_id} does not use contract revision {CONTRACT_REVISION}")
    models = _model_ids(spec.get("model_contract"))
    missing_models = sorted(required - models)
    if missing_models:
        raise ValueError(f"{train_id} is missing frozen models: {missing_models}")
    commands = [
        str(entrypoint["command"])
        for entrypoint in spec.get("entrypoints", [])
        if entrypoint.get("command") is not None
    ]
    for command in commands:
        forbidden = [fragment for fragment in FORBIDDEN_EXECUTABLE_FRAGMENTS if fragment in command]
        if forbidden:
            raise ValueError(f"{train_id} contains retired executable paths: {forbidden}")
    if train_id == "fire-pointing-lora-v1":
        verifier = spec["model_contract"]["verifier"]
        if (
            verifier.get("role") != "verifier_only"
            or verifier.get("coordinate_authority") is not False
        ):
            raise ValueError("Qwen must remain verifier-only without coordinate authority")
    if train_id == "cross-view-localization-v1" and commands:
        raise ValueError("ConGeo and PLGeo remain blocked until their dedicated trainers exist")
    return {
        "train_id": train_id,
        "contract_revision": CONTRACT_REVISION,
        "model_ids": sorted(models),
        "executable_entrypoints": len(commands),
    }


def validate_bundle(bundle: Path, expected_spec: dict[str, Any]) -> dict[str, Any]:
    train_id = str(expected_spec["train_id"])
    member = f"{train_id}/TRAIN_BUNDLE.json"
    with zipfile.ZipFile(bundle, mode="r", allowZip64=True) as archive:
        try:
            embedded = json.loads(archive.read(member))
        except KeyError as exc:
            raise ValueError(f"{bundle} does not contain {member}") from exc
    validate_contract(embedded)
    if embedded.get("entrypoints") != expected_spec.get("entrypoints"):
        raise ValueError(f"{bundle} embeds stale entrypoints")
    if embedded.get("model_contract") != expected_spec.get("model_contract"):
        raise ValueError(f"{bundle} embeds a stale model contract")
    return {
        "bundle": str(bundle.resolve()),
        "train_id": train_id,
        "contract_revision": CONTRACT_REVISION,
        "embedded_contract_matches_spec": True,
    }


def validate_spec_directory(spec_dir: Path) -> dict[str, Any]:
    reports = []
    for train_id in REQUIRED_MODELS:
        reports.append(validate_contract(_load_json(spec_dir / f"{train_id}.json")))
    return {
        "schema_version": 1,
        "contract_revision": CONTRACT_REVISION,
        "contracts": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen FireWarning MVP train contracts")
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--train-id", choices=tuple(REQUIRED_MODELS))
    args = parser.parse_args()
    if args.bundle is not None:
        if args.train_id is None:
            raise ValueError("--bundle requires --train-id")
        report = validate_bundle(
            args.bundle,
            _load_json(args.spec_dir / f"{args.train_id}.json"),
        )
    else:
        report = validate_spec_directory(args.spec_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
