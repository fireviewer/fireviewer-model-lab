"""Produce an evidence-bound sequential A40 training campaign plan.

The runner is deliberately not an auto-promotion mechanism.  It checks the
same preflight contracts as the individual trainers and emits a plan whose
blocked stages remain blocked until their input evidence exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.train_prithvi_burnscars import build_preflight_report as build_burnscar_report
from fireviewer_model_lab.training.train_rtdetr import build_preflight_report, load_records


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_plan(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    detector_manifests = [
        dataset_root / "corpus" / "fasdd" / "manifest.jsonl",
        dataset_root / "corpus" / "pyro-sdis-v0.1.0" / "manifest.jsonl",
        dataset_root / "additional" / "alarmod-forest-fire" / "manifest.rtdetr.jsonl",
        dataset_root / "sources" / "boreal-forest-fire-detection-v1" / "manifest.jsonl",
    ]
    hls_manifest = dataset_root / "corpus" / "hls-burn-scars-v1" / "manifest.jsonl"
    eo4_manifest = dataset_root / "additional" / "eo4wildfires" / "manifest.jsonl"
    missing_detector_manifests = [
        f"missing_detector_manifest:{path.relative_to(dataset_root).as_posix()}"
        for path in detector_manifests
        if not path.is_file()
    ]
    detector = (
        build_preflight_report(
            load_records(detector_manifests, verify_files=False),
            profile="media_filter_v1",
        )
        if not missing_detector_manifests
        else {
            "training_ready": False,
            "deployment_ready": False,
            "errors": missing_detector_manifests,
        }
    )
    burnscar = (
        build_burnscar_report(hls_manifest, eo4_manifest, verify_files=False)
        if hls_manifest.is_file() and eo4_manifest.is_file()
        else {
            "training_ready": False,
            "promotion_ready": False,
            "training_errors": [
                name
                for name, path in (
                    ("missing_hls_manifest", hls_manifest),
                    ("missing_eo4_manifest", eo4_manifest),
                )
                if not path.is_file()
            ],
            "promotion_errors": ["trained_model_independent_evaluation_missing"],
        }
    )
    stages = [
        {
            "id": "media_filter_dfine_v1",
            "order": 1,
            "command": "python -m fireviewer_model_lab.training.train_dfine ... train",
            "training_ready": detector["training_ready"],
            "promotion_ready": detector["deployment_ready"],
            "reason": detector["errors"],
            "benchmark": {
                "candidates": ["D-FINE", "Pyronear YOLO11s", "RT-DETRv2-R50"],
                "selection": "one immutable selection_sha256 for every candidate",
            },
        },
        {
            "id": "burned_area_prithvi_v1",
            "order": 2,
            "command": "python -m fireviewer_model_lab.training.train_prithvi_burnscars ... train",
            "training_ready": burnscar["training_ready"],
            "promotion_ready": burnscar["promotion_ready"],
            "reason": burnscar["training_errors"] + burnscar["promotion_errors"],
            "benchmark": {"challenger": "TerraMind-base-Fire"},
        },
        {
            "id": "cross_view_congeo_plgeo_v1",
            "order": 3,
            "command": None,
            "training_ready": False,
            "promotion_ready": False,
            "reason": [
                "congeo_trainer_not_implemented",
                "plgeo_trainer_not_implemented",
                "independent_double_validated_geographic_test_missing",
            ],
        },
        {
            "id": "molmopoint_fire_pointing_v1",
            "order": 4,
            "command": None,
            "training_ready": False,
            "promotion_ready": False,
            "reason": [
                "molmopoint_trainer_not_implemented",
                "point_labels_are_weak_supervision_only",
            ],
            "verifier": {
                "model": "Qwen/Qwen3.5-9B",
                "role": "verifier_only",
                "coordinate_authority": False,
            },
        },
    ]
    return {
        "schema_version": 1,
        "hardware": {"gpu": "NVIDIA A40", "vram_gib": 48, "execution": "strictly_sequential"},
        "dataset_root": str(dataset_root),
        "stages": stages,
        "promotion_policy": "human_review_required",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FireWarning A40 campaign plan")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.dataset_root)
    _write_json(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
