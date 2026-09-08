from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import fireviewer_model_lab.training.train_prithvi_burnscars as burnscars


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_geographic_gate(root: Path) -> Path:
    manifest = root / "geographic-critical-test" / "manifest.jsonl"
    rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for index in range(100):
        sample_id = f"sample-{index:03d}"
        image_sha = _digest(f"geo-image-{index}")
        mask_sha = _digest(f"geo-mask-{index}")
        event_id = f"event-{index % 3}"
        site_id = f"site-{index % 3}"
        rows.append(
            {
                "sample_id": sample_id,
                "event_id": event_id,
                "site_id": site_id,
                "split": "test",
                "crs": "EPSG:2154",
                "bounds": [index, index, index + 1, index + 1],
                "image_sha256": image_sha,
                "mask_sha256": mask_sha,
                "validation_status": "dual_automated_validation_passed",
                "automated_validators": [
                    "eo4-official-source-contract-v1",
                    "geotiff-geospatial-reopen-v1",
                ],
                "validator_count": 2,
            }
        )
        selection_rows.append(
            {
                "event_id": event_id,
                "image_sha256": image_sha,
                "mask_sha256": mask_sha,
                "sample_id": sample_id,
                "site_id": site_id,
            }
        )
    _write_jsonl(manifest, rows)
    selection_sha = hashlib.sha256(
        (
            json.dumps(
                sorted(selection_rows, key=lambda row: str(row["sample_id"])),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    report_path = manifest.parent / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_path": "manifest.jsonl",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "selection_sha256": selection_sha,
                "independent_from_training": True,
                "georeferencing_verified": True,
                "official_reference_verified": True,
                "independent_validation_complete": True,
                "validation_policy": "dual_automated_official_source_v1",
                "automated_validator_count": 2,
                "training_group_overlap": 0,
                "split_leakage": 0,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report_path


def _hls_row(split: str) -> dict[str, object]:
    return {
        "split": split,
        "split_group": f"hls:{split}",
        "sha256": _digest(f"hls-image:{split}"),
        "mask_sha256": _digest(f"hls-mask:{split}"),
        "image_relpath": f"images/{split}.tif",
        "mask_relpath": f"masks/{split}.tif",
        "mask_values": {"burned": 1, "not_burned": 0, "ignore": -1},
        "raster": {"burned_pixels": 10},
        "source_asset": {
            "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
        },
    }


def _eo4_row(split: str) -> dict[str, object]:
    return {
        "split": split,
        "sha256": _digest(f"eo4:{split}"),
        "source_member": f"{split}.nc",
        "source_revision": burnscars.EO4_SOURCE_REVISION,
        "variables": {
            "S2A": {"shape": [6, 32, 32]},
            "burned_mask": {"shape": [32, 32]},
        },
        "burned_mask": {"positive_pixels": 10},
    }


def test_prithvi_preflight_separates_training_and_promotion_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        burnscars,
        "EXPECTED_EO4_SPLITS",
        {"train": 1, "validation": 1, "test": 1},
    )
    hls = tmp_path / "hls" / "manifest.jsonl"
    eo4 = tmp_path / "eo4" / "manifest.jsonl"
    _write_jsonl(hls, [_hls_row("train"), _hls_row("validation")])
    _write_jsonl(eo4, [_eo4_row("train"), _eo4_row("validation"), _eo4_row("test")])

    report = burnscars.build_preflight_report(
        hls,
        eo4,
        verify_files=False,
        geographic_test_report=_write_geographic_gate(tmp_path),
    )

    assert report["training_ready"] is True
    assert report["promotion_ready"] is False
    assert report["training_errors"] == []
    assert report["promotion_errors"] == ["trained_model_independent_evaluation_missing"]


def test_prithvi_preflight_blocks_training_without_independent_geographic_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        burnscars,
        "EXPECTED_EO4_SPLITS",
        {"train": 1, "validation": 1, "test": 1},
    )
    hls = tmp_path / "hls" / "manifest.jsonl"
    eo4 = tmp_path / "eo4" / "manifest.jsonl"
    _write_jsonl(hls, [_hls_row("train"), _hls_row("validation")])
    _write_jsonl(eo4, [_eo4_row("train"), _eo4_row("validation"), _eo4_row("test")])

    report = burnscars.build_preflight_report(hls, eo4, verify_files=False)

    assert report["training_ready"] is False
    assert report["training_errors"] == ["independent_geographic_critical_test_missing"]


def test_prithvi_geographic_gate_rejects_single_automated_validator(
    tmp_path: Path,
) -> None:
    report_path = _write_geographic_gate(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = report_path.parent / report["manifest_path"]
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["automated_validators"] = ["eo4-official-source-contract-v1"]
    rows[0]["validator_count"] = 1
    rows[0]["validation_status"] = "source_contract_validated"
    _write_jsonl(manifest, rows)
    report["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    gate = burnscars._validate_geographic_test_report(
        report_path,
        verify_files=False,
    )

    assert gate["ready"] is False
    assert gate["errors"] == ["geographic_test_row_dual_validation_missing"]


def test_prithvi_config_uses_bf16_six_hls_bands_and_checkpoints(tmp_path: Path) -> None:
    config = burnscars.build_terratorch_config(
        tmp_path / "dataset",
        tmp_path / "output",
        batch_size=8,
        workers=4,
        epochs=100,
        checkpoint_steps=500,
    )

    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["model"]["init_args"]["model_args"]["backbone"] == "prithvi_eo_v2_300"
    assert config["data"]["init_args"]["dataset_bands"] == burnscars.EXPECTED_HLS_BANDS
    assert config["data"]["init_args"]["check_stackability"] is False
    assert config["data"]["class_path"] == (
        "fireviewer_model_lab.training.prithvi_optimized_datamodule.OptimizedGenericNonGeoSegmentationDataModule"
    )
    assert config["data"]["init_args"]["pin_memory"] is True
    assert config["data"]["init_args"]["persistent_workers"] is True
    assert config["data"]["init_args"]["prefetch_factor"] == 2
    callbacks = config["trainer"]["callbacks"]
    checkpoint = next(
        item
        for item in callbacks
        if item["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"
    )
    assert checkpoint["init_args"]["monitor"] == "step"
    assert checkpoint["init_args"]["mode"] == "max"
    assert checkpoint["init_args"]["save_last"] is True
    assert checkpoint["init_args"]["save_top_k"] == 4
    assert checkpoint["init_args"]["every_n_train_steps"] == 500
    assert checkpoint["init_args"]["save_on_train_epoch_end"] is False
    assert config["trainer"]["limit_val_batches"] == 0
    train_transforms = config["data"]["init_args"]["train_transform"]
    val_transforms = config["data"]["init_args"]["val_transform"]
    test_transforms = config["data"]["init_args"]["test_transform"]
    assert [item["class_path"] for item in train_transforms[:2]] == [
        "albumentations.PadIfNeeded",
        "albumentations.RandomCrop",
    ]
    assert [item["class_path"] for item in test_transforms[:2]] == [
        "albumentations.PadIfNeeded",
        "albumentations.CenterCrop",
    ]
    assert val_transforms == test_transforms
    assert train_transforms[0]["init_args"]["fill_mask"] == -1
    assert test_transforms[0]["init_args"]["fill_mask"] == -1


def test_prithvi_full_evaluation_config_reenables_all_batches(tmp_path: Path) -> None:
    training = burnscars.build_terratorch_config(
        tmp_path / "dataset",
        tmp_path / "output",
        batch_size=8,
        workers=4,
        epochs=100,
        checkpoint_steps=500,
    )

    evaluation = burnscars.build_full_evaluation_config(training)

    assert training["trainer"]["limit_val_batches"] == 0
    assert evaluation["trainer"]["limit_val_batches"] == 1.0
    assert evaluation["trainer"]["limit_test_batches"] == 1.0
    assert evaluation["trainer"]["callbacks"] == []


def test_prithvi_resume_auto_uses_last_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert burnscars._resolve_resume_checkpoint(tmp_path, "auto") == checkpoint.resolve()
    assert burnscars._resolve_resume_checkpoint(tmp_path / "empty", "auto") is None


def test_resolve_terratorch_executable_next_to_venv_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    python = venv_bin / "python"
    executable = venv_bin / "terratorch"
    venv_bin.mkdir(parents=True)
    python.write_bytes(b"")
    executable.write_bytes(b"")
    monkeypatch.setattr(burnscars.shutil, "which", lambda _name: None)
    monkeypatch.setattr(burnscars.sys, "executable", str(python))

    assert burnscars._resolve_terratorch_executable() == str(executable.resolve())


def test_prithvi_smoke_plan_is_bounded_without_changing_the_train_plan(
    tmp_path: Path,
) -> None:
    smoke = burnscars.build_terratorch_config(
        tmp_path / "dataset",
        tmp_path / "smoke",
        batch_size=2,
        workers=2,
        epochs=100,
        checkpoint_steps=500,
        smoke=True,
    )
    train = burnscars.build_terratorch_config(
        tmp_path / "dataset",
        tmp_path / "train",
        batch_size=8,
        workers=8,
        epochs=100,
        checkpoint_steps=500,
    )

    assert smoke["trainer"]["max_epochs"] == 1
    assert smoke["trainer"]["limit_train_batches"] == 2
    assert smoke["trainer"]["limit_val_batches"] == 0
    assert smoke["trainer"]["limit_test_batches"] == 0
    assert smoke["trainer"]["num_sanity_val_steps"] == 0
    assert train["trainer"]["max_epochs"] == 100
    assert "limit_train_batches" not in train["trainer"]


def test_materialized_dataset_gate_rejects_incompatible_eo4_scale(tmp_path: Path) -> None:
    for split in ("train", "validation", "test"):
        split_path = tmp_path / "splits" / f"{split}.txt"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(f"{split}-sample\n", encoding="utf-8")
    report = {
        "combined_split_counts": {"train": 1, "validation": 1, "test": 1},
        "normalization": {
            "eo4_audit": {"compatible_with_hls_normalization": False},
        },
    }
    (tmp_path / "materialization-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="reflectance"):
        burnscars._validate_materialized_dataset(tmp_path)


def test_materialized_preflight_does_not_require_source_manifests(tmp_path: Path) -> None:
    for split in ("train", "validation", "test"):
        split_path = tmp_path / "splits" / f"{split}.txt"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(f"{split}-sample\n", encoding="utf-8")
    report = {
        "combined_split_counts": {"train": 1, "validation": 1, "test": 1},
        "normalization": {
            "eo4_audit": {"compatible_with_hls_normalization": True},
        },
    }
    (tmp_path / "materialization-report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    preflight = burnscars.build_materialized_preflight_report(
        tmp_path,
        geographic_test_report=_write_geographic_gate(tmp_path),
        verify_files=False,
    )

    assert preflight["training_ready"] is True
    assert preflight["promotion_ready"] is False
    assert preflight["materialized_dataset"]["split_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_reflectance_gate_accepts_rare_bright_outliers_but_not_scale_drift() -> None:
    rare_outliers = [[0.0004, 0.00001, 0.00000001] for _ in range(6)]

    assert burnscars._reflectance_scale_compatible(
        [0.0] * 6,
        [0.08] * 6,
        [1.6] * 6,
        rare_outliers,
    )
    assert not burnscars._reflectance_scale_compatible(
        [0.0] * 6,
        [0.08] * 6,
        [1.6] * 6,
        [[0.006, 0.00001, 0.00000001] for _ in range(6)],
    )
