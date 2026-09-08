from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fireviewer_model_lab.training.spatial_register_roma import (
    RegistrationSetupError,
    _held_out_match_diagnostics,
    preflight,
)

from fireviewer_geolocation.roma_registration import (
    AssetSpec,
    RomaAssetError,
    _download_asset,
    verify_asset,
)


def test_asset_download_is_atomic_and_digest_verified(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = b"pinned model bytes"
    spec = AssetSpec(
        filename="model.pth",
        url="https://models.invalid/model.pth",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        license="MIT",
    )

    path = _download_asset(tmp_path, spec, opener=lambda *_args, **_kwargs: io.BytesIO(payload))

    assert path.read_bytes() == payload
    assert not path.with_suffix(".pth.partial").exists()
    verify_asset(path, spec)
    output = capsys.readouterr().out
    assert "downloading spatial model asset=model.pth" in output
    assert "spatial model progress asset=model.pth" in output
    assert "percent=100" in output
    assert "spatial model ready asset=model.pth" in output


def test_altered_asset_is_rejected_and_partial_is_removed(tmp_path: Path) -> None:
    spec = AssetSpec(
        filename="model.pth",
        url="https://models.invalid/model.pth",
        size=8,
        sha256="0" * 64,
        license="MIT",
    )

    with pytest.raises(RomaAssetError, match="SHA-256 mismatch"):
        _download_asset(tmp_path, spec, opener=lambda *_args, **_kwargs: io.BytesIO(b"altered!"))

    assert not (tmp_path / "weights/model.pth").exists()
    assert not (tmp_path / "weights/model.pth.partial").exists()


def _write_corpus(tmp_path: Path, *, operational: bool = False) -> None:
    corpus = tmp_path / "corpus/cross-view-registration-v0.1.0"
    corpus.mkdir(parents=True)
    rows = []
    for index, source_id in enumerate(
        ("aerialextrematch_localization", "odm_sance_mountain", "odm_seneca_rural")
    ):
        rows.append(
            {
                "operational_incident": operational,
                "sample_id": f"sample-{index}",
                "source_id": source_id,
                "split": "validation" if index == 0 else "train",
                "split_group": f"group-{index}",
            }
        )
    manifest = corpus / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "gates": {"deployment_ready": False, "training_ready": True},
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "rows": len(rows),
        "split_counts": {"train": 2, "validation": 1},
    }
    (corpus / "build-report.json").write_text(json.dumps(report), encoding="utf-8")


def test_cross_view_preflight_keeps_training_disabled(tmp_path: Path) -> None:
    _write_corpus(tmp_path)

    report = preflight(tmp_path)

    assert report["rows"] == 3
    assert report["deployment_ready"] is False
    assert report["training_command_available"] is False
    assert report["critical_lot_included"] is False


def test_cross_view_preflight_rejects_operational_media(tmp_path: Path) -> None:
    _write_corpus(tmp_path, operational=True)

    with pytest.raises(RegistrationSetupError, match="operational registration row denied"):
        preflight(tmp_path)


class _FlatTerrain:
    def sample_many(self, eastings: object, _northings: object) -> np.ndarray:
        return np.zeros(len(np.asarray(eastings).reshape(-1)), dtype=np.float64)


def test_held_out_match_diagnostic_measures_truth_without_admitting_it() -> None:
    world_xy = np.asarray([(-2.0, -1.0), (-1.0, 2.0), (0.0, 0.0), (1.5, -2.0), (2.0, 1.0)])
    bounds = (-5.0, -5.0, 5.0, 5.0)
    map_pixels = np.column_stack(
        (
            (world_xy[:, 0] - bounds[0]) / (bounds[2] - bounds[0]) * 100.0,
            (bounds[3] - world_xy[:, 1]) / (bounds[3] - bounds[1]) * 100.0,
        )
    )
    source_pixels = np.column_stack(
        (100.0 * world_xy[:, 0] / 10.0 + 50.0, 100.0 * world_xy[:, 1] / 10.0 + 50.0)
    )
    row = {
        "ground_truth": {
            "quaternion_wxyz_world_to_camera": [1.0, 0.0, 0.0, 0.0],
            "translation_xyz_world_to_camera": [0.0, 0.0, 10.0],
        },
        "source_view": {
            "width": 100,
            "height": 100,
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
        },
    }

    result = _held_out_match_diagnostics(
        source_pixels=source_pixels,
        map_pixels=map_pixels,
        certainties=np.ones(len(source_pixels)),
        map_image_size=(100, 100),
        map_bounds=bounds,
        terrain=_FlatTerrain(),  # type: ignore[arg-type]
        row=row,
    )

    assert result["admission_input"] is False
    assert result["projected_inside_image_count"] == len(source_pixels)
    assert result["inside_source_image"]["p95_error_px"] == pytest.approx(0.0)
    assert result["inside_source_image"]["within_px"]["4"]["ratio"] == 1.0


def test_held_out_match_diagnostic_exposes_incorrect_matches() -> None:
    row = {
        "ground_truth": {
            "quaternion_wxyz_world_to_camera": [1.0, 0.0, 0.0, 0.0],
            "translation_xyz_world_to_camera": [0.0, 0.0, 10.0],
        },
        "source_view": {
            "width": 200,
            "height": 200,
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
        },
    }
    result = _held_out_match_diagnostics(
        source_pixels=np.asarray([[150.0, 150.0], [160.0, 160.0]]),
        map_pixels=np.asarray([[50.0, 50.0], [50.0, 50.0]]),
        certainties=np.ones(2),
        map_image_size=(100, 100),
        map_bounds=(-5.0, -5.0, 5.0, 5.0),
        terrain=_FlatTerrain(),  # type: ignore[arg-type]
        row=row,
    )

    assert result["all_in_front"]["within_px"]["32"]["count"] == 0
    assert result["all_in_front"]["median_error_px"] > 100.0
