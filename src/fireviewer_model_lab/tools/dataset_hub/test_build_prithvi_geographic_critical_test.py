from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import pytest
from build_prithvi_geographic_critical_test import (
    EXPECTED_SPLITS,
    MATERIALIZED_VALIDATOR_ID,
    SOURCE_VALIDATOR_ID,
    _audit_materialized_candidate,
    _decode_candidate,
    _load_split,
    _mark_dual_automated_validation,
    _selection_sha256,
    _write_candidate,
)


def test_selection_digest_matches_prithvi_gate_contract() -> None:
    rows = [
        {
            "sample_id": "b",
            "event_id": "event-b",
            "site_id": "site-b",
            "image_sha256": "1" * 64,
            "mask_sha256": "2" * 64,
        },
        {
            "sample_id": "a",
            "event_id": "event-a",
            "site_id": "site-a",
            "image_sha256": "3" * 64,
            "mask_sha256": "4" * 64,
        },
    ]
    selection = [
        {
            "event_id": row["event_id"],
            "image_sha256": row["image_sha256"],
            "mask_sha256": row["mask_sha256"],
            "sample_id": row["sample_id"],
            "site_id": row["site_id"],
        }
        for row in rows
    ]
    import hashlib

    expected = hashlib.sha256(
        (
            json.dumps(
                sorted(selection, key=lambda row: row["sample_id"]),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        usedforsecurity=False,
    ).hexdigest()
    assert _selection_sha256(rows) == expected


def test_load_split_rejects_digest_drift(tmp_path: Path) -> None:
    split = tmp_path / "split.csv.gz"
    with gzip.open(split, "wb") as handle:
        handle.write(b"one.nc\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        _load_split(
            str(split),
            {"count": 1, "sha256": "0" * 64},
        )


def test_official_split_contract_is_disjoint_by_declared_size() -> None:
    assert EXPECTED_SPLITS["train"]["count"] == 20_307
    assert EXPECTED_SPLITS["validation"]["count"] == 5_077
    assert EXPECTED_SPLITS["test"]["count"] == 6_346
    assert sum(value["count"] for value in EXPECTED_SPLITS.values()) == 31_730


def test_gzip_fixture_shape_is_supported(tmp_path: Path) -> None:
    payload = io.BytesIO()
    with gzip.GzipFile(fileobj=payload, mode="wb") as archive:
        archive.write(b"one.nc\ntwo.nc\n")
    split = tmp_path / "split.csv.gz"
    split.write_bytes(payload.getvalue())
    import hashlib

    digest = hashlib.sha256(payload.getvalue(), usedforsecurity=False).hexdigest()
    _compressed, rows = _load_split(
        str(split),
        {"count": 2, "sha256": digest},
    )
    assert rows == ["one.nc", "two.nc"]


def test_write_candidate_materializes_valid_image_and_mask_geotiffs(
    tmp_path: Path,
) -> None:
    np = pytest.importorskip("numpy")
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    row = _write_candidate(
        tmp_path,
        "event-001.nc",
        b"official-source-payload",
        {
            "image": np.arange(6 * 4 * 5, dtype=np.float32).reshape(6, 4, 5),
            "mask": np.asarray(
                [
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int16,
            ),
            "transform": Affine(0.01, 0, 2.0, 0, -0.01, 49.0),
            "crs": CRS.from_epsg(4326),
            "bounds": [2.0, 48.96, 2.05, 49.0],
            "site_id": "eo4-grid-+48-+002",
            "positive_pixels": 6,
            "valid_pixels": 20,
        },
    )

    with rasterio.open(tmp_path / row["image_relpath"]) as image:
        assert image.count == 6
        assert image.dtypes == ("float32",) * 6
        assert image.descriptions == (
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        )
        assert image.crs.to_epsg() == 4326
    with rasterio.open(tmp_path / row["mask_relpath"]) as mask:
        assert mask.count == 1
        assert mask.dtypes == ("int16",)
        assert mask.nodata == -1
        assert set(np.unique(mask.read(1))) == {0, 1}
    assert _audit_materialized_candidate(tmp_path, row) == []
    _mark_dual_automated_validation(row)
    assert row["validation_status"] == "dual_automated_validation_passed"
    assert row["validator_count"] == 2
    assert row["automated_validators"] == [
        SOURCE_VALIDATOR_ID,
        MATERIALIZED_VALIDATOR_ID,
    ]


def test_decode_candidate_uses_official_eo4_nan_background_semantics() -> None:
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.transform import Affine, xy

    transform = Affine(0.01, 0, 2.0, 0, -0.01, 49.0)
    x = np.asarray(xy(transform, 0, range(3), offset="center")[0])
    y = np.asarray(xy(transform, range(2), 0, offset="center")[1])
    payload = io.BytesIO()
    with h5py.File(payload, "w") as dataset:
        dataset.create_dataset(
            "S2A",
            data=np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3),
        )
        dataset.create_dataset(
            "burned_mask",
            data=np.asarray([[np.nan, 1, 1], [np.nan, np.nan, 1]], dtype=np.float32),
        )
        dataset.create_dataset("x", data=x)
        dataset.create_dataset("y", data=y)
        spatial_ref = dataset.create_dataset("spatial_ref", data=0)
        spatial_ref.attrs["GeoTransform"] = " ".join(str(value) for value in transform.to_gdal())
        spatial_ref.attrs["crs_wkt"] = CRS.from_epsg(4326).to_wkt()

    decoded = _decode_candidate(payload.getvalue())

    assert set(np.unique(decoded["mask"])) == {0, 1}
    assert decoded["positive_pixels"] == 3
    assert decoded["valid_pixels"] == 6
