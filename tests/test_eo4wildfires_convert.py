from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import fireviewer_model_lab.training.eo4wildfires_convert as eo4wildfires_convert
from fireviewer_model_lab.training.eo4wildfires_convert import convert, inspect_netcdf, load_official_splits


def _scene() -> bytes:
    payload = io.BytesIO()
    with h5py.File(payload, "w") as dataset:
        dataset.create_dataset("S1_GRD_A", data=np.zeros((3, 2, 2), dtype="float32"))
        dataset.create_dataset("S1_GRD_D", data=np.zeros((3, 2, 2), dtype="float32"))
        dataset.create_dataset("S2A", data=np.zeros((6, 2, 2), dtype="float32"))
        dataset.create_dataset("BURNED_AREA", data=np.array(4.0, dtype="float64"))
        dataset.create_dataset("burned_mask", data=np.array([[0, 1], [1, 0]], dtype="float32"))
        dataset.create_dataset("x", data=np.array([10.0, 20.0], dtype="float64"))
        dataset.create_dataset("y", data=np.array([30.0, 40.0], dtype="float64"))
        for variable in (
            "RH2M",
            "T2M",
            "PRECTOTCORR",
            "WS2M",
            "FRSNO",
            "GWETROOT",
            "SNODP",
            "PRECSNOLAND",
            "GWETTOP",
        ):
            dataset.create_dataset(variable, data=np.zeros(31, dtype="float32"))
        dataset.attrs["crs"] = "EPSG:4326"
    return payload.getvalue()


def _write_split(path: Path, names: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write("".join(f"{name}\n" for name in names))


def test_drop_page_cache_keeps_materialized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "shard.tar.gz"
    shard.write_bytes(b"materialized")
    calls: list[tuple[int, int, int]] = []
    sync_calls: list[int] = []

    monkeypatch.setattr(
        eo4wildfires_convert.os,
        "fsync",
        lambda fd: sync_calls.append(fd),
    )
    monkeypatch.setattr(
        eo4wildfires_convert.os,
        "posix_fadvise",
        lambda _fd, offset, length, advice: calls.append((offset, length, advice)),
        raising=False,
    )
    monkeypatch.setattr(eo4wildfires_convert.os, "POSIX_FADV_DONTNEED", 4, raising=False)

    eo4wildfires_convert._drop_page_cache(shard)

    assert len(sync_calls) == 1
    assert calls == [(0, 0, 4)]
    assert shard.read_bytes() == b"materialized"


def test_official_splits_reject_cross_split_leakage(tmp_path: Path) -> None:
    for split in ("train", "validation", "test"):
        _write_split(tmp_path / f"{split}.csv.gz", ["same.nc"])

    with pytest.raises(ValueError, match="Cross-split leakage"):
        load_official_splits(
            {split: str(tmp_path / f"{split}.csv.gz") for split in ("train", "validation", "test")},
            enforce_declared_counts=False,
        )


def test_netcdf_inspection_reports_mask_coordinates_and_crs() -> None:
    report = inspect_netcdf(_scene())

    assert report["burned_mask"]["positive_pixels"] == 2
    assert report["burned_mask"]["positive_fraction"] == 0.5
    assert report["coordinates"]["x"] == {"count": 2, "min": 10.0, "max": 20.0}
    assert report["crs_evidence"]["root.crs"] == "EPSG:4326"


def test_conversion_materializes_unchanged_scene_in_official_split(tmp_path: Path) -> None:
    split_locations: dict[str, str] = {}
    for split, names in (("train", ["1.nc"]), ("validation", []), ("test", [])):
        split_path = tmp_path / f"{split}.csv.gz"
        _write_split(split_path, names)
        split_locations[split] = str(split_path)
    archive = tmp_path / "source.tar.gz"
    scene = _scene()
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("eo4wildfires/1.nc")
        member.size = len(scene)
        output.addfile(member, io.BytesIO(scene))

    report = convert(
        archive=str(archive),
        split_locations=split_locations,
        output_root=tmp_path / "output",
        samples_per_shard=1,
        max_samples=1,
    )

    shard = tmp_path / "output" / "train" / "eo4wildfires-train-00000.tar.gz"
    with tarfile.open(shard, "r:gz") as materialized:
        extracted = materialized.extractfile("eo4wildfires/1.nc")
        assert extracted is not None
        assert extracted.read() == scene
    assert report["processed_scenes"] == 1
    assert report["materialized_split_counts"] == {"train": 1, "validation": 0, "test": 0}
