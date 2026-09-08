from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, TiffImagePlugin
from fireviewer_model_lab.training.thu_wildfire_ninuo_setup import _parse_dji_xmp, prepare


def _write_jpeg(path: Path, *, timestamp: str, serial: str, latitude: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[306] = timestamp
    Image.new("RGB", (8, 8), (20, 40, 60)).save(path, exif=exif)
    xmp = (
        f'<rdf drone-dji:DroneSerialNumber="{serial}" drone-dji:DroneModel="M30T" '
        f'drone-dji:GpsStatus="RTK" drone-dji:RtkFlag="50" '
        f'drone-dji:GpsLatitude="{latitude}" drone-dji:GpsLongitude="2.0" '
        'drone-dji:AbsoluteAltitude="100.0" drone-dji:RelativeAltitude="50.0" '
        'drone-dji:GimbalRollDegree="0.0" drone-dji:GimbalYawDegree="0.0" '
        'drone-dji:GimbalPitchDegree="-90.0" drone-dji:FlightXSpeed="0.0" '
        'drone-dji:FlightYSpeed="0.0" drone-dji:FlightZSpeed="0.0" />'
    ).encode()
    with path.open("ab") as handle:
        handle.write(xmp)


def _write_tiff(path: Path, *, projected: bool, rgb: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB" if rgb else "F", (8, 8), (1, 2, 3) if rgb else 1.0)
    tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
    if projected:
        tiffinfo[34264] = (
            0.00001,
            0.0,
            0.0,
            2.0,
            0.0,
            -0.00001,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    image.save(path, tiffinfo=tiffinfo)


def _build_fixture(root: Path) -> Path:
    source = root / "Ninuo"
    image_root = source / "Active-fire/Image"
    for frame_id, serial, latitude in (
        (1, "camera-a", 1.0),
        (2, "camera-b", 1.0),
        (3, "camera-a", 1.0),
    ):
        stem = f"{frame_id:06d}"
        timestamp = f"2025:02:14 12:00:0{frame_id}"
        _write_jpeg(
            image_root / "Optical" / f"{stem}.jpg",
            timestamp=timestamp,
            serial=serial,
            latitude=latitude,
        )
        _write_jpeg(
            image_root / "InfraredJPG" / f"{stem}.jpg",
            timestamp=timestamp,
            serial=serial,
            latitude=latitude,
        )
        _write_tiff(image_root / "Optical_projected" / f"{stem}.tif", projected=True, rgb=True)
        _write_tiff(image_root / "ThermalTIFF" / f"{stem}.tif", projected=False)
        _write_tiff(image_root / "ThermalTIFF_projected" / f"{stem}.tif", projected=True)
    annotation_root = image_root / "Annotation"
    annotation_root.mkdir(parents=True)
    Image.new("L", (8, 8), 255).save(annotation_root / "000001.png")
    Image.new("L", (8, 8), 255).save(annotation_root / "000003.png")
    return source


def test_dji_xmp_parser_extracts_pose_fields() -> None:
    parsed = _parse_dji_xmp(
        b'<rdf drone-dji:DroneSerialNumber="camera-a" Camera:GpsStatus="RTK" />'
    )

    assert parsed == {"DroneSerialNumber": "camera-a", "GpsStatus": "RTK"}


def test_prepare_quarantines_single_view_temporal_subset(tmp_path: Path) -> None:
    source = _build_fixture(tmp_path)
    output = tmp_path / "output"

    report = prepare(
        source,
        output,
        expected_frames=3,
        expected_annotations=2,
    )

    assert report["quarantine_manifest_ready"] is True
    assert report["temporal_cross_modal_data_present"] is True
    assert report["true_multiview_training_ready"] is False
    assert report["geometry"]["camera_count"] == 2
    assert "declared_license_missing" in report["blocking_reasons"]
    assert "viewpoint_baseline_below_multiview_requirement" in report["blocking_reasons"]

    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["split"] == "quarantine" for row in rows)
    assert all(row["training_membership"] is False for row in rows)
    assert sum(row["annotation"] is not None for row in rows) == 2
    assert not any("DroneSerialNumber" in json.dumps(row) for row in rows)
