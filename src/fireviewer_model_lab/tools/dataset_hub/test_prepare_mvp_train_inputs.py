from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from prepare_mvp_train_inputs import (
    PRITHVI_BUNDLE,
    BundleContract,
    _safe_archive_relative,
    extract_required_payloads,
    prepare_prithvi,
    verify_bundle,
)


def test_prithvi_contract_matches_the_current_public_materialized_bundle() -> None:
    assert PRITHVI_BUNDLE.size_bytes == 27_926_097_197
    assert (
        PRITHVI_BUNDLE.sha256 == "85c2f17248528ebbd5aa8395e72435ba5a12626bb5a53f5730109b11ea5dde36"
    )
    assert PRITHVI_BUNDLE.prefixes == ("materialized",)


def _contract(path: Path) -> BundleContract:
    return BundleContract(
        train_id="train-v1",
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        prefixes=("sources/useful",),
    )


def test_verified_extraction_keeps_only_the_required_subtree(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, mode="w") as output:
        output.writestr("train-v1/sources/useful/manifest.jsonl", "{}\n")
        output.writestr("train-v1/sources/unused/data.bin", b"unused")
    contract = _contract(archive)

    verified = verify_bundle(archive, contract)
    report = extract_required_payloads(archive, tmp_path / "output", contract)

    assert verified["sha256"] == contract.sha256
    assert report["extracted_files"] == 1
    assert (tmp_path / "output/sources/useful/manifest.jsonl").is_file()
    assert not (tmp_path / "output/sources/unused").exists()


def test_archive_path_traversal_is_rejected() -> None:
    contract = BundleContract("train-v1", "bundle.zip", 1, "0" * 64, ("sources",))

    with pytest.raises(ValueError, match="Unsafe"):
        _safe_archive_relative("train-v1/sources/../../escape", contract)


def test_existing_preparation_must_still_match_the_public_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "bundle.zip"
    archive.write_bytes(b"exact-public-bundle")
    destination = tmp_path / "prepared"
    destination.mkdir()
    (destination / "preparation-report.json").write_text(
        '{"bundle":{"filename":"different.zip"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "prepare_mvp_train_inputs.PRITHVI_BUNDLE",
        BundleContract(
            "train-v1",
            archive.name,
            archive.stat().st_size,
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            ("corpus",),
        ),
    )

    with pytest.raises(ValueError, match="does not match"):
        prepare_prithvi(tmp_path, destination)
