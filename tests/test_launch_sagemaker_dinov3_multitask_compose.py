from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fireviewer_model_lab.tools.launch_sagemaker_dinov3_multitask_compose import (
    BOOTSTRAP_RELATIVE_PATH,
    CODE_BUNDLE_BOOTSTRAP_VERIFIER,
    CODE_BUNDLE_PINS,
    DEFAULT_CODE_BUNDLE_ROOT,
    build_request,
)
from fireviewer_model_lab.training.dinov3_sagemaker_bundle_bootstrap import EXPECTED_BUNDLE_SHA256


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "bucket": "bucket",
        "role": "role",
        "image": "image",
        "pyro_control_prefix": "pointing/pyro/run",
        "boreal_prefix": "pointing/boreal/run",
        "camp_swift_prefix": "pointing/camp/run",
        "kit_prefix": "pointing/kit/run",
        "ijmond_prefix": "pointing/ijmond/run",
        "code_prefix": "pointing/code/compose",
        "code_bundle_root": DEFAULT_CODE_BUNDLE_ROOT,
        "output_prefix": "pointing/composition",
        "instance_type": "ml.t3.large",
        "volume_size": 30,
        "max_runtime": 3600,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_compose_launcher_keeps_training_and_publication_closed() -> None:
    request = build_request(_args(), "job")

    inputs = {item["InputName"]: item for item in request["ProcessingInputs"]}
    assert set(inputs) == {
        "code",
        "pyro-control",
        "boreal",
        "camp-swift",
        "kit",
        "ijmond",
    }
    arguments = request["AppSpecification"]["ContainerArguments"]
    assert arguments[:2] == ["-c", CODE_BUNDLE_BOOTSTRAP_VERIFIER]
    assert arguments[2].endswith(f"/{BOOTSTRAP_RELATIVE_PATH}")
    assert arguments[3] == CODE_BUNDLE_PINS[BOOTSTRAP_RELATIVE_PATH]
    assert max(map(len, arguments)) <= 256
    assert arguments.count("--overlay") == 4
    assert arguments.count("--control") == 1
    assert arguments[arguments.index("--registry") + 1].endswith(
        "/training/registries/dinov3-multitask-composition-v1.json"
    )
    assert arguments[arguments.index("--benchmark-denylist") + 1].endswith(
        "/training/registries/dinov3-independent-benchmark-denylist-v1.json"
    )
    assert {
        key: value for key, value in CODE_BUNDLE_PINS.items() if key != BOOTSTRAP_RELATIVE_PATH
    } == EXPECTED_BUNDLE_SHA256
    environment = request["Environment"]
    assert (
        environment["FIREVIEWER_IDENTITY_MODULE_SHA256"]
        == CODE_BUNDLE_PINS["training/dinov3_corpus_identity.py"]
    )
    assert (
        environment["FIREVIEWER_BENCHMARK_DENYLIST_SHA256"]
        == CODE_BUNDLE_PINS["training/registries/dinov3-independent-benchmark-denylist-v1.json"]
    )
    tags = {item["Key"]: item["Value"] for item in request["Tags"]}
    assert tags["fireviewer:publication-allowed"] == "false"
    assert tags["fireviewer:training-ready"] == "false"


def test_compose_launcher_rejects_benchmark_inputs() -> None:
    with pytest.raises(ValueError, match="independent benchmark"):
        build_request(_args(kit_prefix="independent-benchmark/kit"), "job")


def _copy_bundle(destination: Path) -> None:
    for relative in CODE_BUNDLE_PINS:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEFAULT_CODE_BUNDLE_ROOT / relative, target)


@pytest.mark.parametrize(
    "relative",
    [
        "training/dinov3_corpus_identity.py",
        "training/registries/dinov3-independent-benchmark-denylist-v1.json",
    ],
)
def test_compose_launcher_fails_closed_for_missing_bundle_file(
    tmp_path: Path, relative: str
) -> None:
    _copy_bundle(tmp_path)
    (tmp_path / relative).unlink()

    with pytest.raises(ValueError, match="required code bundle file is missing"):
        build_request(_args(code_bundle_root=tmp_path), "job")


@pytest.mark.parametrize(
    "relative",
    [
        "training/dinov3_corpus_identity.py",
        "training/registries/dinov3-independent-benchmark-denylist-v1.json",
    ],
)
def test_compose_launcher_fails_closed_for_bundle_sha_drift(tmp_path: Path, relative: str) -> None:
    _copy_bundle(tmp_path)
    with (tmp_path / relative).open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="code bundle SHA-256 mismatch"):
        build_request(_args(code_bundle_root=tmp_path), "job")


@pytest.mark.parametrize(
    "relative",
    [
        "training/dinov3_corpus_identity.py",
        "training/registries/dinov3-independent-benchmark-denylist-v1.json",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "sha_drift"])
def test_runtime_bootstrap_rejects_missing_or_drift_before_execution(
    tmp_path: Path, relative: str, mutation: str
) -> None:
    _copy_bundle(tmp_path)
    target = tmp_path / relative
    if mutation == "missing":
        target.unlink()
        expected_error = f"required code bundle file is missing: {relative}"
    else:
        with target.open("ab") as handle:
            handle.write(b"\n")
        expected_error = f"code bundle SHA-256 mismatch: {relative}"

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned arguments.
        [
            sys.executable,
            "-c",
            CODE_BUNDLE_BOOTSTRAP_VERIFIER,
            str(tmp_path / BOOTSTRAP_RELATIVE_PATH),
            CODE_BUNDLE_PINS[BOOTSTRAP_RELATIVE_PATH],
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr


@pytest.mark.parametrize("mutation", ["missing", "sha_drift"])
def test_runtime_verifier_rejects_missing_or_drifted_bootstrap(
    tmp_path: Path, mutation: str
) -> None:
    _copy_bundle(tmp_path)
    bootstrap = tmp_path / BOOTSTRAP_RELATIVE_PATH
    if mutation == "missing":
        bootstrap.unlink()
    else:
        with bootstrap.open("ab") as handle:
            handle.write(b"\n")

    completed = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned arguments.
        [
            sys.executable,
            "-c",
            CODE_BUNDLE_BOOTSTRAP_VERIFIER,
            str(bootstrap),
            CODE_BUNDLE_PINS[BOOTSTRAP_RELATIVE_PATH],
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "bootstrap integrity failure" in completed.stderr
