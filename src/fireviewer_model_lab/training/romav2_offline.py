"""Fail-closed offline loader for the pinned RoMaV2 FireViewer benchmark."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROMAV2_REVISION = "7151f3846ad0c89c213afb6803966484a6dd76e0"
DINOV3_TORCHHUB_REVISION = "adc254450203739c8149213a7a69d8d905b4fcfa"
ROMAV2_WEIGHTS_SHA256 = "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"
ROMAV2_WEIGHTS_URL = "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt"
ROMAV2_RELEASE_URLS = {
    ROMAV2_WEIGHTS_URL,
    "https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt",
}
DINOV3_TORCHHUB_REF = "facebookresearch/dinov3:" + DINOV3_TORCHHUB_REVISION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git executable is required")
    result = subprocess.run(  # noqa: S603 - resolved git executable and fixed arguments
        [git_executable, "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_assets(*, romav2_source: Path, dinov3_source: Path, weights: Path) -> dict[str, Any]:
    romav2_source = romav2_source.resolve()
    dinov3_source = dinov3_source.resolve()
    weights = weights.resolve()
    if not (romav2_source / "src" / "romav2" / "romav2.py").is_file():
        raise FileNotFoundError(f"RoMaV2 source is incomplete: {romav2_source}")
    if not (dinov3_source / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 torch-hub source is incomplete: {dinov3_source}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    roma_revision = git_revision(romav2_source)
    dino_revision = git_revision(dinov3_source)
    weights_sha = sha256(weights)
    if roma_revision != ROMAV2_REVISION:
        raise RuntimeError(f"RoMaV2 revision mismatch: {roma_revision}")
    if dino_revision != DINOV3_TORCHHUB_REVISION:
        raise RuntimeError(f"DINOv3 torch-hub revision mismatch: {dino_revision}")
    if weights_sha != ROMAV2_WEIGHTS_SHA256:
        raise RuntimeError(f"RoMaV2 weights SHA-256 mismatch: {weights_sha}")
    return {
        "romav2_revision": roma_revision,
        "dinov3_torchhub_revision": dino_revision,
        "weights_sha256": weights_sha,
        "weights_bytes": weights.stat().st_size,
    }


def load_romav2_offline(
    *,
    romav2_source: Path,
    dinov3_source: Path,
    weights: Path,
    setting: str = "turbo",
) -> tuple[Any, dict[str, Any]]:
    """Instantiate RoMaV2 without permitting an implicit network fallback."""

    provenance = verify_assets(
        romav2_source=romav2_source,
        dinov3_source=dinov3_source,
        weights=weights,
    )
    source_dir = str(romav2_source.resolve() / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    import torch
    from romav2 import RoMaV2

    original_load = torch.hub.load
    original_load_state_dict = torch.hub.load_state_dict_from_url

    def local_dinov3_load(repo_or_dir: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir != DINOV3_TORCHHUB_REF or model != "dinov3_vitl16":
            raise RuntimeError(f"unexpected torch-hub dependency: {repo_or_dir}:{model}")
        kwargs.pop("skip_validation", None)
        return original_load(
            str(dinov3_source.resolve()),
            model,
            *args,
            source="local",
            **kwargs,
        )

    def local_weights(url: str, *args: Any, **kwargs: Any) -> Any:
        if url not in ROMAV2_RELEASE_URLS:
            raise RuntimeError(f"unexpected RoMaV2 weights URL: {url}")
        map_location = kwargs.get("map_location", "cpu")
        return torch.load(
            weights.resolve(),
            map_location=map_location,
            weights_only=True,
        )

    torch.hub.load = local_dinov3_load
    torch.hub.load_state_dict_from_url = local_weights
    try:
        model = RoMaV2(RoMaV2.Cfg(setting=setting, compile=False))
    finally:
        torch.hub.load = original_load
        torch.hub.load_state_dict_from_url = original_load_state_dict
    return model, provenance
