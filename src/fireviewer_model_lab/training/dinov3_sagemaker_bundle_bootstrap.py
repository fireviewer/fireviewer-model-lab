"""Verify the pinned SageMaker composition bundle before executing it."""

from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path

EXPECTED_BUNDLE_SHA256 = {'training/__init__.py': '5347f39488a1caf80541f05685b35d2cb110c25ab3756396e38999d3b64be71c', 'training/dinov3_corpus_identity.py': '0fe55f7f5f5437bf25580dd3b70b2626a3d85d48c0f4ddad4241a74e3fd47637', 'training/dinov3_multitask_compose.py': 'c6f2b028f8a40f40791519037a9b376de4aa23b0b0688cb5281ebdcfe46de696', 'training/registries/dinov3-independent-benchmark-denylist-v1.json': '7368b1337e4ea363b0fb9e711fff94c5795aaea329a5f87a7afc0b46c1659a40', 'training/registries/dinov3-multitask-composition-v1.json': '27de9d64c3d758f2078148fa51ab95d97df9f86c5f4b6e56d05732477462a8d4'}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("code bundle root is missing")
    root = Path(sys.argv[1]).resolve()
    for relative, expected_sha256 in EXPECTED_BUNDLE_SHA256.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise SystemExit(f"unsafe code bundle path: {relative}")
        if not path.is_file():
            raise SystemExit(f"required code bundle file is missing: {relative}")
        if _sha256(path) != expected_sha256:
            raise SystemExit(f"code bundle SHA-256 mismatch: {relative}")

    script = (root / "training/dinov3_multitask_compose.py").resolve()
    sys.path.insert(0, str(root))
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
