from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from huggingface_hub import hf_hub_download

from fireviewer_vision_runtime.mvp.vision.yolo import (
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact = Path(
        hf_hub_download(
            MODEL_ID,
            MODEL_FILENAME,
            revision=MODEL_REVISION,
            cache_dir=args.cache_dir,
        )
    )
    digest = file_sha256(artifact)
    if digest != MODEL_SHA256:
        raise SystemExit(f"unexpected model SHA-256: {digest}")
    print(f"prefetched {MODEL_ID}@{MODEL_REVISION} sha256:{digest}")


if __name__ == "__main__":
    main()
