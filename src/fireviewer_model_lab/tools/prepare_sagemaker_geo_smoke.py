from __future__ import annotations

import argparse
import base64
import json
from hashlib import sha256
from pathlib import Path

from fireviewer_orchestrator.mvp.gpu.sagemaker_service import GeoGpuRequest


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()
    if not 1 <= len(args.images) <= 8:
        raise ValueError("the bounded smoke accepts one to eight images")
    payloads = []
    for index, path in enumerate(args.images, start=1):
        resolved = path.resolve(strict=True)
        content = resolved.read_bytes()
        payloads.append(
            {
                "input_id": f"GPU-SMOKE-IMAGE-{index}",
                "content_type": "image/png",
                "content_sha256": sha256(content).hexdigest(),
                "content_base64": base64.b64encode(content).decode("ascii"),
            }
        )
    request = GeoGpuRequest.model_validate(
        {
            "schema": "fireviewer.geo-gpu-request.v1",
            "request_id": "FIREVIEWER-FIRST-GPU-SMOKE",
            "operation": "megaloc.encode",
            "payloads": payloads,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        request.model_dump_json(by_alias=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": "fireviewer.aws-geo-smoke-request-receipt.v1",
                "path": str(args.output.resolve()),
                "sha256": _digest(args.output),
                "byte_size": args.output.stat().st_size,
                "image_count": len(args.images),
                "operation": request.operation,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
