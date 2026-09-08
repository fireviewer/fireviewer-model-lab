from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_vision_runtime.mvp.vision import inspect_local_grounding_dino_bundle


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a digest-qualified manifest for an explicit local Grounding DINO snapshot."
        )
    )
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    manifest = inspect_local_grounding_dino_bundle(args.directory, revision=args.revision)
    payload = json.dumps(
        manifest.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "file_count": len(manifest.files),
                "model_id": manifest.model_id,
                "revision": manifest.revision,
                "status": "qualified",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
