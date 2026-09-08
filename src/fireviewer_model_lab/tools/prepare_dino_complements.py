from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.dino_complements import (
    REGISTRY_PATH,
    build_plan,
    download_assets,
    extract_archives,
    find_source,
    load_registry,
    probe_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare bounded DINOv3 dataset complements")
    parser.add_argument("action", choices=("plan", "probe", "download", "extract"))
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--source", default="firestereo-firesgl")
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--maximum-assets", type=int)
    parser.add_argument("--delete-archives", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if args.action == "plan":
        result = build_plan(registry)
    else:
        source = find_source(registry, args.source)
        if args.action == "probe":
            result = {"source_id": args.source, "assets": probe_assets(source)}
        elif args.action == "download":
            if args.archive_root is None:
                parser.error("--archive-root is required for download")
            result = {
                "source_id": args.source,
                "assets": download_assets(
                    source, args.archive_root, maximum_assets=args.maximum_assets
                ),
            }
        else:
            if args.archive_root is None or args.output_root is None:
                parser.error("--archive-root and --output-root are required for extract")
            result = {
                "source_id": args.source,
                "assets": extract_archives(
                    source,
                    args.archive_root,
                    args.output_root,
                    delete_archives=args.delete_archives,
                    maximum_assets=args.maximum_assets,
                ),
            }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
