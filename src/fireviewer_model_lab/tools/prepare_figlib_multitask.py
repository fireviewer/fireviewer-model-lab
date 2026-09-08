from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.figlib_multitask import (
    acquire_figlib_archive,
    acquire_figlib_selection,
    materialize_figlib,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare HPWREN FIgLib for DINOv3")
    parser.add_argument("action", choices=("acquire", "acquire-local", "materialize"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path)
    parser.add_argument("--acquisition-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()
    if args.action == "acquire":
        if args.index_manifest is None:
            parser.error("acquire requires --index-manifest")
        result = acquire_figlib_selection(
            index_manifest=args.index_manifest.resolve(),
            campaign_root=args.campaign_root.resolve(),
            output_root=args.output_root.resolve(),
            workers=args.workers,
        )
    elif args.action == "acquire-local":
        if args.index_manifest is None or args.archive is None:
            parser.error("acquire-local requires --index-manifest and --archive")
        result = acquire_figlib_archive(
            archive=args.archive.resolve(),
            index_manifest=args.index_manifest.resolve(),
            campaign_root=args.campaign_root.resolve(),
            output_root=args.output_root.resolve(),
            delete_archive=not args.keep_archive,
        )
    else:
        if args.acquisition_manifest is None:
            parser.error("materialize requires --acquisition-manifest")
        result = materialize_figlib(
            acquisition_manifest=args.acquisition_manifest.resolve(),
            campaign_root=args.campaign_root.resolve(),
            output_root=args.output_root.resolve(),
            batch_size=args.batch_size,
            device_name=args.device,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
