from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.firesgl_multitask import (
    acquire_firesgl_archives,
    acquire_firesgl_selection,
    materialize_firesgl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare FIReStereo FiresGL supervision")
    parser.add_argument("action", choices=("acquire", "acquire-local", "materialize"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-per-sequence", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--acquisition-manifest", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    if args.action == "acquire":
        result = acquire_firesgl_selection(
            campaign_root=args.campaign_root.resolve(),
            output_root=args.output_root.resolve(),
            maximum_per_sequence=args.maximum_per_sequence,
            workers=args.workers,
        )
    elif args.action == "acquire-local":
        if args.archive_root is None:
            parser.error("acquire-local requires --archive-root")
        result = acquire_firesgl_archives(
            campaign_root=args.campaign_root.resolve(),
            archive_root=args.archive_root.resolve(),
            output_root=args.output_root.resolve(),
            maximum_per_sequence=args.maximum_per_sequence,
            workers=args.workers,
            delete_archives=not args.keep_archives,
        )
    else:
        if args.acquisition_manifest is None:
            parser.error("materialize requires --acquisition-manifest")
        result = materialize_firesgl(
            acquisition_manifest=args.acquisition_manifest.resolve(),
            campaign_root=args.campaign_root.resolve(),
            output_root=args.output_root.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
