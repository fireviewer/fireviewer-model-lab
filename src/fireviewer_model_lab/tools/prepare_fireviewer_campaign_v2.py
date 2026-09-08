"""CLI for the shared FireViewer v2 campaign datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.fireviewer_campaign_v2 import build_segmentation_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("segmentation", choices=("segmentation",))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--boreal-root", type=Path, required=True)
    parser.add_argument("--firesentry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = build_segmentation_corpus(
        campaign_root=args.campaign_root,
        boreal_root=args.boreal_root,
        firesentry_root=args.firesentry_root,
        output_root=args.output,
        additional_manifests=tuple(args.additional_manifest),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
