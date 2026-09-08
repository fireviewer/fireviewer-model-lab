"""Prepare the Gaussians on Fire sparse-depth benchmark for MoGe-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.gaussians_moge import build_gaussians_moge_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=30)
    args = parser.parse_args()
    report = build_gaussians_moge_manifest(
        campaign_root=args.campaign_root,
        shared_root=args.shared_root,
        source_root=args.source_root,
        output_root=args.output,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
