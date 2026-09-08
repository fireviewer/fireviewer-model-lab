from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.gaussians_cross_view import build_gaussians_cross_view_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Gaussians on Fire cross-view pairs")
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    args = parser.parse_args()
    report = build_gaussians_cross_view_manifest(
        campaign_root=args.campaign_root,
        shared_root=args.shared_root,
        source_root=args.source_root,
        output_root=args.output,
        frame_stride=args.frame_stride,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
