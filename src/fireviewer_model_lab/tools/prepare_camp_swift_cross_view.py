"""Prepare synchronized Camp Swift ground-camera pairs for cross-view training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.camp_swift_cross_view import build_camp_swift_cross_view_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-sync-score", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_camp_swift_cross_view_manifest(
                campaign_root=args.campaign_root,
                video_root=args.video_root,
                output_root=args.output_root,
                stride_seconds=args.stride_seconds,
                minimum_sync_score=args.minimum_sync_score,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
