"""Prepare the shared Camp Swift EO/IR segmentation payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.camp_swift_thermal import build_camp_swift_thermal_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--geodatabase-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-delta-ms", type=int, default=2000)
    parser.add_argument("--red-threshold", type=int, default=40)
    parser.add_argument("--minimum-component-pixels", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    report = build_camp_swift_thermal_corpus(
        campaign_root=args.campaign_root,
        geodatabase_root=args.geodatabase_root,
        output_root=args.output,
        frame_stride=args.frame_stride,
        max_delta_ms=args.max_delta_ms,
        red_threshold=args.red_threshold,
        minimum_component_pixels=args.minimum_component_pixels,
        jobs=args.jobs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
