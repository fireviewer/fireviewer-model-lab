"""Prepare bounded RxCADRE validation data with the immutable v1 teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.rxcadre_teacher import build_rxcadre_teacher_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-repository-revision", required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--maximum-edge", type=int, default=896)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-component-pixels", type=int, default=4)
    parser.add_argument("--pair-max-delta-seconds", type=int, default=20)
    parser.add_argument("--pre-fire-margin-seconds", type=int, default=300)
    parser.add_argument("--negative-limit-per-camera", type=int, default=12)
    args = parser.parse_args()
    report = build_rxcadre_teacher_corpus(
        campaign_root=args.campaign_root,
        source_root=args.source_root,
        output_root=args.output,
        model_path=args.model_path,
        model_revision=args.model_revision,
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_repository_revision=args.teacher_repository_revision,
        interval_seconds=args.interval_seconds,
        maximum_edge=args.maximum_edge,
        batch_size=args.batch_size,
        probability_threshold=args.probability_threshold,
        minimum_component_pixels=args.minimum_component_pixels,
        pair_max_delta_seconds=args.pair_max_delta_seconds,
        pre_fire_margin_seconds=args.pre_fire_margin_seconds,
        negative_limit_per_camera=args.negative_limit_per_camera,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
