#!/usr/bin/env python3
"""Create the premium ground-view RF-DETR Small corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.rfdetr_premium_ground import prepare_elite_ground, prepare_premium_ground


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("premium", "elite"), default="premium")
    args = parser.parse_args()
    prepare = prepare_elite_ground if args.profile == "elite" else prepare_premium_ground
    print(
        json.dumps(
            prepare(args.source_root, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
