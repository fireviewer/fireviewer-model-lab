from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.pyro_sdis_multitask import materialize_pyro_sdis


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Pyro-SDIS for DINOv3 multi-task")
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = materialize_pyro_sdis(
        parquet_root=args.parquet_root.resolve(),
        campaign_root=args.campaign_root.resolve(),
        output_root=args.output_root.resolve(),
        batch_size=args.batch_size,
        threshold=args.threshold,
        device_name=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
