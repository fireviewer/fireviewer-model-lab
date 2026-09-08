from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.figlib_index import FIGLIB_INDEX_URL, build_figlib_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Index HPWREN FIgLib without payload download")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index-url", default=FIGLIB_INDEX_URL)
    parser.add_argument("--minimum-sequences", type=int, default=400)
    args = parser.parse_args()
    report = build_figlib_index(
        args.output_root,
        index_url=args.index_url,
        minimum_sequences=args.minimum_sequences,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
