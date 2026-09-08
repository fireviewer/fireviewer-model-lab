from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_model_lab.training.dinov3_v4_campaign import (
    REGISTRY_PATH,
    acquire_hf_sources,
    adapt_baseline_manifest,
    load_registry,
    merge_manifests,
)


def _token(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("Hugging Face token file is empty")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare FireViewer DINOv3 v4 dataset")
    parser.add_argument("action", choices=("plan", "acquire-hf", "adapt-baseline", "merge"))
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--source",
        action="append",
        choices=("fireviewer-v3-baseline", "pyronear-pyro-sdis"),
        default=[],
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    if args.action == "plan":
        result = registry
    else:
        if args.campaign_root is None:
            parser.error(f"{args.action} requires --campaign-root")
        campaign_root = args.campaign_root.resolve()
        if args.action == "acquire-hf":
            if args.token_file is None:
                parser.error("acquire-hf requires --token-file")
            result = acquire_hf_sources(
                registry=registry,
                campaign_root=campaign_root,
                token=_token(args.token_file),
                source_ids=tuple(args.source)
                if args.source
                else (
                    "fireviewer-v3-baseline",
                    "pyronear-pyro-sdis",
                ),
                workers=args.workers,
            )
        elif args.action == "adapt-baseline":
            if args.output is None:
                parser.error("adapt-baseline requires --output")
            result = adapt_baseline_manifest(
                campaign_root=campaign_root,
                baseline_root=campaign_root / "sources" / "fireviewer-v3-baseline",
                output=args.output.resolve(),
            )
        else:
            if args.output is None or not args.manifest:
                parser.error("merge requires --output and at least one --manifest")
            result = merge_manifests(
                campaign_root=campaign_root,
                manifests=[path.resolve() for path in args.manifest],
                output_root=args.output.resolve(),
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
