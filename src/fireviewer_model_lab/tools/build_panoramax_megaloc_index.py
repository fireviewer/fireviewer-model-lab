from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_geolocation.mvp.localization.faiss_index import write_faiss_bundle
from fireviewer_geolocation.mvp.localization.local_megaloc_bundle import (
    LocalMegaLocBundleManifest,
    LocalMegaLocModelLoader,
)
from fireviewer_geolocation.mvp.localization.megaloc import MegaLocConfig, TorchMegaLocEncoder
from fireviewer_geolocation.mvp.localization.panoramax_cache import (
    CachedPanoramaxImageLoader,
    PanoramaxCacheManifest,
)
from fireviewer_geolocation.mvp.localization.regional_index import PanoramaxRegionalIndexBuilder


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a versioned MegaLoc/FAISS index from a qualified Panoramax cache."
    )
    parser.add_argument("--panoramax-cache", required=True, type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    cache_manifest = PanoramaxCacheManifest.model_validate_json(
        (args.panoramax_cache / "cache-manifest.json").read_text(encoding="utf-8")
    )
    model_manifest = LocalMegaLocBundleManifest.model_validate_json(
        args.model_manifest.read_text(encoding="utf-8")
    )
    config = json.loads((args.model_directory / "config.json").read_text(encoding="utf-8"))
    expected_dimension = config.get("feat_dim") if isinstance(config, dict) else None
    if isinstance(expected_dimension, bool) or not isinstance(expected_dimension, int):
        raise ValueError("MegaLoc bundle config has no valid feat_dim")
    encoder = TorchMegaLocEncoder(
        model_loader=LocalMegaLocModelLoader(
            directory=args.model_directory,
            manifest=model_manifest,
        ),
        model_version=model_manifest.revision,
        config=MegaLocConfig(
            expected_dimension=expected_dimension,
            batch_size=args.batch_size,
            device=args.device,
        ),
    )
    builder = PanoramaxRegionalIndexBuilder(
        image_loader=CachedPanoramaxImageLoader(
            directory=args.panoramax_cache,
            manifest=cache_manifest,
        ),
        encoder=encoder,
        panoramax_revision=f"sha256:{cache_manifest.canonical_sha256()}",
    )
    index = builder.build(cache_manifest.search_result)
    index_manifest = write_faiss_bundle(index, args.output)
    print(
        json.dumps(
            {
                "dimension": index_manifest.dimension,
                "index_sha256": index_manifest.index_sha256,
                "model_revision": index_manifest.model_version,
                "output": str(args.output),
                "status": "ready",
                "vector_count": index_manifest.vector_count,
                "zone_id": index_manifest.zone_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
