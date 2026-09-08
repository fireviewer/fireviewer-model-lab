from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_geolocation.mvp.localization import (
    EventLocalizationConfig,
    LocalEvidenceImageLoader,
    MegaLocFaissEventLocalizer,
    MegaLocFaissRetriever,
    PanoramaxCacheManifest,
    RetrievalConfig,
    abstain_for_missing_reference_coverage,
)
from fireviewer_geolocation.mvp.localization.faiss_index import load_faiss_bundle
from fireviewer_geolocation.mvp.localization.local_megaloc_bundle import (
    LocalMegaLocBundleManifest,
    LocalMegaLocModelLoader,
)
from fireviewer_geolocation.mvp.localization.megaloc import MegaLocConfig, TorchMegaLocEncoder
from fireviewer_orchestrator.mvp.orchestration import CorpusEventRuntimeInput


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localize one materialized EventEvidence with MegaLoc and regional FAISS."
    )
    parser.add_argument("--runtime-input", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--panoramax-cache", type=Path)
    parser.add_argument("--model-directory", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def _finish(result: EventEvidenceV1, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    status = "localized" if result.location_candidates else "abstained"
    print(
        json.dumps(
            {
                "candidate_count": len(result.location_candidates),
                "cluster_count": len(result.candidate_clusters),
                "event_id": result.event_id,
                "needs_human_review": result.needs_human_review,
                "output": str(output),
                "status": status,
                "uncertainty_codes": [item.code for item in result.uncertainties],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = _arguments()
    runtime_input = CorpusEventRuntimeInput.model_validate_json(
        args.runtime_input.read_text(encoding="utf-8")
    )
    if args.panoramax_cache is not None:
        cache_manifest = PanoramaxCacheManifest.model_validate_json(
            (args.panoramax_cache / "cache-manifest.json").read_text(encoding="utf-8")
        )
        if not cache_manifest.assets:
            return _finish(
                abstain_for_missing_reference_coverage(runtime_input.evidence),
                args.output,
            )
    if args.index is None:
        raise ValueError("a regional FAISS index is required when Panoramax coverage exists")
    model_manifest = LocalMegaLocBundleManifest.model_validate_json(
        args.model_manifest.read_text(encoding="utf-8")
    )
    model_config = json.loads((args.model_directory / "config.json").read_text(encoding="utf-8"))
    expected_dimension = model_config.get("feat_dim") if isinstance(model_config, dict) else None
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
    index = load_faiss_bundle(args.index)
    localizer = MegaLocFaissEventLocalizer(
        image_loader=LocalEvidenceImageLoader(
            root=args.repository_root,
            relative_paths_by_media_id=runtime_input.relative_paths_by_media_id,
        ),
        encoder=encoder,
        retriever=MegaLocFaissRetriever(
            index,
            config=RetrievalConfig(top_k=args.top_k),
        ),
        config=EventLocalizationConfig(
            query_batch_size=min(len(runtime_input.evidence.media), 256),
        ),
    )
    result = localizer.localize(runtime_input.evidence)
    return _finish(result, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
