from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from fireviewer_contracts.mvp.benchmarks.corpus import Summer2026Corpus
from fireviewer_geolocation.mvp.localization.panoramax import PanoramaxClient, PanoramaxQuery
from fireviewer_geolocation.mvp.localization.panoramax_cache import materialize_panoramax_cache


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize one regional Panoramax cache for the Part.3 runtime."
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--panoramax-api", default="https://panoramax.ign.fr/api")
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    corpus = Summer2026Corpus.model_validate_json(args.corpus.read_text(encoding="utf-8"))
    matching = [case for case in corpus.cases if case.case_id == args.case_id]
    if len(matching) != 1:
        raise ValueError("requested case identifier is not unique in the corpus")
    case = matching[0]
    result = PanoramaxClient(api_url=args.panoramax_api).search(
        PanoramaxQuery(
            zone_id=case.case_id,
            bbox_wgs84=case.collection_aoi.bbox_wgs84,
            limit=args.limit,
        ),
        retrieved_at=datetime.now(UTC),
    )
    manifest = materialize_panoramax_cache(result, args.output)
    print(
        json.dumps(
            {
                "case_id": case.case_id,
                "image_count": len(manifest.assets),
                "output": str(args.output),
                "query_sha256": result.query_sha256,
                "status": "ready" if manifest.assets else "empty",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
