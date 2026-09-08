from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_orchestrator.mvp.orchestration import merge_event_evidence


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge additive Part.2 and Part.3 outputs into one EventEvidence graph."
    )
    parser.add_argument("--input", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_name = stream.name
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    args = _arguments()
    records = tuple(
        EventEvidenceV1.model_validate_json(path.read_text(encoding="utf-8")) for path in args.input
    )
    merged = merge_event_evidence(*records)
    payload = (
        json.dumps(
            merged.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(args.output, payload)
    print(
        json.dumps(
            {
                "candidate_count": len(merged.location_candidates),
                "cluster_count": len(merged.candidate_clusters),
                "event_id": merged.event_id,
                "media_count": len(merged.media),
                "needs_human_review": merged.needs_human_review,
                "output": str(args.output),
                "uncertainty_codes": [item.code for item in merged.uncertainties],
                "visual_observation_count": len(merged.visual_observations),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
