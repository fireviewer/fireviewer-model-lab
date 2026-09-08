from __future__ import annotations

import argparse
import json
from pathlib import Path

from fireviewer_contracts.mvp.benchmarks.corpus import Summer2026Corpus
from fireviewer_orchestrator.mvp.orchestration import prepare_corpus_event


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one materialized summer 2026 case for the Part.3 runtime.",
    )
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    corpus = Summer2026Corpus.model_validate_json(args.corpus.read_text(encoding="utf-8"))
    matching = [case for case in corpus.cases if case.case_id == args.case_id]
    if len(matching) != 1:
        raise ValueError("requested case identifier is not unique in the corpus")
    runtime_input = prepare_corpus_event(matching[0])
    payload = json.dumps(
        runtime_input.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "media_count": len(runtime_input.evidence.media),
                "output": str(args.output),
                "status": "ready",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
