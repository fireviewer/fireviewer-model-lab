from __future__ import annotations

import json
from pathlib import Path

from fireviewer_contracts.mvp.contracts import GeographicHypothesisResultV1

from fireviewer_contracts.resources import schema_path
OUTPUT_PATH = schema_path("geographic-hypotheses/v1/geographic-hypotheses.schema.json")


def rendered_schema() -> str:
    return (
        json.dumps(
            GeographicHypothesisResultV1.model_json_schema(
                by_alias=True,
                mode="serialization",
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(rendered_schema(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
