from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from fireviewer_contracts.mvp.contracts import PointAssessmentV1, PointEvidenceBundleV1

from fireviewer_contracts.resources import schema_path
OUTPUT_ROOT = schema_path("point-supervisor/v1/point-assessment.schema.json").parent
SCHEMAS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("point-evidence-bundle.schema.json", PointEvidenceBundleV1),
    ("point-assessment.schema.json", PointAssessmentV1),
)


def rendered_schema(model: type[BaseModel]) -> str:
    return (
        json.dumps(
            model.model_json_schema(by_alias=True, mode="serialization"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, model in SCHEMAS:
        (OUTPUT_ROOT / filename).write_text(rendered_schema(model), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
