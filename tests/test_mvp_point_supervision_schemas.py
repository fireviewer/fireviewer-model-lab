from __future__ import annotations

from pathlib import Path

from fireviewer_model_lab.tools.export_point_supervision_schemas import rendered_schema, OUTPUT_ROOT

from fireviewer_contracts.mvp.contracts import PointAssessmentV1, PointEvidenceBundleV1

SCHEMA_ROOT = OUTPUT_ROOT


def test_point_supervision_schema_snapshots_match_the_frozen_models() -> None:
    expected = {
        "point-evidence-bundle.schema.json": rendered_schema(PointEvidenceBundleV1),
        "point-assessment.schema.json": rendered_schema(PointAssessmentV1),
    }

    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in SCHEMA_ROOT.glob("*.schema.json")
    } == expected


def test_point_supervision_schemas_cannot_output_map_or_perimeter_geometry() -> None:
    schemas = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SCHEMA_ROOT.glob("*.schema.json"))
    )

    assert '"geometry_mutation_allowed"' in schemas
    assert '"const": false' in schemas
    assert '"perimeter"' not in schemas.casefold()
    assert '"polygon"' not in schemas.casefold()
