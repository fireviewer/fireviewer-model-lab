from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from fireviewer_model_lab.tools.export_geographic_hypothesis_schema import OUTPUT_PATH, rendered_schema

from fireviewer_contracts.mvp.contracts import (
    DetectionResultV1,
    EventEvidenceV1,
    GeographicHypothesis,
    GeographicReference,
    PriorFireStateReference,
    UploadLocationEvidence,
)
from fireviewer_geolocation.mvp.localization.geographic_endpoint import (
    DurableGeographicHypothesisService,
)
from fireviewer_geolocation.mvp.localization.geographic_hypotheses import (
    GeographicHypothesisEngine,
)
from fireviewer_orchestrator.mvp.orchestration.point_bundle_pipeline import (
    GeographicPointBundlePipeline,
)
from fireviewer_contracts.mvp.supervision.backend_event_evidence import (
    DurableEventEvidence,
    DurableTerrainReference,
)

NOW = datetime(2026, 8, 23, 18, tzinfo=UTC)
MEDIA_SHA = "a" * 64


class _Terrain:
    reference_revision = "DEM-IMMUTABLE-1"
    resolution_m = 25.0

    def __init__(self, *, blocked: bool = False) -> None:
        self.blocked = blocked

    def elevation_m(self, longitude: float, _latitude: float) -> float | None:
        if self.blocked and 5.004 < longitude < 5.009:
            return 150.0
        return 100.0


def _event() -> EventEvidenceV1:
    return EventEvidenceV1.model_validate(
        {
            "schema": "fireviewer.event-evidence.v1",
            "event_id": "EVENT-GEO-1",
            "time_window": {
                "from_at": (NOW - timedelta(minutes=10)).isoformat(),
                "to_at": NOW.isoformat(),
            },
            "sources": [
                {
                    "source_id": "SOURCE-1",
                    "origin_id": "ORIGIN-1",
                    "publisher": "FireViewer test",
                    "retrieved_at": NOW.isoformat(),
                    "source_type": "witness",
                    "independence_weight": 1,
                }
            ],
            "media": [
                {
                    "media_id": "MEDIA-1",
                    "source_id": "SOURCE-1",
                    "media_group_id": "GROUP-1",
                    "origin_id": "ORIGIN-1",
                    "kind": "photo",
                    "sha256": MEDIA_SHA,
                    "captured_at": NOW.isoformat(),
                }
            ],
            "visual_observations": [
                {
                    "observation_id": "OBS-1",
                    "media_id": "MEDIA-1",
                    "observation_type": "detection",
                    "result_reference": "RESULT-1",
                    "confidence": 0.9,
                }
            ],
            "needs_human_review": True,
        }
    )


def _artifact() -> DetectionResultV1:
    return DetectionResultV1.model_validate(
        {
            "schema": "fireviewer.detection.v1",
            "media_id": "MEDIA-1",
            "provider_run": {
                "provider_id": "yolo-fire-smoke-cpu",
                "provider_version": "1.0.0",
                "model_id": "test/fire-smoke",
                "model_version": "immutable-1",
                "config": {"device": "cpu"},
                "input_hash": MEDIA_SHA,
                "runtime_ms": 10,
                "cost_usd": 0,
                "generated_at": NOW.isoformat(),
            },
            "detections": [
                {
                    "detection_id": "DET-1",
                    "detection_class": "fire",
                    "bbox": [0.45, 0.35, 0.55, 0.65],
                    "score": 0.9,
                    "prompt": "fire",
                }
            ],
            "status": "fire",
            "review_status": "candidate",
            "needs_human_review": True,
        }
    )


def _location(*, oriented: bool = True) -> UploadLocationEvidence:
    return UploadLocationEvidence(
        location_id="UPLOAD-1",
        media_id="MEDIA-1",
        longitude=5.0,
        latitude=44.0,
        accuracy_m=10,
        location_origin="user_declared",
        captured_at=NOW,
        heading_deg=90 if oriented else None,
        horizontal_fov_deg=60 if oriented else None,
        heading_uncertainty_deg=3 if oriented else None,
        altitude_m=120,
        altitude_uncertainty_m=5,
        source_record_sha256="b" * 64,
    )


def _satellite() -> GeographicReference:
    return GeographicReference(
        reference_id="SAT-1",
        reference_kind="satellite_hotspot",
        geometry_geojson={"type": "Point", "coordinates": [5.0127, 44.0]},
        observed_at=NOW - timedelta(minutes=10),
        horizontal_uncertainty_m=375,
        confidence=0.85,
        artifact_revision="VIIRS-GRANULE-1",
        lineage_family_id="VIIRS-FAMILY-1",
    )


def _history(*, reversed_progression: bool = False) -> tuple[GeographicReference, ...]:
    older = 5.012 if reversed_progression else 5.006
    newer = 5.011 if reversed_progression else 5.010
    return (
        GeographicReference(
            reference_id="HISTORY-1",
            reference_kind="prior_active_point",
            geometry_geojson={"type": "Point", "coordinates": [older, 44.0]},
            observed_at=NOW - timedelta(hours=2),
            horizontal_uncertainty_m=100,
            artifact_revision="HISTORY-REV-1",
        ),
        GeographicReference(
            reference_id="HISTORY-2",
            reference_kind="prior_active_point",
            geometry_geojson={"type": "Point", "coordinates": [newer, 44.0]},
            observed_at=NOW - timedelta(hours=1),
            horizontal_uncertainty_m=100,
            artifact_revision="HISTORY-REV-2",
        ),
    )


def _locate(
    *,
    terrain: _Terrain | None = None,
    location: UploadLocationEvidence | None = None,
    references: tuple[GeographicReference, ...] | None = None,
):
    return GeographicHypothesisEngine(terrain or _Terrain()).locate(
        _event(),
        vision_artifacts=(_artifact(),),
        upload_locations=((location or _location()),),
        geographic_references=(references or (_satellite(), *_history())),
        source_revision_sha256="c" * 64,
        generated_at=NOW,
    )


def test_visual_box_is_only_a_bearing_and_satellite_supplies_the_gps() -> None:
    result = _locate()

    assert result.status == "hypotheses"
    assert result.provider_run.cost_usd == 0
    assert result.geometry_mutation_allowed is False
    hypothesis = result.hypotheses[0]
    assert (hypothesis.longitude, hypothesis.latitude) == (5.0127, 44.0)
    assert hypothesis.geometry_geojson == {
        "type": "Point",
        "coordinates": [5.0127, 44.0],
    }
    assert hypothesis.source_point_normalized == (0.5, 0.65)
    assert hypothesis.supporting_reference_ids == (
        "HISTORY-1",
        "HISTORY-2",
        "SAT-1",
    )
    assert hypothesis.score_breakdown.visual == 0.9
    assert hypothesis.score_breakdown.camera_bearing > 0.99
    assert hypothesis.score_breakdown.terrain_visibility > 0
    assert hypothesis.score_breakdown.satellite == 0.85
    assert hypothesis.score_breakdown.history_progression is not None
    assert "terrain_line_of_sight_supported" in hypothesis.reason_codes


def test_missing_orientation_abstains_without_any_coordinate() -> None:
    result = _locate(location=_location(oriented=False))

    assert result.status == "abstained"
    assert result.hypotheses == ()
    assert result.abstentions[0].reason_codes == ("missing_camera_orientation",)
    payload = result.model_dump(mode="json", by_alias=True)
    assert payload["hypotheses"] == []


def test_missing_durable_terrain_abstains_without_any_coordinate() -> None:
    result = GeographicHypothesisEngine(None).locate(
        _event(),
        vision_artifacts=(_artifact(),),
        upload_locations=(_location(),),
        geographic_references=(_satellite(), *_history()),
        source_revision_sha256="c" * 64,
        generated_at=NOW,
    )

    assert result.status == "abstained"
    assert result.hypotheses == ()
    assert result.abstentions[0].reason_codes == ("missing_terrain_reference",)


def test_yolo_and_camera_without_satellite_never_invent_distance() -> None:
    result = _locate(references=_history())

    assert result.status == "abstained"
    assert result.hypotheses == ()
    assert result.abstentions[0].reason_codes == ("missing_satellite_reference",)


def test_terrain_occlusion_rejects_the_satellite_seed() -> None:
    result = _locate(terrain=_Terrain(blocked=True))

    assert result.status == "abstained"
    assert result.hypotheses == ()
    assert result.abstentions[0].reason_codes == ("terrain_line_of_sight_blocked",)


def test_point_opposed_to_recent_front_progression_is_rejected() -> None:
    result = _locate(references=(_satellite(), *_history(reversed_progression=True)))

    assert result.status == "abstained"
    assert result.hypotheses == ()
    assert result.abstentions[0].reason_codes == ("history_progression_contradicted",)


def test_contract_rejects_map_or_perimeter_payloads() -> None:
    payload = _locate().hypotheses[0].model_dump(mode="json")
    payload["perimeter"] = {"type": "Polygon", "coordinates": []}

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GeographicHypothesis.model_validate(payload)


def test_contract_rejects_geojson_that_differs_from_deterministic_coordinates() -> None:
    payload = _locate().hypotheses[0].model_dump(mode="json")
    payload["geometry_geojson"] = {"type": "Point", "coordinates": [0, 0]}

    with pytest.raises(ValidationError, match="GeoJSON must match"):
        GeographicHypothesis.model_validate(payload)


def test_separate_service_reads_durable_evidence_and_returns_only_hypotheses() -> None:
    class Repository:
        def read(self, event_id: str) -> DurableEventEvidence:
            assert event_id == "EVENT-GEO-1"
            return DurableEventEvidence(
                event=_event(),
                media_locations=(),
                vision_artifacts=(_artifact(),),
                upload_locations=(_location(),),
                prior_fire_states=(),
                geospatial_checks=(),
                geographic_references=(_satellite(), *_history()),
                source_revision_sha256="c" * 64,
            )

    service = DurableGeographicHypothesisService(
        Repository(),
        GeographicHypothesisEngine(_Terrain()),
        clock=lambda: NOW,
    )

    payload = service.locate_payload({"event_id": "EVENT-GEO-1"})

    assert payload["schema"] == "fireviewer.geographic-hypotheses.v1"
    assert payload["status"] == "hypotheses"
    assert payload["geometry_mutation_allowed"] is False
    assert "perimeter" not in payload


def test_geographic_hypotheses_become_compact_read_only_point_bundles() -> None:
    event_payload = _event().model_dump(mode="json", by_alias=True)
    event_payload["satellite_observations"] = [
        {
            "observation_id": "SAT-1",
            "source_id": "SOURCE-1",
            "media_id": None,
            "observation_type": "hotspot",
            "result_reference": "VIIRS-GRANULE-1",
            "acquired_at": (NOW - timedelta(minutes=10)).isoformat(),
            "confidence": 0.85,
        }
    ]
    event = EventEvidenceV1.model_validate(event_payload)
    prior_state = PriorFireStateReference(
        state_id="PRIOR-STATE-1",
        state_kind="active_points",
        observed_at=NOW - timedelta(hours=1),
        artifact_reference="backend://events/EVENT-GEO-1/history/1",
        artifact_sha256="d" * 64,
    )

    class Repository:
        def read(self, event_id: str) -> DurableEventEvidence:
            assert event_id == "EVENT-GEO-1"
            return DurableEventEvidence(
                event=event,
                media_locations=(),
                vision_artifacts=(_artifact(),),
                upload_locations=(_location(),),
                prior_fire_states=(prior_state,),
                geospatial_checks=(),
                geographic_references=(_satellite(), *_history()),
                source_revision_sha256="c" * 64,
            )

    pipeline = GeographicPointBundlePipeline(
        DurableGeographicHypothesisService(
            Repository(),
            GeographicHypothesisEngine(_Terrain()),
            clock=lambda: NOW,
        )
    )

    payload = pipeline.build_payload("EVENT-GEO-1", generated_at=NOW)

    assert payload["schema"] == "fireviewer.point-evidence-bundle-batch.v1"
    assert payload["geographic_status"] == "hypotheses"
    assert payload["geometry_mutation_allowed"] is False
    assert len(payload["bundles"]) == 1
    bundle = payload["bundles"][0]
    assert bundle["point"]["source_candidate_ids"][0].startswith("GEO-")
    assert bundle["upload_locations"][0]["heading_deg"] == 90
    assert {item["reference_id"] for item in bundle["geographic_references"]} == {
        "HISTORY-1",
        "HISTORY-2",
        "SAT-1",
    }
    assert {item["check_type"] for item in bundle["geospatial_checks"]} == {
        "camera_bearing",
        "camera_distance",
        "history_progression",
        "satellite_overlap",
        "temporal_alignment",
        "terrain_visibility",
    }
    assert bundle["missing_evidence_codes"] == []


def test_service_resolves_the_terrain_reference_from_durable_event_evidence() -> None:
    terrain_reference = DurableTerrainReference(
        terrain_id="TERRAIN-1",
        package_id="PACKAGE-1",
        sha256="d" * 64,
        size_bytes=1_024,
        media_type="application/vnd.fireviewer.terrain",
        crs="EPSG:2154",
        resolution_m=25,
        content_url=(
            "https://api.example.test/api/v1/internal/event-evidence/"
            "EVENT-GEO-1/terrain/content"
        ),
    )

    class Repository:
        def read(self, event_id: str) -> DurableEventEvidence:
            assert event_id == "EVENT-GEO-1"
            return DurableEventEvidence(
                event=_event(),
                media_locations=(),
                vision_artifacts=(_artifact(),),
                upload_locations=(_location(),),
                prior_fire_states=(),
                geospatial_checks=(),
                geographic_references=(_satellite(), *_history()),
                source_revision_sha256="c" * 64,
                terrain_reference=terrain_reference,
            )

    class Resolver:
        def __init__(self) -> None:
            self.references: list[DurableTerrainReference] = []

        def resolve(self, reference: DurableTerrainReference) -> _Terrain:
            self.references.append(reference)
            return _Terrain()

    resolver = Resolver()
    service = DurableGeographicHypothesisService(
        Repository(),
        terrain_resolver=resolver,
        clock=lambda: NOW,
    )

    payload = service.locate_payload({"event_id": "EVENT-GEO-1"})

    assert payload["status"] == "hypotheses"
    assert resolver.references == [terrain_reference]


def test_exported_geographic_contract_matches_the_frozen_model() -> None:
    schema = OUTPUT_PATH.read_text(encoding="utf-8")

    assert schema == rendered_schema()
    assert '"geometry_mutation_allowed"' in schema
    assert '"const": false' in schema
    assert '"perimeter"' not in schema.casefold()
