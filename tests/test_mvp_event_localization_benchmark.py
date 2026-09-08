from __future__ import annotations

from fireviewer_model_lab.mvp.benchmarks import (
    BenchmarkThresholds,
    EventBenchmarkCase,
    GroundTruthArea,
    evaluate_event_localization,
)
from fireviewer_contracts.mvp.contracts import EventEvidenceV1
from fireviewer_geolocation.mvp.localization import DeterministicEvidenceFusion, FusionConfig


def _event(
    case_number: int,
    candidate_groups: tuple[tuple[float, float, float, int], ...],
) -> EventEvidenceV1:
    prefix = f"CASE-{case_number}"
    sources = []
    media = []
    candidates = []
    for source_index, suffix in enumerate(("A", "B"), start=1):
        source_id = f"SOURCE-{prefix}-{suffix}"
        media_id = f"MEDIA-{prefix}-{suffix}"
        sources.append(
            {
                "source_id": source_id,
                "origin_id": f"ORIGIN-{prefix}-{suffix}",
                "publisher": f"Synthetic source {suffix}",
                "retrieved_at": "2026-08-21T10:00:00Z",
                "source_type": "witness",
                "independence_weight": 1,
            }
        )
        media.append(
            {
                "media_id": media_id,
                "source_id": source_id,
                "media_group_id": f"GROUP-{prefix}-{suffix}",
                "origin_id": f"ORIGIN-{prefix}-{suffix}",
                "kind": "photo",
                "sha256": f"{case_number + source_index:x}" * 64,
            }
        )
        for group_index, (longitude, latitude, score, rank) in enumerate(
            candidate_groups,
            start=1,
        ):
            candidates.append(
                {
                    "candidate_id": f"CANDIDATE-{prefix}-{suffix}-{group_index}",
                    "longitude": longitude + source_index * 0.0001,
                    "latitude": latitude + source_index * 0.0001,
                    "radius_m": 100,
                    "score": score,
                    "rank": rank,
                    "evidence_kind": "visual_retrieval",
                    "provider_id": "synthetic-retrieval",
                    "provider_version": "test-only",
                    "media_id": media_id,
                    "reference_id": f"REFERENCE-{prefix}-{suffix}-{group_index}",
                }
            )
    evidence = EventEvidenceV1.model_validate(
        {
            "schema": "fireviewer.event-evidence.v1",
            "event_id": f"EVENT-{prefix}",
            "sources": sources,
            "media": media,
            "location_candidates": candidates,
        }
    )
    return DeterministicEvidenceFusion(
        FusionConfig(
            cluster_distance_m=500,
            human_review_threshold=0,
            ambiguity_margin=0,
        )
    ).fuse(evidence)


def test_five_event_synthetic_campaign_exercises_event_and_image_metrics() -> None:
    truth = GroundTruthArea(
        center=(5.37, 44.75),
        radius_m=500,
        provenance="Synthetic contract test; not a product-quality benchmark.",
    )
    cases = (
        EventBenchmarkCase(
            case_id="CASE-1",
            evidence=_event(1, ((5.37, 44.75, 0.9, 1),)),
            ground_truth=truth,
            retrieval_media_ids=("MEDIA-CASE-1-A", "MEDIA-CASE-1-B"),
        ),
        EventBenchmarkCase(
            case_id="CASE-2",
            evidence=_event(2, ((5.37, 44.75, 0.8, 1),)),
            ground_truth=truth,
            retrieval_media_ids=("MEDIA-CASE-2-A", "MEDIA-CASE-2-B"),
        ),
        EventBenchmarkCase(
            case_id="CASE-3",
            evidence=_event(
                3,
                (
                    (5.50, 44.90, 0.95, 1),
                    (5.37, 44.75, 0.7, 2),
                ),
            ),
            ground_truth=truth,
            retrieval_media_ids=("MEDIA-CASE-3-A", "MEDIA-CASE-3-B"),
        ),
        EventBenchmarkCase(
            case_id="CASE-4",
            evidence=_event(4, ((5.50, 44.90, 0.9, 1),)),
            ground_truth=truth,
            retrieval_media_ids=("MEDIA-CASE-4-A", "MEDIA-CASE-4-B"),
        ),
        EventBenchmarkCase(
            case_id="CASE-5",
            evidence=_event(5, ()),
            ground_truth=truth,
            retrieval_media_ids=("MEDIA-CASE-5-A", "MEDIA-CASE-5-B"),
        ),
    )
    report = evaluate_event_localization(
        cases,
        thresholds=BenchmarkThresholds(
            pass_event_recall_at_3=0.6,
            pass_event_recall_at_5=0.8,
            pass_usable_event_fraction=0.8,
            pass_median_candidate_radius_m=1_000,
            fail_event_recall_at_5_below=0.2,
            fail_usable_event_fraction_below=0.2,
        ),
    )

    assert report.case_count == 5
    assert report.evaluated_media_count == 10
    assert report.event_recall_at_1 == 0.4
    assert report.event_recall_at_3 == 0.6
    assert report.event_recall_at_5 == 0.6
    assert report.usable_event_fraction == 0.8
    assert report.unresolved_event_fraction == 0.4
    assert report.image_recall_at_1 == 0.4
    assert report.image_recall_at_5 == 0.6
    assert report.verdict == "PARTIAL"
    assert report.cases[2].correct_cluster_rank == 2
    assert "event_recall_at_5_below_pass_threshold" in report.gate_reasons
