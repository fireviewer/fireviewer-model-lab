from __future__ import annotations

from statistics import median
from typing import Literal

from pydantic import Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, StrictModel
from fireviewer_contracts.mvp.contracts import EventEvidenceV1, LocationCandidate
from fireviewer_contracts.mvp.contracts.common import validate_lon_lat
from fireviewer_geolocation.mvp.localization import haversine_m


class GroundTruthArea(StrictModel):
    center: tuple[float, float]
    radius_m: float = Field(gt=0, le=1_000_000, allow_inf_nan=False)
    provenance: str = Field(min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_center(self) -> GroundTruthArea:
        validate_lon_lat(self.center, label="ground-truth center")
        return self


class EventBenchmarkCase(StrictModel):
    case_id: SafeIdentifierV2
    evidence: EventEvidenceV1
    ground_truth: GroundTruthArea
    retrieval_media_ids: tuple[SafeIdentifierV2, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def validate_media(self) -> EventBenchmarkCase:
        if len(self.retrieval_media_ids) != len(set(self.retrieval_media_ids)):
            raise ValueError("benchmark retrieval media identifiers must be unique")
        known_media = {item.media_id for item in self.evidence.media}
        if not set(self.retrieval_media_ids).issubset(known_media):
            raise ValueError("benchmark references unknown retrieval media")
        return self


class BenchmarkThresholds(StrictModel):
    pass_event_recall_at_3: float = Field(ge=0, le=1)
    pass_event_recall_at_5: float = Field(ge=0, le=1)
    pass_usable_event_fraction: float = Field(ge=0, le=1)
    pass_median_candidate_radius_m: float = Field(gt=0, le=1_000_000)
    fail_event_recall_at_5_below: float = Field(ge=0, le=1)
    fail_usable_event_fraction_below: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> BenchmarkThresholds:
        if self.fail_event_recall_at_5_below > self.pass_event_recall_at_5:
            raise ValueError("event recall fail floor cannot exceed its pass threshold")
        if self.fail_usable_event_fraction_below > self.pass_usable_event_fraction:
            raise ValueError("usable-event fail floor cannot exceed its pass threshold")
        return self


class EventCaseMetrics(StrictModel):
    case_id: SafeIdentifierV2
    correct_cluster_rank: int | None = Field(default=None, ge=1)
    leading_candidate_radius_m: float | None = Field(default=None, gt=0)
    leading_distance_to_true_area_m: float | None = Field(default=None, ge=0)
    usable_candidate_cluster: bool
    unresolved: bool
    needs_human_review: bool


class EventLocalizationBenchmarkReport(StrictModel):
    case_count: int = Field(ge=1)
    evaluated_media_count: int = Field(ge=0)
    event_recall_at_1: float = Field(ge=0, le=1)
    event_recall_at_3: float = Field(ge=0, le=1)
    event_recall_at_5: float = Field(ge=0, le=1)
    median_candidate_radius_m: float | None = Field(default=None, ge=0)
    median_distance_to_true_area_m: float | None = Field(default=None, ge=0)
    usable_event_fraction: float = Field(ge=0, le=1)
    unresolved_event_fraction: float = Field(ge=0, le=1)
    human_review_fraction: float = Field(ge=0, le=1)
    image_recall_at_1: float | None = Field(default=None, ge=0, le=1)
    image_recall_at_5: float | None = Field(default=None, ge=0, le=1)
    image_recall_at_10: float | None = Field(default=None, ge=0, le=1)
    image_recall_at_20: float | None = Field(default=None, ge=0, le=1)
    verdict: Literal["PASS", "PARTIAL", "FAIL"]
    gate_reasons: tuple[str, ...] = Field(min_length=1, max_length=32)
    cases: tuple[EventCaseMetrics, ...]


def _overlaps(
    center: tuple[float, float],
    radius_m: float,
    truth: GroundTruthArea,
) -> bool:
    return haversine_m(center, truth.center) <= radius_m + truth.radius_m


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_event_localization(
    cases: tuple[EventBenchmarkCase, ...],
    *,
    thresholds: BenchmarkThresholds,
) -> EventLocalizationBenchmarkReport:
    if not cases:
        raise ValueError("event localization benchmark requires at least one case")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("event localization benchmark case identifiers must be unique")

    case_metrics: list[EventCaseMetrics] = []
    correct_ranks: list[int | None] = []
    candidate_radii: list[float] = []
    distances_to_truth: list[float] = []
    image_hits = {1: 0, 5: 0, 10: 0, 20: 0}
    evaluated_media_count = 0
    for case in cases:
        clusters = case.evidence.candidate_clusters
        correct_rank = next(
            (
                rank
                for rank, cluster in enumerate(clusters, start=1)
                if _overlaps(cluster.center, cluster.radius_m, case.ground_truth)
            ),
            None,
        )
        correct_ranks.append(correct_rank)
        leading_radius = clusters[0].radius_m if clusters else None
        leading_distance = (
            max(
                0.0,
                haversine_m(clusters[0].center, case.ground_truth.center)
                - case.ground_truth.radius_m,
            )
            if clusters
            else None
        )
        if leading_radius is not None:
            candidate_radii.append(leading_radius)
        if leading_distance is not None:
            distances_to_truth.append(leading_distance)
        case_metrics.append(
            EventCaseMetrics(
                case_id=case.case_id,
                correct_cluster_rank=correct_rank,
                leading_candidate_radius_m=leading_radius,
                leading_distance_to_true_area_m=leading_distance,
                usable_candidate_cluster=bool(clusters),
                unresolved=correct_rank is None or correct_rank > 5,
                needs_human_review=case.evidence.needs_human_review,
            )
        )

        candidates_by_media: dict[str, list[LocationCandidate]] = {}
        for media_id in case.retrieval_media_ids:
            candidates_by_media[media_id] = []
        for candidate in case.evidence.location_candidates:
            if candidate.media_id in candidates_by_media:
                candidates_by_media[candidate.media_id].append(candidate)
        for media_id in case.retrieval_media_ids:
            evaluated_media_count += 1
            ranked = sorted(
                candidates_by_media[media_id],
                key=lambda item: (item.rank, -item.score, item.candidate_id),
            )
            for cutoff in image_hits:
                if any(
                    _overlaps(
                        (candidate.longitude, candidate.latitude),
                        candidate.radius_m,
                        case.ground_truth,
                    )
                    for candidate in ranked[:cutoff]
                ):
                    image_hits[cutoff] += 1

    case_count = len(cases)
    event_recall_at_1 = _fraction(sum(rank == 1 for rank in correct_ranks), case_count)
    event_recall_at_3 = _fraction(
        sum(rank is not None and rank <= 3 for rank in correct_ranks),
        case_count,
    )
    event_recall_at_5 = _fraction(
        sum(rank is not None and rank <= 5 for rank in correct_ranks),
        case_count,
    )
    usable_event_fraction = _fraction(
        sum(item.usable_candidate_cluster for item in case_metrics),
        case_count,
    )
    unresolved_event_fraction = _fraction(sum(item.unresolved for item in case_metrics), case_count)
    human_review_fraction = _fraction(
        sum(item.needs_human_review for item in case_metrics),
        case_count,
    )
    median_candidate_radius_m = median(candidate_radii) if candidate_radii else None
    median_distance_to_true_area_m = median(distances_to_truth) if distances_to_truth else None

    pass_failures: list[str] = []
    if event_recall_at_3 < thresholds.pass_event_recall_at_3:
        pass_failures.append("event_recall_at_3_below_pass_threshold")
    if event_recall_at_5 < thresholds.pass_event_recall_at_5:
        pass_failures.append("event_recall_at_5_below_pass_threshold")
    if usable_event_fraction < thresholds.pass_usable_event_fraction:
        pass_failures.append("usable_event_fraction_below_pass_threshold")
    if (
        median_candidate_radius_m is None
        or median_candidate_radius_m > thresholds.pass_median_candidate_radius_m
    ):
        pass_failures.append("median_candidate_radius_above_pass_threshold")

    gate_reasons: tuple[str, ...]
    if not pass_failures:
        verdict: Literal["PASS", "PARTIAL", "FAIL"] = "PASS"
        gate_reasons = ("all_configured_pass_thresholds_met",)
    elif (
        event_recall_at_5 < thresholds.fail_event_recall_at_5_below
        or usable_event_fraction < thresholds.fail_usable_event_fraction_below
    ):
        verdict = "FAIL"
        gate_reasons = tuple(pass_failures)
    else:
        verdict = "PARTIAL"
        gate_reasons = tuple(pass_failures)

    image_recalls = {
        cutoff: (_fraction(hits, evaluated_media_count) if evaluated_media_count else None)
        for cutoff, hits in image_hits.items()
    }
    return EventLocalizationBenchmarkReport(
        case_count=case_count,
        evaluated_media_count=evaluated_media_count,
        event_recall_at_1=event_recall_at_1,
        event_recall_at_3=event_recall_at_3,
        event_recall_at_5=event_recall_at_5,
        median_candidate_radius_m=median_candidate_radius_m,
        median_distance_to_true_area_m=median_distance_to_true_area_m,
        usable_event_fraction=usable_event_fraction,
        unresolved_event_fraction=unresolved_event_fraction,
        human_review_fraction=human_review_fraction,
        image_recall_at_1=image_recalls[1],
        image_recall_at_5=image_recalls[5],
        image_recall_at_10=image_recalls[10],
        image_recall_at_20=image_recalls[20],
        verdict=verdict,
        gate_reasons=gate_reasons,
        cases=tuple(case_metrics),
    )
