from __future__ import annotations

from datetime import datetime

from pydantic import AnyHttpUrl, Field, model_validator

from fireviewer_contracts.contracts import SafeIdentifierV2, Sha256HexV2, StrictModel
from fireviewer_contracts.mvp.benchmarks.corpus import Summer2026Corpus
from fireviewer_contracts.mvp.contracts.common import is_timezone_aware
from fireviewer_geolocation.mvp.localization import haversine_m
from fireviewer_geolocation.mvp.localization.panoramax import PanoramaxClient, PanoramaxQuery


class PanoramaxCoverageCase(StrictModel):
    case_id: SafeIdentifierV2
    query_sha256: Sha256HexV2
    reference_count: int = Field(ge=0)
    distinct_sequence_count: int = Field(ge=0)
    licensed_reference_count: int = Field(ge=0)
    downloadable_reference_count: int = Field(ge=0)
    nearest_reference_distance_m: float | None = Field(default=None, ge=0)
    oldest_capture_at: datetime | None = None
    newest_capture_at: datetime | None = None

    @model_validator(mode="after")
    def validate_capture_range(self) -> PanoramaxCoverageCase:
        captures = tuple(
            value for value in (self.oldest_capture_at, self.newest_capture_at) if value is not None
        )
        if any(not is_timezone_aware(value) for value in captures):
            raise ValueError("coverage capture times must include a timezone")
        if (
            self.oldest_capture_at is not None
            and self.newest_capture_at is not None
            and self.newest_capture_at < self.oldest_capture_at
        ):
            raise ValueError("coverage capture range must be ordered")
        return self


class PanoramaxCoverageReceipt(StrictModel):
    schema_name: str = Field(
        default="fireviewer.panoramax-coverage-receipt.v1",
        alias="schema",
        serialization_alias="schema",
    )
    api_url: AnyHttpUrl
    retrieved_at: datetime
    corpus_sha256: Sha256HexV2
    query_limit_per_case: int = Field(ge=1, le=1_000)
    downloaded_media_count: int = Field(default=0, ge=0, le=0)
    cases: tuple[PanoramaxCoverageCase, ...] = Field(min_length=5, max_length=10)

    @model_validator(mode="after")
    def validate_retrieval_time(self) -> PanoramaxCoverageReceipt:
        if not is_timezone_aware(self.retrieved_at):
            raise ValueError("coverage retrieval time must include a timezone")
        return self


def probe_panoramax_coverage(
    corpus: Summer2026Corpus,
    *,
    client: PanoramaxClient,
    retrieved_at: datetime,
    limit_per_case: int = 25,
) -> PanoramaxCoverageReceipt:
    if not 1 <= limit_per_case <= 1_000:
        raise ValueError("Panoramax coverage limit must be between 1 and 1000")
    cases: list[PanoramaxCoverageCase] = []
    for case in corpus.cases:
        if case.stage == "excluded":
            continue
        result = client.search(
            PanoramaxQuery(
                zone_id=case.case_id,
                bbox_wgs84=case.collection_aoi.bbox_wgs84,
                limit=limit_per_case,
            ),
            retrieved_at=retrieved_at,
        )
        capture_times = [image.captured_at for image in result.images]
        distances = [
            haversine_m(
                case.collection_aoi.center_wgs84,
                (image.longitude, image.latitude),
            )
            for image in result.images
        ]
        cases.append(
            PanoramaxCoverageCase(
                case_id=case.case_id,
                query_sha256=result.query_sha256,
                reference_count=len(result.images),
                distinct_sequence_count=len({image.sequence_id for image in result.images}),
                licensed_reference_count=sum(
                    image.license_url is not None for image in result.images
                ),
                downloadable_reference_count=sum(
                    image.image_url is not None for image in result.images
                ),
                nearest_reference_distance_m=min(distances, default=None),
                oldest_capture_at=min(capture_times, default=None),
                newest_capture_at=max(capture_times, default=None),
            )
        )
    return PanoramaxCoverageReceipt.model_validate(
        {
            "api_url": client.api_url,
            "retrieved_at": retrieved_at,
            "corpus_sha256": corpus.canonical_sha256(),
            "query_limit_per_case": limit_per_case,
            "cases": tuple(cases),
        }
    )
