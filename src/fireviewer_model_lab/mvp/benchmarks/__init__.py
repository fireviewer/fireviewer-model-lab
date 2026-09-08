"""Event-level benchmark contracts and metrics for the Part.3 coarse-localization gate."""

from fireviewer_contracts.mvp.benchmarks.corpus import (
    CorpusReadinessReport,
    Summer2026Corpus,
    Summer2026EventCase,
)
from fireviewer_model_lab.mvp.benchmarks.coverage import (
    PanoramaxCoverageReceipt,
    probe_panoramax_coverage,
)
from fireviewer_model_lab.mvp.benchmarks.event_localization import (
    BenchmarkThresholds,
    EventBenchmarkCase,
    EventLocalizationBenchmarkReport,
    GroundTruthArea,
    evaluate_event_localization,
)
from fireviewer_model_lab.mvp.benchmarks.ground_truth import (
    ObservedEventGroundTruth,
    summarize_observed_event_geojson,
)

__all__ = [
    "BenchmarkThresholds",
    "CorpusReadinessReport",
    "EventBenchmarkCase",
    "EventLocalizationBenchmarkReport",
    "GroundTruthArea",
    "ObservedEventGroundTruth",
    "PanoramaxCoverageReceipt",
    "Summer2026Corpus",
    "Summer2026EventCase",
    "evaluate_event_localization",
    "probe_panoramax_coverage",
    "summarize_observed_event_geojson",
]
