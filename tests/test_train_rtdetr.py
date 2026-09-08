from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fireviewer_model_lab.training.corpus_pipeline import CLASS_NAMES
from fireviewer_model_lab.training.train_rtdetr import (
    LoadedRecord,
    StreamingDetectionMetrics,
    _build_optimizer_plan,
    _clip_annotation_to_window,
    _dataset_rows,
    _metric_batch,
    _prediction_logits_and_boxes,
    _resolve_resume_checkpoint,
    _select_benchmark_records,
    _select_smoke_records,
    _trainer_step_budget,
    build_object_size_audit,
    build_parser,
    build_preflight_report,
    generate_tile_windows,
)


def _record(
    *,
    identifier: str,
    split: str,
    role: str,
    class_id: int | None,
    validation: str = "source_provided",
    width: int = 1280,
    height: int = 720,
    bbox_xywh: list[float] | None = None,
) -> LoadedRecord:
    annotations = []
    if class_id is not None:
        annotations.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "validation_status": validation,
                "bbox_xywh": bbox_xywh or [128.0, 72.0, 64.0, 36.0],
            }
        )
    return LoadedRecord(
        record={
            "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            "near_duplicate_of": None,
            "corpus_role": role,
            "split": split,
            "split_group": f"group:{identifier}",
            "annotations": annotations,
            "sample_validation_status": validation,
            "consent_basis": {"kind": "source_license", "reference": "test"},
            "image_relpath": f"images/{identifier.replace(':', '-')}.jpg",
            "width": width,
            "height": height,
        },
        corpus_root=Path("."),
    )


def _ready_records() -> list[LoadedRecord]:
    records: list[LoadedRecord] = []
    for split in ("train", "validation", "test"):
        for class_id in CLASS_NAMES:
            records.append(
                _record(
                    identifier=f"{split}:{class_id}",
                    split=split,
                    role="detector_training",
                    class_id=class_id,
                )
            )
    records.extend(
        [
            _record(
                identifier="train:negative",
                split="train",
                role="detector_training",
                class_id=None,
            ),
            _record(
                identifier="validation:negative",
                split="validation",
                role="detector_training",
                class_id=None,
            ),
        ]
    )
    for class_id in CLASS_NAMES:
        records.append(
            _record(
                identifier=f"critical:{class_id}",
                split="critical_test",
                role="detector_critical_test",
                class_id=class_id,
                validation="double_validated",
            )
        )
    records.append(
        _record(
            identifier="critical:negative",
            split="critical_test",
            role="detector_critical_test",
            class_id=None,
            validation="double_validated",
        )
    )
    return records


def test_preflight_accepts_complete_grouped_double_validated_corpus() -> None:
    report = build_preflight_report(_ready_records())

    assert report["training_ready"] is True
    assert report["deployment_ready"] is True
    assert report["training_profile"] == "operational_four_class_v1"
    assert report["errors"] == []
    assert report["critical_test_rows"] == 5


def test_preflight_rejects_missing_classes_and_unreviewed_critical_samples() -> None:
    records = _ready_records()
    records = [
        loaded
        for loaded in records
        if not (
            loaded.record["split"] == "validation"
            and loaded.record["annotations"]
            and loaded.record["annotations"][0]["class_id"] == 3
        )
    ]
    records[-1].record["sample_validation_status"] = "candidate_unreviewed"

    report = build_preflight_report(records)

    assert report["training_ready"] is False
    assert report["deployment_ready"] is False
    assert "missing_classes:validation:fire_response_vehicle_visible" in report["errors"]
    assert "critical_samples_not_double_validated:1" in report["errors"]


def test_preflight_rejects_cross_split_near_duplicates() -> None:
    records = _ready_records()
    reference = records[0].record["sha256"]
    validation_record = next(loaded for loaded in records if loaded.record["split"] == "validation")
    validation_record.record["near_duplicate_of"] = reference

    report = build_preflight_report(records)

    assert report["training_ready"] is False
    assert "cross_split_near_duplicates:1" in report["errors"]


def test_media_filter_can_train_before_critical_deployment_gate() -> None:
    records = [
        loaded
        for loaded in _ready_records()
        if loaded.record["corpus_role"] != "detector_critical_test"
        and (
            not loaded.record["annotations"]
            or loaded.record["annotations"][0]["class_id"] in {0, 1}
        )
    ]

    report = build_preflight_report(records, profile="media_filter_v1")

    assert report["required_classes"] == ["flame_visible", "smoke_visible"]
    assert report["training_ready"] is True
    assert report["deployment_ready"] is False
    assert report["training_errors"] == []
    assert report["deployment_errors"] == ["missing_detector_critical_test"]


def test_object_size_audit_reports_scaled_minimum_dimension_percentiles() -> None:
    record = _record(
        identifier="train:small-fire",
        split="train",
        role="detector_training",
        class_id=0,
        width=1280,
        height=640,
        bbox_xywh=[100.0, 100.0, 16.0, 8.0],
    )
    held_out = _record(
        identifier="critical:large-fire",
        split="critical_test",
        role="detector_critical_test",
        class_id=0,
        width=1280,
        height=640,
        bbox_xywh=[100.0, 100.0, 800.0, 400.0],
        validation="double_validated",
    )

    audit = build_object_size_audit(
        [record, held_out],
        allowed_class_names=frozenset({CLASS_NAMES[0]}),
        target_sizes=(640, 768, 960),
    )

    assert audit["included_splits"] == ["train", "validation"]
    assert audit["sizes"]["640"]["all_classes"]["objects"] == 1
    assert audit["sizes"]["640"]["all_classes"]["min_dimension_percentiles_px"]["p50"] == 4.0
    assert audit["sizes"]["640"]["all_classes"]["width_percentiles_px"]["p50"] == 8.0
    assert audit["sizes"]["640"]["all_classes"]["height_percentiles_px"]["p50"] == 4.0
    assert audit["sizes"]["640"]["all_classes"]["area_percentiles_px2"]["p50"] == 32.0
    assert audit["sizes"]["768"]["all_classes"]["min_dimension_percentiles_px"]["p50"] == 4.8
    assert audit["sizes"]["960"]["all_classes"]["min_dimension_percentiles_px"]["p50"] == 6.0
    assert audit["sizes"]["640"]["all_classes"]["under_threshold_ratios"]["lt_8px"] == 1.0


def test_preflight_rejects_incoherent_resolution_plan() -> None:
    records = _ready_records()

    try:
        build_preflight_report(
            records,
            global_image_size=768,
            multiscale_sizes=(640, 960),
        )
    except ValueError as exc:
        assert str(exc) == "global_image_size must be present in multiscale_sizes"
    else:
        raise AssertionError("Expected an invalid resolution plan to be rejected")


def test_overlapping_tile_windows_cover_the_full_image() -> None:
    windows = generate_tile_windows(2048, 1024, tile_size=1024, overlap=0.25)

    assert windows == [
        (0, 0, 1024, 1024),
        (768, 0, 1792, 1024),
        (1024, 0, 2048, 1024),
    ]


def test_annotation_is_kept_only_when_half_visible_in_tile() -> None:
    annotation = {"bbox_xywh": [900.0, 100.0, 200.0, 100.0]}

    clipped = _clip_annotation_to_window(annotation, (0, 0, 1024, 1024))
    excluded = _clip_annotation_to_window(annotation, (1024, 0, 2048, 1024))

    assert clipped is not None
    assert clipped[0]["bbox_xywh"] == [900.0, 100.0, 124.0, 100.0]
    assert clipped[1] == 0.62
    assert excluded is None


def test_dataset_rows_keep_global_view_and_add_training_tiles() -> None:
    positive = _record(
        identifier="train:positive-large",
        split="train",
        role="detector_training",
        class_id=0,
        width=2048,
        height=1024,
        bbox_xywh=[900.0, 100.0, 200.0, 100.0],
    )
    negative = _record(
        identifier="train:negative-large",
        split="train",
        role="detector_training",
        class_id=None,
        width=2048,
        height=1024,
    )

    rows, report = _dataset_rows(
        [positive, negative],
        allowed_class_ids=frozenset({0}),
        tile_size=1024,
        tile_overlap=0.25,
        max_positive_tiles_per_image=4,
        negative_tile_fraction=1.0,
    )

    assert [row["view_kind"] for row in rows["train"]].count("global") == 2
    assert [row["view_kind"] for row in rows["train"]].count("positive_tile") == 2
    assert [row["view_kind"] for row in rows["train"]].count("negative_tile") == 1
    assert report["view_counts"]["train"] == {
        "global": 2,
        "negative_tile": 1,
        "positive_tile": 2,
    }


def test_parser_defaults_to_768_global_and_640_to_960_multiscale() -> None:
    args = build_parser().parse_args(["preflight", "--manifest", "manifest.jsonl"])

    assert args.global_image_size == 768
    assert args.multiscale_sizes == (640, 768, 896, 960)
    assert args.checkpoint_steps == 500
    assert args.tile_size == 1024
    assert args.tile_overlap == 0.25
    assert args.max_positive_tiles_per_image == 1
    assert args.negative_tile_fraction == 0.05
    assert args.epochs == 3
    assert args.batch_size == 5
    assert args.eval_batch_size == 8
    assert args.gradient_accumulation_steps == 4
    assert args.max_optimizer_steps == 100_000
    assert args.max_host_ram_gb == 10.0
    assert args.benchmark_records == 128
    assert args.benchmark_steps == 10
    assert not hasattr(args, "early_stopping_patience")
    assert not hasattr(args, "early_stopping_threshold")
    assert args.gradient_checkpointing is False
    assert args.workers == 4


def test_resume_checkpoint_auto_selects_highest_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-1500").mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert _resolve_resume_checkpoint(tmp_path, "auto") == str(
        (tmp_path / "checkpoint-1500").resolve()
    )
    assert _resolve_resume_checkpoint(tmp_path / "empty", "auto") is None


def test_resume_checkpoint_explicit_path_must_exist(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-500"
    checkpoint.mkdir()

    assert _resolve_resume_checkpoint(tmp_path, str(checkpoint)) == str(checkpoint.resolve())
    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        _resolve_resume_checkpoint(tmp_path, str(tmp_path / "missing"))


def test_optimizer_plan_accepts_benchmark_and_rejects_abandoned_long_run() -> None:
    accepted = _build_optimizer_plan(
        train_views=156_095,
        batch_size=5,
        gradient_accumulation_steps=4,
        epochs=3,
        max_optimizer_steps=100_000,
    )

    assert accepted == {
        "train_views": 156_095,
        "micro_batches_per_epoch": 31_219,
        "optimizer_steps_per_epoch": 7_805,
        "planned_max_optimizer_steps": 23_415,
        "effective_batch_size": 20,
    }

    try:
        _build_optimizer_plan(
            train_views=210_960,
            batch_size=1,
            gradient_accumulation_steps=16,
            epochs=30,
            max_optimizer_steps=100_000,
        )
    except ValueError as exc:
        assert "395550 > 100000" in str(exc)
    else:
        raise AssertionError("The abandoned 215-hour plan must be rejected")


def test_trainer_step_budget_uses_the_guarded_preflight_plan() -> None:
    assert (
        _trainer_step_budget(
            smoke_mode=False,
            benchmark_mode=False,
            benchmark_steps=10,
            planned_max_optimizer_steps=24_042,
        )
        == 24_042
    )
    assert (
        _trainer_step_budget(
            smoke_mode=False,
            benchmark_mode=True,
            benchmark_steps=10,
            planned_max_optimizer_steps=24_042,
        )
        == 10
    )
    assert (
        _trainer_step_budget(
            smoke_mode=True,
            benchmark_mode=False,
            benchmark_steps=10,
            planned_max_optimizer_steps=24_042,
        )
        == 1
    )


def test_smoke_selection_is_small_balanced_and_split_aware() -> None:
    selected = _select_smoke_records(
        _ready_records(),
        allowed_class_ids=frozenset({0, 1}),
    )

    assert len(selected) == 6
    for split in ("train", "validation"):
        split_records = [loaded.record for loaded in selected if loaded.record["split"] == split]
        assert len(split_records) == 3
        assert sum(not record["annotations"] for record in split_records) == 1
        assert {
            annotation["class_id"]
            for record in split_records
            for annotation in record["annotations"]
        } == {0, 1}


def test_benchmark_selection_is_bounded_stratified_and_deterministic() -> None:
    records = _ready_records()
    records.extend(
        _record(
            identifier=f"train:extra:{index}",
            split="train",
            role="detector_training",
            class_id=index % 2,
            width=640 + index,
            height=480 + index,
        )
        for index in range(20)
    )

    first = _select_benchmark_records(
        records,
        allowed_class_ids=frozenset({0, 1}),
        limit=8,
    )
    second = _select_benchmark_records(
        records,
        allowed_class_ids=frozenset({0, 1}),
        limit=8,
    )

    assert len(first) == 8
    assert [record.record["sha256"] for record in first] == [
        record.record["sha256"] for record in second
    ]
    assert {
        annotation["class_id"]
        for loaded in first
        for annotation in loaded.record["annotations"]
        if annotation["class_id"] in {0, 1}
    } == {0, 1}
    assert any(not loaded.record["annotations"] for loaded in first)


def test_prediction_extraction_accepts_extended_rtdetr_output_tuple() -> None:
    outputs = tuple(f"field-{index}" for index in range(14))

    logits, boxes = _prediction_logits_and_boxes(outputs)

    assert logits == "field-1"
    assert boxes == "field-2"


class _FakeImageProcessor:
    def post_process_object_detection(
        self,
        output: SimpleNamespace,
        *,
        threshold: float,
        target_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        del output, threshold
        predictions = []
        for _ in target_sizes:
            scores = torch.linspace(0.0, 1.0, 150)
            predictions.append(
                {
                    "scores": scores,
                    "boxes": torch.arange(600, dtype=torch.float32).reshape(150, 4),
                    "labels": torch.zeros(150, dtype=torch.int64),
                }
            )
        return predictions


def test_metric_batch_moves_state_to_cpu_and_keeps_coco_top_100() -> None:
    predictions = (
        torch.tensor(0.0),
        torch.zeros((1, 150, 2)),
        torch.zeros((1, 150, 4)),
    )
    targets = [
        {
            "orig_size": torch.tensor([480, 640]),
            "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            "class_labels": torch.tensor([0]),
        }
    ]

    metric_predictions, metric_targets = _metric_batch(
        predictions,
        targets,
        image_processor=_FakeImageProcessor(),
    )

    assert len(metric_predictions[0]["scores"]) == 100
    assert metric_predictions[0]["scores"].device.type == "cpu"
    assert metric_predictions[0]["scores"].min().item() > 0.32
    assert metric_targets[0]["boxes"].device.type == "cpu"
    assert metric_targets[0]["boxes"].tolist() == [[240.0, 180.0, 400.0, 300.0]]


def test_streaming_metrics_only_computes_and_resets_on_final_batch() -> None:
    class FakeMetric:
        def __init__(self) -> None:
            self.updates = 0
            self.resets = 0

        def update(self, predictions: object, targets: object) -> None:
            del predictions, targets
            self.updates += 1

        def compute(self) -> dict[str, torch.Tensor]:
            return {
                "map": torch.tensor(0.5),
                "map_50": torch.tensor(0.75),
                "classes": torch.tensor([0]),
                "map_per_class": torch.tensor([0.5]),
                "mar_100_per_class": torch.tensor([0.6]),
            }

        def reset(self) -> None:
            self.resets += 1

    accumulator = object.__new__(StreamingDetectionMetrics)
    accumulator.image_processor = _FakeImageProcessor()
    accumulator.class_names = {0: "fire"}
    accumulator.metric = FakeMetric()
    evaluation = SimpleNamespace(
        predictions=(
            torch.tensor(0.0),
            torch.zeros((1, 150, 2)),
            torch.zeros((1, 150, 4)),
        ),
        label_ids=[
            {
                "orig_size": torch.tensor([480, 640]),
                "boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
                "class_labels": torch.tensor([0]),
            }
        ],
    )

    assert accumulator(evaluation, compute_result=False) == {}
    result = accumulator(evaluation, compute_result=True)

    assert accumulator.metric.updates == 2
    assert accumulator.metric.resets == 1
    assert result == {
        "map": 0.5,
        "map_50": 0.75,
        "map_fire": 0.5,
        "mar_100_fire": 0.6,
    }
