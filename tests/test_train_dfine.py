from pathlib import Path

from fireviewer_model_lab.training.train_dfine import DFINE_MODEL_REVISION, DFINE_MODEL_SPEC
from fireviewer_model_lab.training.train_rtdetr import build_parser, build_preflight_report


def test_dfine_trainer_uses_pinned_transformers_object_detection_model() -> None:
    assert DFINE_MODEL_SPEC.family == "D-FINE-X-Objects365"
    assert DFINE_MODEL_SPEC.model_id == "ustc-community/dfine-xlarge-obj365"
    assert DFINE_MODEL_SPEC.revision == DFINE_MODEL_REVISION
    assert len(DFINE_MODEL_REVISION) == 40
    assert DFINE_MODEL_SPEC.output_prefix == "FW_DFINE"


def test_dfine_parser_inherits_the_common_multiscale_training_contract() -> None:
    args = build_parser(DFINE_MODEL_SPEC).parse_args(["preflight", "--manifest", "manifest.jsonl"])

    assert args.output == Path("data/training/dfine-x-objects365-firewarning")
    assert args.profile == "media_filter_v1"
    assert args.global_image_size == 768
    assert args.multiscale_sizes == (640, 768, 896, 960)


def test_dfine_preflight_records_the_actual_candidate_model() -> None:
    report = build_preflight_report([], model_spec=DFINE_MODEL_SPEC, profile="media_filter_v1")

    assert report["base_model"] == DFINE_MODEL_SPEC.model_id
    assert report["base_model_revision"] == DFINE_MODEL_SPEC.revision
