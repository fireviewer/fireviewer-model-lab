from __future__ import annotations

from pathlib import Path

from fireviewer_model_lab.training.train_rtdetr import DetectorModelSpec, main_for_model

DFINE_MODEL = "ustc-community/dfine-xlarge-obj365"
DFINE_MODEL_REVISION = "a99923ec2727499f769a82edba864299fceaca48"

DFINE_MODEL_SPEC = DetectorModelSpec(
    family="D-FINE-X-Objects365",
    model_id=DFINE_MODEL,
    revision=DFINE_MODEL_REVISION,
    output_prefix="FW_DFINE",
    default_output=Path("data/training/dfine-x-objects365-firewarning"),
)


def main() -> None:
    main_for_model(DFINE_MODEL_SPEC)


if __name__ == "__main__":
    main()
