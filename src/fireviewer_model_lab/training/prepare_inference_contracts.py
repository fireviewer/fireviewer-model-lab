"""Write pinned inference integration contracts for OCR and Ministral.

This does not download weights or claim runtime activation. It creates the
operator-facing contract that the worker integration must satisfy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_IMMUTABLE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})$")


def _validate_revision(name: str, revision: str) -> None:
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError(f"{name} revision must be a 40-character commit SHA or sha256 digest")


def build_contract(*, ppocr_revision: str, ministral_revision: str) -> dict[str, object]:
    _validate_revision("PP-OCRv6", ppocr_revision)
    _validate_revision("Ministral 3", ministral_revision)
    return {
        "schema_version": 1,
        "models": [
            {
                "name": "PP-OCRv6 Small",
                "role": "conditional_cpu_ocr",
                "revision": ppocr_revision,
                "fallback": "ambiguous_crop_review",
                "forbidden_outputs": ["latitude", "longitude", "perimeter"],
            },
            {
                "name": "Ministral 3 8B Instruct FP8",
                "role": "structured_multimodal_analysis",
                "revision": ministral_revision,
                "input_contract": [
                    "evidence_ids",
                    "detector_outputs",
                    "ocr_outputs",
                    "pointing_outputs",
                ],
                "output_contract": ["schema_valid_json", "evidence_spans", "abstention_reason"],
                "forbidden_outputs": ["authoritative_coordinates", "unverified_perimeter"],
            },
        ],
        "activation_gates": [
            "pinned_revision",
            "offline_cache_digest",
            "contract_tests",
            "human_gate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OCR/Ministral runtime contracts")
    parser.add_argument("--ppocr-revision", required=True)
    parser.add_argument("--ministral-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_contract(
        ppocr_revision=args.ppocr_revision,
        ministral_revision=args.ministral_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
