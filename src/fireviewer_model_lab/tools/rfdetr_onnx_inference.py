#!/usr/bin/env python3
"""Run FireViewer RF-DETR ONNX inference and emit JSON detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CLASS_NAMES = ("flame_visible", "smoke_visible")


def _detections_to_json(detections: Any) -> dict[str, Any]:
    rows = []
    confidences = detections.confidence
    class_ids = detections.class_id
    for index, box in enumerate(detections.xyxy):
        class_id = int(class_ids[index])
        rows.append(
            {
                "box_xyxy": [float(value) for value in box],
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": float(confidences[index]),
            }
        )
    return {"classes": list(CLASS_NAMES), "detections": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    from rfdetr.export._onnx.inference import _create_onnx_session, _run_inference

    session = _create_onnx_session(args.model)
    detections, _ = _run_inference(session, args.image, threshold=args.threshold)
    payload = json.dumps(_detections_to_json(detections), ensure_ascii=False, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
