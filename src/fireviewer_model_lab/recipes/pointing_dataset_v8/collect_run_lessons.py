"""Consolidate seven actually retained training experiments, keeping protocols distinct."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.pointing_dataset_v7.split_registry import digest, read_rows

MAP_KEYS = ("map", "map_50", "map_75", "map_small", "map_fire", "map_smoke")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    local = root / "artifacts/local"
    runs = []

    def add(name, source, protocol, metrics, notes):
        runs.append({"run": name, "source": str(source), "source_sha256": digest(source),
                     "protocol": protocol, "metrics": metrics, "limitations": notes})

    for version, source in (("V3", root / "artifacts/runpod/pointing-rtdetr-v3-r5/pointing-rtdetr-v3-rtx4000ada-4h30-20260823-r5/artifact/test_metrics.json"),
                            ("V4", local / "pointing-rtdetr-v4-local-run-2810-20260824/test_metrics.json")):
        data = json.loads(source.read_text())["combined"]
        keys = MAP_KEYS + ("operating_precision", "operating_recall", "operating_f1", "operating_best_threshold")
        add(f"RT-DETR {version}", source, f"legacy_internal_test_{version}",
            {key: data.get("test_combined_" + key) for key in keys},
            ["Different corpora and thresholds: no causal cross-version ranking.",
             "Legacy source metrics/calibration require their original protocol caveats."])
    source = local / "pointing-rtdetr-v5-small800-local-run-2909-20260824/checkpoint-1050/trainer_state.json"
    state = json.loads(source.read_text())
    evaluated = [r for r in state["log_history"] if "eval_map" in r]
    selected = next((r for r in evaluated if r["step"] == state["best_global_step"]), max(evaluated, key=lambda r: r["eval_map"]))
    add("RT-DETR V5 small800", source, "validation_selected_checkpoint", {key: selected.get("eval_" + key) for key in MAP_KEYS} | {"step": selected["step"]},
        ["Validation, not a held-out benchmark."])
    source = local / "pointing-rfdetr-large-v5-704-run-2909-20260824/metrics.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        values = [r for r in csv.DictReader(handle) if r.get("val/ema_mAP_50_95")]
    best = max(values, key=lambda r: float(r["val/ema_mAP_50_95"]))
    add("RF-DETR Large V5", source, "best_EMA_validation", {"map": float(best["val/ema_mAP_50_95"]), "map_50": float(best["val/ema_mAP_50"]), "epoch": float(best["epoch"])},
        ["No interchangeable held-out test score; regular and EMA metrics must not be mixed."])
    for version, folder in (("V5", "pointing-deim-dfine-large-v5-704-run-2909-20260824"), ("V6", "pointing-deim-dfine-large-v6-704-run-3163-20260825")):
        source = local / folder / "log.txt"
        values = read_rows(source)
        best = max(values, key=lambda r: r["test_coco_eval_bbox"][0])
        stats = best["test_coco_eval_bbox"]
        add(f"DEIM D-FINE Large {version}", source, "best_validation_despite_legacy_test_key", {"map": stats[0], "map_50": stats[1], "map_75": stats[2], "map_small": stats[3], "epoch": best["epoch"]},
            ["The training log key test_coco_eval_bbox evaluates validation, not the final test."])
    source = local / "pointing-v7-audited-benchmark-20260827/benchmark_report.json"
    benchmark = json.loads(source.read_text())
    model = benchmark["models"][0]
    add("DEIM D-FINE Large V7", source, "audited_test_validation_calibration", {key: model["test"]["map"].get(key) for key in MAP_KEYS} | {"threshold": model["selected_threshold"],
        "precision": model["test"]["operating"]["precision"], "recall": model["test"]["operating"]["recall"]},
        ["13 active test images and 11 validation images still have conflicting review evidence.", "37 test negatives: insufficient for a precise operational false-alarm estimate."])
    paired_path = local / "pointing-deim-dfine-v6-heldout-benchmark-293-20260825/benchmark_report.json"
    paired = json.loads(paired_path.read_text())
    sliced = {"by_scene_bin": {}, "by_source": {}}
    for field in sliced:
        for name, result in model["test"].get(field, {}).items():
            sliced[field][name] = {"images": result["images"], "map": result["map"]["map"], "map_small": result["map"].get("map_small"),
                                  **{key: result["operating_iou_50" if field == "by_scene_bin" else "operating"][key] for key in ("precision", "recall", "tp", "fp", "fn")}}
    report = {"schema": "fireviewer.pointing-seven-runs-lessons.v1", "runs": runs,
              "scope": "Seven retained experiments across corpus versions V3-V7, not seven independently comparable V1-V7 tests.",
              "v1_v2_final_scores": None, "v1_v2_status": "No final score artifact established in the retained local run roots; do not invent.",
              "v5_v6_paired_test": {"source": str(paired_path), "source_sha256": digest(paired_path), "comparison": paired["comparison"], "corpus": paired["corpus"]},
              "v7_slices": sliced,
              "actionable_gaps": [
                  "Small/faint smoke is the main extension priority; measure fire and smoke separately.",
                  "Do not sacrifice contextual small flames while adding smoke-only sequences.",
                  "Prioritize independently reviewed difficult negatives and report confuser families.",
                  "Correct incomplete/loose boxes and fire/smoke ambiguity; captions are not detector supervision.",
                  "Cap sequence recurrence and source concentration; an augmentation is not a new scene.",
                  "Maintain frozen holdouts and an additional independent external panel.",
                  "Low light/compression/downscale robustness require separate controlled evaluation, not inflated corpus counts."],
              "training_started": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"runs": [{"run": r["run"], "protocol": r["protocol"], "metrics": r["metrics"]} for r in runs], "v7_slices": sliced}, indent=2))


if __name__ == "__main__":
    main()
