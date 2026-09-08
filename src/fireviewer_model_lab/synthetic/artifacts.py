"""Atomic artifact helpers shared by the synthetic case generators."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(volume_root: Path, path: Path, *, kind: str) -> dict[str, Any]:
    resolved_root = volume_root.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents:
        raise ValueError("artifact escapes production volume")
    if not resolved.is_file():
        raise ValueError(f"artifact is absent: {resolved}")
    return {
        "kind": kind,
        "relpath": resolved.relative_to(resolved_root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def case_record_path(batch_root: Path, case_id: str) -> Path:
    return batch_root / "records" / f"{case_id}.json"


def finalize_case_record(
    *,
    batch_root: Path,
    record: dict[str, Any],
    started_monotonic: float,
    gpu_memory: dict[str, Any] | None = None,
) -> Path:
    """Write a case record with measured, self-consistent output byte counts."""

    elapsed_seconds = time.perf_counter() - started_monotonic
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise RuntimeError("case elapsed time measurement is invalid")
    artifact_bytes = sum(
        int(item.get("bytes", 0)) for item in record.get("artifacts", [])
    )
    if artifact_bytes <= 0:
        raise RuntimeError("case artifact byte measurement is empty")
    performance: dict[str, Any] = {
        "measurement": "observed_pilot_case_v1",
        "elapsed_seconds": round(elapsed_seconds, 6),
        "artifact_bytes": artifact_bytes,
        "record_bytes": 0,
        "case_output_bytes": artifact_bytes,
        **(
            gpu_memory
            if gpu_memory is not None
            else {
                "vram_measurement": "not_applicable_non_gpu_case",
                "vram_baseline_bytes": 0,
                "vram_peak_bytes": 0,
                "vram_delta_peak_bytes": 0,
                "vram_sample_count": 0,
            }
        ),
    }
    record["performance"] = performance
    destination = case_record_path(batch_root, str(record["case_id"]))
    for _attempt in range(4):
        write_json(destination, record)
        record_bytes = destination.stat().st_size
        case_output_bytes = artifact_bytes + record_bytes
        if (
            performance["record_bytes"] == record_bytes
            and performance["case_output_bytes"] == case_output_bytes
        ):
            return destination
        performance["record_bytes"] = record_bytes
        performance["case_output_bytes"] = case_output_bytes
    write_json(destination, record)
    final_record_bytes = destination.stat().st_size
    if performance["record_bytes"] != final_record_bytes:
        raise RuntimeError("case record byte count did not stabilize")
    return destination
