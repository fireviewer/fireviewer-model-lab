"""Build genuine synchronized multi-camera pairs from Camp Swift ground videos."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import correlate

SOURCE_ID = "usfs-rmrs/camp-swift-fire-experiment-2014-rds-2018-0042"
SOURCE_REVISION = "RDS-2018-0042"
SOURCE_CITATION = (
    "Butler, Bret W.; Jimenez, Daniel M.; Teske, Casey C. 2018. "
    "Camp Swift Fire Experiment 2014: Fire behavior packages and videos. "
    "https://doi.org/10.2737/RDS-2018-0042"
)
BLOCK_SPLITS = {1: "train", 2: "validation", 3: "test"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    if not command or Path(command[0]).name.lower() not in {
        "ffmpeg",
        "ffmpeg.exe",
        "ffprobe",
        "ffprobe.exe",
    }:
        raise ValueError("only ffmpeg and ffprobe commands are permitted")
    try:
        return subprocess.run(  # noqa: S603 - executable is allowlisted above
            command, capture_output=True, check=True
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{message}") from exc


def probe_duration(video: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ]
    )
    value = json.loads(result.stdout.decode("utf-8"))["format"]["duration"]
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid video duration: {video}")
    return duration


def audio_envelope(
    video: Path, *, sample_rate: int = 8_000, hop_seconds: float = 0.05
) -> np.ndarray:
    hop = round(sample_rate * hop_seconds)
    if hop <= 0:
        raise ValueError("audio hop must be positive")
    result = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ]
    )
    audio = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    frames = len(audio) // hop
    if frames < 10:
        raise ValueError(f"audio track is too short: {video}")
    windows = audio[: frames * hop].reshape(frames, hop)
    rms = np.sqrt(np.mean(windows * windows, axis=1) + 1e-10)
    return np.log(rms + 1e-5)


def contained_audio_offset(
    reference: np.ndarray, query: np.ndarray, *, hop_seconds: float = 0.05
) -> tuple[float, float]:
    """Locate a shorter query envelope inside a reference using normalized correlation."""

    if query.ndim != 1 or reference.ndim != 1:
        raise ValueError("audio envelopes must be one-dimensional")
    if len(query) > len(reference):
        raise ValueError("reference audio must be at least as long as the query")
    centered_query = query - query.mean()
    query_norm = float(np.sqrt(np.sum(centered_query * centered_query)))
    if query_norm <= 1e-8:
        raise ValueError("query audio envelope has no usable variation")
    numerator = correlate(reference, centered_query, mode="valid", method="fft")
    prefix = np.concatenate(([0.0], np.cumsum(reference, dtype=np.float64)))
    prefix2 = np.concatenate(([0.0], np.cumsum(reference * reference, dtype=np.float64)))
    size = len(query)
    sums = prefix[size:] - prefix[:-size]
    sums2 = prefix2[size:] - prefix2[:-size]
    variance = np.maximum(sums2 - sums * sums / size, 1e-12)
    scores = numerator / (np.sqrt(variance) * query_norm)
    index = int(np.argmax(scores))
    return index * hop_seconds, float(scores[index])


def align_block_videos(
    videos: list[Path], *, minimum_score: float = 0.15
) -> tuple[Path, dict[Path, float], dict[Path, float], dict[Path, float]]:
    if len(videos) < 2:
        raise ValueError("a Camp Swift block needs at least two cameras")
    durations = {video: probe_duration(video) for video in videos}
    envelopes = {video: audio_envelope(video) for video in videos}
    reference = max(videos, key=lambda video: len(envelopes[video]))
    offsets = {reference: 0.0}
    scores = {reference: 1.0}
    for video in videos:
        if video == reference:
            continue
        offset, score = contained_audio_offset(envelopes[reference], envelopes[video])
        if score < minimum_score:
            raise RuntimeError(
                f"audio synchronization confidence is too low for {video.name}: {score:.3f}"
            )
        offsets[video] = offset
        scores[video] = score
    return reference, offsets, scores, durations


def common_timeline(
    offsets: dict[Path, float],
    durations: dict[Path, float],
    *,
    stride_seconds: float,
    minimum_cameras: int = 2,
) -> list[tuple[float, list[Path]]]:
    if stride_seconds <= 0:
        raise ValueError("frame stride must be positive")
    end = max(offsets[video] + durations[video] for video in offsets)
    result: list[tuple[float, list[Path]]] = []
    ref_time = 0.0
    while ref_time <= end:
        active = sorted(
            (
                video
                for video, offset in offsets.items()
                if offset <= ref_time <= offset + durations[video] - 0.5
            ),
            key=lambda path: path.name,
        )
        if len(active) >= minimum_cameras:
            result.append((round(ref_time, 6), active))
        ref_time += stride_seconds
    return result


def extract_frame(video: Path, timestamp: float, output: Path) -> None:
    if output.is_file() and output.stat().st_size > 0:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-threads",
            "1",
            "-vf",
            "yadif=0:-1:0,setsar=1",
            "-q:v",
            "2",
            str(output),
        ]
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"frame extraction failed: {output}")


def _block_number(video: Path) -> int:
    marker = "BurnBlock"
    name = video.stem
    start = name.index(marker) + len(marker)
    return int(name[start : start + 1])


def _camera_name(video: Path) -> str:
    return video.stem.rsplit("_", 1)[-1]


def build_camp_swift_cross_view_manifest(
    *,
    campaign_root: Path,
    video_root: Path,
    output_root: Path,
    stride_seconds: float = 2.0,
    minimum_sync_score: float = 0.15,
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    video_root = video_root.resolve()
    output_root = output_root.resolve()
    videos = sorted(video_root.glob("CS_BurnBlock*_*.mpg"))
    if not videos:
        raise FileNotFoundError(f"no Camp Swift ground videos found under {video_root}")
    by_block: dict[int, list[Path]] = {block: [] for block in BLOCK_SPLITS}
    for video in videos:
        by_block[_block_number(video)].append(video)
    if any(len(block_videos) < 2 for block_videos in by_block.values()):
        raise ValueError("each Camp Swift block must contain at least two cameras")

    rows: list[dict[str, Any]] = []
    image_cache: dict[tuple[Path, int], dict[str, Any]] = {}
    synchronization: dict[str, Any] = {}
    for block, block_videos in sorted(by_block.items()):
        reference, offsets, scores, durations = align_block_videos(
            block_videos, minimum_score=minimum_sync_score
        )
        timeline = common_timeline(
            offsets, durations, stride_seconds=stride_seconds, minimum_cameras=2
        )
        synchronization[str(block)] = {
            "reference_camera": _camera_name(reference),
            "offset_seconds": {
                _camera_name(video): round(offsets[video], 6)
                for video in sorted(block_videos, key=lambda path: path.name)
            },
            "correlation_score": {
                _camera_name(video): round(scores[video], 6)
                for video in sorted(block_videos, key=lambda path: path.name)
            },
            "synchronised_instants": len(timeline),
        }
        for reference_time, active_videos in timeline:
            time_milliseconds = round(reference_time * 1_000)
            assets: dict[Path, dict[str, Any]] = {}
            for video in active_videos:
                key = (video, time_milliseconds)
                if key not in image_cache:
                    local_time = reference_time - offsets[video]
                    camera = _camera_name(video)
                    frame = (
                        output_root
                        / "images"
                        / f"block-{block}"
                        / camera
                        / f"{time_milliseconds:08d}.jpg"
                    )
                    extract_frame(video, local_time, frame)
                    image_cache[key] = {
                        "image_relpath": _relative(campaign_root, frame),
                        "sha256": _sha256(frame),
                        "camera": camera,
                        "reference_time_seconds": reference_time,
                        "video_time_seconds": round(local_time, 6),
                    }
                assets[video] = image_cache[key]
            retrieval_group = f"camp-swift:block-{block}:t-{time_milliseconds:08d}"
            for source_video, map_video in permutations(active_videos, 2):
                source_asset = assets[source_video]
                map_asset = assets[map_video]
                rows.append(
                    {
                        "sample_id": (
                            f"{retrieval_group}:{source_asset['camera']}-to-{map_asset['camera']}"
                        ),
                        "family": "cross_view_registration",
                        "split": BLOCK_SPLITS[block],
                        "split_group": f"camp-swift:block-{block}",
                        "retrieval_group": retrieval_group,
                        "source_id": SOURCE_ID,
                        "source_revision": SOURCE_REVISION,
                        "source_citation": SOURCE_CITATION,
                        "license": "US-Government-funded-data-citation-required",
                        "consent_basis": "public_research_archive_no_additional_permission_or_fee",
                        "operational_incident": False,
                        "dynamic_fire_scene": True,
                        "transient_mask_status": "required_before_full_train",
                        "point_target_valid": False,
                        "source_view": source_asset,
                        "map_view": map_asset,
                        "geometry": {
                            "pose_source": "unpublished_camera_pose",
                            "same_event_time": True,
                            "synchronization_source": "normalized_audio_envelope_correlation",
                            "camera_positions_known": False,
                        },
                    }
                )

    rows.sort(key=lambda row: str(row["sample_id"]))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "dataset_family": "camp-swift-synchronised-cross-view-v1",
        "manifest": str(manifest),
        "rows": len(rows),
        "unique_images": len(image_cache),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
        "split_groups": {
            split: sorted({row["split_group"] for row in rows if row["split"] == split})
            for split in sorted(set(row["split"] for row in rows))
        },
        "stride_seconds": stride_seconds,
        "synchronization": synchronization,
        "point_target_rows": 0,
        "retrieval_only_rows": len(rows),
        "training_ready": False,
        "training_blockers": ["transient_fire_smoke_masks_not_generated"],
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
