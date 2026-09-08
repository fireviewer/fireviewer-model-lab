from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import data_root, ensure_layout, load_config, phash_file, sha256_file, write_json, write_jsonl


USER_AGENT = "FireViewerScientificBenchmark/1.0 (+local research acquisition)"


def request_bytes(url: str, retries: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as exc:  # network errors are reported in the receipt
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to fetch {url}: {last_error}")


def request_json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url).decode("utf-8"))


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_flickr_taken(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def flickr_api(api_key: str, method: str, **params: Any) -> dict[str, Any]:
    query = {
        "method": method,
        "api_key": api_key,
        "format": "json",
        "nojsoncallback": "1",
        **{key: str(value) for key, value in params.items()},
    }
    url = "https://www.flickr.com/services/rest/?" + urllib.parse.urlencode(query)
    payload = request_json(url)
    if payload.get("stat") != "ok":
        raise RuntimeError(f"Flickr API error for {method}: {payload}")
    return payload


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(request_bytes(url))
    temporary.replace(destination)


def best_size(sizes: list[dict[str, Any]], media: str) -> dict[str, Any]:
    matches = [item for item in sizes if item.get("media") == media]
    if not matches:
        raise RuntimeError(f"No {media} source in Flickr size response")
    return max(matches, key=lambda item: int(item.get("width") or 0) * int(item.get("height") or 0))


def source_group(title: str, description: str, media_id: str) -> str:
    text = f"{title} {html.unescape(description)}".lower()
    groups = (
        ("sand-creek-fire", "sand creek"),
        ("lakeview-seat-base", "lakeview seat"),
        ("little-giant-fire", "little giant"),
        ("deer-creek-fire", "deer creek"),
        ("moose-mountain-fire", "moose mountain"),
        ("rio-escondido-fire", "rio escondido"),
        ("crooked-fire", "crooked fire"),
        ("paradise-fire", "paradise fire"),
        ("hagen-fire", "hagen fire"),
        ("cherry-peak-fire", "cherry peak"),
        ("skillet-adams-fire", "skillet"),
        ("oregon-multi-fire-briefing", "briefing on oregon fires"),
    )
    for group, marker in groups:
        if marker in text:
            return f"nifc-2026:{group}"
    return f"nifc-2026:media-{media_id}"


def extract_sequence(video_path: Path, output_dir: Path, fps: int, quality: int) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "frame_%06d.jpg"
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        return existing
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(quality),
        str(pattern),
    ]
    subprocess.run(command, check=True)
    return sorted(output_dir.glob("frame_*.jpg"))


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return paths
    indexes = sorted({round(index * (len(paths) - 1) / (count - 1)) for index in range(count)})
    return [paths[index] for index in indexes]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = data_root(config)
    ensure_layout(root)
    source = config["source"]

    album_html = request_bytes(source["album_url"]).decode("utf-8", errors="replace")
    key_match = re.search(r'site_key\s*=\s*"([0-9a-f]+)"', album_html)
    if not key_match:
        raise RuntimeError("Flickr public site key was not found in album page")
    api_key = key_match.group(1)

    album = flickr_api(
        api_key,
        "flickr.photosets.getPhotos",
        photoset_id=source["album_id"],
        user_id=source["owner_nsid"],
        extras="date_upload,date_taken,license,media,url_o,url_k,url_h,description,tags",
        per_page=500,
    )["photoset"]
    captured_after = parse_utc(source["captured_after_utc"])
    published_after = parse_utc(source["published_after_utc"])

    selected: list[dict[str, Any]] = []
    rejected_counts: dict[str, int] = {}
    for item in album["photo"]:
        reasons: list[str] = []
        captured = parse_flickr_taken(item["datetaken"])
        published = datetime.fromtimestamp(int(item["dateupload"]), tz=timezone.utc)
        if item.get("license") != source["license_id"]:
            reasons.append("license")
        if captured <= captured_after:
            reasons.append("capture_cutoff")
        if published <= published_after:
            reasons.append("publication_cutoff")
        if reasons:
            for reason in reasons:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
            continue
        selected.append(item)

    still_candidates: list[dict[str, Any]] = []
    sequence_candidates: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for position, item in enumerate(sorted(selected, key=lambda row: row["id"]), 1):
        media_id = item["id"]
        info = flickr_api(api_key, "flickr.photos.getInfo", photo_id=media_id)["photo"]
        sizes = flickr_api(api_key, "flickr.photos.getSizes", photo_id=media_id)["sizes"]["size"]
        description = html.unescape(info.get("description", {}).get("_content", ""))
        title = info.get("title", {}).get("_content", item.get("title", ""))
        group_id = source_group(title, description, media_id)
        page_url = info["urls"]["url"][0]["_content"]
        receipt = {
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "album_id": source["album_id"],
            "info": info,
            "license_evidence": {
                "id": source["license_id"],
                "name": source["license_name"],
                "url": source["license_url"],
                "policy_evidence_url": source["policy_evidence_url"],
            },
            "sizes": sizes,
        }
        write_json(root / "receipts" / "flickr" / f"{media_id}.json", receipt)

        poster_source = best_size(sizes, "photo")
        poster_suffix = Path(urllib.parse.urlparse(poster_source["source"]).path).suffix or ".jpg"
        if info["media"] == "photo":
            native_path = root / "raw" / "photos" / f"{media_id}{poster_suffix}"
            download(poster_source["source"], native_path)
            asset_sha = sha256_file(native_path)
            asset_phash = phash_file(native_path)
            row = {
                "sample_id": f"nifc-{media_id}",
                "panel": "still_detection",
                "media_kind": "photo",
                "image_path": str(native_path),
                "source_media_id": media_id,
                "source_group_id": group_id,
                "source_page_url": page_url,
                "source_title": title,
                "source_description_hint": description,
                "captured_at_utc": parse_flickr_taken(item["datetaken"]).isoformat(),
                "published_at_utc": datetime.fromtimestamp(int(item["dateupload"]), tz=timezone.utc).isoformat(),
                "license": source["license_name"],
                "license_url": source["license_url"],
                "sha256": asset_sha,
                "phash": asset_phash,
                "width": int(poster_source["width"]),
                "height": int(poster_source["height"]),
                "annotation_status": "pending_human_review",
                "objects": [],
            }
            still_candidates.append(row)
            source_records.append(row)
        else:
            poster_path = root / "raw" / "posters" / f"{media_id}{poster_suffix}"
            download(poster_source["source"], poster_path)
            video_source = best_size(sizes, "video")
            video_path = root / "raw" / "videos" / f"{media_id}.mp4"
            download(video_source["source"], video_path)
            video_sha = sha256_file(video_path)
            frames = extract_sequence(
                video_path,
                root / "sequences" / media_id,
                int(config["video"]["sequence_fps"]),
                int(config["video"]["jpeg_quality"]),
            )
            frame_rows: list[dict[str, Any]] = []
            for frame_number, frame_path in enumerate(frames, 1):
                frame_rows.append(
                    {
                        "frame_number": frame_number,
                        "image_path": str(frame_path),
                        "sha256": sha256_file(frame_path),
                        "phash": phash_file(frame_path),
                    }
                )
            sequence = {
                "sequence_id": f"nifc-{media_id}",
                "panel": "temporal_sequences",
                "source_media_id": media_id,
                "source_group_id": group_id,
                "source_page_url": page_url,
                "source_title": title,
                "source_description_hint": description,
                "captured_at_utc": parse_flickr_taken(item["datetaken"]).isoformat(),
                "published_at_utc": datetime.fromtimestamp(int(item["dateupload"]), tz=timezone.utc).isoformat(),
                "license": source["license_name"],
                "license_url": source["license_url"],
                "video_path": str(video_path),
                "video_sha256": video_sha,
                "duration_seconds": int(info.get("video", {}).get("duration", 0)),
                "fps": int(config["video"]["sequence_fps"]),
                "frames": frame_rows,
                "annotation_status": "pending_human_review",
                "event_label": None,
                "first_visible_target_frame": None,
            }
            sequence_candidates.append(sequence)
            source_records.append(sequence)
            selected_frames = evenly_spaced(frames, int(config["video"]["still_samples_per_video"]))
            selected_names = {path.name for path in selected_frames}
            for frame in frame_rows:
                frame_path = Path(frame["image_path"])
                if frame_path.name not in selected_names:
                    continue
                still_candidates.append(
                    {
                        "sample_id": f"nifc-{media_id}-{frame_path.stem}",
                        "panel": "still_detection",
                        "media_kind": "video_frame",
                        "image_path": frame["image_path"],
                        "source_media_id": media_id,
                        "source_group_id": group_id,
                        "source_page_url": page_url,
                        "source_title": title,
                        "source_description_hint": description,
                        "captured_at_utc": parse_flickr_taken(item["datetaken"]).isoformat(),
                        "published_at_utc": datetime.fromtimestamp(int(item["dateupload"]), tz=timezone.utc).isoformat(),
                        "license": source["license_name"],
                        "license_url": source["license_url"],
                        "sha256": frame["sha256"],
                        "phash": frame["phash"],
                        "sequence_id": sequence["sequence_id"],
                        "frame_number": frame["frame_number"],
                        "annotation_status": "pending_human_review",
                        "objects": [],
                    }
                )
        print(f"[{position}/{len(selected)}] acquired {media_id} {info['media']}", flush=True)

    write_jsonl(root / "manifests" / "source_records.jsonl", source_records)
    write_jsonl(root / "candidates" / "still_candidates.jsonl", still_candidates)
    write_jsonl(root / "candidates" / "sequence_candidates.jsonl", sequence_candidates)
    summary = {
        "benchmark_id": config["benchmark_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "album_total": int(album["total"]),
        "selected_source_media": len(selected),
        "selected_photos": sum(1 for item in selected if item["media"] == "photo"),
        "selected_videos": sum(1 for item in selected if item["media"] == "video"),
        "still_candidates": len(still_candidates),
        "sequence_candidates": len(sequence_candidates),
        "sequence_frames": sum(len(row["frames"]) for row in sequence_candidates),
        "rejected_counts": rejected_counts,
        "release_state": "candidate_only_pending_human_annotation_and_independence_audit",
    }
    write_json(root / "manifests" / "acquisition_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
