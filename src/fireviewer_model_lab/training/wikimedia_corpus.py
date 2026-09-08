from __future__ import annotations

import argparse
import html
import json
import re
import time
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from fireviewer_model_lab.training.corpus_pipeline import (
    _inspect_image,
    deterministic_split,
    normalized_identifier,
    sha256_bytes,
    validate_manifest,
)

USER_AGENT = "FireWarningCorpusBuilder/0.1 (https://github.com/unicornwhodev/fireviewer)"
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024
DOWNLOAD_DELAY_SECONDS = 1.0
MAX_RATE_LIMIT_RETRIES = 4
METADATA_FILTER = "|".join(
    (
        "Artist",
        "Credit",
        "LicenseShortName",
        "LicenseUrl",
        "UsageTerms",
        "DateTimeOriginal",
        "DateTime",
        "GPSLatitude",
        "GPSLongitude",
    )
)
HTML_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Lot:
    lot_id: str
    source_id: str
    category: str
    recursive_depth: int
    corpus_role: str
    candidate_classes: tuple[str, ...]
    default_location: dict[str, Any] | None


def _plain_text(value: object) -> str:
    unescaped = html.unescape(str(value or ""))
    return " ".join(HTML_TAG.sub(" ", unescaped).split())[:4096]


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    raw = metadata.get(key)
    if not isinstance(raw, dict):
        return ""
    return _plain_text(raw.get("value"))


def canonical_open_license(short_name: str, license_url: str) -> tuple[str, str] | None:
    normalized_name = " ".join(short_name.strip().lower().split())
    if not license_url and normalized_name == "cc0":
        return "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"
    if not license_url and normalized_name in {"public domain", "public domain mark"}:
        return "PDM-1.0", "https://creativecommons.org/publicdomain/mark/1.0/"
    parsed = urlparse(license_url.replace("http://", "https://", 1))
    if parsed.hostname not in {"creativecommons.org", "www.creativecommons.org"}:
        return None
    path = parsed.path.rstrip("/").lower()
    match = re.fullmatch(r"/licenses/(by|by-sa)/(2\.0|2\.5|3\.0|4\.0)(?:/.*)?", path)
    if match:
        family, version = match.groups()
        identifier = f"CC-{family.upper()}-{version}"
        return identifier, f"https://creativecommons.org/licenses/{family}/{version}/"
    if path == "/publicdomain/zero/1.0" or path.startswith("/publicdomain/zero/1.0/"):
        return "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"
    if path == "/publicdomain/mark/1.0" or path.startswith("/publicdomain/mark/1.0/"):
        return "PDM-1.0", "https://creativecommons.org/publicdomain/mark/1.0/"
    return None


def _request_json(client: httpx.Client, api_url: str, params: dict[str, object]) -> dict[str, Any]:
    response: httpx.Response | None = None
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = client.get(api_url, params=params)
        if response.status_code != 429:
            break
        if attempt == MAX_RATE_LIMIT_RETRIES:
            response.raise_for_status()
        _wait_after_rate_limit(response, attempt)
    if response is None:
        raise RuntimeError("Wikimedia API request did not produce a response")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError(f"Unexpected Wikimedia API response: {payload!r}")
    return payload


def category_files(
    client: httpx.Client,
    api_url: str,
    category: str,
    *,
    recursive_depth: int,
) -> list[str]:
    visited: set[str] = set()
    files: set[str] = set()

    def walk(current: str, remaining_depth: int) -> None:
        if current in visited:
            return
        visited.add(current)
        continuation: dict[str, object] = {}
        while True:
            payload = _request_json(
                client,
                api_url,
                {
                    "action": "query",
                    "format": "json",
                    "formatversion": 2,
                    "list": "categorymembers",
                    "cmtitle": f"Category:{current}",
                    "cmtype": "file|subcat",
                    "cmlimit": "max",
                    **continuation,
                },
            )
            members = payload.get("query", {}).get("categorymembers", [])
            for member in members:
                namespace = int(member.get("ns", -1))
                title = str(member.get("title", ""))
                if namespace == 6:
                    files.add(title)
                elif namespace == 14 and remaining_depth > 0 and title.startswith("Category:"):
                    walk(title.removeprefix("Category:"), remaining_depth - 1)
            next_page = payload.get("continue")
            if not isinstance(next_page, dict):
                break
            continuation = {str(key): value for key, value in next_page.items()}

    walk(category, recursive_depth)
    return sorted(files)


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def file_metadata(
    client: httpx.Client,
    api_url: str,
    titles: list[str],
    *,
    thumbnail_width: int,
) -> Iterable[dict[str, Any]]:
    for chunk in _chunks(titles, 20):
        payload = _request_json(
            client,
            api_url,
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "prop": "imageinfo|coordinates",
                "titles": "|".join(chunk),
                "iiprop": "url|size|sha1|mime|extmetadata",
                "iiurlwidth": thumbnail_width,
                "iiextmetadatalanguage": "en",
                "iiextmetadatafilter": METADATA_FILTER,
                "colimit": "max",
            },
        )
        pages = payload.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            raise ValueError("Wikimedia pages response is not a list")
        yield from pages


def _point_location(page: dict[str, Any], default: dict[str, Any] | None) -> dict[str, Any] | None:
    coordinates = page.get("coordinates")
    if isinstance(coordinates, list) and coordinates:
        coordinate = coordinates[0]
        if isinstance(coordinate, dict) and "lat" in coordinate and "lon" in coordinate:
            location = (
                deepcopy(default)
                if default is not None
                else {
                    "massif_id": None,
                    "massif_name": None,
                }
            )
            location.update(
                {
                    "origin": "source_page_coordinates",
                    "precision": "point",
                    "latitude": float(coordinate["lat"]),
                    "longitude": float(coordinate["lon"]),
                    "reference": str(page["description_url"]),
                    "reference_version": str(page["retrieved_at"]),
                }
            )
            return location
    return deepcopy(default)


def _wait_after_rate_limit(response: httpx.Response, attempt: int) -> None:
    raw_retry_after = response.headers.get("retry-after", "").strip()
    try:
        requested_delay = float(raw_retry_after)
    except ValueError:
        requested_delay = 10.0 * (attempt + 1)
    time.sleep(min(max(requested_delay, 1.0), 60.0))


def _download_image(client: httpx.Client, url: str) -> bytes:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        with client.stream("GET", url) as response:
            if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                _wait_after_rate_limit(response, attempt)
                continue
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length is not None and int(content_length) > MAX_IMAGE_BYTES:
                raise ValueError("download exceeds the 25 MiB candidate limit")
            payload = bytearray()
            for chunk in response.iter_bytes():
                payload.extend(chunk)
                if len(payload) > MAX_IMAGE_BYTES:
                    raise ValueError("download exceeds the 25 MiB candidate limit")
        time.sleep(DOWNLOAD_DELAY_SECONDS)
        return bytes(payload)
    raise RuntimeError("Wikimedia download retry loop exited unexpectedly")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} in {path} is not an object")
            rows.append(value)
    return rows


def _title_from_description_url(description_url: str) -> str:
    path = unquote(urlparse(description_url).path)
    if not path.startswith("/wiki/"):
        return ""
    return path.removeprefix("/wiki/").replace("_", " ")


def _prepare_page(page: dict[str, Any], retrieved_at: str) -> dict[str, Any] | None:
    imageinfo = page.get("imageinfo")
    if not isinstance(imageinfo, list) or not imageinfo:
        return None
    info = imageinfo[0]
    if not isinstance(info, dict):
        return None
    metadata = info.get("extmetadata", {})
    if not isinstance(metadata, dict):
        return None
    description_url = str(info.get("descriptionurl", ""))
    return {
        **page,
        "info": info,
        "metadata": metadata,
        "description_url": description_url,
        "retrieved_at": retrieved_at,
    }


def _record_from_page(
    client: httpx.Client,
    page: dict[str, Any],
    lot: Lot,
    output_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    info = page["info"]
    metadata = page["metadata"]
    mime = str(info.get("mime", "")).lower()
    if mime not in ALLOWED_MIME_TYPES:
        return None, f"unsupported_mime:{mime or 'missing'}"
    original_sha1 = str(info.get("sha1", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{40}", original_sha1):
        return None, "missing_original_sha1"

    short_license = _metadata_value(metadata, "LicenseShortName")
    license_url = _metadata_value(metadata, "LicenseUrl")
    canonical = canonical_open_license(short_license, license_url)
    if canonical is None:
        return None, f"license_not_allowlisted:{short_license or 'missing'}"
    license_id, canonical_license_url = canonical

    original_url = str(info.get("url", ""))
    thumbnail_url = str(info.get("thumburl", ""))
    download_url = thumbnail_url or original_url
    if not original_url.startswith("https://") or not download_url.startswith("https://"):
        return None, "non_https_media_url"
    payload = _download_image(client, download_url)
    width, height, extension, perceptual_hash = _inspect_image(payload)
    digest = sha256_bytes(payload)
    relative_path = Path("images") / digest[:2] / f"{digest}.{extension}"
    destination = output_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_bytes(destination.read_bytes()) != digest:
        raise ValueError(f"Existing candidate image has a bad digest: {destination}")
    if not destination.exists():
        destination.write_bytes(payload)

    title = str(page.get("title", ""))
    page_id = str(page.get("pageid", ""))
    group = f"{lot.source_id}:{normalized_identifier(lot.category)}"
    location = _point_location(page, lot.default_location)
    split = (
        "critical_test"
        if lot.corpus_role == "geo_context_evaluation"
        else deterministic_split(group)
    )
    capture_literal = _metadata_value(metadata, "DateTimeOriginal") or _metadata_value(
        metadata, "DateTime"
    )
    return (
        {
            "sample_id": f"{lot.source_id}:{digest[:24]}",
            "source_id": lot.source_id,
            "source_record_id": f"wikimedia-page:{page_id}:sha1:{original_sha1}",
            "corpus_role": lot.corpus_role,
            "image_relpath": relative_path.as_posix(),
            "sha256": digest,
            "phash": perceptual_hash,
            "near_duplicate_of": None,
            "width": width,
            "height": height,
            "event_id": group,
            "sequence_id": f"{group}:{normalized_identifier(title)}",
            "split_group": group,
            "captured_at_literal": capture_literal or None,
            "split": split,
            "license": license_id,
            "consent_basis": {
                "kind": "source_license",
                "reference": str(page["description_url"]),
            },
            "sample_validation_status": "candidate_unreviewed",
            "source_asset": {
                "original_url": original_url,
                "description_url": str(page["description_url"]),
                "original_sha1": original_sha1,
                "variant": "thumbnail_2048" if thumbnail_url else "original",
                "artist": _metadata_value(metadata, "Artist"),
                "credit": _metadata_value(metadata, "Credit"),
                "license_url": canonical_license_url,
            },
            "candidate_classes": list(lot.candidate_classes),
            "annotations": [],
            "negative_tags": [],
            "location": location,
        },
        None,
    )


def _load_lots(config_path: Path, selected: set[str]) -> tuple[str, int, list[Lot]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lots: list[Lot] = []
    for raw in config["lots"]:
        if selected and raw["lot_id"] not in selected:
            continue
        lots.append(
            Lot(
                lot_id=str(raw["lot_id"]),
                source_id=str(raw["source_id"]),
                category=str(raw["category"]),
                recursive_depth=int(raw["recursive_depth"]),
                corpus_role=str(raw["corpus_role"]),
                candidate_classes=tuple(str(value) for value in raw["candidate_classes"]),
                default_location=raw["default_location"],
            )
        )
    unknown = selected - {lot.lot_id for lot in lots}
    if unknown:
        raise ValueError(f"Unknown Wikimedia lots: {sorted(unknown)}")
    return str(config["api_url"]), int(config["thumbnail_width"]), lots


def resume_candidate_titles(
    titles: list[str],
    *,
    accepted_titles: set[str],
    permanent_rejection_titles: set[str],
) -> list[str]:
    return sorted(set(titles) - accepted_titles - permanent_rejection_titles)


def acquire_wikimedia(
    output_dir: Path,
    config_path: Path,
    *,
    selected_lots: set[str],
    max_per_lot: int,
    resume: bool,
) -> dict[str, Any]:
    api_url, thumbnail_width, lots = _load_lots(config_path, selected_lots)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    rejection_path = output_dir / "rejections.jsonl"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"Candidate manifest already exists: {manifest_path}")

    retrieved_at = datetime.now(UTC).isoformat()
    records = _load_jsonl(manifest_path) if resume else []
    existing_rows_before_run = len(records)
    seen_digests = {str(record["sha256"]) for record in records}
    accepted_titles = {
        _title_from_description_url(str(record.get("source_asset", {}).get("description_url", "")))
        for record in records
    }
    previous_rejections = _load_jsonl(rejection_path) if resume else []
    permanent_rejection_titles_by_lot: dict[str, set[str]] = {}
    rejections: list[dict[str, str]] = []
    for rejection in previous_rejections:
        reason = str(rejection.get("reason", ""))
        title = str(rejection.get("title", ""))
        if reason.startswith("acquisition_error:") and title not in accepted_titles:
            continue
        else:
            rejections.append({str(key): str(value) for key, value in rejection.items()})
            permanent_rejection_titles_by_lot.setdefault(str(rejection["lot_id"]), set()).add(title)
    lot_reports: dict[str, dict[str, int]] = {}
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(60.0, connect=20.0),
        follow_redirects=False,
    ) as client:
        for lot in lots:
            titles = category_files(
                client,
                api_url,
                lot.category,
                recursive_depth=lot.recursive_depth,
            )
            catalog_count = len(titles)
            if resume:
                titles = resume_candidate_titles(
                    titles,
                    accepted_titles=accepted_titles,
                    permanent_rejection_titles=permanent_rejection_titles_by_lot.get(
                        lot.lot_id, set()
                    ),
                )
            if max_per_lot > 0:
                titles = titles[:max_per_lot]
            counters: Counter[str] = Counter(
                catalog_discovered=catalog_count,
                acquisition_candidates=len(titles),
            )
            for raw_page in file_metadata(
                client,
                api_url,
                titles,
                thumbnail_width=thumbnail_width,
            ):
                page = _prepare_page(raw_page, retrieved_at)
                if page is None:
                    counters["rejected"] += 1
                    rejections.append(
                        {
                            "lot_id": lot.lot_id,
                            "title": str(raw_page.get("title", "")),
                            "reason": "missing_imageinfo",
                        }
                    )
                    continue
                try:
                    record, rejection = _record_from_page(client, page, lot, output_dir)
                except (httpx.HTTPError, OSError, ValueError) as exc:
                    record, rejection = None, f"acquisition_error:{type(exc).__name__}:{exc}"
                if record is None:
                    counters["rejected"] += 1
                    rejections.append(
                        {
                            "lot_id": lot.lot_id,
                            "title": str(page.get("title", "")),
                            "reason": str(rejection),
                        }
                    )
                    continue
                digest = str(record["sha256"])
                if digest in seen_digests:
                    counters["exact_duplicate"] += 1
                    continue
                seen_digests.add(digest)
                records.append(record)
                counters["accepted"] += 1
                if record["location"] is not None:
                    counters["location_pairs"] += 1
            lot_reports[lot.lot_id] = dict(sorted(counters.items()))

    manifest_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    rejection_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rejections),
        encoding="utf-8",
    )
    validation = validate_manifest(manifest_path, output_dir=output_dir, verify_files=True)
    report = {
        "schema_version": 1,
        "retrieved_at": retrieved_at,
        "api_url": api_url,
        "thumbnail_width": thumbnail_width,
        "resume": resume,
        "existing_rows_before_run": existing_rows_before_run,
        "lots": lot_reports,
        "accepted_rows": len(records),
        "rejected_rows": len(rejections),
        "validation": validation,
    }
    (output_dir / "acquisition-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire license-gated Wikimedia candidate lots")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "corpus" / "wikimedia_lots.json",
    )
    parser.add_argument("--lot", action="append", default=[])
    parser.add_argument("--max-per-lot", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_per_lot < 0:
        raise ValueError("--max-per-lot cannot be negative")
    report = acquire_wikimedia(
        args.output,
        args.config,
        selected_lots=set(args.lot),
        max_per_lot=args.max_per_lot,
        resume=args.resume,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

