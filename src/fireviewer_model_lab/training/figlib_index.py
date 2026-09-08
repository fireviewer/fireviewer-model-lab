"""Index HPWREN FIgLib sequences without downloading their image payloads."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fireviewer_model_lab.training.remote_zip import require_http_url

FIGLIB_INDEX_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html"
DATE_PREFIX = re.compile(r"^(?P<date>\d{8})(?:[.-]\d{6})?[-_](?P<body>.+)$")


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def parse_sequence_name(name: str) -> dict[str, str]:
    clean = urllib.parse.unquote(name.strip().strip("/"))
    match = DATE_PREFIX.match(clean)
    if match is None:
        raise ValueError(f"unrecognized FIgLib sequence name: {name}")
    date = match.group("date")
    body = match.group("body")
    if "_" in body:
        event_name, camera_id = body.split("_", 1)
    else:
        fields = body.split("-")
        if len(fields) < 3:
            raise ValueError(f"missing FIgLib camera identifier: {name}")
        event_name = fields[0]
        camera_id = "-".join(fields[1:])
    event_name = event_name.strip("-_")
    camera_id = camera_id.strip("-_")
    if not event_name or not camera_id:
        raise ValueError(f"incomplete FIgLib sequence name: {name}")
    return {
        "sequence_id": clean,
        "event_date": date,
        "event_name": event_name,
        "event_key": f"{date}:{event_name.lower()}",
        "camera_id": camera_id,
    }


def parse_index_html(html: str, *, index_url: str = FIGLIB_INDEX_URL) -> list[dict[str, Any]]:
    parser = _LinkParser()
    parser.feed(html)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for href in parser.links:
        path = urllib.parse.urlparse(href).path.rstrip("/")
        name = Path(path).name
        if name.lower() == "index.html":
            name = Path(path).parent.name
        if not DATE_PREFIX.match(name) or name in seen:
            continue
        seen.add(name)
        row: dict[str, Any] = parse_sequence_name(name)
        row.update(
            {
                "sequence_url": urllib.parse.urljoin(index_url, href),
                "split_group": f"figlib:event:{row['event_key']}",
                "source_id": "hpwren-figlib",
                "smoke_onset_offset_seconds": 0,
            }
        )
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["sequence_id"]))


def build_figlib_index(
    output_root: Path,
    *,
    index_url: str = FIGLIB_INDEX_URL,
    minimum_sequences: int = 400,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    index_url = require_http_url(index_url)
    request = urllib.request.Request(  # noqa: S310 - URL validated above
        index_url, headers={"User-Agent": "FireViewer/1.0"}
    )
    with urllib.request.urlopen(  # noqa: S310 - URL validated above
        request, timeout=timeout_seconds
    ) as response:
        html = response.read().decode("utf-8", errors="replace")
    rows = parse_index_html(html, index_url=index_url)
    if len(rows) < minimum_sequences:
        raise ValueError(
            f"FIgLib index unexpectedly contains only {len(rows)} sequences (< {minimum_sequences})"
        )
    event_cameras: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        event_cameras[str(row["event_key"])].add(str(row["camera_id"]))
    for row in rows:
        row["event_camera_count"] = len(event_cameras[str(row["event_key"])])
        row["cross_view_candidate"] = row["event_camera_count"] >= 2

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "index.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    multi_camera_events = {
        event: sorted(cameras) for event, cameras in event_cameras.items() if len(cameras) >= 2
    }
    report = {
        "schema_version": 1,
        "source_id": "hpwren-figlib",
        "index_url": index_url,
        "sequences": len(rows),
        "events": len(event_cameras),
        "multi_camera_events": len(multi_camera_events),
        "cross_view_candidate_sequences": sum(
            1 for row in rows if bool(row["cross_view_candidate"])
        ),
        "year_counts": dict(sorted(Counter(str(row["event_date"])[:4] for row in rows).items())),
        "manifest": str(manifest),
        "payload_downloaded": False,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
