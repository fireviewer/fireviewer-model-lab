from training.pointing_dataset_v8.acquire_commons import legacy_sha1_adapter

import json
import sys

import pytest

from training.pointing_dataset_v7.split_registry import digest
from training.pointing_dataset_v8 import acquire_commons as commons


def test_sha1_legacy_adapter_is_lossless_and_does_not_mutate_api_payload():
    raw = 'a4034fa83c100a406b2afb199adb2a8a8da8cf81'
    page = {'imageinfo': [{'sha1': raw, 'size': 1127899}]}
    adapted = legacy_sha1_adapter(page)
    assert int(adapted['imageinfo'][0]['sha1'], 36) == int(raw, 16)
    assert adapted['imageinfo'][0]['upstream_sha1_hex'] == raw
    assert page['imageinfo'][0]['sha1'] == raw


def test_invalid_digest_is_not_repaired_or_replaced():
    for raw in ('', 'not-a-sha', 'x' * 40):
        assert legacy_sha1_adapter({'imageinfo': [{'sha1': raw}]})['imageinfo'][0]['sha1'] == raw


def test_429_stops_one_image_immediately_without_retry_or_file_creation(tmp_path, monkeypatch):
    class Response:
        status_code = 429
        headers = {"Retry-After": "600"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    calls, sleeps = [], []

    def get(url, **_):
        calls.append(url)
        return Response()

    monkeypatch.setattr(commons.requests, "get", get)
    monkeypatch.setattr(commons.time, "sleep", sleeps.append)
    with pytest.raises(commons.RateLimited):
        commons.acquire({"pageid": 1, "current_file_version": {"original_url": "https://example.test/photo.jpg"}},
                        tmp_path, commons.ByteBudget(1000), request_delay=10)
    assert calls == ["https://example.test/photo.jpg"] and sleeps == [10]
    assert not list(tmp_path.iterdir())


def test_paused_batch_preserves_prior_receipts_and_skips_remaining_requests(tmp_path, monkeypatch):
    source = tmp_path / "preserved.jpg"
    source.write_bytes(b"previously-acquired-bytes")
    receipt = {"pageid": 99, "source_image": str(source), "sha256": digest(source)}
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "commons-99.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(commons, "catalogue", lambda _: [{"pageid": 1}, {"pageid": 2}])
    calls = []

    def throttled(row, *_):
        calls.append(row["pageid"])
        raise commons.RateLimited("HTTP 429")

    monkeypatch.setattr(commons, "acquire", throttled)
    monkeypatch.setattr(sys, "argv", ["commons", "--output", str(tmp_path), "--limit", "2"])
    commons.main()
    assert len(calls) == 1
    assert json.loads((tmp_path / "candidate_manifest.jsonl").read_text()) == receipt
    status = json.loads((tmp_path / "acquisition_status.json").read_text())
    assert status["status"] == "paused_rate_limited_not_complete"
    assert status["automatic_retry_scheduled"] is False and status["training_started"] is False
    errors = [json.loads(line) for line in (tmp_path / "acquisition_errors.jsonl").read_text().splitlines()]
    assert {error["type"] for error in errors} == {"RateLimited", "SkippedAfterRateLimit"}
    assert digest(source) == receipt["sha256"]


@pytest.mark.parametrize("size,cap,remaining,match", [(101, 100, 200, "individual"), (100, 200, 99, "storage")])
def test_download_budgets_fail_before_a_network_request(tmp_path, monkeypatch, size, cap, remaining, match):
    def unexpected_request(*_, **__):
        raise AssertionError("An over-budget request must never be sent")
    monkeypatch.setattr(commons.requests, "get", unexpected_request)
    with pytest.raises(ValueError, match=match):
        commons.acquire({"pageid": 1, "current_file_version": {"bytes": size}}, tmp_path,
                        commons.ByteBudget(remaining), max_original_bytes=cap)
    assert not list(tmp_path.iterdir())


def test_source_priority_does_not_change_or_admit_candidate_metadata():
    rows = [{"pageid": 1, "title": "File:smoke photograph.jpg"},
            {"pageid": 2, "title": "File:Wildfire flames at night.jpg"}]
    original = json.loads(json.dumps(rows))
    assert sorted(rows, key=commons.acquisition_priority)[0]["pageid"] == 2
    assert rows == original
    assert all("v8_corpus_admitted" not in row for row in rows)


def test_external_search_cache_is_read_only_and_query_bound(tmp_path, monkeypatch):
    cache = tmp_path / "original_metadata"
    cache.mkdir()
    cached = cache / "search-00.json"
    cached.write_text(json.dumps({"query": "fixed query", "response": {"query": {"pages": []}}}))
    before = digest(cached)
    monkeypatch.setattr(commons, "QUERIES", ("fixed query",))
    monkeypatch.setattr(commons.CommonsAPI, "request", lambda *_: pytest.fail("Cached query must not be downloaded"))
    assert commons.catalogue(tmp_path / "completion", metadata_cache=cache) == []
    assert digest(cached) == before
    assert not list((tmp_path / "completion" / "metadata").iterdir())
    monkeypatch.setattr(commons, "QUERIES", ("different query",))
    with pytest.raises(ValueError, match="Cached Commons search changed"):
        commons.catalogue(tmp_path / "completion", metadata_cache=cache)
