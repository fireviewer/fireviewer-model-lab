"""Loopback-only backend test double for the Eve integration evaluation."""

from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import quote

from fireviewer_evidence_supervisor.mvp.supervision import BackendEventEvidenceSnapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--token-env", default="FIREVIEWER_BACKEND_TOKEN")
    return parser


def _canonical_sha256(payload: object) -> str:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(body).hexdigest()


def _handler_for(
    payload: dict[str, object],
    *,
    expected_token: str,
) -> type[BaseHTTPRequestHandler]:
    candidate_id = str(payload["candidate_id"])
    expected_path = (
        "/api/v1/internal/event-evidence/" + quote(candidate_id, safe="")
    )

    class FakeBackendHandler(BaseHTTPRequestHandler):
        server_version = "FireViewerFakeEventEvidenceBackend/1.0"

        def log_message(self, _format: str, *_args: object) -> None:
            return None

        def _write(self, status: HTTPStatus, body: dict[str, object]) -> None:
            encoded = json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            self.send_response(status.value)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(encoded)))
            self.send_header("cache-control", "private, no-store")
            if status == HTTPStatus.OK:
                checksum = str(body["source_sha256"])
                self.send_header("etag", f'"{checksum}"')
                self.send_header("x-checksum-sha256", checksum)
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if not compare_digest(
                self.headers.get("authorization", ""),
                f"Bearer {expected_token}",
            ):
                self._write(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if self.path != expected_path:
                self._write(HTTPStatus.NOT_FOUND, {"error": "event_not_found"})
                return
            self._write(HTTPStatus.OK, payload)

    return FakeBackendHandler


def main() -> int:
    args = _parser().parse_args()
    try:
        if not ip_address(args.host).is_loopback:
            raise ValueError("the fake backend must only listen on loopback")
    except ValueError as exc:
        raise ValueError("the fake backend host must be a loopback IP") from exc
    token = os.environ.get(args.token_env)
    if token is None or len(token) < 32:
        raise ValueError(f"{args.token_env} must contain at least 32 characters")
    raw = json.loads(args.fixture.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("fixture root must be a JSON object")
    raw.pop("source_sha256", None)
    raw["source_sha256"] = _canonical_sha256(raw)
    snapshot = BackendEventEvidenceSnapshot.model_validate(raw)
    payload = snapshot.model_dump(mode="json")
    payload.pop("source_sha256")
    payload["source_sha256"] = _canonical_sha256(payload)
    BackendEventEvidenceSnapshot.model_validate(payload)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        _handler_for(payload, expected_token=token),
    )
    host, port = server.server_address[:2]
    host_text = host.decode() if isinstance(host, bytes) else host
    print(f"fake-event-evidence-backend ready http://{host_text}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
