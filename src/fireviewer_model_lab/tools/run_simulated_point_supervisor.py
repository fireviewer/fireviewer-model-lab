from __future__ import annotations

import argparse
import os

from pydantic import SecretStr

from fireviewer_evidence_supervisor.mvp.supervision import (
    AzureBackendEventEvidenceAdapter,
    AzureBackendEventEvidenceConfig,
    BackendPointAssessmentPublisher,
    create_point_supervisor_server,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the loopback-only, zero-cost FireViewer point supervisor simulation.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--backend-base-url",
        default=os.environ.get("FIREVIEWER_BACKEND_BASE_URL"),
    )
    parser.add_argument(
        "--backend-token-env",
        default="FIREVIEWER_BACKEND_TOKEN",
        help="Environment variable containing the internal backend bearer token.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.backend_base_url:
        raise ValueError("FIREVIEWER_BACKEND_BASE_URL or --backend-base-url is required")
    token = os.environ.get(args.backend_token_env)
    if token is None:
        raise ValueError(f"{args.backend_token_env} is required")
    backend_config = AzureBackendEventEvidenceConfig(
        base_url=args.backend_base_url,
        bearer_token=SecretStr(token),
    )
    repository = AzureBackendEventEvidenceAdapter(backend_config)
    publisher = BackendPointAssessmentPublisher(backend_config)
    server = create_point_supervisor_server(
        repository,
        host=args.host,
        port=args.port,
        publisher=publisher,
    )
    host, port = server.server_address[:2]
    host_text = host.decode() if isinstance(host, bytes) else host
    print(
        f"simulated-point-supervisor ready http://{host_text}:{port} "
        "evidence=azure-backend publication=event-2.0-policy",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
