from __future__ import annotations

import argparse
import json
from decimal import Decimal

from fireviewer_orchestrator.mvp.managed_costs import (
    ManagedMvpScenario,
    calculate_managed_mvp_cost,
)


def _json_default(value: object) -> str:
    if isinstance(value, Decimal):
        return format(value, "f")
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate the managed FireViewer MVP pilot cost.")
    parser.add_argument("--events", type=int, choices=(9, 30), default=9)
    parser.add_argument(
        "--nova-verification-per-event",
        type=int,
        choices=range(0, 21),
        default=0,
        metavar="0..20",
        help="Optional Nova checks after RT-DETR; zero keeps the detector path CPU-only.",
    )
    args = parser.parse_args()

    scenario = ManagedMvpScenario(
        name="initial-9-events" if args.events == 9 else "extended-30-events",
        event_count=args.events,
        nova_verification_media_per_event=args.nova_verification_per_event,
    )
    print(
        json.dumps(
            calculate_managed_mvp_cost(scenario),
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
