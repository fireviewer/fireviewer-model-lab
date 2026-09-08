from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fireviewer_model_lab.synthetic.incident_days import generate_incident_day
from fireviewer_model_lab.synthetic.review_store import CaseStore


class IncidentDayTests(unittest.TestCase):
    def test_fixture_is_complete_fictional_and_reviewable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = Path(directory)
            batch = volume / "production" / "batches" / "incident-pilot"
            record_path = generate_incident_day(
                volume_root=volume,
                batch_root=batch,
                case_index=0,
                seed=640000000,
                production_stage="pilot",
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["data_origin"], "new_synthetic_generation")
            self.assertFalse(record["truth"]["real_world_claim"])
            self.assertGreater(record["performance"]["elapsed_seconds"], 0.0)
            self.assertGreater(record["performance"]["case_output_bytes"], 0)
            self.assertEqual(record["performance"]["vram_peak_bytes"], 0)
            self.assertEqual(
                record["performance"]["vram_measurement"],
                "not_applicable_non_gpu_case",
            )
            self.assertEqual(
                record["truth"]["fixture_kind"],
                "fictional_synthetic_incident_day",
            )
            kinds = {item["kind"] for item in record["artifacts"]}
            self.assertTrue(
                {
                    "source_packet",
                    "research_log",
                    "fact_ledger",
                    "contradiction_log",
                    "fire_zone_overlay",
                }.issubset(kinds)
            )
            store = CaseStore(volume)
            store.register(record)
            self.assertEqual(
                store.list(category="france_incident_days", offset=0, limit=10)[
                    "total"
                ],
                1,
            )


if __name__ == "__main__":
    unittest.main()
