from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.data.smoke_data import run_smoke_data_pipeline


class SmokeDataTests(unittest.TestCase):
    def test_smoke_data_pipeline_writes_expected_artifacts(self) -> None:
        output_dir = Path("tests/_tmp/smoke_data")
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = run_smoke_data_pipeline(output_dir=output_dir)

        self.assertGreater(summary["counts"]["canonical_examples"], 0)
        self.assertEqual(summary["counts"]["schema_shift_examples"], 2)
        self.assertEqual(summary["counts"]["control_examples"], 2)
        self.assertEqual(summary["evaluation_manifest"]["split_counts"]["eval"], 5)
        self.assertNotIn("unassigned", summary["evaluation_manifest"]["split_counts"])

        golden_path = Path(summary["golden_fixture_path"])
        leakage_path = Path(summary["leakage_audit_path"])
        self.assertTrue(golden_path.exists())
        self.assertTrue(leakage_path.exists())

        golden_payload = json.loads(golden_path.read_text(encoding="utf-8"))
        self.assertEqual(golden_payload["prompt_contract_version"], "fc_v1")


if __name__ == "__main__":
    unittest.main()
