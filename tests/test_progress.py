from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.utils.progress import RunHeartbeat


class ProgressTests(unittest.TestCase):
    def test_heartbeat_interval_throttles_non_forced_updates(self) -> None:
        output_dir = Path("tests/_tmp/progress")
        output_dir.mkdir(parents=True, exist_ok=True)

        heartbeat = RunHeartbeat(
            output_dir=output_dir,
            phase="eval_main",
            total_units=10,
            unit_name="examples",
            heartbeat_interval_seconds=10.0,
        )

        with patch("src.utils.progress.time.monotonic", side_effect=[0.0, 1.0, 11.0]):
            heartbeat.start(completed_units=0, message="started")
            first_payload = json.loads(heartbeat.paths.heartbeat_path.read_text(encoding="utf-8"))
            skipped_payload = heartbeat.maybe_update(
                completed_units=1,
                message="too soon",
                metrics={"strict_full_call_success_so_far": 1.0},
            )
            second_payload = heartbeat.maybe_update(
                completed_units=2,
                message="written after interval",
                metrics={"strict_full_call_success_so_far": 0.5},
            )

        self.assertEqual(skipped_payload, {})
        self.assertEqual(second_payload["completed_units"], 2)
        current_payload = json.loads(heartbeat.paths.heartbeat_path.read_text(encoding="utf-8"))
        self.assertEqual(first_payload["completed_units"], 0)
        self.assertEqual(current_payload["completed_units"], 2)


if __name__ == "__main__":
    unittest.main()
