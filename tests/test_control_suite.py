from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.data.build_control_suite import build_control_example, write_control_suite
from src.eval.control_metrics import aggregate_control_scores, score_control_prediction


class ControlSuiteTests(unittest.TestCase):
    def test_control_example_ids_are_deterministic(self) -> None:
        first = build_control_example(
            source="synthetic",
            prompt="Rewrite to lowercase: HELLO",
            target_text="hello",
            source_row_id="ctrl-1",
        )
        second = build_control_example(
            source="synthetic",
            prompt="Rewrite to lowercase: HELLO",
            target_text="hello",
            source_row_id="ctrl-1",
        )
        self.assertEqual(first.example_id, second.example_id)

    def test_control_scoring_uses_exact_match_after_basic_normalization(self) -> None:
        example = build_control_example(
            source="synthetic",
            prompt="Return hello.",
            target_text="hello",
        )
        score = score_control_prediction(raw_output="  hello\r\n", example=example)
        self.assertTrue(score.exact_match)
        self.assertEqual(aggregate_control_scores([score]), 1.0)

    def test_control_suite_writer_writes_jsonl(self) -> None:
        output_path = Path("tests/_tmp/control_suite/controls.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        example = build_control_example(
            source="synthetic",
            prompt="Return 583",
            target_text="583",
        )
        written = write_control_suite(examples=[example], output_path=output_path)
        payload = [json.loads(line) for line in written.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(payload[0]["target_text"], "583")


if __name__ == "__main__":
    unittest.main()
