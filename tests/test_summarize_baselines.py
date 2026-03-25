from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.analysis.donor_gap_gate import build_donor_gap_gate
from src.analysis.summarize_baselines import build_baseline_summary


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _prediction_row(*, example_id: str, variant: str, strict_correct: bool) -> dict[str, object]:
    return {
        "example_id": example_id,
        "split": "eval",
        "variant": variant,
        "raw_output": "{}",
        "score": {
            "arg_exact_match": strict_correct,
            "example_id": example_id,
            "gold_name": "NO_TOOL" if "nocall" in variant else "tool",
            "is_nocall_example": "nocall" in variant,
            "json_valid": True,
            "parse_status": "ok",
            "predicted_is_nocall": "nocall" in variant,
            "predicted_name": "NO_TOOL" if "nocall" in variant else "tool",
            "semantic_correct": strict_correct,
            "semantic_error": None,
            "semantic_gold_name": "NO_TOOL" if "nocall" in variant else "tool",
            "semantic_predicted_name": "NO_TOOL" if "nocall" in variant else "tool",
            "strict_correct": strict_correct,
            "strict_error": None,
        },
    }


class SummarizeBaselinesTests(unittest.TestCase):
    def test_summary_computes_primary_metric_and_gate(self) -> None:
        output_dir = Path("tests/_tmp/summarize_baselines")
        base_predictions = _write_predictions(
            output_dir / "base.jsonl",
            [
                _prediction_row(example_id="a", variant="schema_shift", strict_correct=False),
                _prediction_row(
                    example_id="b", variant="nocall_missing_tool", strict_correct=False
                ),
                _prediction_row(example_id="c", variant="nocall_unsupported", strict_correct=False),
                _prediction_row(example_id="d", variant="canonical", strict_correct=False),
            ],
        )
        donor_predictions = _write_predictions(
            output_dir / "donor.jsonl",
            [
                _prediction_row(example_id="a", variant="schema_shift", strict_correct=True),
                _prediction_row(example_id="b", variant="nocall_missing_tool", strict_correct=True),
                _prediction_row(example_id="c", variant="nocall_unsupported", strict_correct=True),
                _prediction_row(example_id="d", variant="canonical", strict_correct=False),
            ],
        )

        summary = build_baseline_summary(
            {
                "analysis": {
                    "base_predictions_path": str(base_predictions),
                    "donor_predictions_path": str(donor_predictions),
                    "bootstrap_samples": 200,
                    "bootstrap_seed": 17,
                }
            }
        )

        self.assertEqual(summary["primary_metric"]["base_value"], 0.0)
        self.assertEqual(summary["primary_metric"]["donor_value"], 1.0)
        self.assertEqual(summary["gate_decision"]["status"], "pass")
        self.assertEqual(summary["gate_decision"]["reason"], "delta_at_least_15pp")

        summary_path = output_dir / "baseline_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        gate_payload = build_donor_gap_gate({"analysis": {"summary_path": str(summary_path)}})
        self.assertEqual(gate_payload["gate_decision"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
