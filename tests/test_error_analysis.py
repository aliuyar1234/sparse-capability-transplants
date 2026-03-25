from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.analysis.error_analysis import write_error_analysis_report


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _score(
    *,
    example_id: str,
    strict_correct: bool,
    semantic_correct: bool,
    strict_error: str | None,
    semantic_error: str | None,
    predicted_name: str | None,
    semantic_predicted_name: str | None,
    gold_name: str,
    semantic_gold_name: str,
    is_nocall: bool,
    predicted_is_nocall: bool | None,
) -> dict:
    return {
        "example_id": example_id,
        "parse_status": "ok",
        "json_valid": True,
        "strict_correct": strict_correct,
        "semantic_correct": semantic_correct,
        "strict_error": strict_error,
        "semantic_error": semantic_error,
        "predicted_name": predicted_name,
        "semantic_predicted_name": semantic_predicted_name,
        "gold_name": gold_name,
        "semantic_gold_name": semantic_gold_name,
        "arg_exact_match": strict_correct and semantic_correct,
        "is_nocall_example": is_nocall,
        "predicted_is_nocall": predicted_is_nocall,
    }


class ErrorAnalysisTests(unittest.TestCase):
    def test_write_error_analysis_report_builds_tables_and_examples(self) -> None:
        root = Path("tests/_tmp/error_analysis")
        root.mkdir(parents=True, exist_ok=True)

        eval_examples_path = _write_jsonl(
            root / "eval_examples.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "gold": {"name": "display_location_map", "arguments": {"query": "museum"}},
                    "tools": [{"tool_id": "show_map", "name": "display_location_map"}],
                    "user_request": "Show me the museum on a map.",
                },
                {
                    "example_id": "ex_missing",
                    "gold": {"name": "NO_TOOL", "arguments": {}},
                    "tools": [{"tool_id": "show_map", "name": "show_map"}],
                    "user_request": "Use a tool that does not exist.",
                },
                {
                    "example_id": "ex_unsupported",
                    "gold": {"name": "NO_TOOL", "arguments": {}},
                    "tools": [{"tool_id": "show_map", "name": "show_map"}],
                    "user_request": "Write a limerick.",
                },
            ],
        )
        eval_manifest_path = _write_json(
            root / "eval_manifest.json",
            {
                "dataset_path": str(eval_examples_path.resolve()),
                "prompt_contract_version": "fc_v1",
            },
        )
        control_examples_path = _write_jsonl(
            root / "control_examples.jsonl",
            [{"example_id": "ctrl_1", "target_text": "hello world"}],
        )
        control_manifest_path = _write_json(
            root / "control_manifest.json",
            {"dataset_path": str(control_examples_path.resolve())},
        )

        base_predictions_path = _write_jsonl(
            root / "base_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "raw_output": '{"name": "open_wifi_settings", "arguments": {}}',
                    "score": _score(
                        example_id="ex_schema",
                        strict_correct=False,
                        semantic_correct=False,
                        strict_error="wrong_tool",
                        semantic_error="wrong_tool",
                        predicted_name="open_wifi_settings",
                        semantic_predicted_name="open_wifi_settings",
                        gold_name="display_location_map",
                        semantic_gold_name="show_map",
                        is_nocall=False,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "raw_output": '{"name": "show_map", "arguments": {}}',
                    "score": _score(
                        example_id="ex_missing",
                        strict_correct=False,
                        semantic_correct=False,
                        strict_error="wrong_nocall_decision",
                        semantic_error="wrong_nocall_decision",
                        predicted_name="show_map",
                        semantic_predicted_name="show_map",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "raw_output": '{"name": "NO_TOOL", "arguments": {}}',
                    "score": _score(
                        example_id="ex_unsupported",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="NO_TOOL",
                        semantic_predicted_name="NO_TOOL",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=True,
                    ),
                },
            ],
        )
        donor_predictions_path = _write_jsonl(
            root / "donor_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "raw_output": (
                        '{"name": "display_location_map", "arguments": {"query": "museum"}}'
                    ),
                    "score": _score(
                        example_id="ex_schema",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="display_location_map",
                        semantic_predicted_name="show_map",
                        gold_name="display_location_map",
                        semantic_gold_name="show_map",
                        is_nocall=False,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "raw_output": '{"name": "NO_TOOL", "arguments": {}}',
                    "score": _score(
                        example_id="ex_missing",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="NO_TOOL",
                        semantic_predicted_name="NO_TOOL",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=True,
                    ),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "raw_output": '{"name": "NO_TOOL", "arguments": {}}',
                    "score": _score(
                        example_id="ex_unsupported",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="NO_TOOL",
                        semantic_predicted_name="NO_TOOL",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=True,
                    ),
                },
            ],
        )
        sparse_predictions_path = _write_jsonl(
            root / "sparse_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "raw_output": '{"name": "show_map", "arguments": {"query": "museum"}}',
                    "score": _score(
                        example_id="ex_schema",
                        strict_correct=False,
                        semantic_correct=True,
                        strict_error="wrong_tool",
                        semantic_error=None,
                        predicted_name="show_map",
                        semantic_predicted_name="show_map",
                        gold_name="display_location_map",
                        semantic_gold_name="show_map",
                        is_nocall=False,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "raw_output": '{"name": "NO_TOOL", "arguments": {}}',
                    "score": _score(
                        example_id="ex_missing",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="NO_TOOL",
                        semantic_predicted_name="NO_TOOL",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=True,
                    ),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "raw_output": '{"name": "NO_TOOL", "arguments": {}}',
                    "score": _score(
                        example_id="ex_unsupported",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="NO_TOOL",
                        semantic_predicted_name="NO_TOOL",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=True,
                    ),
                },
            ],
        )
        dense_predictions_path = _write_jsonl(
            root / "dense_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "raw_output": (
                        '{"name": "display_location_map", "arguments": {"query": "museum"}}'
                    ),
                    "score": _score(
                        example_id="ex_schema",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="display_location_map",
                        semantic_predicted_name="show_map",
                        gold_name="display_location_map",
                        semantic_gold_name="show_map",
                        is_nocall=False,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "raw_output": '{"name": "show_map", "arguments": {}}',
                    "score": _score(
                        example_id="ex_missing",
                        strict_correct=False,
                        semantic_correct=False,
                        strict_error="wrong_nocall_decision",
                        semantic_error="wrong_nocall_decision",
                        predicted_name="show_map",
                        semantic_predicted_name="show_map",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "raw_output": '{"name": "show_map", "arguments": {}}',
                    "score": _score(
                        example_id="ex_unsupported",
                        strict_correct=False,
                        semantic_correct=False,
                        strict_error="wrong_nocall_decision",
                        semantic_error="wrong_nocall_decision",
                        predicted_name="show_map",
                        semantic_predicted_name="show_map",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=False,
                    ),
                },
            ],
        )
        steering_predictions_path = _write_jsonl(
            root / "steering_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "raw_output": '{"name": "open_wifi_settings", "arguments": {}}',
                    "score": _score(
                        example_id="ex_schema",
                        strict_correct=False,
                        semantic_correct=False,
                        strict_error="wrong_tool",
                        semantic_error="wrong_tool",
                        predicted_name="open_wifi_settings",
                        semantic_predicted_name="open_wifi_settings",
                        gold_name="display_location_map",
                        semantic_gold_name="show_map",
                        is_nocall=False,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "raw_output": '{"name": "show_map", "arguments": {}}',
                    "score": _score(
                        example_id="ex_missing",
                        strict_correct=False,
                        semantic_correct=False,
                        strict_error="wrong_nocall_decision",
                        semantic_error="wrong_nocall_decision",
                        predicted_name="show_map",
                        semantic_predicted_name="show_map",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=False,
                    ),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "raw_output": '{"name": "NO_TOOL", "arguments": {}}',
                    "score": _score(
                        example_id="ex_unsupported",
                        strict_correct=True,
                        semantic_correct=True,
                        strict_error=None,
                        semantic_error=None,
                        predicted_name="NO_TOOL",
                        semantic_predicted_name="NO_TOOL",
                        gold_name="NO_TOOL",
                        semantic_gold_name="NO_TOOL",
                        is_nocall=True,
                        predicted_is_nocall=True,
                    ),
                },
            ],
        )

        base_control_predictions = _write_jsonl(
            root / "base_control_predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello world",
                        "normalized_target": "hello world",
                    },
                }
            ],
        )
        sparse_control_predictions = _write_jsonl(
            root / "sparse_control_predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": False,
                        "normalized_prediction": "bye world",
                        "normalized_target": "hello world",
                    },
                }
            ],
        )
        dense_control_predictions = _write_jsonl(
            root / "dense_control_predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello world",
                        "normalized_target": "hello world",
                    },
                }
            ],
        )

        baseline_summary_path = _write_json(
            root / "baseline_summary.json",
            {
                "base_predictions_path": str(base_predictions_path.resolve()),
                "donor_predictions_path": str(donor_predictions_path.resolve()),
            },
        )
        sparse_summary_path = _write_json(
            root / "sparse_summary.json",
            {
                "task_eval": {"predictions_path": str(sparse_predictions_path.resolve())},
                "control_eval": {
                    "base_predictions_path": str(base_control_predictions.resolve()),
                    "candidate_predictions_path": str(sparse_control_predictions.resolve()),
                },
            },
        )
        dense_summary_path = _write_json(
            root / "dense_summary.json",
            {
                "task_eval": {"predictions_path": str(dense_predictions_path.resolve())},
                "control_eval": {
                    "base_predictions_path": str(base_control_predictions.resolve()),
                    "candidate_predictions_path": str(dense_control_predictions.resolve()),
                },
            },
        )
        steering_summary_path = _write_json(
            root / "steering_summary.json",
            {
                "task_eval": {"predictions_path": str(steering_predictions_path.resolve())},
                "control_eval": {
                    "base_predictions_path": str(base_control_predictions.resolve()),
                    "candidate_predictions_path": str(dense_control_predictions.resolve()),
                },
            },
        )
        shortcut_summary_path = _write_json(
            root / "shortcut_summary.json",
            {
                "dense_control": {
                    "frozen_eval": {"summary_path": str(dense_summary_path.resolve())}
                },
                "steering_control": {
                    "frozen_eval": {"summary_path": str(steering_summary_path.resolve())}
                },
            },
        )

        summary_path = write_error_analysis_report(
            config={
                "paper_artifacts": {
                    "baseline_summary_path": str(baseline_summary_path),
                    "eval_manifest_path": str(eval_manifest_path),
                    "control_manifest_path": str(control_manifest_path),
                    "sparse_selected_eval_summary_path": str(sparse_summary_path),
                    "shortcut_summary_path": str(shortcut_summary_path),
                },
                "error_analysis": {"max_examples_per_bucket": 2},
            },
            output_dir=root / "report",
        )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        strict_table = json.loads(
            Path(summary["strict_vs_semantic_table_path"]).read_text(encoding="utf-8")
        )
        error_table = json.loads(
            Path(summary["error_category_table_path"]).read_text(encoding="utf-8")
        )
        appendix_examples = json.loads(
            Path(summary["appendix_examples_path"]).read_text(encoding="utf-8")
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["prompt_contract_version"], "fc_v1")
        self.assertGreater(len(strict_table["rows"]), 0)
        self.assertGreater(len(error_table["rows"]), 0)
        self.assertEqual(len(appendix_examples["dense_beats_sparse"]), 1)
        self.assertEqual(len(appendix_examples["sparse_beats_dense"]), 2)
        self.assertEqual(summary["control_damage_counts"]["sparse"], 1)


if __name__ == "__main__":
    unittest.main()
