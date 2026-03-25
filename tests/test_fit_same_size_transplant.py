from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.param_budget import sparse_same_size_params
from src.data.canonical import ArgSpec, CanonicalExample, ToolSpec
from src.data.manifest import load_manifest_payload, write_manifest
from src.eval.run_control_eval import ControlEvalArtifacts
from src.eval.run_eval import EvalArtifacts
from src.train.fit_same_size_transplant import run_same_size_fit_pipeline


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


def _tool() -> ToolSpec:
    return ToolSpec(
        tool_id="show_map",
        name="show_map",
        description="Show a place on a map.",
        arguments=[
            ArgSpec(
                name="query",
                type="string",
                required=True,
                description="Location query.",
            )
        ],
    )


def _canonical_example(*, example_id: str, split: str, user_request: str) -> CanonicalExample:
    return CanonicalExample(
        example_id=example_id,
        split=split,
        user_request=user_request,
        tools=[_tool()],
        gold={"name": "show_map", "arguments": {"query": "Vienna"}},
        meta={"source": "test", "variant": "canonical", "raw_split": split},
    )


def _score_row(*, example_id: str, variant: str, strict: bool) -> dict:
    return {
        "example_id": example_id,
        "variant": variant,
        "score": {
            "example_id": example_id,
            "parse_status": "ok",
            "json_valid": True,
            "strict_correct": strict,
            "semantic_correct": strict,
            "strict_error": None if strict else "wrong_tool",
            "semantic_error": None if strict else "wrong_tool",
            "predicted_name": "show_map" if strict else "NO_TOOL",
            "semantic_predicted_name": "show_map" if strict else "NO_TOOL",
            "gold_name": "show_map",
            "semantic_gold_name": "show_map",
            "arg_exact_match": strict,
            "is_nocall_example": False,
            "predicted_is_nocall": not strict,
        },
    }


def _nocall_row(*, example_id: str, variant: str, strict: bool) -> dict:
    predicted_name = "NO_TOOL" if strict else "show_map"
    return {
        "example_id": example_id,
        "variant": variant,
        "score": {
            "example_id": example_id,
            "parse_status": "ok",
            "json_valid": True,
            "strict_correct": strict,
            "semantic_correct": strict,
            "strict_error": None if strict else "wrong_nocall_decision",
            "semantic_error": None if strict else "wrong_nocall_decision",
            "predicted_name": predicted_name,
            "semantic_predicted_name": predicted_name,
            "gold_name": "NO_TOOL",
            "semantic_gold_name": "NO_TOOL",
            "arg_exact_match": strict,
            "is_nocall_example": True,
            "predicted_is_nocall": strict,
        },
    }


def _control_row(*, example_id: str, exact_match: bool) -> dict:
    return {
        "example_id": example_id,
        "score": {
            "example_id": example_id,
            "exact_match": exact_match,
            "normalized_prediction": "ok" if exact_match else "bad",
            "normalized_target": "ok",
        },
    }


class FitSameSizeTransplantTests(unittest.TestCase):
    def test_run_same_size_fit_pipeline_builds_calibration_bundle_and_selects_best_gain(
        self,
    ) -> None:
        root = Path("tests/_tmp/fit_same_size_transplant")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = _write_json(
            root / "fit_summary.json",
            {
                "checkpoint_path": str((root / "module_checkpoint.pt").resolve()),
                "layer_id": 12,
                "cache_version": "cache_v1",
                "topk": 2,
                "latent_width": 4,
                "input_dim": 8,
            },
        )
        canonical_manifest = write_manifest(
            examples=[
                _canonical_example(
                    example_id="ex_calib_a",
                    split="calib",
                    user_request="Show me Vienna on the map",
                ),
                _canonical_example(
                    example_id="ex_calib_b",
                    split="calib",
                    user_request="Open a map for Graz",
                ),
                _canonical_example(
                    example_id="ex_eval_a",
                    split="eval",
                    user_request="Show Linz on the map",
                ),
            ],
            output_dir=root / "canonical_manifest",
            manifest_id="manifest_test_canonical",
            prompt_contract_version="fc_v1",
            metadata={"split_manifest_hash": "split_hash_test"},
        )

        gain0_predictions = _write_jsonl(
            root / "gain_0" / "task_eval" / "predictions.jsonl",
            [
                _score_row(example_id="schema_1", variant="schema_shift", strict=False),
                _nocall_row(
                    example_id="nocall_1",
                    variant="nocall_missing_tool",
                    strict=False,
                ),
            ],
        )
        gain05_predictions = _write_jsonl(
            root / "gain_05" / "task_eval" / "predictions.jsonl",
            [
                _score_row(example_id="schema_1", variant="schema_shift", strict=True),
                _nocall_row(
                    example_id="nocall_1",
                    variant="nocall_missing_tool",
                    strict=True,
                ),
            ],
        )
        gain0_control = _write_jsonl(
            root / "gain_0" / "control_eval" / "predictions.jsonl",
            [
                _control_row(example_id="ctrl_1", exact_match=True),
                _control_row(example_id="ctrl_2", exact_match=True),
            ],
        )
        gain05_control = _write_jsonl(
            root / "gain_05" / "control_eval" / "predictions.jsonl",
            [
                _control_row(example_id="ctrl_1", exact_match=True),
                _control_row(example_id="ctrl_2", exact_match=True),
            ],
        )
        for path in (
            root / "gain_0" / "task_eval" / "summary.json",
            root / "gain_0" / "task_eval" / "metrics.json",
            root / "gain_05" / "task_eval" / "summary.json",
            root / "gain_05" / "task_eval" / "metrics.json",
            root / "gain_0" / "control_eval" / "summary.json",
            root / "gain_0" / "control_eval" / "metrics.json",
            root / "gain_05" / "control_eval" / "summary.json",
            root / "gain_05" / "control_eval" / "metrics.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        with patch(
            "src.train.fit_same_size_transplant.run_eval_pipeline",
            side_effect=[
                EvalArtifacts(
                    summary_path=str((root / "gain_0" / "task_eval" / "summary.json").resolve()),
                    metrics_path=str((root / "gain_0" / "task_eval" / "metrics.json").resolve()),
                    predictions_path=str(gain0_predictions.resolve()),
                ),
                EvalArtifacts(
                    summary_path=str((root / "gain_05" / "task_eval" / "summary.json").resolve()),
                    metrics_path=str((root / "gain_05" / "task_eval" / "metrics.json").resolve()),
                    predictions_path=str(gain05_predictions.resolve()),
                ),
            ],
        ):
            with patch(
                "src.train.fit_same_size_transplant.run_control_eval_pipeline",
                side_effect=[
                    ControlEvalArtifacts(
                        summary_path=str(
                            (root / "gain_0" / "control_eval" / "summary.json").resolve()
                        ),
                        metrics_path=str(
                            (root / "gain_0" / "control_eval" / "metrics.json").resolve()
                        ),
                        predictions_path=str(gain0_control.resolve()),
                    ),
                    ControlEvalArtifacts(
                        summary_path=str(
                            (root / "gain_05" / "control_eval" / "summary.json").resolve()
                        ),
                        metrics_path=str(
                            (root / "gain_05" / "control_eval" / "metrics.json").resolve()
                        ),
                        predictions_path=str(gain05_control.resolve()),
                    ),
                ],
            ):
                artifacts = run_same_size_fit_pipeline(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "same_size": {
                            "fit_summary_path": str(fit_summary_path),
                            "canonical_manifest_path": str(canonical_manifest.manifest_path),
                            "gain_grid": [0.0, 0.5],
                        },
                    },
                    output_dir=root / "report",
                )

        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        calibration_manifest_payload = load_manifest_payload(artifacts.calibration_manifest_path)
        self.assertEqual(calibration_manifest_payload["example_count"], 6)
        self.assertEqual(summary["calibration_bundle"]["counts"]["control_examples"], 2)
        self.assertEqual(summary["selected_result"]["gain"], 0.5)
        self.assertTrue(Path(artifacts.checkpoint_path).exists())
        self.assertEqual(
            summary["parameter_budget"]["added_params"],
            sparse_same_size_params(hidden_size=8, bottleneck_size=4, layer_count=1),
        )

    def test_run_same_size_fit_pipeline_records_donor_gap_recovery_when_donor_model_is_provided(
        self,
    ) -> None:
        root = Path("tests/_tmp/fit_same_size_transplant_donor")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = _write_json(
            root / "fit_summary.json",
            {
                "checkpoint_path": str((root / "module_checkpoint.pt").resolve()),
                "layer_id": 12,
                "cache_version": "cache_v1",
                "topk": 2,
                "latent_width": 4,
                "input_dim": 8,
            },
        )
        canonical_manifest = write_manifest(
            examples=[
                _canonical_example(
                    example_id="ex_calib_a",
                    split="calib",
                    user_request="Show Vienna on the map",
                ),
                _canonical_example(
                    example_id="ex_calib_b",
                    split="calib",
                    user_request="Open a map for Salzburg",
                ),
            ],
            output_dir=root / "canonical_manifest",
            manifest_id="manifest_test_canonical",
            prompt_contract_version="fc_v1",
            metadata={"split_manifest_hash": "split_hash_test"},
        )

        donor_predictions = _write_jsonl(
            root / "donor" / "predictions.jsonl",
            [
                _score_row(example_id="schema_1", variant="schema_shift", strict=True),
                _nocall_row(
                    example_id="nocall_1",
                    variant="nocall_missing_tool",
                    strict=True,
                ),
            ],
        )
        gain0_predictions = _write_jsonl(
            root / "gain_0" / "task_eval" / "predictions.jsonl",
            [
                _score_row(example_id="schema_1", variant="schema_shift", strict=False),
                _nocall_row(
                    example_id="nocall_1",
                    variant="nocall_missing_tool",
                    strict=True,
                ),
            ],
        )
        gain1_predictions = _write_jsonl(
            root / "gain_1" / "task_eval" / "predictions.jsonl",
            [
                _score_row(example_id="schema_1", variant="schema_shift", strict=True),
                _nocall_row(
                    example_id="nocall_1",
                    variant="nocall_missing_tool",
                    strict=True,
                ),
            ],
        )
        gain0_control = _write_jsonl(
            root / "gain_0" / "control_eval" / "predictions.jsonl",
            [_control_row(example_id="ctrl_1", exact_match=True)],
        )
        gain1_control = _write_jsonl(
            root / "gain_1" / "control_eval" / "predictions.jsonl",
            [_control_row(example_id="ctrl_1", exact_match=True)],
        )
        for path in (
            root / "donor" / "summary.json",
            root / "donor" / "metrics.json",
            root / "gain_0" / "task_eval" / "summary.json",
            root / "gain_0" / "task_eval" / "metrics.json",
            root / "gain_1" / "task_eval" / "summary.json",
            root / "gain_1" / "task_eval" / "metrics.json",
            root / "gain_0" / "control_eval" / "summary.json",
            root / "gain_0" / "control_eval" / "metrics.json",
            root / "gain_1" / "control_eval" / "summary.json",
            root / "gain_1" / "control_eval" / "metrics.json",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        with patch(
            "src.train.fit_same_size_transplant.run_eval_pipeline",
            side_effect=[
                EvalArtifacts(
                    summary_path=str((root / "donor" / "summary.json").resolve()),
                    metrics_path=str((root / "donor" / "metrics.json").resolve()),
                    predictions_path=str(donor_predictions.resolve()),
                ),
                EvalArtifacts(
                    summary_path=str((root / "gain_0" / "task_eval" / "summary.json").resolve()),
                    metrics_path=str((root / "gain_0" / "task_eval" / "metrics.json").resolve()),
                    predictions_path=str(gain0_predictions.resolve()),
                ),
                EvalArtifacts(
                    summary_path=str((root / "gain_1" / "task_eval" / "summary.json").resolve()),
                    metrics_path=str((root / "gain_1" / "task_eval" / "metrics.json").resolve()),
                    predictions_path=str(gain1_predictions.resolve()),
                ),
            ],
        ):
            with patch(
                "src.train.fit_same_size_transplant.run_control_eval_pipeline",
                side_effect=[
                    ControlEvalArtifacts(
                        summary_path=str(
                            (root / "gain_0" / "control_eval" / "summary.json").resolve()
                        ),
                        metrics_path=str(
                            (root / "gain_0" / "control_eval" / "metrics.json").resolve()
                        ),
                        predictions_path=str(gain0_control.resolve()),
                    ),
                    ControlEvalArtifacts(
                        summary_path=str(
                            (root / "gain_1" / "control_eval" / "summary.json").resolve()
                        ),
                        metrics_path=str(
                            (root / "gain_1" / "control_eval" / "metrics.json").resolve()
                        ),
                        predictions_path=str(gain1_control.resolve()),
                    ),
                ],
            ):
                artifacts = run_same_size_fit_pipeline(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "donor_model": {"loader": "transformers", "id": "dummy_donor"},
                        "same_size": {
                            "fit_summary_path": str(fit_summary_path),
                            "canonical_manifest_path": str(canonical_manifest.manifest_path),
                            "gain_grid": [0.0, 1.0],
                        },
                    },
                    output_dir=root / "report",
                )

        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        self.assertIsNotNone(summary["donor_reference"])
        self.assertAlmostEqual(summary["base_reference"]["primary_strict_full_call_success"], 0.5)
        self.assertAlmostEqual(summary["selected_result"]["donor_gap_recovery"], 1.0)


if __name__ == "__main__":
    unittest.main()
