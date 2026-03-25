from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.eval_layer_candidate import build_layer_candidate_summary
from src.eval.run_control_eval import ControlEvalArtifacts
from src.eval.run_eval import EvalArtifacts


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


class EvalLayerCandidateTests(unittest.TestCase):
    def test_build_layer_candidate_summary_computes_recovery_and_control_drop(self) -> None:
        root = Path("tests/_tmp/eval_layer_candidate")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = root / "fit_summary.json"
        fit_summary_path.write_text(
            json.dumps(
                {
                    "checkpoint_path": str((root / "module_checkpoint.pt").resolve()),
                    "layer_id": 6,
                    "topk": 8,
                    "latent_width": 256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        baseline_summary_path = root / "baseline_summary.json"
        baseline_summary_path.write_text(
            json.dumps(
                {
                    "primary_metric": {
                        "base_value": 0.10,
                        "donor_value": 0.30,
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        candidate_predictions_path = _write_jsonl(
            root / "candidate_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": {
                        "example_id": "ex_schema",
                        "parse_status": "ok",
                        "json_valid": True,
                        "strict_correct": True,
                        "semantic_correct": True,
                        "strict_error": None,
                        "semantic_error": None,
                        "predicted_name": "show_map",
                        "semantic_predicted_name": "show_map",
                        "gold_name": "show_map",
                        "semantic_gold_name": "show_map",
                        "arg_exact_match": True,
                        "is_nocall_example": False,
                        "predicted_is_nocall": False,
                    },
                },
                {
                    "example_id": "ex_nocall",
                    "variant": "nocall_missing_tool",
                    "score": {
                        "example_id": "ex_nocall",
                        "parse_status": "ok",
                        "json_valid": True,
                        "strict_correct": False,
                        "semantic_correct": False,
                        "strict_error": "wrong_nocall_decision",
                        "semantic_error": "wrong_nocall_decision",
                        "predicted_name": "show_map",
                        "semantic_predicted_name": "show_map",
                        "gold_name": "NO_TOOL",
                        "semantic_gold_name": "NO_TOOL",
                        "arg_exact_match": False,
                        "is_nocall_example": True,
                        "predicted_is_nocall": False,
                    },
                },
            ],
        )
        candidate_metrics_path = root / "candidate_eval" / "metrics.json"
        candidate_metrics_path.write_text("{}", encoding="utf-8")
        candidate_summary_path = root / "candidate_eval" / "summary.json"
        candidate_summary_path.write_text("{}", encoding="utf-8")

        base_control_predictions_path = _write_jsonl(
            root / "base_control_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        candidate_control_predictions_path = _write_jsonl(
            root / "candidate_control_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": False,
                        "normalized_prediction": "bye",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        for path in (
            root / "base_control_eval" / "metrics.json",
            root / "base_control_eval" / "summary.json",
            root / "candidate_control_eval" / "metrics.json",
            root / "candidate_control_eval" / "summary.json",
        ):
            path.write_text("{}", encoding="utf-8")

        with patch(
            "src.analysis.eval_layer_candidate.run_eval_pipeline",
            return_value=EvalArtifacts(
                summary_path=str(candidate_summary_path.resolve()),
                metrics_path=str(candidate_metrics_path.resolve()),
                predictions_path=str(candidate_predictions_path.resolve()),
            ),
        ) as eval_mock:
            with patch(
                "src.analysis.eval_layer_candidate.run_control_eval_pipeline",
                side_effect=[
                    ControlEvalArtifacts(
                        summary_path=str((root / "base_control_eval" / "summary.json").resolve()),
                        metrics_path=str((root / "base_control_eval" / "metrics.json").resolve()),
                        predictions_path=str(base_control_predictions_path.resolve()),
                    ),
                    ControlEvalArtifacts(
                        summary_path=str(
                            (root / "candidate_control_eval" / "summary.json").resolve()
                        ),
                        metrics_path=str(
                            (root / "candidate_control_eval" / "metrics.json").resolve()
                        ),
                        predictions_path=str(candidate_control_predictions_path.resolve()),
                    ),
                ],
            ) as control_eval_mock:
                summary = build_layer_candidate_summary(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "candidate_eval": {
                            "fit_summary_path": str(fit_summary_path),
                            "eval_manifest_path": str(root / "eval_manifest.json"),
                            "control_manifest_path": str(root / "control_manifest.json"),
                            "baseline_summary_path": str(baseline_summary_path),
                            "batch_size": 8,
                            "control_batch_size": 4,
                            "gain": 1.0,
                            "control_drop_tolerance": 0.02,
                        },
                    },
                    output_dir=root / "report",
                )

        self.assertEqual(summary["candidate"]["layer_id"], 6)
        self.assertAlmostEqual(
            summary["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"], 0.5
        )
        self.assertAlmostEqual(summary["donor_gap_recovery"], 2.0)
        self.assertAlmostEqual(summary["control_eval"]["control_drop"], 1.0)
        self.assertEqual(summary["proceed_decision"]["status"], "pass")
        self.assertEqual(eval_mock.call_args.kwargs["config"]["eval"]["batch_size"], 8)
        self.assertEqual(
            control_eval_mock.call_args_list[0].kwargs["config"]["control_eval"]["batch_size"],
            4,
        )
        self.assertEqual(
            control_eval_mock.call_args_list[1].kwargs["config"]["control_eval"]["batch_size"],
            4,
        )

    def test_build_layer_candidate_summary_reuses_base_control_predictions_when_provided(
        self,
    ) -> None:
        root = Path("tests/_tmp/eval_layer_candidate_reuse")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = root / "fit_summary.json"
        fit_summary_path.write_text(
            json.dumps(
                {
                    "checkpoint_path": str((root / "module_checkpoint.pt").resolve()),
                    "layer_id": 16,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        base_control_predictions_path = _write_jsonl(
            root / "shared_base_control" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        candidate_predictions_path = _write_jsonl(
            root / "candidate_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": {
                        "example_id": "ex_schema",
                        "parse_status": "ok",
                        "json_valid": True,
                        "strict_correct": True,
                        "semantic_correct": True,
                        "strict_error": None,
                        "semantic_error": None,
                        "predicted_name": "show_map",
                        "semantic_predicted_name": "show_map",
                        "gold_name": "show_map",
                        "semantic_gold_name": "show_map",
                        "arg_exact_match": True,
                        "is_nocall_example": False,
                        "predicted_is_nocall": False,
                    },
                }
            ],
        )
        candidate_control_predictions_path = _write_jsonl(
            root / "candidate_control_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": False,
                        "normalized_prediction": "bye",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        for path in (
            root / "candidate_eval" / "metrics.json",
            root / "candidate_eval" / "summary.json",
            root / "candidate_control_eval" / "metrics.json",
            root / "candidate_control_eval" / "summary.json",
        ):
            path.write_text("{}", encoding="utf-8")

        with patch(
            "src.analysis.eval_layer_candidate.run_eval_pipeline",
            return_value=EvalArtifacts(
                summary_path=str((root / "candidate_eval" / "summary.json").resolve()),
                metrics_path=str((root / "candidate_eval" / "metrics.json").resolve()),
                predictions_path=str(candidate_predictions_path.resolve()),
            ),
        ) as eval_mock:
            with patch(
                "src.analysis.eval_layer_candidate.run_control_eval_pipeline",
                return_value=ControlEvalArtifacts(
                    summary_path=str((root / "candidate_control_eval" / "summary.json").resolve()),
                    metrics_path=str((root / "candidate_control_eval" / "metrics.json").resolve()),
                    predictions_path=str(candidate_control_predictions_path.resolve()),
                ),
            ) as control_eval_mock:
                summary = build_layer_candidate_summary(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "candidate_eval": {
                            "fit_summary_path": str(fit_summary_path),
                            "eval_manifest_path": str(root / "eval_manifest.json"),
                            "control_manifest_path": str(root / "control_manifest.json"),
                            "base_control_predictions_path": str(base_control_predictions_path),
                            "batch_size": 6,
                            "control_batch_size": 5,
                        },
                    },
                    output_dir=root / "report",
                )

        self.assertEqual(control_eval_mock.call_count, 1)
        self.assertEqual(
            summary["control_eval"]["base_predictions_path"],
            str(base_control_predictions_path.resolve()),
        )
        self.assertEqual(eval_mock.call_args.kwargs["config"]["eval"]["batch_size"], 6)
        self.assertEqual(
            control_eval_mock.call_args.kwargs["config"]["control_eval"]["batch_size"],
            5,
        )

    def test_build_layer_candidate_summary_forwards_feature_subset_and_eval_caps(self) -> None:
        root = Path("tests/_tmp/eval_layer_candidate_feature_subset")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = root / "fit_summary.json"
        fit_summary_path.write_text(
            json.dumps(
                {
                    "checkpoint_path": str((root / "module_checkpoint.pt").resolve()),
                    "layer_id": 12,
                    "topk": 16,
                    "latent_width": 256,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        base_control_predictions_path = _write_jsonl(
            root / "shared_base_control" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        candidate_predictions_path = _write_jsonl(
            root / "candidate_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": {
                        "example_id": "ex_schema",
                        "parse_status": "ok",
                        "json_valid": True,
                        "strict_correct": True,
                        "semantic_correct": True,
                        "strict_error": None,
                        "semantic_error": None,
                        "predicted_name": "show_map",
                        "semantic_predicted_name": "show_map",
                        "gold_name": "show_map",
                        "semantic_gold_name": "show_map",
                        "arg_exact_match": True,
                        "is_nocall_example": False,
                        "predicted_is_nocall": False,
                    },
                }
            ],
        )
        candidate_control_predictions_path = _write_jsonl(
            root / "candidate_control_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        for path in (
            root / "candidate_eval" / "metrics.json",
            root / "candidate_eval" / "summary.json",
            root / "candidate_control_eval" / "metrics.json",
            root / "candidate_control_eval" / "summary.json",
        ):
            path.write_text("{}", encoding="utf-8")

        with patch(
            "src.analysis.eval_layer_candidate.run_eval_pipeline",
            return_value=EvalArtifacts(
                summary_path=str((root / "candidate_eval" / "summary.json").resolve()),
                metrics_path=str((root / "candidate_eval" / "metrics.json").resolve()),
                predictions_path=str(candidate_predictions_path.resolve()),
            ),
        ) as eval_mock:
            with patch(
                "src.analysis.eval_layer_candidate.run_control_eval_pipeline",
                return_value=ControlEvalArtifacts(
                    summary_path=str((root / "candidate_control_eval" / "summary.json").resolve()),
                    metrics_path=str((root / "candidate_control_eval" / "metrics.json").resolve()),
                    predictions_path=str(candidate_control_predictions_path.resolve()),
                ),
            ) as control_eval_mock:
                summary = build_layer_candidate_summary(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "candidate_eval": {
                            "fit_summary_path": str(fit_summary_path),
                            "eval_manifest_path": str(root / "eval_manifest.json"),
                            "control_manifest_path": str(root / "control_manifest.json"),
                            "base_control_predictions_path": str(base_control_predictions_path),
                            "feature_ids": [2, 5, 7],
                            "batch_size": 6,
                            "control_batch_size": 5,
                            "max_examples": 12,
                            "control_max_examples": 9,
                        },
                    },
                    output_dir=root / "report",
                )

        self.assertEqual(summary["candidate"]["feature_ids"], [2, 5, 7])
        self.assertEqual(summary["candidate"]["feature_count"], 3)
        self.assertEqual(
            eval_mock.call_args.kwargs["config"]["eval"]["transplant"]["layers"][0]["feature_ids"],
            [2, 5, 7],
        )
        self.assertEqual(eval_mock.call_args.kwargs["config"]["eval"]["max_examples"], 12)
        self.assertEqual(
            control_eval_mock.call_args.kwargs["config"]["control_eval"]["transplant"]["layers"][0][
                "feature_ids"
            ],
            [2, 5, 7],
        )
        self.assertEqual(
            control_eval_mock.call_args.kwargs["config"]["control_eval"]["max_examples"], 9
        )

    def test_build_layer_candidate_summary_forwards_intervention_kind(self) -> None:
        root = Path("tests/_tmp/eval_layer_candidate_dense_kind")
        root.mkdir(parents=True, exist_ok=True)

        candidate_predictions_path = _write_jsonl(
            root / "candidate_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": {
                        "example_id": "ex_schema",
                        "parse_status": "ok",
                        "json_valid": True,
                        "strict_correct": True,
                        "semantic_correct": True,
                        "strict_error": None,
                        "semantic_error": None,
                        "predicted_name": "show_map",
                        "semantic_predicted_name": "show_map",
                        "gold_name": "show_map",
                        "semantic_gold_name": "show_map",
                        "arg_exact_match": True,
                        "is_nocall_example": False,
                        "predicted_is_nocall": False,
                    },
                }
            ],
        )
        base_control_predictions_path = _write_jsonl(
            root / "shared_base_control" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        candidate_control_predictions_path = _write_jsonl(
            root / "candidate_control_eval" / "predictions.jsonl",
            [
                {
                    "example_id": "ctrl_1",
                    "score": {
                        "example_id": "ctrl_1",
                        "exact_match": True,
                        "normalized_prediction": "hello",
                        "normalized_target": "hello",
                    },
                }
            ],
        )
        for path in (
            root / "candidate_eval" / "metrics.json",
            root / "candidate_eval" / "summary.json",
            root / "candidate_control_eval" / "metrics.json",
            root / "candidate_control_eval" / "summary.json",
        ):
            path.write_text("{}", encoding="utf-8")

        with patch(
            "src.analysis.eval_layer_candidate.run_eval_pipeline",
            return_value=EvalArtifacts(
                summary_path=str((root / "candidate_eval" / "summary.json").resolve()),
                metrics_path=str((root / "candidate_eval" / "metrics.json").resolve()),
                predictions_path=str(candidate_predictions_path.resolve()),
            ),
        ) as eval_mock:
            with patch(
                "src.analysis.eval_layer_candidate.run_control_eval_pipeline",
                return_value=ControlEvalArtifacts(
                    summary_path=str((root / "candidate_control_eval" / "summary.json").resolve()),
                    metrics_path=str((root / "candidate_control_eval" / "metrics.json").resolve()),
                    predictions_path=str(candidate_control_predictions_path.resolve()),
                ),
            ) as control_eval_mock:
                summary = build_layer_candidate_summary(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "candidate_eval": {
                            "checkpoint_path": str((root / "dense_checkpoint.pt").resolve()),
                            "layer_id": 12,
                            "kind": "dense_mlp",
                            "eval_manifest_path": str(root / "eval_manifest.json"),
                            "control_manifest_path": str(root / "control_manifest.json"),
                            "base_control_predictions_path": str(base_control_predictions_path),
                        },
                    },
                    output_dir=root / "report",
                )

        self.assertEqual(summary["candidate"]["kind"], "dense_mlp")
        self.assertEqual(
            eval_mock.call_args.kwargs["config"]["eval"]["transplant"]["layers"][0]["kind"],
            "dense_mlp",
        )
        self.assertEqual(
            control_eval_mock.call_args.kwargs["config"]["control_eval"]["transplant"]["layers"][0][
                "kind"
            ],
            "dense_mlp",
        )


if __name__ == "__main__":
    unittest.main()
