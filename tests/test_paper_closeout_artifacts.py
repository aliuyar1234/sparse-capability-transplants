from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.analysis.export_final_registry import export_final_registry
from src.analysis.plot_recovery import write_recovery_artifacts
from src.analysis.plot_tradeoffs import write_tradeoff_artifacts


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


def _score(example_id: str, strict: bool, semantic: bool, variant: str) -> dict:
    return {
        "example_id": example_id,
        "parse_status": "ok",
        "json_valid": True,
        "strict_correct": strict,
        "semantic_correct": semantic,
        "strict_error": None if strict else "wrong_tool",
        "semantic_error": None if semantic else "wrong_tool",
        "predicted_name": "show_map",
        "semantic_predicted_name": "show_map",
        "gold_name": "show_map",
        "semantic_gold_name": "show_map",
        "arg_exact_match": strict and semantic,
        "is_nocall_example": variant.startswith("nocall"),
        "predicted_is_nocall": False,
    }


class PaperCloseoutArtifactTests(unittest.TestCase):
    def test_recovery_tradeoff_and_registry_exports(self) -> None:
        root = Path("tests/_tmp/paper_closeout")
        root.mkdir(parents=True, exist_ok=True)

        eval_examples_path = _write_jsonl(
            root / "eval_examples.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "gold": {"name": "show_map", "arguments": {"query": "museum"}},
                    "tools": [{"tool_id": "show_map", "name": "show_map"}],
                    "user_request": "Show me the museum.",
                },
                {
                    "example_id": "ex_missing",
                    "gold": {"name": "NO_TOOL", "arguments": {}},
                    "tools": [{"tool_id": "show_map", "name": "show_map"}],
                    "user_request": "Use a missing tool.",
                },
                {
                    "example_id": "ex_unsupported",
                    "gold": {"name": "NO_TOOL", "arguments": {}},
                    "tools": [{"tool_id": "show_map", "name": "show_map"}],
                    "user_request": "Write a poem.",
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
        control_manifest_path = _write_json(
            root / "control_manifest.json",
            {"dataset_path": str((root / "control_examples.jsonl").resolve())},
        )
        _write_jsonl(
            root / "control_examples.jsonl",
            [{"example_id": "ctrl_1", "target_text": "hello"}],
        )

        base_predictions = _write_jsonl(
            root / "base_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": _score("ex_schema", False, False, "schema_shift"),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "score": _score("ex_missing", False, False, "nocall_missing_tool"),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "score": _score("ex_unsupported", True, True, "nocall_unsupported"),
                },
            ],
        )
        donor_predictions = _write_jsonl(
            root / "donor_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": _score("ex_schema", True, True, "schema_shift"),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "score": _score("ex_missing", True, True, "nocall_missing_tool"),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "score": _score("ex_unsupported", True, True, "nocall_unsupported"),
                },
            ],
        )
        sparse_predictions = _write_jsonl(
            root / "sparse_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": _score("ex_schema", False, True, "schema_shift"),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "score": _score("ex_missing", True, True, "nocall_missing_tool"),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "score": _score("ex_unsupported", True, True, "nocall_unsupported"),
                },
            ],
        )
        dense_predictions = _write_jsonl(
            root / "dense_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": _score("ex_schema", True, True, "schema_shift"),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "score": _score("ex_missing", True, True, "nocall_missing_tool"),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "score": _score("ex_unsupported", True, True, "nocall_unsupported"),
                },
            ],
        )
        steering_predictions = _write_jsonl(
            root / "steering_predictions.jsonl",
            [
                {
                    "example_id": "ex_schema",
                    "variant": "schema_shift",
                    "score": _score("ex_schema", False, False, "schema_shift"),
                },
                {
                    "example_id": "ex_missing",
                    "variant": "nocall_missing_tool",
                    "score": _score("ex_missing", False, False, "nocall_missing_tool"),
                },
                {
                    "example_id": "ex_unsupported",
                    "variant": "nocall_unsupported",
                    "score": _score("ex_unsupported", True, True, "nocall_unsupported"),
                },
            ],
        )

        baseline_summary_path = _write_json(
            root / "baseline_summary.json",
            {
                "base_predictions_path": str(base_predictions.resolve()),
                "donor_predictions_path": str(donor_predictions.resolve()),
                "primary_metric": {"base_value": 0.04, "donor_value": 0.18},
                "base_metrics": {
                    "primary": {
                        "strict_full_call_success": 0.04,
                        "semantic_full_call_success": 0.08,
                    },
                    "schema_shift": {
                        "strict_full_call_success": 0.0,
                        "semantic_full_call_success": 0.1,
                    },
                },
                "donor_metrics": {
                    "primary": {
                        "strict_full_call_success": 0.18,
                        "semantic_full_call_success": 0.22,
                    },
                    "schema_shift": {
                        "strict_full_call_success": 0.53,
                        "semantic_full_call_success": 0.62,
                    },
                },
            },
        )
        sparse_selected_summary_path = _write_json(
            root / "sparse_selected_summary.json",
            {
                "task_eval": {
                    "predictions_path": str(sparse_predictions.resolve()),
                    "grouped_metrics": {
                        "primary": {
                            "strict_full_call_success": 0.20,
                            "semantic_full_call_success": 0.26,
                        }
                    },
                },
                "control_eval": {"control_drop": -0.01},
                "donor_gap_recovery": 1.2,
            },
        )
        dense_frozen_summary_path = _write_json(
            root / "dense_frozen_summary.json",
            {
                "task_eval": {"predictions_path": str(dense_predictions.resolve())},
                "control_eval": {"control_drop": -0.002},
            },
        )
        steering_frozen_summary_path = _write_json(
            root / "steering_frozen_summary.json",
            {
                "task_eval": {"predictions_path": str(steering_predictions.resolve())},
                "control_eval": {"control_drop": -0.006},
            },
        )

        sparse_gain_sweep_path = _write_json(
            root / "same_size" / "gain_sweep.json",
            {
                "results": [
                    {
                        "gain": 0.0,
                        "task_eval": {
                            "grouped_metrics": {
                                "primary": {
                                    "strict_full_call_success": 0.04,
                                    "semantic_full_call_success": 0.09,
                                }
                            }
                        },
                        "control_eval": {"control_drop": 0.0},
                    },
                    {
                        "gain": 1.25,
                        "task_eval": {
                            "grouped_metrics": {
                                "primary": {
                                    "strict_full_call_success": 0.20,
                                    "semantic_full_call_success": 0.26,
                                }
                            }
                        },
                        "control_eval": {"control_drop": -0.01},
                    },
                ]
            },
        )
        _write_json(
            root / "shortcut" / "dense_control" / "calibration" / "gain_sweep.json",
            {
                "results": [
                    {
                        "gain": 0.0,
                        "task_eval": {
                            "grouped_metrics": {
                                "primary": {
                                    "strict_full_call_success": 0.04,
                                    "semantic_full_call_success": 0.09,
                                }
                            }
                        },
                        "control_eval": {"control_drop": 0.0},
                    },
                    {
                        "gain": 1.25,
                        "task_eval": {
                            "grouped_metrics": {
                                "primary": {
                                    "strict_full_call_success": 0.185,
                                    "semantic_full_call_success": 0.23,
                                }
                            }
                        },
                        "control_eval": {"control_drop": -0.002},
                    },
                ]
            },
        )
        _write_json(
            root / "shortcut" / "steering_control" / "calibration" / "gain_sweep.json",
            {
                "results": [
                    {
                        "gain": 0.0,
                        "task_eval": {
                            "grouped_metrics": {
                                "primary": {
                                    "strict_full_call_success": 0.04,
                                    "semantic_full_call_success": 0.09,
                                }
                            }
                        },
                        "control_eval": {"control_drop": 0.0},
                    },
                    {
                        "gain": 1.25,
                        "task_eval": {
                            "grouped_metrics": {
                                "primary": {
                                    "strict_full_call_success": 0.108,
                                    "semantic_full_call_success": 0.15,
                                }
                            }
                        },
                        "control_eval": {"control_drop": -0.006},
                    },
                ]
            },
        )

        same_size_summary_path = _write_json(
            root / "same_size" / "summary.json",
            {
                "candidate": {"latent_width": 256},
                "parameter_budget": {"added_params": 590081},
                "gain_sweep_path": str(sparse_gain_sweep_path.resolve()),
            },
        )
        prune_summary_path = _write_json(
            root / "prune_summary.json",
            {
                "summary_path": str((root / "prune_summary.json").resolve()),
                "full_same_size_reference": {"primary_strict": 0.2068},
                "selected_subset": {
                    "feature_count": 1,
                    "frozen_primary_strict": 0.1172,
                    "frozen_control_drop": 0.0,
                    "retained_gain_fraction_vs_full": 0.4708,
                },
                "random_subset_controls": {
                    "mean_primary_strict": 0.0566,
                    "mean_control_drop": -0.0008,
                },
            },
        )
        shortcut_summary_path = _write_json(
            root / "shortcut" / "summary.json",
            {
                "summary_path": str((root / "shortcut" / "summary.json").resolve()),
                "dense_control": {
                    "budget": {"dense_params": 591232},
                    "frozen_eval": {
                        "primary_strict": 0.1854,
                        "donor_gap_recovery": 1.0519,
                        "control_drop": -0.0016,
                        "summary_path": str(dense_frozen_summary_path.resolve()),
                    },
                },
                "steering_control": {
                    "frozen_eval": {
                        "primary_strict": 0.1078,
                        "donor_gap_recovery": 0.5,
                        "control_drop": -0.0063,
                        "summary_path": str(steering_frozen_summary_path.resolve()),
                    },
                    "vector_summary_path": str(
                        _write_json(
                            root / "shortcut" / "steering_control" / "vector" / "summary.json",
                            {"input_dim": 1152},
                        ).resolve()
                    ),
                },
            },
        )
        sparse_multiseed_summary_path = _write_json(
            root / "sparse_multiseed_summary.json",
            {
                "aggregate": {
                    "primary_strict": {"mean": 0.1658},
                    "donor_gap_recovery": {"mean": 0.9123},
                    "control_drop": {"mean": -0.0057},
                }
            },
        )
        dense_multiseed_summary_path = _write_json(
            root / "dense_multiseed_summary.json",
            {
                "aggregate": {
                    "primary_strict": {"mean": 0.2177},
                    "donor_gap_recovery": {"mean": 1.2815},
                    "control_drop": {"mean": -0.0109},
                },
                "comparison_to_sparse_multiseed": {
                    "sparse_beats_dense_mean": False,
                    "mean_primary_delta_sparse_minus_dense": -0.0519,
                },
            },
        )

        paper_config = {
            "paper_artifacts": {
                "baseline_summary_path": str(baseline_summary_path),
                "eval_manifest_path": str(eval_manifest_path),
                "control_manifest_path": str(control_manifest_path),
                "same_size_summary_path": str(same_size_summary_path),
                "sparse_selected_eval_summary_path": str(sparse_selected_summary_path),
                "prune_summary_path": str(prune_summary_path),
                "shortcut_summary_path": str(shortcut_summary_path),
                "sparse_multiseed_summary_path": str(sparse_multiseed_summary_path),
                "dense_multiseed_summary_path": str(dense_multiseed_summary_path),
            }
        }

        recovery_summary_path = write_recovery_artifacts(
            config=paper_config,
            output_dir=root / "m7_recovery",
        )
        tradeoff_summary_path = write_tradeoff_artifacts(
            config=paper_config,
            output_dir=root / "m7_tradeoffs",
        )
        error_summary_path = _write_json(
            root / "m7_error_analysis" / "summary.json",
            {
                "semantic_transfer_snapshot": {
                    "base_schema_strict": 0.0,
                    "sparse_schema_strict": 0.0,
                    "base_schema_semantic": 0.1,
                    "sparse_schema_semantic": 0.2,
                }
            },
        )
        claims_matrix_path = _write_json(
            root / "CLAIMS_MATRIX.md",
            {},
        )
        claims_matrix_path.write_text(
            "\n".join(
                [
                    "## C1 - Same-size existence proof",
                    "- **Status**  ",
                    "  partially supported",
                    "## C3 - Semantic transfer",
                    "- **Status**  ",
                    "  intended",
                    "## C5 - Narrowness",
                    "- **Status**  ",
                    "  intended",
                    "## C6 - Sparse subset",
                    "- **Status**  ",
                    "  partially supported",
                    "## C7 - Mechanism matters",
                    "- **Status**  ",
                    "  partially supported",
                    "## C8 - Scope-limited",
                    "- **Status**  ",
                    "  supported by design",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        registry_summary_path = export_final_registry(
            config={
                "execution_variant": "V24",
                **paper_config,
                "final_registry": {
                    "error_analysis_summary_path": str(error_summary_path),
                    "plot_recovery_summary_path": str(recovery_summary_path),
                    "plot_tradeoffs_summary_path": str(tradeoff_summary_path),
                    "claims_matrix_path": str(claims_matrix_path),
                },
            },
            output_dir=root / "m8_registry",
        )

        recovery_summary = json.loads(recovery_summary_path.read_text(encoding="utf-8"))
        tradeoff_summary = json.loads(tradeoff_summary_path.read_text(encoding="utf-8"))
        registry_summary = json.loads(registry_summary_path.read_text(encoding="utf-8"))
        final_claim_audit = json.loads(
            Path(registry_summary["final_claim_audit_path"]).read_text(encoding="utf-8")
        )

        self.assertEqual(recovery_summary["status"], "passed")
        self.assertTrue(Path(recovery_summary["recovery_vs_parameters_figure_path"]).exists())
        self.assertEqual(tradeoff_summary["status"], "passed")
        self.assertTrue(Path(tradeoff_summary["per_slice_figure_path"]).exists())
        self.assertEqual(registry_summary["m5_decision"]["status"], "failure")
        claim_ids = {row["claim_id"] for row in final_claim_audit["claims"]}
        self.assertIn("C1", claim_ids)
        c1_row = next(row for row in final_claim_audit["claims"] if row["claim_id"] == "C1")
        self.assertEqual(c1_row["new_status"], "weakened")


if __name__ == "__main__":
    unittest.main()
