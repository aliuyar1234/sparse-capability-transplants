from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.multiseed_dense_control import write_dense_control_multiseed_report


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class DenseControlMultiseedTests(unittest.TestCase):
    def test_write_dense_control_multiseed_report_reuses_seed17_and_compares_against_sparse(
        self,
    ) -> None:
        root = Path("tests/_tmp/multiseed_dense_control")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = _write_json(
            root / "fit_summary.json",
            {
                "layer_id": 12,
                "topk": 16,
                "latent_width": 256,
                "input_dim": 1152,
                "cache_manifest_path": str((root / "cache_manifest.json").resolve()),
            },
        )
        same_size_summary_path = _write_json(
            root / "same_size_summary.json",
            {
                "candidate": {"layer_id": 12, "position_policy": "last_token_only"},
                "selected_result": {"gain": 1.25},
                "gain_results": [{"gain": 0.0}, {"gain": 1.25}],
                "calibration_bundle": {
                    "primary_manifest": {
                        "manifest_path": str((root / "calib_eval.json").resolve())
                    },
                    "control_manifest": {
                        "manifest_path": str((root / "calib_control.json").resolve())
                    },
                },
            },
        )
        reference_selected_eval_summary_path = _write_json(
            root / "selected_eval_summary.json",
            {
                "control_eval": {
                    "base_predictions_path": str((root / "base_predictions.jsonl").resolve()),
                    "base_summary_path": str((root / "base_summary.json").resolve()),
                    "base_metrics_path": str((root / "base_metrics.json").resolve()),
                    "control_drop": -0.0140625,
                }
            },
        )
        shortcut_summary_path = _write_json(
            root / "shortcut_summary.json",
            {
                "dense_control": {
                    "training_summary_path": str((root / "dense_seed17_train.json").resolve()),
                    "calibration_selected_gain": 1.25,
                    "frozen_eval": {
                        "primary_strict": 0.1854166667,
                        "control_drop": -0.0015625,
                        "donor_gap_recovery": 1.0518518519,
                        "summary_path": str((root / "dense_seed17_frozen.json").resolve()),
                    },
                }
            },
        )
        sparse_multiseed_summary_path = _write_json(
            root / "sparse_multiseed_summary.json",
            {
                "aggregate": {"primary_strict": {"mean": 0.1657986111}},
                "seed_results": [
                    {"seed": 17, "primary_strict": 0.2067708333, "control_drop": -0.0140625},
                    {"seed": 29, "primary_strict": 0.1182291667, "control_drop": -0.0046875},
                    {"seed": 43, "primary_strict": 0.1723958333, "control_drop": 0.0015625},
                ],
            },
        )
        baseline_summary_path = _write_json(
            root / "baseline_summary.json",
            {"primary_metric": {"base_value": 0.0375, "donor_value": 0.178125}},
        )
        dense_seed17_summary_path = _write_json(
            root / "dense_seed17_frozen.json",
            {
                "summary_path": str((root / "dense_seed17_frozen.json").resolve()),
                "baseline_reference": {"primary_metric": {"base_value": 0.0375}},
                "task_eval": {
                    "grouped_metrics": {"primary": {"strict_full_call_success": 0.1854166667}},
                    "summary_path": str((root / "dense_seed17_candidate_eval.json").resolve()),
                },
                "control_eval": {"control_drop": -0.0015625},
                "donor_gap_recovery": 1.0518518519,
            },
        )

        fit_calls: list[int] = []
        gain_sweep_calls: list[int] = []
        frozen_eval_calls: list[int] = []

        def fake_fit_dense_shortcut_control(
            *, fit_summary: dict, shortcut_config: dict, output_dir: Path
        ):
            seed = int(shortcut_config["dense_control"]["train_seed"])
            fit_calls.append(seed)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = output_dir / "summary.json"
            payload = {
                "summary_path": str(summary_path.resolve()),
                "checkpoint_path": str((output_dir / f"checkpoint_seed_{seed}.pt").resolve()),
                "target_sparse_params": 590081,
                "dense_params": 591232,
                "relative_param_diff": 0.0019,
            }
            summary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return payload

        def fake_run_gain_sweep(
            *,
            kind: str,
            checkpoint_path: str,
            layer_id: int,
            position_policy: str,
            gain_grid: list[float],
            model_config: dict,
            same_size_summary: dict,
            baseline_summary_path: str,
            shortcut_config: dict,
            output_dir: Path,
        ):
            seed = int(shortcut_config["dense_control"]["train_seed"])
            gain_sweep_calls.append(seed)
            return [], {"candidate": {"gain": 1.25}}

        def fake_run_candidate_summary(
            *,
            model_config: dict,
            output_dir: Path,
            layer_config: dict,
            eval_manifest_path: str,
            control_manifest_path: str,
            baseline_summary_path: str | None,
            base_control_reference: dict[str, str] | None,
            shortcut_config: dict,
        ):
            seed = int(shortcut_config["dense_control"]["train_seed"])
            frozen_eval_calls.append(seed)
            primary = 0.1015625 if seed == 29 else 0.1234375
            control_drop = -0.003125 if seed == 29 else 0.0
            output_dir.mkdir(parents=True, exist_ok=True)
            return {
                "baseline_reference": {"primary_metric": {"base_value": 0.0375}},
                "task_eval": {
                    "grouped_metrics": {"primary": {"strict_full_call_success": primary}},
                    "summary_path": str((output_dir / "candidate_eval_summary.json").resolve()),
                },
                "control_eval": {
                    "control_drop": control_drop,
                    "base_predictions_path": str(
                        base_control_reference["base_control_predictions_path"]
                    ),
                    "base_summary_path": str(base_control_reference["base_control_summary_path"]),
                    "base_metrics_path": str(base_control_reference["base_control_metrics_path"]),
                },
                "donor_gap_recovery": 0.55 if seed == 29 else 0.68,
            }

        with patch(
            "src.analysis.multiseed_dense_control._fit_dense_shortcut_control",
            side_effect=fake_fit_dense_shortcut_control,
        ):
            with patch(
                "src.analysis.multiseed_dense_control._run_gain_sweep",
                side_effect=fake_run_gain_sweep,
            ):
                with patch(
                    "src.analysis.multiseed_dense_control._run_candidate_summary",
                    side_effect=fake_run_candidate_summary,
                ):
                    summary_path = write_dense_control_multiseed_report(
                        config={
                            "execution_variant": "V24",
                            "model": {"loader": "transformers", "id": "dummy"},
                            "multiseed_dense_control": {
                                "fit_summary_path": str(fit_summary_path),
                                "same_size_summary_path": str(same_size_summary_path),
                                "reference_selected_eval_summary_path": str(
                                    reference_selected_eval_summary_path
                                ),
                                "shortcut_summary_path": str(shortcut_summary_path),
                                "sparse_multiseed_summary_path": str(sparse_multiseed_summary_path),
                                "baseline_summary_path": str(baseline_summary_path),
                                "eval_manifest_path": str((root / "eval_manifest.json").resolve()),
                                "control_manifest_path": str(
                                    (root / "control_manifest.json").resolve()
                                ),
                                "seeds": [17, 29, 43],
                                "reuse_existing_seed_summaries": {
                                    "17": str(dense_seed17_summary_path)
                                },
                                "reuse_existing_training_summaries": {
                                    "17": str((root / "dense_seed17_train.json").resolve())
                                },
                                "dense_control": {
                                    "batch_size": 1024,
                                    "epochs": 4,
                                    "learning_rate": 0.0005,
                                },
                            },
                        },
                        output_dir=root / "report",
                    )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(fit_calls, [29, 43])
        self.assertEqual(gain_sweep_calls, [29, 43])
        self.assertEqual(frozen_eval_calls, [29, 43])
        self.assertEqual(summary["seed_policy"]["evaluated_seeds"], [17, 29, 43])
        self.assertEqual(summary["seed_results"][0]["source"], "reused_existing")
        self.assertTrue(summary["comparison_to_sparse_multiseed"]["sparse_beats_dense_mean"])
        self.assertEqual(summary["confirmatory_decision"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
