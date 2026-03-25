from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.multiseed_same_size import write_same_size_multiseed_report


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class SameSizeMultiseedTests(unittest.TestCase):
    def test_write_same_size_multiseed_report_reuses_existing_seed_and_aggregates_confirmatory_runs(
        self,
    ) -> None:
        root = Path("tests/_tmp/same_size_multiseed")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = _write_json(
            root / "fit_summary.json",
            {
                "layer_id": 12,
                "topk": 16,
                "latent_width": 256,
                "cache_manifest_path": str((root / "cache_manifest.json").resolve()),
                "checkpoint_path": str((root / "template_checkpoint.pt").resolve()),
            },
        )
        reference_same_size_summary_path = _write_json(
            root / "same_size_summary.json",
            {
                "candidate": {"position_policy": "last_token_only"},
                "selected_result": {"gain": 1.25},
            },
        )
        reference_selected_eval_summary_path = _write_json(
            root / "selected_eval_summary.json",
            {
                "summary_path": str((root / "selected_eval_summary.json").resolve()),
                "baseline_reference": {"primary_metric": {"base_value": 0.0375}},
                "task_eval": {
                    "grouped_metrics": {"primary": {"strict_full_call_success": 0.2067708333}}
                },
                "control_eval": {
                    "base_predictions_path": str(
                        (root / "base_control_predictions.jsonl").resolve()
                    ),
                    "base_summary_path": str((root / "base_control_summary.json").resolve()),
                    "base_metrics_path": str((root / "base_control_metrics.json").resolve()),
                    "control_drop": -0.0140625,
                },
                "donor_gap_recovery": 1.2037037037,
            },
        )
        baseline_summary_path = _write_json(
            root / "baseline_summary.json",
            {
                "primary_metric": {
                    "base_value": 0.0375,
                    "donor_value": 0.178125,
                }
            },
        )
        shortcut_summary_path = _write_json(
            root / "shortcut_summary.json",
            {
                "dense_control": {
                    "frozen_eval": {
                        "primary_strict": 0.1854166667,
                        "control_drop": -0.0015625,
                        "donor_gap_recovery": 1.0518518519,
                    }
                },
                "steering_control": {
                    "frozen_eval": {
                        "primary_strict": 0.1078125,
                        "control_drop": -0.00625,
                        "donor_gap_recovery": 0.5,
                    }
                },
            },
        )

        fit_calls: list[int] = []

        def fake_fit_layer_delta_module(
            *,
            config: dict,
            output_dir: str | Path,
            layer_id: int,
            topk: int,
        ):
            seed = int(config["seed"])
            fit_calls.append(seed)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            summary_path = output_path / "summary.json"
            payload = {
                "status": "passed",
                "summary_path": str(summary_path.resolve()),
                "layer_id": layer_id,
                "topk": topk,
                "latent_width": 256,
                "cache_manifest_path": str((root / "cache_manifest.json").resolve()),
                "checkpoint_path": str((output_path / f"checkpoint_seed_{seed}.pt").resolve()),
            }
            summary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return payload

        def fake_write_layer_candidate_summary(*, config: dict, output_dir: str | Path) -> Path:
            candidate = dict(config["candidate_eval"])
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            seed = 29 if "seed_29" in str(output_path) else 43
            primary = 0.1916666667 if seed == 29 else 0.1885416667
            control_drop = -0.003125 if seed == 29 else -0.0015625
            recovery = 1.095 if seed == 29 else 1.072
            summary_path = output_path / "layer_candidate_summary.json"
            payload = {
                "status": "passed",
                "summary_path": str(summary_path.resolve()),
                "baseline_reference": {"primary_metric": {"base_value": 0.0375}},
                "task_eval": {
                    "grouped_metrics": {"primary": {"strict_full_call_success": primary}},
                    "summary_path": str((output_path / "candidate_eval_summary.json").resolve()),
                },
                "control_eval": {
                    "control_drop": control_drop,
                    "base_predictions_path": str(candidate["base_control_predictions_path"]),
                    "base_summary_path": str(candidate["base_control_summary_path"]),
                    "base_metrics_path": str(candidate["base_control_metrics_path"]),
                },
                "donor_gap_recovery": recovery,
            }
            summary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return summary_path

        with patch(
            "src.analysis.multiseed_same_size.fit_layer_delta_module",
            side_effect=fake_fit_layer_delta_module,
        ):
            with patch(
                "src.analysis.multiseed_same_size.write_layer_candidate_summary",
                side_effect=fake_write_layer_candidate_summary,
            ):
                summary_path = write_same_size_multiseed_report(
                    config={
                        "execution_variant": "V24",
                        "model": {"loader": "transformers", "id": "dummy"},
                        "multiseed_same_size": {
                            "fit_summary_path": str(fit_summary_path),
                            "reference_same_size_summary_path": str(
                                reference_same_size_summary_path
                            ),
                            "reference_selected_eval_summary_path": str(
                                reference_selected_eval_summary_path
                            ),
                            "shortcut_summary_path": str(shortcut_summary_path),
                            "baseline_summary_path": str(baseline_summary_path),
                            "eval_manifest_path": str((root / "eval_manifest.json").resolve()),
                            "control_manifest_path": str(
                                (root / "control_manifest.json").resolve()
                            ),
                            "seeds": [17, 29, 43],
                            "reuse_existing_seed_summaries": {
                                "17": str(reference_selected_eval_summary_path)
                            },
                            "gain": 1.25,
                            "control_drop_tolerance": 0.02,
                            "layer_training": {
                                "validation_fraction": 0.1,
                                "batch_size": 1024,
                                "epochs": 4,
                            },
                        },
                    },
                    output_dir=root / "report",
                )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(fit_calls, [29, 43])
        self.assertEqual(summary["seed_policy"]["evaluated_seeds"], [17, 29, 43])
        self.assertEqual(summary["seed_results"][0]["source"], "reused_existing")
        self.assertEqual(summary["seed_results"][1]["source"], "confirmatory_rerun")
        self.assertEqual(summary["aggregate"]["primary_strict"]["seed_count"], 3)
        self.assertEqual(summary["confirmatory_decision"]["status"], "pass")
        self.assertGreater(
            summary["aggregate"]["primary_strict"]["mean"],
            summary["single_seed_references"]["dense_shortcut_seed17"]["primary_strict"],
        )


if __name__ == "__main__":
    unittest.main()
