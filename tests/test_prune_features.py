from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.analysis.prune_features import write_pruned_feature_report


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class PruneFeaturesTests(unittest.TestCase):
    def test_write_pruned_feature_report_uses_capped_calibration_before_frozen_eval(self) -> None:
        root = Path("tests/_tmp/prune_features")
        root.mkdir(parents=True, exist_ok=True)

        checkpoint_path = root / "module_checkpoint.pt"
        checkpoint_path.write_bytes(b"checkpoint")
        feature_stats_path = _write_json(
            root / "feature_stats.json",
            {
                "top_features": [
                    {"feature_id": 0},
                    {"feature_id": 1},
                    {"feature_id": 2},
                    {"feature_id": 3},
                ]
            },
        )
        fit_summary_path = _write_json(
            root / "fit_summary.json",
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "layer_id": 12,
                "latent_width": 4,
                "topk": 2,
                "feature_stats_path": str(feature_stats_path.resolve()),
                "cache_manifest_path": str((root / "cache_manifest.json").resolve()),
            },
        )
        same_size_summary_path = _write_json(
            root / "same_size_summary.json",
            {
                "candidate": {
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "layer_id": 12,
                    "position_policy": "last_token_only",
                },
                "selected_result": {"gain": 1.25},
                "calibration_bundle": {
                    "primary_manifest": {
                        "manifest_path": str((root / "calibration_primary_manifest.json").resolve())
                    },
                    "control_manifest": {
                        "manifest_path": str((root / "calibration_control_manifest.json").resolve())
                    },
                },
            },
        )
        baseline_summary_path = _write_json(
            root / "baseline_summary.json",
            {"primary_metric": {"base_value": 0.10, "donor_value": 0.30}},
        )
        selected_eval_summary_path = _write_json(
            root / "selected_eval_summary.json",
            {
                "baseline_reference": {"primary_metric": {"base_value": 0.10, "donor_value": 0.30}},
                "control_eval": {
                    "base_predictions_path": str((root / "base_predictions.jsonl").resolve()),
                    "base_summary_path": str((root / "base_summary.json").resolve()),
                    "base_metrics_path": str((root / "base_metrics.json").resolve()),
                    "control_drop": -0.01,
                },
                "task_eval": {"grouped_metrics": {"primary": {"strict_full_call_success": 0.26}}},
            },
        )

        seen_candidate_configs: list[dict[str, object]] = []

        def fake_write_layer_candidate_summary(*, config: dict, output_dir: str | Path) -> Path:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            candidate_config = dict(config["candidate_eval"])
            seen_candidate_configs.append(candidate_config)
            feature_ids = tuple(candidate_config.get("feature_ids", []))
            location = str(output_path)
            is_frozen = "evaluation_slices" in str(candidate_config["eval_manifest_path"])

            if float(candidate_config.get("gain", 1.25)) == 0.0:
                primary = 0.10
            elif "single_feature_ablation" in location:
                primary = {
                    (1, 2, 3): 0.13,
                    (0, 2, 3): 0.20,
                    (0, 1, 3): 0.19,
                    (0, 1, 2): 0.23,
                }[feature_ids]
            elif "selection\\greedy" in location or "selection/greedy" in location:
                primary = {
                    (0,): 0.18,
                    (1,): 0.13,
                    (2,): 0.15,
                    (3,): 0.11,
                    (0, 1): 0.19,
                    (0, 2): 0.205,
                    (0, 3): 0.17,
                    (1, 2): 0.16,
                    (1, 3): 0.12,
                    (2, 3): 0.14,
                    (0, 1, 2): 0.203,
                    (0, 1, 3): 0.192,
                    (0, 2, 3): 0.199,
                }[feature_ids]
            elif is_frozen:
                primary = {
                    (0, 2): 0.22,
                    (1, 2): 0.15,
                    (1, 3): 0.12,
                }[feature_ids]
            else:
                primary = 0.23

            summary_path = output_path / "layer_candidate_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "task_eval": {
                            "grouped_metrics": {"primary": {"strict_full_call_success": primary}}
                        },
                        "control_eval": {
                            "control_drop": 0.0,
                            "candidate_exact_match_average": 0.01,
                            "base_predictions_path": str(
                                (output_path / "base_control_predictions.jsonl").resolve()
                            ),
                            "base_summary_path": str(
                                (output_path / "base_control_summary.json").resolve()
                            ),
                            "base_metrics_path": str(
                                (output_path / "base_control_metrics.json").resolve()
                            ),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return summary_path

        with patch(
            "src.analysis.prune_features.write_layer_candidate_summary",
            side_effect=fake_write_layer_candidate_summary,
        ):
            with patch(
                "src.analysis.prune_features.load_layer_training_data",
                return_value=SimpleNamespace(
                    val_indices=torch.tensor([0, 1], dtype=torch.long),
                    x_b=torch.zeros((2, 1), dtype=torch.float32),
                    metadata=[
                        {"token_class": "tool"},
                        {"token_class": "argval"},
                    ],
                ),
            ):
                with patch(
                    "src.analysis.prune_features._layer_module_from_fit_summary",
                    return_value=object(),
                ):
                    with patch(
                        "src.analysis.prune_features._activation_profiles_for_features",
                        return_value=(
                            {
                                0: [1.0, 0.0],
                                1: [0.0, 1.0],
                                2: [0.5, 0.5],
                                3: [0.2, 0.1],
                            },
                            {
                                0: {"act_target": 1.0, "act_format": 0.1},
                                1: {"act_target": 0.6, "act_format": 0.4},
                                2: {"act_target": 0.7, "act_format": 0.2},
                                3: {"act_target": 0.2, "act_format": 0.2},
                            },
                        ),
                    ):
                        summary_path = write_pruned_feature_report(
                            config={
                                "model": {"loader": "transformers", "id": "dummy"},
                                "prune_features": {
                                    "same_size_summary_path": str(same_size_summary_path),
                                    "fit_summary_path": str(fit_summary_path),
                                    "selected_eval_summary_path": str(selected_eval_summary_path),
                                    "baseline_summary_path": str(baseline_summary_path),
                                    "frozen_eval_manifest_path": str(
                                        (root / "evaluation_slices" / "manifest.json").resolve()
                                    ),
                                    "frozen_control_manifest_path": str(
                                        (root / "control_suite" / "manifest.json").resolve()
                                    ),
                                    "shortlist_size": 4,
                                    "max_selected_features": 3,
                                    "selection_max_examples": 2,
                                    "selection_control_max_examples": 1,
                                    "random_subset_count": 2,
                                    "random_seed": 17,
                                },
                            },
                            output_dir=root / "report",
                        )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["selected_subset"]["feature_ids"], [0, 2])
        self.assertAlmostEqual(summary["selected_subset"]["retained_gain_fraction_vs_full"], 0.75)
        self.assertEqual(summary["random_subset_controls"]["count"], 2)
        self.assertTrue(
            summary["selected_subset"]["summary_path"].endswith("layer_candidate_summary.json")
        )

        calibration_calls = [
            call
            for call in seen_candidate_configs
            if str(call["eval_manifest_path"]).endswith("calibration_primary_manifest.json")
        ]
        self.assertGreaterEqual(len(calibration_calls), 2)
        self.assertEqual(calibration_calls[0]["max_examples"], 2)
        self.assertEqual(calibration_calls[0]["control_max_examples"], 1)
        self.assertEqual(calibration_calls[1]["max_examples"], 2)
        self.assertEqual(calibration_calls[1]["control_max_examples"], 1)

        frozen_calls = [
            call
            for call in seen_candidate_configs
            if "evaluation_slices" in str(call["eval_manifest_path"])
        ]
        self.assertGreaterEqual(len(frozen_calls), 3)
        self.assertTrue(all("max_examples" not in call for call in frozen_calls))
        self.assertTrue(all("control_max_examples" not in call for call in frozen_calls))


if __name__ == "__main__":
    unittest.main()
