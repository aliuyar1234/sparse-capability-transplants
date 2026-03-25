from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.analysis.shortcut_controls import write_same_size_shortcut_control_report


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


class ShortcutControlTests(unittest.TestCase):
    def test_write_same_size_shortcut_control_report_uses_locked_gain_grid_and_records_controls(
        self,
    ) -> None:
        root = Path("tests/_tmp/shortcut_controls")
        root.mkdir(parents=True, exist_ok=True)

        fit_summary_path = _write_json(
            root / "fit_summary.json",
            {
                "layer_id": 12,
                "input_dim": 8,
                "latent_width": 8,
                "cache_manifest_path": str((root / "cache_manifest.json").resolve()),
            },
        )
        same_size_summary_path = _write_json(
            root / "same_size_summary.json",
            {
                "candidate": {"layer_id": 12, "position_policy": "last_token_only"},
                "selected_result": {"gain": 1.25},
                "gain_results": [{"gain": 0.0}, {"gain": 0.5}],
                "calibration_bundle": {
                    "primary_manifest": {
                        "manifest_path": str((root / "calibration_primary.json").resolve())
                    },
                    "control_manifest": {
                        "manifest_path": str((root / "calibration_control.json").resolve())
                    },
                },
            },
        )
        selected_eval_summary_path = _write_json(
            root / "selected_eval_summary.json",
            {
                "task_eval": {"grouped_metrics": {"primary": {"strict_full_call_success": 0.2}}},
                "control_eval": {
                    "base_predictions_path": str((root / "base_predictions.jsonl").resolve()),
                    "base_summary_path": str((root / "base_summary.json").resolve()),
                    "base_metrics_path": str((root / "base_metrics.json").resolve()),
                    "control_drop": -0.01,
                },
                "donor_gap_recovery": 1.1,
            },
        )
        baseline_summary_path = _write_json(
            root / "baseline_summary.json",
            {"primary_metric": {"base_value": 0.05, "donor_value": 0.15}},
        )
        prune_summary_path = _write_json(
            root / "prune_summary.json",
            {
                "selected_subset": {
                    "feature_ids": [104],
                    "feature_count": 1,
                    "frozen_primary_strict": 0.11,
                    "frozen_control_drop": 0.0,
                }
            },
        )

        def fake_write_layer_candidate_summary(*, config: dict, output_dir: str | Path) -> Path:
            candidate = dict(config["candidate_eval"])
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            gain = float(candidate["gain"])
            kind = str(candidate["kind"])
            is_frozen = "frozen_eval" in str(output_path)
            primary = (
                0.09
                if gain == 0.0
                else (
                    0.16
                    if kind == "dense_mlp" and not is_frozen
                    else 0.13
                    if kind == "steering_vector" and not is_frozen
                    else 0.12
                    if kind == "dense_mlp"
                    else 0.08
                )
            )
            control_drop = 0.0 if kind == "dense_mlp" else 0.005
            summary_path = output_path / "layer_candidate_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "candidate": {"gain": gain, "kind": kind},
                        "task_eval": {
                            "grouped_metrics": {"primary": {"strict_full_call_success": primary}}
                        },
                        "control_eval": {
                            "control_drop": control_drop,
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
                        "validation_objective": {"score": primary},
                        "donor_gap_recovery": 0.7,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return summary_path

        with patch(
            "src.analysis.shortcut_controls.load_layer_training_data",
            return_value=SimpleNamespace(
                layer_id=12,
                cache_version="cache_v1",
                input_dim=8,
                x_b=torch.tensor(
                    [
                        [0.0] * 8,
                        [1.0] + ([0.0] * 7),
                        [0.0, 1.0] + ([0.0] * 6),
                        [1.0, 1.0] + ([0.0] * 6),
                    ],
                    dtype=torch.float32,
                ),
                target_delta=torch.tensor(
                    [
                        [0.0] * 8,
                        [1.0] + ([0.0] * 7),
                        [0.0, 1.0] + ([0.0] * 6),
                        [1.0, 1.0] + ([0.0] * 6),
                    ],
                    dtype=torch.float32,
                ),
                row_weights=torch.ones(4, dtype=torch.float32),
                metadata=[{"example_id": f"ex_{index}"} for index in range(4)],
                train_indices=torch.tensor([0, 1], dtype=torch.long),
                val_indices=torch.tensor([2, 3], dtype=torch.long),
            ),
        ):
            with patch(
                "src.analysis.shortcut_controls.write_layer_candidate_summary",
                side_effect=fake_write_layer_candidate_summary,
            ):
                summary_path = write_same_size_shortcut_control_report(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "shortcut_controls": {
                            "same_size_summary_path": str(same_size_summary_path),
                            "fit_summary_path": str(fit_summary_path),
                            "selected_eval_summary_path": str(selected_eval_summary_path),
                            "baseline_summary_path": str(baseline_summary_path),
                            "prune_summary_path": str(prune_summary_path),
                            "frozen_eval_manifest_path": str((root / "frozen_eval.json").resolve()),
                            "frozen_control_manifest_path": str(
                                (root / "frozen_control.json").resolve()
                            ),
                            "batch_size": 8,
                            "control_batch_size": 4,
                            "layer_training": {"validation_fraction": 0.1, "split_seed": 17},
                            "dense_control": {
                                "batch_size": 2,
                                "epochs": 1,
                                "learning_rate": 0.001,
                                "train_seed": 17,
                            },
                        },
                    },
                    output_dir=root / "report",
                )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["progressive_ablation"]["status"], "not_applicable_single_layer")
        self.assertEqual(summary["dense_control"]["calibration_gain_grid"], [0.0, 0.5])
        self.assertEqual(summary["dense_control"]["calibration_selected_gain"], 0.5)
        self.assertEqual(summary["steering_control"]["calibration_selected_gain"], 0.5)
        self.assertEqual(summary["pruned_subset_reference"]["feature_ids"], [104])
        self.assertGreater(
            summary["dense_control"]["frozen_eval"]["primary_strict"],
            summary["steering_control"]["frozen_eval"]["primary_strict"],
        )


if __name__ == "__main__":
    unittest.main()
