from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.analysis.param_budget import (
    choose_lora_rank_for_budget,
    sparse_same_size_params,
    stitch_pair_params,
    write_budget_report,
)


class ParamBudgetTests(unittest.TestCase):
    def test_sparse_same_size_formula_matches_locked_spec(self) -> None:
        self.assertEqual(
            sparse_same_size_params(hidden_size=2048, bottleneck_size=512, layer_count=1),
            2097665,
        )

    def test_stitch_pair_formula_counts_both_maps(self) -> None:
        self.assertEqual(
            stitch_pair_params(donor_dim=2048, recipient_dim=1536, rank=64, pair_count=1),
            98304 + 131072 + 2048 + 131072 + 98304 + 1536,
        )

    def test_lora_rank_choice_returns_nearest_rank(self) -> None:
        choice = choose_lora_rank_for_budget(
            target_params=1_000_000,
            num_layers=4,
            hidden_size=512,
            intermediate_size=2048,
            rank_grid=[1, 2, 4, 8, 16],
        )
        self.assertIn(choice.selected_rank, {8, 16})

    def test_budget_report_writer_outputs_json(self) -> None:
        output_dir = Path("tests/_tmp/param_budget")
        path = write_budget_report(
            config={
                "param_budget": {
                    "sparse_same_size": {
                        "hidden_size": 2048,
                        "bottleneck_size": 512,
                        "layer_count": 2,
                    },
                    "dense_match": {"mlp_hidden_size": 512},
                    "stitch_pair": {
                        "donor_dim": 2048,
                        "recipient_dim": 1536,
                        "rank": 64,
                        "pair_count": 2,
                    },
                    "lora": {
                        "num_layers": 26,
                        "hidden_size": 2048,
                        "intermediate_size": 16384,
                        "rank_grid": [1, 2, 4, 8, 16, 32],
                    },
                }
            },
            output_dir=output_dir,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("sparse_same_size_params", payload)
        self.assertIn("lora_budget_choice", payload)


if __name__ == "__main__":
    unittest.main()
