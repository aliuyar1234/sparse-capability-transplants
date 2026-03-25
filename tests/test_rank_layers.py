from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.analysis.rank_layers import build_layer_ranking_report
from src.train.train_delta_module import SparseDeltaModule


def _write_cache_manifest(
    *,
    root: Path,
    layer_payloads: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> Path:
    chunk_dir = root / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_records = []
    per_layer_row_counts = {}
    for layer_id, (x_b, target_delta) in sorted(layer_payloads.items()):
        metadata = [
            {
                "example_id": f"ex_{row_index // 2:04d}",
                "token_index": int(row_index % 2),
                "token_class": "tool",
                "layer_id": int(layer_id),
                "split": "train",
                "variant": "canonical",
                "cache_version": "test_cache_v1",
            }
            for row_index in range(x_b.shape[0])
        ]
        chunk_path = chunk_dir / f"layer_{layer_id:02d}_chunk_0000.pt"
        torch.save(
            {
                "layer_id": int(layer_id),
                "cache_version": "test_cache_v1",
                "row_count": int(x_b.shape[0]),
                "metadata": metadata,
                "token_class_counts": {"tool": int(x_b.shape[0])},
                "x_b": x_b.to(dtype=torch.float32),
                "u_b": torch.zeros_like(target_delta, dtype=torch.float32),
                "u_d": target_delta.to(dtype=torch.float32),
            },
            chunk_path,
        )
        chunk_records.append(
            {
                "chunk_index": 0,
                "layer_id": int(layer_id),
                "path": str(chunk_path.resolve()),
                "row_count": int(x_b.shape[0]),
                "token_class_counts": {"tool": int(x_b.shape[0])},
            }
        )
        per_layer_row_counts[str(layer_id)] = int(x_b.shape[0])

    manifest_path = root / "cache_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cache_version": "test_cache_v1",
                "hook_library": "torch_forward_hooks",
                "source_manifest_id": "manifest_test",
                "source_manifest_hash": "abc123",
                "layer_ids": [int(layer_id) for layer_id in sorted(layer_payloads)],
                "selected_token_classes": ["tool"],
                "per_layer_row_counts": per_layer_row_counts,
                "chunk_records": chunk_records,
                "summary_path": str((root / "summary.json").resolve()),
                "hook_audit_path": str((root / "hook_audit.json").resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


class RankLayersTests(unittest.TestCase):
    def test_build_layer_ranking_report_orders_layers_by_proxy_score(self) -> None:
        root = Path("tests/_tmp/rank_layers")
        root.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(1)
        true_module = SparseDeltaModule(input_dim=4, latent_width=4, topk=1)
        with torch.no_grad():
            true_module.encoder.weight.zero_()
            true_module.encoder.bias.zero_()
            true_module.encoder.weight[0, 0] = 1.7
            true_module.decoder.weight.zero_()
            true_module.decoder.weight[0, 0] = 1.4

        x_easy = torch.randn(128, 4)
        y_easy, _ = true_module(x_easy)
        x_hard = torch.randn(128, 4)
        y_hard = torch.randn(128, 4) * 0.5
        manifest_path = _write_cache_manifest(
            root=root / "cache",
            layer_payloads={3: (x_easy, y_easy), 7: (x_hard, y_hard)},
        )

        report = build_layer_ranking_report(
            config={
                "seed": 19,
                "layer_scan": {
                    "cache_manifest_path": str(manifest_path),
                    "latent_width": 4,
                    "topk_values": [1, 2],
                    "validation_fraction": 0.2,
                    "batch_size": 32,
                    "epochs": 50,
                    "learning_rate": 0.02,
                    "lambda_act": 0.0,
                    "lambda_dec": 0.0,
                    "device": "cpu",
                    "heartbeat_interval_seconds": 0.0,
                    "feature_report_limit": 8,
                    "max_feature_stats_rows": 64,
                },
            },
            output_dir=root / "report",
        )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["ranking_mode"], "reconstruction_proxy")
        self.assertFalse(report["claim_bearing"])
        self.assertEqual(report["layer_rankings"][0]["layer_id"], 3)
        self.assertEqual(report["fit_count"], 4)


if __name__ == "__main__":
    unittest.main()
