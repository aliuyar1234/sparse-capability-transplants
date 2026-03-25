from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from src.train.train_delta_module import SparseDeltaModule, fit_layer_delta_module, topk_sparsify


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


class SparseDeltaModuleTests(unittest.TestCase):
    def test_topk_sparsify_keeps_exactly_k_entries_per_row(self) -> None:
        activations = torch.tensor(
            [[0.1, 0.8, -0.2, 0.4], [0.3, 0.2, 0.7, -0.1]],
            dtype=torch.float32,
        )
        sparse_latents, active_indices, _ = topk_sparsify(activations, topk=2)

        self.assertEqual(active_indices.shape, (2, 2))
        self.assertTrue(torch.equal((sparse_latents != 0).sum(dim=-1), torch.tensor([2, 2])))

    def test_fit_layer_delta_module_recovers_known_sparse_mapping(self) -> None:
        root = Path("tests/_tmp/train_delta_module")
        root.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(0)
        true_module = SparseDeltaModule(input_dim=4, latent_width=6, topk=2)
        with torch.no_grad():
            true_module.encoder.weight.zero_()
            true_module.encoder.bias.zero_()
            true_module.encoder.weight[0, 0] = 1.5
            true_module.encoder.weight[1, 1] = -1.0
            true_module.encoder.weight[2, 2] = 0.8
            true_module.decoder.weight.zero_()
            true_module.decoder.weight[0, 0] = 1.1
            true_module.decoder.weight[1, 1] = -0.9
            true_module.decoder.weight[2, 2] = 0.6

        x_b = torch.randn(240, 4)
        target_delta, _ = true_module(x_b)
        manifest_path = _write_cache_manifest(root=root, layer_payloads={5: (x_b, target_delta)})

        summary = fit_layer_delta_module(
            config={
                "seed": 17,
                "layer_scan": {
                    "cache_manifest_path": str(manifest_path),
                    "latent_width": 6,
                    "validation_fraction": 0.2,
                    "batch_size": 32,
                    "epochs": 80,
                    "learning_rate": 0.02,
                    "lambda_act": 0.0,
                    "lambda_dec": 0.0,
                    "device": "cpu",
                    "feature_report_limit": 8,
                    "max_feature_stats_rows": 120,
                },
            },
            output_dir=root / "fit_layer_05_k02",
            layer_id=5,
            topk=2,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertGreater(summary["val_metrics"]["explained_fraction_vs_zero"], 0.9)
        self.assertTrue(summary["val_metrics"]["beats_mean_delta_control"])


if __name__ == "__main__":
    unittest.main()
