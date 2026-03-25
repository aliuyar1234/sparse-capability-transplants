from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from src.models.transplant import (
    _predict_delta,
    inject_sparse_delta_modules,
    load_sparse_transplant_layers,
)
from src.train.train_delta_module import DenseDeltaModule, SparseDeltaModule


class ToyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.mlp.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ToyModel(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyLayer(width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


def _write_sparse_checkpoint(*, path: Path) -> Path:
    module = SparseDeltaModule(input_dim=1, latent_width=1, topk=1)
    with torch.no_grad():
        module.encoder.weight.zero_()
        module.encoder.bias.fill_(1.0)
        module.decoder.weight.fill_(4.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer_id": 0,
            "input_dim": 1,
            "latent_width": 1,
            "topk": 1,
            "state_dict": {key: value.cpu() for key, value in module.state_dict().items()},
        },
        path,
    )
    return path


def _write_dense_checkpoint(*, path: Path) -> Path:
    module = DenseDeltaModule(input_dim=1, hidden_width=1)
    with torch.no_grad():
        module.encoder.weight.zero_()
        module.encoder.bias.fill_(1.0)
        module.decoder.weight.fill_(2.0)
        module.decoder.bias.fill_(1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module_kind": "dense_mlp",
            "layer_id": 0,
            "input_dim": 1,
            "hidden_width": 1,
            "state_dict": {key: value.cpu() for key, value in module.state_dict().items()},
        },
        path,
    )
    return path


def _write_steering_checkpoint(*, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "module_kind": "steering_vector",
            "layer_id": 0,
            "input_dim": 1,
            "vector": torch.tensor([3.0], dtype=torch.float32),
        },
        path,
    )
    return path


class SparseTransplantHookTests(unittest.TestCase):
    def test_inject_sparse_delta_modules_changes_only_last_token_by_default(self) -> None:
        model = ToyModel(width=1, depth=1)
        inputs = torch.zeros((1, 3, 1), dtype=torch.float32)
        checkpoint_path = _write_sparse_checkpoint(
            path=Path("tests/_tmp/transplant_hook/layer_00_checkpoint.pt")
        )
        layers = load_sparse_transplant_layers(
            transplant_config={"layers": [{"checkpoint_path": str(checkpoint_path), "gain": 1.0}]},
            device=torch.device("cpu"),
        )

        base_output = model(inputs)
        with inject_sparse_delta_modules(model, layers):
            transplanted_output = model(inputs)

        self.assertTrue(torch.allclose(base_output[:, :2, :], transplanted_output[:, :2, :]))
        self.assertGreater(
            float(transplanted_output[0, 2, 0].detach()),
            float(base_output[0, 2, 0].detach()),
        )

    def test_feature_subset_masks_unselected_features_before_topk(self) -> None:
        checkpoint_path = Path("tests/_tmp/transplant_hook/feature_subset_checkpoint.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        module = SparseDeltaModule(input_dim=1, latent_width=2, topk=1)
        with torch.no_grad():
            module.encoder.weight.copy_(torch.tensor([[2.0], [1.0]]))
            module.encoder.bias.zero_()
            module.decoder.weight.copy_(torch.tensor([[10.0, 5.0]]))
        torch.save(
            {
                "layer_id": 0,
                "input_dim": 1,
                "latent_width": 2,
                "topk": 1,
                "state_dict": {key: value.cpu() for key, value in module.state_dict().items()},
            },
            checkpoint_path,
        )

        full_layer = load_sparse_transplant_layers(
            transplant_config={"layers": [{"checkpoint_path": str(checkpoint_path), "gain": 1.0}]},
            device=torch.device("cpu"),
        )[0]
        subset_layer = load_sparse_transplant_layers(
            transplant_config={
                "layers": [
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "gain": 1.0,
                        "feature_ids": [1],
                    }
                ]
            },
            device=torch.device("cpu"),
        )[0]

        inputs = torch.tensor([[1.0]], dtype=torch.float32)
        full_delta = _predict_delta(layer=full_layer, inputs=inputs)
        subset_delta = _predict_delta(layer=subset_layer, inputs=inputs)

        self.assertGreater(float(full_delta.item()), float(subset_delta.item()))
        self.assertGreater(float(subset_delta.item()), 0.0)

    def test_dense_shortcut_checkpoint_injects_at_the_same_hook_site(self) -> None:
        model = ToyModel(width=1, depth=1)
        inputs = torch.zeros((1, 2, 1), dtype=torch.float32)
        checkpoint_path = _write_dense_checkpoint(
            path=Path("tests/_tmp/transplant_hook/dense_checkpoint.pt")
        )
        layers = load_sparse_transplant_layers(
            transplant_config={
                "layers": [
                    {
                        "kind": "dense_mlp",
                        "checkpoint_path": str(checkpoint_path),
                        "gain": 1.0,
                    }
                ]
            },
            device=torch.device("cpu"),
        )

        with inject_sparse_delta_modules(model, layers):
            output = model(inputs)

        self.assertAlmostEqual(float(output[0, 1, 0].detach()), 2.4621172, places=5)

    def test_steering_vector_checkpoint_applies_a_constant_last_token_delta(self) -> None:
        model = ToyModel(width=1, depth=1)
        inputs = torch.zeros((1, 3, 1), dtype=torch.float32)
        checkpoint_path = _write_steering_checkpoint(
            path=Path("tests/_tmp/transplant_hook/steering_checkpoint.pt")
        )
        layers = load_sparse_transplant_layers(
            transplant_config={
                "layers": [
                    {
                        "kind": "steering_vector",
                        "checkpoint_path": str(checkpoint_path),
                        "gain": 0.5,
                    }
                ]
            },
            device=torch.device("cpu"),
        )

        base_output = model(inputs)
        with inject_sparse_delta_modules(model, layers):
            output = model(inputs)

        self.assertTrue(torch.allclose(base_output[:, :2, :], output[:, :2, :]))
        self.assertAlmostEqual(float(output[0, 2, 0].detach()), 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
