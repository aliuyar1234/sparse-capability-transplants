from __future__ import annotations

import unittest

import torch
from torch import nn

from src.models.hooks import candidate_layer_ids, capture_mlp_io, resolve_mlp_modules


class ToyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class ToyModel(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyLayer(width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return x


class HookTests(unittest.TestCase):
    def test_capture_mlp_io_records_pre_and_post_residual_tensors(self) -> None:
        torch.manual_seed(0)
        model = ToyModel(width=4, depth=1)
        inputs = torch.randn(2, 3, 4)
        expected_output = model.model.layers[0].mlp(inputs)

        with capture_mlp_io(model, [0]) as captures:
            model(inputs)

        capture = captures[0]
        self.assertEqual(capture.module_path, "model.layers.0.mlp")
        self.assertTrue(torch.allclose(capture.mlp_input, inputs))
        self.assertTrue(torch.allclose(capture.mlp_output, expected_output))

    def test_resolve_mlp_modules_rejects_out_of_range_layer_ids(self) -> None:
        model = ToyModel(width=4, depth=2)
        with self.assertRaises(IndexError):
            resolve_mlp_modules(model, [3])

    def test_candidate_layer_ids_follow_fractional_depths_without_duplicates(self) -> None:
        model = ToyModel(width=4, depth=12)
        self.assertEqual(candidate_layer_ids(model), [3, 6, 7, 9])


if __name__ == "__main__":
    unittest.main()
