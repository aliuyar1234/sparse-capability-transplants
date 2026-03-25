from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

HOOK_LIBRARY = "torch_forward_hooks"


@dataclass
class LayerIOCapture:
    layer_id: int
    module_path: str
    mlp_input: torch.Tensor | None = None
    mlp_output: torch.Tensor | None = None


def _layer_prefix(model: nn.Module) -> str:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model.layers"
    if hasattr(model, "layers"):
        return "layers"
    raise ValueError(
        "Model does not expose transformer layers at model.layers or layers; "
        "cannot resolve MLP hook points."
    )


def resolve_transformer_layers(model: nn.Module) -> list[nn.Module]:
    prefix = _layer_prefix(model)
    root = model.model if prefix == "model.layers" else model.layers
    if prefix == "model.layers":
        layers = root.layers
    else:
        layers = root
    if not isinstance(layers, (list, nn.ModuleList, tuple)):
        raise TypeError("Transformer layers must be a sequence-like container.")
    return list(layers)


def resolve_mlp_modules(
    model: nn.Module, layer_ids: list[int] | None = None
) -> dict[int, nn.Module]:
    layers = resolve_transformer_layers(model)
    selected_layer_ids = list(range(len(layers))) if layer_ids is None else list(layer_ids)
    if not selected_layer_ids:
        raise ValueError("At least one layer ID must be selected for hook registration.")

    resolved: dict[int, nn.Module] = {}
    for layer_id in selected_layer_ids:
        if layer_id < 0 or layer_id >= len(layers):
            raise IndexError(
                f"Layer index {layer_id} is out of range for a model with {len(layers)} layers."
            )
        layer_module = layers[layer_id]
        mlp_module = getattr(layer_module, "mlp", None)
        if mlp_module is None or not isinstance(mlp_module, nn.Module):
            raise ValueError(f"Layer {layer_id} does not expose an nn.Module at attribute 'mlp'.")
        resolved[layer_id] = mlp_module
    return resolved


def candidate_layer_ids(
    model: nn.Module,
    fractions: list[float] | tuple[float, ...] = (0.25, 0.50, 0.65, 0.85),
) -> list[int]:
    layers = resolve_transformer_layers(model)
    if not layers:
        raise ValueError("Model does not expose any transformer layers.")

    candidate_ids: list[int] = []
    for fraction in fractions:
        if not 0.0 < float(fraction) < 1.0:
            raise ValueError(f"Layer fraction must lie strictly between 0 and 1, got {fraction!r}.")
        layer_id = round((len(layers) - 1) * float(fraction))
        if layer_id not in candidate_ids:
            candidate_ids.append(layer_id)
    return candidate_ids


def _snapshot_tensor(value: object) -> torch.Tensor:
    tensor = value[0] if isinstance(value, tuple) else value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor hook payload, got {type(tensor).__name__}.")
    return tensor.detach().cpu().clone()


@contextmanager
def capture_mlp_io(
    model: nn.Module,
    layer_ids: list[int] | None = None,
) -> Iterator[dict[int, LayerIOCapture]]:
    prefix = _layer_prefix(model)
    resolved = resolve_mlp_modules(model, layer_ids)
    captures = {
        layer_id: LayerIOCapture(layer_id=layer_id, module_path=f"{prefix}.{layer_id}.mlp")
        for layer_id in resolved
    }

    with ExitStack() as stack:
        for layer_id, mlp_module in resolved.items():
            capture = captures[layer_id]

            def _pre_hook(
                module: nn.Module,
                inputs: tuple[object, ...],
                *,
                current_capture: LayerIOCapture = capture,
            ) -> None:
                if not inputs:
                    raise ValueError("MLP forward-pre-hook did not receive any inputs.")
                current_capture.mlp_input = _snapshot_tensor(inputs[0])

            def _post_hook(
                module: nn.Module,
                inputs: tuple[object, ...],
                output: object,
                *,
                current_capture: LayerIOCapture = capture,
            ) -> None:
                current_capture.mlp_output = _snapshot_tensor(output)

            pre_handle = mlp_module.register_forward_pre_hook(_pre_hook)
            post_handle = mlp_module.register_forward_hook(_post_hook)
            stack.callback(pre_handle.remove)
            stack.callback(post_handle.remove)

        yield captures
