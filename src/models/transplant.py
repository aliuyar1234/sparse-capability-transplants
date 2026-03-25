from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TypeAlias

import torch
from torch import nn

from src.models.hooks import resolve_mlp_modules
from src.train.train_delta_module import DenseDeltaModule, SparseDeltaModule


@dataclass(frozen=True)
class LoadedSparseTransplantLayer:
    layer_id: int
    checkpoint_path: str
    gain: float
    input_dim: int
    latent_width: int
    topk: int
    position_policy: str
    feature_ids: tuple[int, ...] | None
    module: SparseDeltaModule

    def to_summary(self) -> dict[str, Any]:
        summary = {
            "kind": "sparse",
            "layer_id": int(self.layer_id),
            "checkpoint_path": str(Path(self.checkpoint_path).resolve()),
            "gain": float(self.gain),
            "input_dim": int(self.input_dim),
            "latent_width": int(self.latent_width),
            "topk": int(self.topk),
            "position_policy": self.position_policy,
        }
        if self.feature_ids is not None:
            summary["feature_ids"] = [int(feature_id) for feature_id in self.feature_ids]
            summary["feature_count"] = len(self.feature_ids)
        return summary


@dataclass(frozen=True)
class LoadedDenseTransplantLayer:
    layer_id: int
    checkpoint_path: str
    gain: float
    input_dim: int
    hidden_width: int
    position_policy: str
    module: DenseDeltaModule

    def to_summary(self) -> dict[str, Any]:
        return {
            "kind": "dense_mlp",
            "layer_id": int(self.layer_id),
            "checkpoint_path": str(Path(self.checkpoint_path).resolve()),
            "gain": float(self.gain),
            "input_dim": int(self.input_dim),
            "hidden_width": int(self.hidden_width),
            "position_policy": self.position_policy,
        }


@dataclass(frozen=True)
class LoadedSteeringVectorLayer:
    layer_id: int
    checkpoint_path: str
    gain: float
    input_dim: int
    position_policy: str
    vector: torch.Tensor

    def to_summary(self) -> dict[str, Any]:
        return {
            "kind": "steering_vector",
            "layer_id": int(self.layer_id),
            "checkpoint_path": str(Path(self.checkpoint_path).resolve()),
            "gain": float(self.gain),
            "input_dim": int(self.input_dim),
            "position_policy": self.position_policy,
        }


LoadedInterventionLayer: TypeAlias = (
    LoadedSparseTransplantLayer | LoadedDenseTransplantLayer | LoadedSteeringVectorLayer
)


def _resolved_feature_ids(
    *,
    layer_config: dict[str, Any],
    latent_width: int,
) -> tuple[int, ...] | None:
    feature_ids_raw = layer_config.get("feature_ids")
    if feature_ids_raw is None:
        return None
    if not isinstance(feature_ids_raw, list) or not feature_ids_raw:
        raise ValueError("transplant layer feature_ids must be a non-empty list when provided.")
    feature_ids = tuple(sorted({int(feature_id) for feature_id in feature_ids_raw}))
    for feature_id in feature_ids:
        if feature_id < 0 or feature_id >= latent_width:
            raise ValueError(
                f"Feature id {feature_id} is out of range for latent width {latent_width}."
            )
    return feature_ids


def _position_policy(layer_config: dict[str, Any], default: str = "last_token_only") -> str:
    position_policy = str(layer_config.get("position_policy", default))
    if position_policy not in {"all_tokens", "last_token_only"}:
        raise ValueError(
            "transplant layer position_policy must be 'all_tokens' or 'last_token_only', "
            f"got {position_policy!r}."
        )
    return position_policy


def _load_single_sparse_transplant_layer(
    *,
    layer_config: dict[str, Any],
    payload: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedSparseTransplantLayer:
    layer_id = int(layer_config.get("layer_id", payload["layer_id"]))
    input_dim = int(payload["input_dim"])
    latent_width = int(payload["latent_width"])
    topk = int(payload["topk"])
    gain = float(layer_config.get("gain", 1.0))
    position_policy = _position_policy(layer_config)

    module = SparseDeltaModule(
        input_dim=input_dim,
        latent_width=latent_width,
        topk=topk,
    )
    module.load_state_dict(payload["state_dict"])
    module.eval()
    module.to(device)

    return LoadedSparseTransplantLayer(
        layer_id=layer_id,
        checkpoint_path=str(checkpoint_path),
        gain=gain,
        input_dim=input_dim,
        latent_width=latent_width,
        topk=topk,
        position_policy=position_policy,
        feature_ids=_resolved_feature_ids(layer_config=layer_config, latent_width=latent_width),
        module=module,
    )


def _load_single_dense_transplant_layer(
    *,
    layer_config: dict[str, Any],
    payload: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedDenseTransplantLayer:
    layer_id = int(layer_config.get("layer_id", payload["layer_id"]))
    input_dim = int(payload["input_dim"])
    hidden_width = int(payload["hidden_width"])
    gain = float(layer_config.get("gain", 1.0))
    position_policy = _position_policy(layer_config)

    module = DenseDeltaModule(input_dim=input_dim, hidden_width=hidden_width)
    module.load_state_dict(payload["state_dict"])
    module.eval()
    module.to(device)

    return LoadedDenseTransplantLayer(
        layer_id=layer_id,
        checkpoint_path=str(checkpoint_path),
        gain=gain,
        input_dim=input_dim,
        hidden_width=hidden_width,
        position_policy=position_policy,
        module=module,
    )


def _load_single_steering_vector_layer(
    *,
    layer_config: dict[str, Any],
    payload: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedSteeringVectorLayer:
    layer_id = int(layer_config.get("layer_id", payload["layer_id"]))
    vector = payload.get("vector")
    if not isinstance(vector, torch.Tensor):
        raise ValueError("steering_vector checkpoint is missing a tensor 'vector' payload.")
    input_dim = int(payload.get("input_dim", vector.shape[-1]))
    if int(vector.shape[-1]) != input_dim:
        raise ValueError(
            f"steering_vector input_dim={input_dim} did not match vector width {vector.shape[-1]}."
        )
    gain = float(layer_config.get("gain", 1.0))
    position_policy = _position_policy(layer_config)
    return LoadedSteeringVectorLayer(
        layer_id=layer_id,
        checkpoint_path=str(checkpoint_path),
        gain=gain,
        input_dim=input_dim,
        position_policy=position_policy,
        vector=vector.to(device=device, dtype=torch.float32),
    )


def _load_single_intervention_layer(
    *,
    layer_config: dict[str, Any],
    device: torch.device,
) -> LoadedInterventionLayer:
    checkpoint_path_raw = layer_config.get("checkpoint_path")
    if not checkpoint_path_raw:
        raise ValueError("Every transplant layer requires checkpoint_path.")
    checkpoint_path = Path(str(checkpoint_path_raw)).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    kind = str(layer_config.get("kind", payload.get("module_kind", "sparse"))).lower()
    if kind == "sparse":
        return _load_single_sparse_transplant_layer(
            layer_config=layer_config,
            payload=payload,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    if kind == "dense_mlp":
        return _load_single_dense_transplant_layer(
            layer_config=layer_config,
            payload=payload,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    if kind == "steering_vector":
        return _load_single_steering_vector_layer(
            layer_config=layer_config,
            payload=payload,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    raise ValueError(f"Unsupported transplant layer kind {kind!r}.")


def load_sparse_transplant_layers(
    *,
    transplant_config: dict[str, Any] | None,
    device: torch.device,
) -> list[LoadedInterventionLayer]:
    if not transplant_config:
        return []
    layer_configs = transplant_config.get("layers")
    if not isinstance(layer_configs, list) or not layer_configs:
        raise ValueError("eval.transplant.layers must be a non-empty list when provided.")
    return [
        _load_single_intervention_layer(layer_config=dict(layer_config), device=device)
        for layer_config in layer_configs
    ]


def _tensor_payload(value: object) -> torch.Tensor:
    if isinstance(value, tuple):
        if not value:
            raise ValueError("Hook payload tuple was empty.")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor hook payload, got {type(value).__name__}.")
    return value


def _predict_sparse_delta(
    *,
    layer: LoadedSparseTransplantLayer,
    inputs: torch.Tensor,
) -> torch.Tensor:
    typed_inputs = inputs.to(dtype=layer.module.encoder.weight.dtype)
    if layer.feature_ids is None:
        delta, _ = layer.module(typed_inputs)
        return delta

    activations = torch.nn.functional.silu(layer.module.encoder(typed_inputs))
    selected_feature_ids = torch.tensor(
        layer.feature_ids,
        dtype=torch.long,
        device=activations.device,
    )
    selected_activations = activations.index_select(dim=1, index=selected_feature_ids)
    selected_topk = max(1, min(layer.topk, selected_activations.shape[1]))
    active_values, active_indices = torch.topk(selected_activations, k=selected_topk, dim=-1)
    sparse_latents = torch.zeros_like(activations)
    sparse_latents.scatter_(
        dim=-1,
        index=selected_feature_ids.index_select(0, active_indices.reshape(-1)).reshape(
            active_indices.shape
        ),
        src=active_values,
    )
    return layer.module.decoder(sparse_latents)


def _predict_dense_delta(
    *,
    layer: LoadedDenseTransplantLayer,
    inputs: torch.Tensor,
) -> torch.Tensor:
    typed_inputs = inputs.to(dtype=layer.module.encoder.weight.dtype)
    return layer.module(typed_inputs)


def _predict_steering_delta(
    *,
    layer: LoadedSteeringVectorLayer,
    inputs: torch.Tensor,
) -> torch.Tensor:
    vector = layer.vector.to(device=inputs.device, dtype=inputs.dtype)
    return vector.unsqueeze(0).expand(inputs.shape[0], -1)


def _predict_delta(
    *,
    layer: LoadedInterventionLayer,
    inputs: torch.Tensor,
) -> torch.Tensor:
    if isinstance(layer, LoadedSparseTransplantLayer):
        return _predict_sparse_delta(layer=layer, inputs=inputs)
    if isinstance(layer, LoadedDenseTransplantLayer):
        return _predict_dense_delta(layer=layer, inputs=inputs)
    return _predict_steering_delta(layer=layer, inputs=inputs)


def _delta_for_positions(
    *,
    layer: LoadedInterventionLayer,
    mlp_input: torch.Tensor,
    output_shape: torch.Size,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if mlp_input.shape[-1] != layer.input_dim:
        raise ValueError(
            f"Transplant layer {layer.layer_id} expects hidden width {layer.input_dim}, "
            f"got {mlp_input.shape[-1]}."
        )

    if layer.position_policy == "all_tokens":
        flat_inputs = mlp_input.reshape(-1, layer.input_dim)
        delta = _predict_delta(layer=layer, inputs=flat_inputs)
        return delta.reshape(output_shape).to(dtype=output_dtype)

    selected_inputs = mlp_input[:, -1:, :].reshape(-1, layer.input_dim)
    delta_last = _predict_delta(layer=layer, inputs=selected_inputs)
    delta = torch.zeros(output_shape, dtype=delta_last.dtype, device=delta_last.device)
    delta[:, -1:, :] = delta_last.reshape(mlp_input.shape[0], 1, layer.input_dim)
    return delta.to(dtype=output_dtype)


@contextmanager
def inject_sparse_delta_modules(
    model: nn.Module,
    layers: list[LoadedInterventionLayer],
) -> Iterator[None]:
    if not layers:
        yield
        return

    resolved = resolve_mlp_modules(model, [layer.layer_id for layer in layers])
    with ExitStack() as stack:
        for layer in layers:
            mlp_module = resolved[layer.layer_id]

            def _post_hook(
                module: nn.Module,
                inputs: tuple[object, ...],
                output: object,
                *,
                current_layer: LoadedInterventionLayer = layer,
            ) -> torch.Tensor:
                if not inputs:
                    raise ValueError("MLP forward hook did not receive any inputs.")
                mlp_input = _tensor_payload(inputs[0])
                mlp_output = _tensor_payload(output)
                delta = _delta_for_positions(
                    layer=current_layer,
                    mlp_input=mlp_input,
                    output_shape=mlp_output.shape,
                    output_dtype=mlp_output.dtype,
                )
                return mlp_output + (current_layer.gain * delta)

            handle = mlp_module.register_forward_hook(_post_hook)
            stack.callback(handle.remove)

        yield
