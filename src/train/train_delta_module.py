from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.train.cache_activations import (
    TOKEN_CLASS_ARGKEY,
    TOKEN_CLASS_ARGVAL,
    TOKEN_CLASS_DECISION,
    TOKEN_CLASS_TOOL,
)

DEFAULT_TOKEN_CLASS_WEIGHTS = {
    TOKEN_CLASS_DECISION: 4.0,
    TOKEN_CLASS_TOOL: 4.0,
    TOKEN_CLASS_ARGKEY: 4.0,
    TOKEN_CLASS_ARGVAL: 4.0,
}


@dataclass(frozen=True)
class SparseLatentOutputs:
    sparse_latents: torch.Tensor
    active_indices: torch.Tensor
    active_values: torch.Tensor


@dataclass(frozen=True)
class LayerTrainingData:
    layer_id: int
    cache_version: str
    input_dim: int
    x_b: torch.Tensor
    target_delta: torch.Tensor
    row_weights: torch.Tensor
    metadata: list[dict[str, Any]]
    train_indices: torch.Tensor
    val_indices: torch.Tensor


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _device_from_layer_scan_config(config: dict[str, Any]) -> torch.device:
    scan_config = dict(config.get("layer_scan", {}))
    device_name = str(scan_config.get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _load_cache_manifest(cache_manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))


def _chunk_records_for_layer(
    cache_manifest: dict[str, Any],
    *,
    layer_id: int,
) -> list[dict[str, Any]]:
    records = [
        dict(record)
        for record in cache_manifest.get("chunk_records", [])
        if int(record["layer_id"]) == int(layer_id)
    ]
    if not records:
        raise ValueError(f"Cache manifest does not contain any chunks for layer {layer_id}.")
    return sorted(records, key=lambda record: int(record["chunk_index"]))


def topk_sparsify(
    activations: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if activations.ndim != 2:
        raise ValueError(
            f"Expected 2D activations for per-token TopK, got shape {tuple(activations.shape)}."
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    if topk > activations.shape[1]:
        raise ValueError(
            f"topk={topk} exceeds latent width {activations.shape[1]} for sparse module."
        )

    active_values, active_indices = torch.topk(activations, k=topk, dim=-1)
    sparse_latents = torch.zeros_like(activations)
    sparse_latents.scatter_(dim=-1, index=active_indices, src=active_values)
    return sparse_latents, active_indices, active_values


class SparseDeltaModule(nn.Module):
    def __init__(self, *, input_dim: int, latent_width: int, topk: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if latent_width <= 0:
            raise ValueError(f"latent_width must be positive, got {latent_width}.")
        if topk <= 0 or topk > latent_width:
            raise ValueError(
                f"topk must lie in [1, latent_width], got topk={topk}, latent_width={latent_width}."
            )
        self.input_dim = int(input_dim)
        self.latent_width = int(latent_width)
        self.topk = int(topk)
        self.encoder = nn.Linear(self.input_dim, self.latent_width, bias=True)
        self.decoder = nn.Linear(self.latent_width, self.input_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, SparseLatentOutputs]:
        if inputs.ndim != 2:
            raise ValueError(
                f"SparseDeltaModule expects [batch, hidden] inputs, got {tuple(inputs.shape)}."
            )
        activations = torch.nn.functional.silu(self.encoder(inputs))
        sparse_latents, active_indices, active_values = topk_sparsify(activations, self.topk)
        delta_hat = self.decoder(sparse_latents)
        return (
            delta_hat,
            SparseLatentOutputs(
                sparse_latents=sparse_latents,
                active_indices=active_indices,
                active_values=active_values,
            ),
        )


class DenseDeltaModule(nn.Module):
    def __init__(self, *, input_dim: int, hidden_width: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if hidden_width <= 0:
            raise ValueError(f"hidden_width must be positive, got {hidden_width}.")
        self.input_dim = int(input_dim)
        self.hidden_width = int(hidden_width)
        self.encoder = nn.Linear(self.input_dim, self.hidden_width, bias=True)
        self.decoder = nn.Linear(self.hidden_width, self.input_dim, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(
                f"DenseDeltaModule expects [batch, hidden] inputs, got {tuple(inputs.shape)}."
            )
        hidden = torch.nn.functional.silu(self.encoder(inputs))
        return self.decoder(hidden)


def _weighted_row_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    row_weights: torch.Tensor,
) -> torch.Tensor:
    squared_error = (prediction - target).pow(2).sum(dim=-1)
    return row_weights * squared_error


def _batch_indices(
    indices: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[torch.Tensor]:
    if indices.numel() == 0:
        return []
    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        order = indices[torch.randperm(indices.numel(), generator=generator)]
    else:
        order = indices
    return [order[start : start + batch_size] for start in range(0, order.numel(), batch_size)]


def _partition_bucket(example_id: str, *, split_seed: int) -> float:
    digest = sha1(f"{split_seed}:{example_id}".encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF


def _resolve_split_indices(
    metadata: list[dict[str, Any]],
    *,
    validation_fraction: float,
    split_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "layer_scan.validation_fraction must lie strictly between 0 and 1, "
            f"got {validation_fraction}."
        )

    train_positions: list[int] = []
    val_positions: list[int] = []
    for position, row in enumerate(metadata):
        bucket = _partition_bucket(str(row["example_id"]), split_seed=split_seed)
        if bucket < validation_fraction:
            val_positions.append(position)
        else:
            train_positions.append(position)

    if not train_positions or not val_positions:
        fallback_val_count = max(1, min(max(len(metadata) // 10, 1), len(metadata) - 1))
        val_positions = list(range(len(metadata) - fallback_val_count, len(metadata)))
        train_positions = list(range(0, len(metadata) - fallback_val_count))

    return (
        torch.tensor(train_positions, dtype=torch.long),
        torch.tensor(val_positions, dtype=torch.long),
    )


def _truncate_rows(
    *,
    x_b_rows: list[torch.Tensor],
    target_rows: list[torch.Tensor],
    metadata_rows: list[dict[str, Any]],
    row_weights: list[float],
    max_rows_per_layer: int | None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], torch.Tensor]:
    x_b = torch.cat(x_b_rows, dim=0).to(dtype=torch.float32)
    target_delta = torch.cat(target_rows, dim=0).to(dtype=torch.float32)
    weights = torch.tensor(row_weights, dtype=torch.float32)

    if max_rows_per_layer is not None:
        capped = max(1, min(int(max_rows_per_layer), x_b.shape[0]))
        x_b = x_b[:capped]
        target_delta = target_delta[:capped]
        metadata_rows = metadata_rows[:capped]
        weights = weights[:capped]

    return x_b, target_delta, metadata_rows, weights


def load_layer_training_data(
    *,
    cache_manifest_path: str | Path,
    layer_id: int,
    layer_scan_config: dict[str, Any],
) -> LayerTrainingData:
    cache_manifest = _load_cache_manifest(cache_manifest_path)
    chunk_records = _chunk_records_for_layer(cache_manifest, layer_id=layer_id)
    token_class_weights = {
        **DEFAULT_TOKEN_CLASS_WEIGHTS,
        **{
            str(key): float(value)
            for key, value in dict(layer_scan_config.get("token_class_weights", {})).items()
        },
    }
    x_b_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    row_weights: list[float] = []
    cache_version: str | None = None
    input_dim: int | None = None

    for record in chunk_records:
        chunk = torch.load(record["path"], map_location="cpu")
        if int(chunk["layer_id"]) != int(layer_id):
            raise ValueError(
                f"Chunk {record['path']} was indexed under layer {layer_id} "
                f"but stored layer {chunk['layer_id']}."
            )
        chunk_cache_version = str(chunk["cache_version"])
        if cache_version is None:
            cache_version = chunk_cache_version
        elif cache_version != chunk_cache_version:
            raise ValueError(
                f"Layer {layer_id} mixes cache versions {cache_version!r} "
                f"and {chunk_cache_version!r}."
            )

        x_b = chunk["x_b"].to(dtype=torch.float32)
        u_b = chunk["u_b"].to(dtype=torch.float32)
        u_d = chunk["u_d"].to(dtype=torch.float32)
        target_delta = u_d - u_b
        chunk_metadata = [dict(row) for row in chunk["metadata"]]
        if x_b.shape != target_delta.shape:
            raise ValueError(
                f"Layer {layer_id} cache rows must align, got x_b {tuple(x_b.shape)} "
                f"vs target {tuple(target_delta.shape)}."
            )
        if x_b.shape[0] != len(chunk_metadata):
            raise ValueError(
                f"Layer {layer_id} metadata count {len(chunk_metadata)} "
                f"did not match tensor rows {x_b.shape[0]}."
            )

        if input_dim is None:
            input_dim = int(x_b.shape[1])
        elif input_dim != int(x_b.shape[1]):
            raise ValueError(
                f"Layer {layer_id} mixed hidden widths {input_dim} and {int(x_b.shape[1])}."
            )

        x_b_rows.append(x_b)
        target_rows.append(target_delta)
        metadata_rows.extend(chunk_metadata)
        row_weights.extend(
            float(token_class_weights.get(str(row["token_class"]), 1.0)) for row in chunk_metadata
        )

    if cache_version is None or input_dim is None or not metadata_rows:
        raise ValueError(f"Layer {layer_id} produced no training rows from the cache manifest.")

    x_b, target_delta, metadata_rows, weight_tensor = _truncate_rows(
        x_b_rows=x_b_rows,
        target_rows=target_rows,
        metadata_rows=metadata_rows,
        row_weights=row_weights,
        max_rows_per_layer=(
            None
            if layer_scan_config.get("max_rows_per_layer") is None
            else int(layer_scan_config["max_rows_per_layer"])
        ),
    )
    train_indices, val_indices = _resolve_split_indices(
        metadata_rows,
        validation_fraction=float(layer_scan_config.get("validation_fraction", 0.1)),
        split_seed=int(layer_scan_config.get("split_seed", 17)),
    )
    return LayerTrainingData(
        layer_id=int(layer_id),
        cache_version=cache_version,
        input_dim=input_dim,
        x_b=x_b,
        target_delta=target_delta,
        row_weights=weight_tensor,
        metadata=metadata_rows,
        train_indices=train_indices,
        val_indices=val_indices,
    )


def _evaluate_split(
    *,
    module: SparseDeltaModule,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    row_weights: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if indices.numel() == 0:
        return {
            "weighted_mse": 0.0,
            "weighted_rmse": 0.0,
            "target_energy": 0.0,
            "l1_mean": 0.0,
            "decoder_fro_norm_sq": float(module.decoder.weight.pow(2).sum().detach().cpu()),
        }

    module.eval()
    weighted_error_total = 0.0
    weighted_target_energy_total = 0.0
    latent_l1_total = 0.0
    row_count = 0
    with torch.inference_mode():
        for batch_indices in _batch_indices(indices, batch_size=batch_size, shuffle=False, seed=0):
            batch_inputs = inputs[batch_indices].to(device)
            batch_targets = targets[batch_indices].to(device)
            batch_weights = row_weights[batch_indices].to(device)
            prediction, latent_outputs = module(batch_inputs)
            weighted_errors = _weighted_row_mse(prediction, batch_targets, batch_weights)
            weighted_error_total += float(weighted_errors.sum().detach().cpu())
            weighted_target_energy_total += float(
                (batch_weights * batch_targets.pow(2).sum(dim=-1)).sum().detach().cpu()
            )
            latent_l1_total += float(
                latent_outputs.sparse_latents.abs().sum(dim=-1).sum().detach().cpu()
            )
            row_count += int(batch_indices.numel())

    weighted_mse = weighted_error_total / row_count
    return {
        "weighted_mse": weighted_mse,
        "weighted_rmse": math.sqrt(max(weighted_mse, 0.0)),
        "target_energy": weighted_target_energy_total / row_count,
        "l1_mean": latent_l1_total / row_count,
        "decoder_fro_norm_sq": float(module.decoder.weight.pow(2).sum().detach().cpu()),
    }


def _weighted_mean_target(
    targets: torch.Tensor,
    row_weights: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    batch_targets = targets[indices]
    batch_weights = row_weights[indices].unsqueeze(-1)
    denominator = float(batch_weights.sum().item())
    if denominator <= 0.0:
        return batch_targets.mean(dim=0)
    return (batch_targets * batch_weights).sum(dim=0) / denominator


def _shortcut_control_metrics(
    *,
    targets: torch.Tensor,
    row_weights: torch.Tensor,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
) -> dict[str, float]:
    val_targets = targets[val_indices]
    val_weights = row_weights[val_indices]
    row_count = max(int(val_indices.numel()), 1)
    zero_weighted_mse = (
        float((val_weights * val_targets.pow(2).sum(dim=-1)).sum().item()) / row_count
    )
    mean_delta = _weighted_mean_target(targets, row_weights, train_indices)
    mean_delta_error = (
        float(
            (val_weights * (val_targets - mean_delta.unsqueeze(0)).pow(2).sum(dim=-1)).sum().item()
        )
        / row_count
    )
    return {
        "zero_predictor_weighted_mse": zero_weighted_mse,
        "mean_delta_predictor_weighted_mse": mean_delta_error,
    }


def _feature_statistics(
    *,
    module: SparseDeltaModule,
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
    max_rows: int | None,
    feature_limit: int,
) -> dict[str, Any]:
    if indices.numel() == 0:
        return {
            "row_count": 0,
            "top_features": [],
        }

    selected = indices
    if max_rows is not None:
        selected = selected[: max(1, min(int(max_rows), int(indices.numel())))]

    active_counts = torch.zeros(module.latent_width, dtype=torch.float64)
    activation_sums = torch.zeros(module.latent_width, dtype=torch.float64)
    activation_abs_sums = torch.zeros(module.latent_width, dtype=torch.float64)

    module.eval()
    with torch.inference_mode():
        for batch_indices in _batch_indices(selected, batch_size=batch_size, shuffle=False, seed=0):
            batch_inputs = inputs[batch_indices].to(device)
            _, latent_outputs = module(batch_inputs)
            flat_indices = latent_outputs.active_indices.reshape(-1).detach().cpu()
            flat_values = latent_outputs.active_values.reshape(-1).detach().cpu().to(torch.float64)
            active_counts.scatter_add_(
                0,
                flat_indices,
                torch.ones_like(flat_values, dtype=torch.float64),
            )
            activation_sums.scatter_add_(0, flat_indices, flat_values)
            activation_abs_sums.scatter_add_(0, flat_indices, flat_values.abs())

    total_activation_mass = float(activation_abs_sums.sum().item())
    rows_evaluated = int(selected.numel())
    top_features = []
    sorted_feature_ids = sorted(
        range(module.latent_width),
        key=lambda feature_id: (
            -float(active_counts[feature_id].item()),
            -float(activation_abs_sums[feature_id].item()),
            feature_id,
        ),
    )
    for feature_id in sorted_feature_ids[: max(1, feature_limit)]:
        active_count = float(active_counts[feature_id].item())
        abs_sum = float(activation_abs_sums[feature_id].item())
        top_features.append(
            {
                "feature_id": int(feature_id),
                "active_count": int(active_count),
                "active_fraction": active_count / rows_evaluated,
                "mean_activation_when_active": (
                    float(activation_sums[feature_id].item()) / active_count
                    if active_count
                    else 0.0
                ),
                "mean_abs_activation_when_active": abs_sum / active_count if active_count else 0.0,
                "activation_mass_fraction": (
                    abs_sum / total_activation_mass if total_activation_mass > 0.0 else 0.0
                ),
            }
        )

    return {
        "row_count": rows_evaluated,
        "top_features": top_features,
    }


def fit_layer_delta_module(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    layer_id: int,
    topk: int,
) -> dict[str, Any]:
    layer_scan_config = dict(config.get("layer_scan", {}))
    cache_manifest_path = layer_scan_config.get("cache_manifest_path")
    if not cache_manifest_path:
        raise ValueError("Config is missing layer_scan.cache_manifest_path.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data = load_layer_training_data(
        cache_manifest_path=cache_manifest_path,
        layer_id=layer_id,
        layer_scan_config=layer_scan_config,
    )
    device = _device_from_layer_scan_config(config)
    batch_size = int(layer_scan_config.get("batch_size", 1024))
    epoch_count = int(layer_scan_config.get("epochs", 4))
    learning_rate = float(layer_scan_config.get("learning_rate", 5e-4))
    lambda_act = float(layer_scan_config.get("lambda_act", 1e-4))
    lambda_dec = float(layer_scan_config.get("lambda_dec", 1e-5))
    weight_decay = float(layer_scan_config.get("weight_decay", 0.0))
    latent_width = int(layer_scan_config.get("latent_width", 256))
    train_seed = int(layer_scan_config.get("train_seed", int(config.get("seed", 17))))

    module = SparseDeltaModule(
        input_dim=data.input_dim,
        latent_width=latent_width,
        topk=int(topk),
    ).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_val_weighted_mse = math.inf
    train_history = []
    val_history = []
    row_class_counts = Counter(str(row["token_class"]) for row in data.metadata)

    for epoch in range(epoch_count):
        module.train()
        train_batches = _batch_indices(
            data.train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=train_seed + epoch,
        )
        batch_losses: list[float] = []
        for batch_indices in train_batches:
            batch_inputs = data.x_b[batch_indices].to(device)
            batch_targets = data.target_delta[batch_indices].to(device)
            batch_weights = data.row_weights[batch_indices].to(device)
            prediction, latent_outputs = module(batch_inputs)
            data_loss = _weighted_row_mse(prediction, batch_targets, batch_weights).mean()
            latent_l1 = latent_outputs.sparse_latents.abs().sum(dim=-1).mean()
            decoder_penalty = module.decoder.weight.pow(2).sum()
            loss = data_loss + (lambda_act * latent_l1) + (lambda_dec * decoder_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_metrics = _evaluate_split(
            module=module,
            inputs=data.x_b,
            targets=data.target_delta,
            row_weights=data.row_weights,
            indices=data.train_indices,
            batch_size=batch_size,
            device=device,
        )
        val_metrics = _evaluate_split(
            module=module,
            inputs=data.x_b,
            targets=data.target_delta,
            row_weights=data.row_weights,
            indices=data.val_indices,
            batch_size=batch_size,
            device=device,
        )
        train_history.append(
            {"epoch": epoch + 1, "loss_mean": sum(batch_losses) / len(batch_losses)}
        )
        val_history.append({"epoch": epoch + 1, **val_metrics})

        if val_metrics["weighted_mse"] < best_val_weighted_mse:
            best_val_weighted_mse = val_metrics["weighted_mse"]
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in module.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(f"Layer {layer_id} TopK {topk} failed to produce a best checkpoint.")

    module.load_state_dict(best_state)
    train_metrics = _evaluate_split(
        module=module,
        inputs=data.x_b,
        targets=data.target_delta,
        row_weights=data.row_weights,
        indices=data.train_indices,
        batch_size=batch_size,
        device=device,
    )
    val_metrics = _evaluate_split(
        module=module,
        inputs=data.x_b,
        targets=data.target_delta,
        row_weights=data.row_weights,
        indices=data.val_indices,
        batch_size=batch_size,
        device=device,
    )
    controls = _shortcut_control_metrics(
        targets=data.target_delta,
        row_weights=data.row_weights,
        train_indices=data.train_indices,
        val_indices=data.val_indices,
    )
    zero_mse = controls["zero_predictor_weighted_mse"]
    mean_delta_mse = controls["mean_delta_predictor_weighted_mse"]
    val_metrics = {
        **val_metrics,
        "explained_fraction_vs_zero": (
            1.0 - (val_metrics["weighted_mse"] / zero_mse) if zero_mse > 0.0 else 0.0
        ),
        "improvement_over_mean_delta_control": mean_delta_mse - val_metrics["weighted_mse"],
        "beats_mean_delta_control": val_metrics["weighted_mse"] < mean_delta_mse,
    }

    feature_stats = _feature_statistics(
        module=module,
        inputs=data.x_b,
        indices=data.val_indices,
        batch_size=batch_size,
        device=device,
        max_rows=(
            None
            if layer_scan_config.get("max_feature_stats_rows") is None
            else int(layer_scan_config["max_feature_stats_rows"])
        ),
        feature_limit=int(layer_scan_config.get("feature_report_limit", 32)),
    )

    checkpoint_path = destination / "module_checkpoint.pt"
    torch.save(
        {
            "layer_id": int(layer_id),
            "cache_version": data.cache_version,
            "input_dim": data.input_dim,
            "latent_width": latent_width,
            "topk": int(topk),
            "state_dict": {key: value.cpu() for key, value in module.state_dict().items()},
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "shortcut_controls": controls,
        },
        checkpoint_path,
    )

    feature_stats_path = destination / "feature_stats.json"
    feature_stats_path.write_text(
        json.dumps(feature_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    training_trace_path = destination / "training_trace.json"
    training_trace_path.write_text(
        json.dumps(
            {
                "train_history": train_history,
                "val_history": val_history,
                "best_epoch": best_epoch,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "status": "passed",
        "layer_id": int(layer_id),
        "cache_manifest_path": str(Path(cache_manifest_path).resolve()),
        "cache_version": data.cache_version,
        "input_dim": data.input_dim,
        "latent_width": latent_width,
        "topk": int(topk),
        "device": str(device),
        "train_row_count": int(data.train_indices.numel()),
        "val_row_count": int(data.val_indices.numel()),
        "row_class_counts": dict(sorted(row_class_counts.items())),
        "learning_rate": learning_rate,
        "lambda_act": lambda_act,
        "lambda_dec": lambda_dec,
        "weight_decay": weight_decay,
        "epochs": epoch_count,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "shortcut_controls": controls,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "feature_stats_path": str(feature_stats_path.resolve()),
        "training_trace_path": str(training_trace_path.resolve()),
        "notes": [
            "This fit summary is a cache-reconstruction artifact for M4 rough scanning.",
            "It does not by itself satisfy the locked end-to-end layer-ranking requirement.",
        ],
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path.resolve())
    return summary
