from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.analysis.eval_layer_candidate import write_layer_candidate_summary
from src.analysis.param_budget import dense_two_layer_params, sparse_same_size_params
from src.train.train_delta_module import (
    DenseDeltaModule,
    _batch_indices,
    _weighted_row_mse,
    load_layer_training_data,
)


@dataclass(frozen=True)
class ShortcutControlArtifacts:
    summary_path: str


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _device_name(shortcut_config: dict[str, Any]) -> str:
    return str(shortcut_config.get("device", "auto"))


def _torch_device(shortcut_config: dict[str, Any]) -> torch.device:
    device_name = _device_name(shortcut_config).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _gain_grid_from_same_size_summary(same_size_summary: dict[str, Any]) -> list[float]:
    gain_grid = sorted(
        {float(result["gain"]) for result in same_size_summary.get("gain_results", [])}
    )
    if not gain_grid:
        raise ValueError("same_size_summary.gain_results must exist to reuse the locked gain grid.")
    return gain_grid


def _selection_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    objective = float(summary["validation_objective"]["score"])
    strict = float(summary["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"])
    control_drop = float(summary["control_eval"]["control_drop"])
    gain = float(summary["candidate"]["gain"])
    return (objective, strict, -control_drop, -abs(gain))


def _dense_hidden_width_for_budget(*, input_dim: int, target_params: int) -> tuple[int, int, float]:
    denominator = (2 * input_dim) + 1
    guess = max(1, int(round((target_params - input_dim) / denominator)))
    candidates = {
        max(1, guess - 1),
        guess,
        guess + 1,
    }
    best_hidden_width = min(
        candidates,
        key=lambda width: abs(
            dense_two_layer_params(hidden_size=input_dim, mlp_hidden_size=width) - target_params
        ),
    )
    matched_params = dense_two_layer_params(
        hidden_size=input_dim,
        mlp_hidden_size=best_hidden_width,
    )
    relative_diff = abs(matched_params - target_params) / target_params
    return best_hidden_width, matched_params, relative_diff


def _fit_dense_shortcut_control(
    *,
    fit_summary: dict[str, Any],
    shortcut_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    dense_config = dict(shortcut_config.get("dense_control", {}))
    layer_training_config = dict(shortcut_config.get("layer_training", {}))
    input_dim = int(fit_summary["input_dim"])
    sparse_params = sparse_same_size_params(
        hidden_size=input_dim,
        bottleneck_size=int(fit_summary["latent_width"]),
        layer_count=1,
    )
    hidden_width, dense_params, relative_diff = _dense_hidden_width_for_budget(
        input_dim=input_dim,
        target_params=sparse_params,
    )
    if relative_diff > 0.10:
        raise ValueError(
            "Dense shortcut control could not match the sparse parameter budget within ±10%."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_layer_training_data(
        cache_manifest_path=fit_summary["cache_manifest_path"],
        layer_id=int(fit_summary["layer_id"]),
        layer_scan_config=layer_training_config,
    )
    device = _torch_device(shortcut_config)
    batch_size = int(dense_config.get("batch_size", 1024))
    epoch_count = int(dense_config.get("epochs", 4))
    learning_rate = float(dense_config.get("learning_rate", 5e-4))
    weight_decay = float(dense_config.get("weight_decay", 0.0))
    train_seed = int(dense_config.get("train_seed", int(shortcut_config.get("seed", 17))))

    module = DenseDeltaModule(input_dim=data.input_dim, hidden_width=hidden_width).to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_weighted_mse = math.inf
    train_history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []

    for epoch in range(epoch_count):
        module.train()
        batch_losses: list[float] = []
        for batch_indices in _batch_indices(
            data.train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=train_seed + epoch,
        ):
            batch_inputs = data.x_b[batch_indices].to(device)
            batch_targets = data.target_delta[batch_indices].to(device)
            batch_weights = data.row_weights[batch_indices].to(device)
            prediction = module(batch_inputs)
            loss = _weighted_row_mse(prediction, batch_targets, batch_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_metrics = _dense_eval_split(
            module=module,
            data=data,
            indices=data.train_indices,
            batch_size=batch_size,
            device=device,
        )
        val_metrics = _dense_eval_split(
            module=module,
            data=data,
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
        raise RuntimeError("Dense shortcut control failed to produce a best checkpoint.")

    module.load_state_dict(best_state)
    train_metrics = _dense_eval_split(
        module=module,
        data=data,
        indices=data.train_indices,
        batch_size=batch_size,
        device=device,
    )
    val_metrics = _dense_eval_split(
        module=module,
        data=data,
        indices=data.val_indices,
        batch_size=batch_size,
        device=device,
    )

    checkpoint_path = output_dir / "module_checkpoint.pt"
    torch.save(
        {
            "module_kind": "dense_mlp",
            "layer_id": int(fit_summary["layer_id"]),
            "input_dim": data.input_dim,
            "hidden_width": hidden_width,
            "state_dict": {key: value.cpu() for key, value in module.state_dict().items()},
            "cache_version": str(data.cache_version),
            "target_sparse_params": sparse_params,
            "dense_params": dense_params,
            "relative_param_diff": relative_diff,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        checkpoint_path,
    )

    training_trace_path = _write_json(
        output_dir / "training_trace.json",
        {
            "train_history": train_history,
            "val_history": val_history,
            "best_epoch": best_epoch,
        },
    )
    summary_path = _write_json(
        output_dir / "summary.json",
        {
            "status": "passed",
            "module_kind": "dense_mlp",
            "layer_id": int(fit_summary["layer_id"]),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "training_trace_path": str(training_trace_path.resolve()),
            "cache_manifest_path": str(Path(str(fit_summary["cache_manifest_path"])).resolve()),
            "cache_version": str(data.cache_version),
            "input_dim": data.input_dim,
            "hidden_width": hidden_width,
            "target_sparse_params": sparse_params,
            "dense_params": dense_params,
            "relative_param_diff": relative_diff,
            "batch_size": batch_size,
            "epochs": epoch_count,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
    )
    payload = _load_json(summary_path)
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def _dense_eval_split(
    *,
    module: DenseDeltaModule,
    data: Any,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if indices.numel() == 0:
        return {
            "weighted_mse": 0.0,
            "weighted_rmse": 0.0,
            "target_energy": 0.0,
        }
    module.eval()
    weighted_error_total = 0.0
    weighted_target_energy_total = 0.0
    row_count = 0
    with torch.inference_mode():
        for batch_indices in _batch_indices(indices, batch_size=batch_size, shuffle=False, seed=0):
            batch_inputs = data.x_b[batch_indices].to(device)
            batch_targets = data.target_delta[batch_indices].to(device)
            batch_weights = data.row_weights[batch_indices].to(device)
            prediction = module(batch_inputs)
            weighted_errors = _weighted_row_mse(prediction, batch_targets, batch_weights)
            weighted_error_total += float(weighted_errors.sum().detach().cpu())
            weighted_target_energy_total += float(
                (batch_weights * batch_targets.pow(2).sum(dim=-1)).sum().detach().cpu()
            )
            row_count += int(batch_indices.numel())
    weighted_mse = weighted_error_total / row_count
    return {
        "weighted_mse": weighted_mse,
        "weighted_rmse": math.sqrt(max(weighted_mse, 0.0)),
        "target_energy": weighted_target_energy_total / row_count,
    }


def _write_steering_vector_control(
    *,
    fit_summary: dict[str, Any],
    shortcut_config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    layer_training_config = dict(shortcut_config.get("layer_training", {}))
    data = load_layer_training_data(
        cache_manifest_path=fit_summary["cache_manifest_path"],
        layer_id=int(fit_summary["layer_id"]),
        layer_scan_config=layer_training_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if data.train_indices.numel() == 0:
        raise ValueError("Steering-vector control requires non-empty training rows.")
    vector = data.target_delta[data.train_indices].mean(dim=0).to(dtype=torch.float32)
    checkpoint_path = output_dir / "vector_checkpoint.pt"
    torch.save(
        {
            "module_kind": "steering_vector",
            "layer_id": int(fit_summary["layer_id"]),
            "input_dim": int(data.input_dim),
            "cache_version": str(data.cache_version),
            "vector": vector.cpu(),
            "vector_norm": float(vector.norm().item()),
            "aggregation": "train_mean_delta",
        },
        checkpoint_path,
    )
    summary_path = _write_json(
        output_dir / "summary.json",
        {
            "status": "passed",
            "module_kind": "steering_vector",
            "layer_id": int(fit_summary["layer_id"]),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "cache_manifest_path": str(Path(str(fit_summary["cache_manifest_path"])).resolve()),
            "cache_version": str(data.cache_version),
            "input_dim": int(data.input_dim),
            "vector_norm": float(vector.norm().item()),
            "train_row_count": int(data.train_indices.numel()),
            "aggregation": "train_mean_delta",
        },
    )
    payload = _load_json(summary_path)
    payload["summary_path"] = str(summary_path.resolve())
    return payload


def _candidate_layer_config(
    *,
    kind: str,
    checkpoint_path: str,
    layer_id: int,
    gain: float,
    position_policy: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "checkpoint_path": str(Path(str(checkpoint_path)).resolve()),
        "layer_id": int(layer_id),
        "gain": float(gain),
        "position_policy": position_policy,
    }


def _run_candidate_summary(
    *,
    model_config: dict[str, Any],
    output_dir: Path,
    layer_config: dict[str, Any],
    eval_manifest_path: str,
    control_manifest_path: str,
    baseline_summary_path: str | None,
    base_control_reference: dict[str, str] | None,
    shortcut_config: dict[str, Any],
) -> dict[str, Any]:
    candidate_config: dict[str, Any] = {
        "checkpoint_path": layer_config["checkpoint_path"],
        "layer_id": int(layer_config["layer_id"]),
        "kind": str(layer_config["kind"]),
        "gain": float(layer_config["gain"]),
        "position_policy": str(layer_config["position_policy"]),
        "eval_manifest_path": str(Path(eval_manifest_path).resolve()),
        "control_manifest_path": str(Path(control_manifest_path).resolve()),
        "prediction_backend": "model_greedy",
        "control_prediction_backend": "model_greedy",
        "batch_size": int(shortcut_config.get("batch_size", 16)),
        "control_batch_size": int(
            shortcut_config.get("control_batch_size", shortcut_config.get("batch_size", 16))
        ),
        "max_new_tokens": int(shortcut_config.get("max_new_tokens", 64)),
        "control_max_new_tokens": int(shortcut_config.get("control_max_new_tokens", 32)),
        "device": _device_name(shortcut_config),
        "heartbeat_interval_seconds": float(
            shortcut_config.get("heartbeat_interval_seconds", 10.0)
        ),
        "control_drop_tolerance": float(shortcut_config.get("control_drop_tolerance", 0.02)),
        "control_penalty_alpha": float(shortcut_config.get("control_penalty_alpha", 1.0)),
        "proceed_min_recovery": float(shortcut_config.get("proceed_min_recovery", 0.10)),
    }
    if baseline_summary_path is not None:
        candidate_config["baseline_summary_path"] = str(Path(baseline_summary_path).resolve())
    if base_control_reference is not None:
        candidate_config.update(base_control_reference)
    summary_path = write_layer_candidate_summary(
        config={"model": dict(model_config), "candidate_eval": candidate_config},
        output_dir=output_dir,
    )
    return _load_json(summary_path)


def _control_reference_from_summary(summary: dict[str, Any]) -> dict[str, str]:
    return {
        "base_control_predictions_path": summary["control_eval"]["base_predictions_path"],
        "base_control_summary_path": summary["control_eval"]["base_summary_path"],
        "base_control_metrics_path": summary["control_eval"]["base_metrics_path"],
    }


def _run_gain_sweep(
    *,
    kind: str,
    checkpoint_path: str,
    layer_id: int,
    position_policy: str,
    gain_grid: list[float],
    model_config: dict[str, Any],
    same_size_summary: dict[str, Any],
    baseline_summary_path: str,
    shortcut_config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration_primary_manifest_path = same_size_summary["calibration_bundle"]["primary_manifest"][
        "manifest_path"
    ]
    calibration_control_manifest_path = same_size_summary["calibration_bundle"]["control_manifest"][
        "manifest_path"
    ]
    gain_results: list[dict[str, Any]] = []
    base_control_reference: dict[str, str] | None = None
    for gain in gain_grid:
        summary = _run_candidate_summary(
            model_config=model_config,
            output_dir=output_dir / f"gain_{str(gain).replace('-', 'neg_').replace('.', '_')}",
            layer_config=_candidate_layer_config(
                kind=kind,
                checkpoint_path=checkpoint_path,
                layer_id=layer_id,
                gain=gain,
                position_policy=position_policy,
            ),
            eval_manifest_path=calibration_primary_manifest_path,
            control_manifest_path=calibration_control_manifest_path,
            baseline_summary_path=baseline_summary_path,
            base_control_reference=base_control_reference,
            shortcut_config=shortcut_config,
        )
        gain_results.append(summary)
        if base_control_reference is None:
            base_control_reference = _control_reference_from_summary(summary)
    selected = max(gain_results, key=_selection_key)
    _write_json(output_dir / "gain_sweep.json", {"results": gain_results})
    return gain_results, selected


def write_same_size_shortcut_control_report(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    shortcut_config = dict(config.get("shortcut_controls", {}))
    for required_key in (
        "same_size_summary_path",
        "fit_summary_path",
        "selected_eval_summary_path",
        "baseline_summary_path",
        "frozen_eval_manifest_path",
        "frozen_control_manifest_path",
    ):
        if not shortcut_config.get(required_key):
            raise ValueError(f"shortcut_controls.{required_key} is required.")

    same_size_summary = _load_json(shortcut_config["same_size_summary_path"])
    fit_summary = _load_json(shortcut_config["fit_summary_path"])
    selected_eval_summary = _load_json(shortcut_config["selected_eval_summary_path"])
    baseline_summary = _load_json(shortcut_config["baseline_summary_path"])
    prune_summary = (
        _load_json(shortcut_config["prune_summary_path"])
        if shortcut_config.get("prune_summary_path")
        else None
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    gain_grid = _gain_grid_from_same_size_summary(same_size_summary)
    layer_id = int(same_size_summary["candidate"]["layer_id"])
    position_policy = str(same_size_summary["candidate"]["position_policy"])

    dense_summary = _fit_dense_shortcut_control(
        fit_summary=fit_summary,
        shortcut_config=shortcut_config,
        output_dir=destination / "dense_control" / "train",
    )
    steering_summary = _write_steering_vector_control(
        fit_summary=fit_summary,
        shortcut_config=shortcut_config,
        output_dir=destination / "steering_control" / "vector",
    )

    _, dense_selected_calibration = _run_gain_sweep(
        kind="dense_mlp",
        checkpoint_path=dense_summary["checkpoint_path"],
        layer_id=layer_id,
        position_policy=position_policy,
        gain_grid=gain_grid,
        model_config=dict(config.get("model", {})),
        same_size_summary=same_size_summary,
        baseline_summary_path=shortcut_config["baseline_summary_path"],
        shortcut_config=shortcut_config,
        output_dir=destination / "dense_control" / "calibration",
    )
    _, steering_selected_calibration = _run_gain_sweep(
        kind="steering_vector",
        checkpoint_path=steering_summary["checkpoint_path"],
        layer_id=layer_id,
        position_policy=position_policy,
        gain_grid=gain_grid,
        model_config=dict(config.get("model", {})),
        same_size_summary=same_size_summary,
        baseline_summary_path=shortcut_config["baseline_summary_path"],
        shortcut_config=shortcut_config,
        output_dir=destination / "steering_control" / "calibration",
    )

    frozen_control_reference = _control_reference_from_summary(selected_eval_summary)
    dense_frozen_summary = _run_candidate_summary(
        model_config=dict(config.get("model", {})),
        output_dir=destination / "dense_control" / "frozen_eval",
        layer_config=_candidate_layer_config(
            kind="dense_mlp",
            checkpoint_path=dense_summary["checkpoint_path"],
            layer_id=layer_id,
            gain=float(dense_selected_calibration["candidate"]["gain"]),
            position_policy=position_policy,
        ),
        eval_manifest_path=shortcut_config["frozen_eval_manifest_path"],
        control_manifest_path=shortcut_config["frozen_control_manifest_path"],
        baseline_summary_path=shortcut_config["baseline_summary_path"],
        base_control_reference=frozen_control_reference,
        shortcut_config=shortcut_config,
    )
    steering_frozen_summary = _run_candidate_summary(
        model_config=dict(config.get("model", {})),
        output_dir=destination / "steering_control" / "frozen_eval",
        layer_config=_candidate_layer_config(
            kind="steering_vector",
            checkpoint_path=steering_summary["checkpoint_path"],
            layer_id=layer_id,
            gain=float(steering_selected_calibration["candidate"]["gain"]),
            position_policy=position_policy,
        ),
        eval_manifest_path=shortcut_config["frozen_eval_manifest_path"],
        control_manifest_path=shortcut_config["frozen_control_manifest_path"],
        baseline_summary_path=shortcut_config["baseline_summary_path"],
        base_control_reference=frozen_control_reference,
        shortcut_config=shortcut_config,
    )

    summary_payload = {
        "status": "passed",
        "claim_bearing": False,
        "selected_layer": {
            "layer_id": layer_id,
            "position_policy": position_policy,
            "same_size_gain": float(same_size_summary["selected_result"]["gain"]),
            "same_size_summary_path": str(
                Path(shortcut_config["same_size_summary_path"]).resolve()
            ),
            "selected_eval_summary_path": str(
                Path(shortcut_config["selected_eval_summary_path"]).resolve()
            ),
        },
        "progressive_ablation": {
            "status": "not_applicable_single_layer",
            "notes": [
                "The active same-size result uses one transplanted layer only.",
                "For a single transplanted layer, a no-progressive same-size ablation is vacuous.",
            ],
        },
        "full_same_size_reference": {
            "primary_strict": float(
                selected_eval_summary["task_eval"]["grouped_metrics"]["primary"][
                    "strict_full_call_success"
                ]
            ),
            "control_drop": float(selected_eval_summary["control_eval"]["control_drop"]),
            "donor_gap_recovery": selected_eval_summary.get("donor_gap_recovery"),
            "summary_path": str(Path(shortcut_config["selected_eval_summary_path"]).resolve()),
        },
        "pruned_subset_reference": (
            {
                "feature_ids": [
                    int(feature_id)
                    for feature_id in prune_summary["selected_subset"]["feature_ids"]
                ],
                "feature_count": int(prune_summary["selected_subset"]["feature_count"]),
                "primary_strict": float(prune_summary["selected_subset"]["frozen_primary_strict"]),
                "control_drop": float(prune_summary["selected_subset"]["frozen_control_drop"]),
                "summary_path": str(Path(shortcut_config["prune_summary_path"]).resolve()),
            }
            if prune_summary is not None
            else None
        ),
        "dense_control": {
            "training_summary_path": dense_summary["summary_path"],
            "calibration_gain_grid": gain_grid,
            "calibration_selected_gain": float(dense_selected_calibration["candidate"]["gain"]),
            "calibration_selected_objective": float(
                dense_selected_calibration["validation_objective"]["score"]
            ),
            "budget": {
                "sparse_params": int(dense_summary["target_sparse_params"]),
                "dense_params": int(dense_summary["dense_params"]),
                "relative_param_diff": float(dense_summary["relative_param_diff"]),
            },
            "frozen_eval": {
                "primary_strict": float(
                    dense_frozen_summary["task_eval"]["grouped_metrics"]["primary"][
                        "strict_full_call_success"
                    ]
                ),
                "control_drop": float(dense_frozen_summary["control_eval"]["control_drop"]),
                "donor_gap_recovery": dense_frozen_summary.get("donor_gap_recovery"),
                "summary_path": str(
                    (
                        destination
                        / "dense_control"
                        / "frozen_eval"
                        / "layer_candidate_summary.json"
                    ).resolve()
                ),
            },
        },
        "steering_control": {
            "vector_summary_path": steering_summary["summary_path"],
            "calibration_gain_grid": gain_grid,
            "calibration_selected_gain": float(steering_selected_calibration["candidate"]["gain"]),
            "calibration_selected_objective": float(
                steering_selected_calibration["validation_objective"]["score"]
            ),
            "frozen_eval": {
                "primary_strict": float(
                    steering_frozen_summary["task_eval"]["grouped_metrics"]["primary"][
                        "strict_full_call_success"
                    ]
                ),
                "control_drop": float(steering_frozen_summary["control_eval"]["control_drop"]),
                "donor_gap_recovery": steering_frozen_summary.get("donor_gap_recovery"),
                "summary_path": str(
                    (
                        destination
                        / "steering_control"
                        / "frozen_eval"
                        / "layer_candidate_summary.json"
                    ).resolve()
                ),
            },
        },
        "baseline_reference": baseline_summary,
        "notes": [
            (
                "Dense and steering shortcut controls reuse the locked same-size "
                "gain grid and calibration bundle."
            ),
            (
                "The frozen control reference is reused from the selected same-size "
                "frozen eval so control-drop comparisons stay aligned."
            ),
        ],
    }
    summary_path = _write_json(destination / "summary.json", summary_payload)
    return summary_path
