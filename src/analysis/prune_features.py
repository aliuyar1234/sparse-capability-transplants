from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from src.analysis.eval_layer_candidate import write_layer_candidate_summary
from src.train.cache_activations import (
    TOKEN_CLASS_ARGKEY,
    TOKEN_CLASS_ARGVAL,
    TOKEN_CLASS_DECISION,
    TOKEN_CLASS_TOOL,
)
from src.train.train_delta_module import SparseDeltaModule, load_layer_training_data

TARGET_TOKEN_CLASSES = {TOKEN_CLASS_TOOL, TOKEN_CLASS_ARGVAL}
FORMAT_TOKEN_CLASSES = {TOKEN_CLASS_DECISION, TOKEN_CLASS_ARGKEY}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _selected_layer_from_same_size_summary(summary: dict[str, Any]) -> dict[str, Any]:
    selected_layers = list(summary.get("same_size_checkpoint", {}).get("selected_layers", []))
    if selected_layers:
        return dict(selected_layers[0])
    selected_result = dict(summary.get("selected_result", {}))
    candidate = dict(summary.get("candidate", {}))
    if candidate and "gain" in selected_result:
        return {
            **candidate,
            "gain": float(selected_result["gain"]),
        }
    raise ValueError("Same-size summary did not expose a selected layer configuration.")


def _layer_module_from_fit_summary(
    *,
    fit_summary: dict[str, Any],
    device: torch.device,
) -> SparseDeltaModule:
    module = SparseDeltaModule(
        input_dim=int(fit_summary["input_dim"]),
        latent_width=int(fit_summary["latent_width"]),
        topk=int(fit_summary["topk"]),
    )
    payload = torch.load(str(fit_summary["checkpoint_path"]), map_location="cpu")
    module.load_state_dict(payload["state_dict"])
    module.eval()
    module.to(device)
    return module


def _subset_objective(
    *,
    primary_value: float,
    control_drop: float,
    control_drop_tolerance: float,
    control_penalty_alpha: float,
    feature_penalty_beta: float,
    feature_count: int,
) -> float:
    return (
        primary_value
        - (control_penalty_alpha * max(0.0, control_drop - control_drop_tolerance))
        - (feature_penalty_beta * feature_count)
    )


def _feature_rank_key(feature_stat: dict[str, Any]) -> tuple[float, float, float, int]:
    specificity = float(feature_stat["act_target"]) - float(feature_stat["act_format"])
    return (
        -float(feature_stat["drop_primary"]),
        float(feature_stat["drop_ctrl"]),
        -specificity,
        int(feature_stat["feature_id"]),
    )


def _activation_profiles_for_features(
    *,
    module: SparseDeltaModule,
    training_data: Any,
    feature_ids: list[int],
    batch_size: int,
    device: torch.device,
    max_rows: int | None,
) -> tuple[dict[int, list[float]], dict[int, dict[str, float]]]:
    selected_indices = training_data.val_indices
    if max_rows is not None:
        selected_indices = selected_indices[
            : max(1, min(int(max_rows), int(selected_indices.numel())))
        ]
    rows = selected_indices.tolist()
    if not rows:
        raise ValueError("Pruning activation profile rows are empty.")

    feature_positions = {feature_id: position for position, feature_id in enumerate(feature_ids)}
    activation_rows: list[torch.Tensor] = []
    module.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            batch_inputs = training_data.x_b[batch_rows].to(device)
            activations = torch.nn.functional.silu(module.encoder(batch_inputs)).detach().cpu()
            activation_rows.append(activations[:, feature_ids])
    activation_matrix = torch.cat(activation_rows, dim=0).to(dtype=torch.float32)

    token_classes = [str(training_data.metadata[row]["token_class"]) for row in rows]
    target_mask = torch.tensor(
        [token_class in TARGET_TOKEN_CLASSES for token_class in token_classes],
        dtype=torch.bool,
    )
    format_mask = torch.tensor(
        [token_class in FORMAT_TOKEN_CLASSES for token_class in token_classes],
        dtype=torch.bool,
    )
    per_feature_signals: dict[int, dict[str, float]] = {}
    for feature_id, position in feature_positions.items():
        values = activation_matrix[:, position].abs()
        per_feature_signals[feature_id] = {
            "act_target": float(values[target_mask].mean().item())
            if bool(target_mask.any())
            else 0.0,
            "act_format": float(values[format_mask].mean().item())
            if bool(format_mask.any())
            else 0.0,
        }

    profiles = {
        feature_id: activation_matrix[:, position].tolist()
        for feature_id, position in feature_positions.items()
    }
    return profiles, per_feature_signals


def _correlation_matrix(profiles: dict[int, list[float]]) -> dict[tuple[int, int], float]:
    feature_ids = sorted(profiles)
    if not feature_ids:
        return {}
    matrix = torch.tensor([profiles[feature_id] for feature_id in feature_ids], dtype=torch.float32)
    centered = matrix - matrix.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, keepdim=True, unbiased=False)
    safe = torch.where(std > 0, centered / std, torch.zeros_like(centered))
    correlations = (safe @ safe.T) / max(matrix.shape[1], 1)
    payload: dict[tuple[int, int], float] = {}
    for row_index, feature_a in enumerate(feature_ids):
        for col_index, feature_b in enumerate(feature_ids):
            payload[(feature_a, feature_b)] = float(correlations[row_index, col_index].item())
    return payload


def _cluster_feature_stats(
    *,
    ranked_feature_stats: list[dict[str, Any]],
    correlation_lookup: dict[tuple[int, int], float],
    threshold: float,
) -> list[dict[str, Any]]:
    clusters: list[list[int]] = []
    clustered_stats: list[dict[str, Any]] = []
    for feature_stat in ranked_feature_stats:
        feature_id = int(feature_stat["feature_id"])
        assigned_cluster = None
        for cluster_id, members in enumerate(clusters):
            if any(
                abs(correlation_lookup.get((feature_id, member), 0.0)) >= threshold
                for member in members
            ):
                members.append(feature_id)
                assigned_cluster = cluster_id
                break
        if assigned_cluster is None:
            clusters.append([feature_id])
            assigned_cluster = len(clusters) - 1
        clustered_stats.append({**feature_stat, "cluster_id": assigned_cluster})
    return clustered_stats


def _selection_eval_config(
    *,
    base_candidate_config: dict[str, Any],
    feature_ids: list[int] | None,
    max_examples: int | None,
    control_max_examples: int | None,
    base_control_reference: dict[str, str] | None,
) -> dict[str, Any]:
    config = dict(base_candidate_config)
    if feature_ids is not None:
        config["feature_ids"] = [int(feature_id) for feature_id in feature_ids]
    else:
        config.pop("feature_ids", None)
    if max_examples is not None:
        config["max_examples"] = int(max_examples)
    if control_max_examples is not None:
        config["control_max_examples"] = int(control_max_examples)
    if base_control_reference is not None:
        config.update(base_control_reference)
    return config


def _summary_primary_and_control(summary: dict[str, Any]) -> tuple[float, float]:
    primary = float(summary["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"])
    control_drop = float(summary["control_eval"]["control_drop"])
    return primary, control_drop


def _run_candidate_summary(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    summary_path = write_layer_candidate_summary(config=config, output_dir=output_dir)
    summary = _load_json(summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def write_pruned_feature_report(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    prune_config = dict(config.get("prune_features", {}))
    same_size_summary_path = prune_config.get("same_size_summary_path")
    fit_summary_path = prune_config.get("fit_summary_path")
    selected_eval_summary_path = prune_config.get("selected_eval_summary_path")
    baseline_summary_path = prune_config.get("baseline_summary_path")
    frozen_eval_manifest_path = prune_config.get("frozen_eval_manifest_path")
    frozen_control_manifest_path = prune_config.get("frozen_control_manifest_path")
    if not all(
        [
            same_size_summary_path,
            fit_summary_path,
            selected_eval_summary_path,
            baseline_summary_path,
            frozen_eval_manifest_path,
            frozen_control_manifest_path,
        ]
    ):
        raise ValueError(
            "prune_features requires same_size_summary_path, fit_summary_path, "
            "selected_eval_summary_path, baseline_summary_path, frozen_eval_manifest_path, "
            "and frozen_control_manifest_path."
        )

    same_size_summary = _load_json(same_size_summary_path)
    fit_summary = _load_json(fit_summary_path)
    selected_eval_summary = _load_json(selected_eval_summary_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    selected_layer = _selected_layer_from_same_size_summary(same_size_summary)
    latent_width = int(fit_summary["latent_width"])
    shortlist_size = max(1, int(prune_config.get("shortlist_size", 8)))
    max_selected_features = max(1, int(prune_config.get("max_selected_features", 4)))
    selection_max_examples = (
        None
        if prune_config.get("selection_max_examples") is None
        else int(prune_config["selection_max_examples"])
    )
    selection_control_max_examples = (
        None
        if prune_config.get("selection_control_max_examples") is None
        else int(prune_config["selection_control_max_examples"])
    )
    correlation_threshold = float(prune_config.get("duplicate_correlation_threshold", 0.995))
    activation_profile_max_rows = int(prune_config.get("activation_profile_max_rows", 1024))
    feature_penalty_beta = float(prune_config.get("feature_penalty_beta", 0.002))
    random_subset_count = max(1, int(prune_config.get("random_subset_count", 4)))
    random_seed = int(prune_config.get("random_seed", 17))
    batch_size = int(prune_config.get("batch_size", 16))
    control_batch_size = int(prune_config.get("control_batch_size", batch_size))

    feature_stats_payload = _load_json(fit_summary["feature_stats_path"])
    shortlist_feature_ids = [
        int(row["feature_id"])
        for row in feature_stats_payload.get("top_features", [])[:shortlist_size]
    ]
    if not shortlist_feature_ids:
        raise ValueError("Feature shortlist was empty; cannot prune selected same-size checkpoint.")

    training_data = load_layer_training_data(
        cache_manifest_path=fit_summary["cache_manifest_path"],
        layer_id=int(fit_summary["layer_id"]),
        layer_scan_config={
            "split_seed": int(prune_config.get("split_seed", 17)),
            "validation_fraction": float(prune_config.get("validation_fraction", 0.1)),
        },
    )
    device_name = str(prune_config.get("device", "cpu")).lower()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    module = _layer_module_from_fit_summary(fit_summary=fit_summary, device=device)
    activation_profiles, activation_signals = _activation_profiles_for_features(
        module=module,
        training_data=training_data,
        feature_ids=shortlist_feature_ids,
        batch_size=max(1, int(prune_config.get("activation_batch_size", 512))),
        device=device,
        max_rows=activation_profile_max_rows,
    )
    correlation_lookup = _correlation_matrix(activation_profiles)

    selection_candidate_config = {
        "checkpoint_path": str(Path(str(selected_layer["checkpoint_path"])).resolve()),
        "layer_id": int(selected_layer["layer_id"]),
        "gain": float(selected_layer["gain"]),
        "position_policy": str(selected_layer.get("position_policy", "last_token_only")),
        "baseline_summary_path": str(Path(str(baseline_summary_path)).resolve()),
        "eval_manifest_path": str(
            Path(
                str(
                    prune_config.get(
                        "selection_eval_manifest_path",
                        same_size_summary["calibration_bundle"]["primary_manifest"][
                            "manifest_path"
                        ],
                    )
                )
            ).resolve()
        ),
        "control_manifest_path": str(
            Path(
                str(
                    prune_config.get(
                        "selection_control_manifest_path",
                        same_size_summary["calibration_bundle"]["control_manifest"][
                            "manifest_path"
                        ],
                    )
                )
            ).resolve()
        ),
        "prediction_backend": "model_greedy",
        "control_prediction_backend": "model_greedy",
        "batch_size": batch_size,
        "control_batch_size": control_batch_size,
        "max_new_tokens": int(prune_config.get("max_new_tokens", 64)),
        "control_max_new_tokens": int(prune_config.get("control_max_new_tokens", 32)),
        "device": str(prune_config.get("eval_device", "auto")),
        "control_drop_tolerance": float(prune_config.get("control_drop_tolerance", 0.02)),
        "control_penalty_alpha": float(prune_config.get("control_penalty_alpha", 1.0)),
        "heartbeat_interval_seconds": float(prune_config.get("heartbeat_interval_seconds", 10.0)),
    }

    base_selection_summary = _run_candidate_summary(
        config={
            "model": dict(config.get("model", {})),
            "candidate_eval": _selection_eval_config(
                base_candidate_config={
                    **selection_candidate_config,
                    "gain": 0.0,
                },
                feature_ids=None,
                max_examples=selection_max_examples,
                control_max_examples=selection_control_max_examples,
                base_control_reference=None,
            ),
        },
        output_dir=destination / "selection" / "base_reference",
    )
    base_control_reference = {
        "base_control_predictions_path": base_selection_summary["control_eval"][
            "base_predictions_path"
        ],
        "base_control_summary_path": base_selection_summary["control_eval"]["base_summary_path"],
        "base_control_metrics_path": base_selection_summary["control_eval"]["base_metrics_path"],
    }
    full_selection_summary = _run_candidate_summary(
        config={
            "model": dict(config.get("model", {})),
            "candidate_eval": _selection_eval_config(
                base_candidate_config=selection_candidate_config,
                feature_ids=None,
                max_examples=selection_max_examples,
                control_max_examples=selection_control_max_examples,
                base_control_reference=base_control_reference,
            ),
        },
        output_dir=destination / "selection" / "full_module",
    )
    full_primary, full_control_drop = _summary_primary_and_control(full_selection_summary)

    feature_stats: list[dict[str, Any]] = []
    all_feature_ids = list(range(latent_width))
    for rank, feature_id in enumerate(shortlist_feature_ids, start=1):
        ablated_feature_ids = [
            candidate for candidate in all_feature_ids if candidate != feature_id
        ]
        ablated_summary = _run_candidate_summary(
            config={
                "model": dict(config.get("model", {})),
                "candidate_eval": _selection_eval_config(
                    base_candidate_config=selection_candidate_config,
                    feature_ids=ablated_feature_ids,
                    max_examples=selection_max_examples,
                    control_max_examples=selection_control_max_examples,
                    base_control_reference=base_control_reference,
                ),
            },
            output_dir=destination
            / "selection"
            / "single_feature_ablation"
            / f"feature_{feature_id:03d}",
        )
        ablated_primary, _ = _summary_primary_and_control(ablated_summary)
        ablated_candidate_control = float(
            ablated_summary["control_eval"]["candidate_exact_match_average"]
        )
        full_candidate_control = float(
            full_selection_summary["control_eval"]["candidate_exact_match_average"]
        )
        signals = activation_signals[feature_id]
        feature_stats.append(
            {
                "layer_id": int(selected_layer["layer_id"]),
                "feature_id": int(feature_id),
                "rank": rank,
                "drop_primary": full_primary - ablated_primary,
                "drop_ctrl": full_candidate_control - ablated_candidate_control,
                "act_target": signals["act_target"],
                "act_format": signals["act_format"],
                "ablation_summary_path": ablated_summary["summary_path"],
            }
        )

    ranked_feature_stats = sorted(feature_stats, key=_feature_rank_key)
    clustered_feature_stats = _cluster_feature_stats(
        ranked_feature_stats=ranked_feature_stats,
        correlation_lookup=correlation_lookup,
        threshold=correlation_threshold,
    )
    feature_stats_path = _write_json(
        destination / "feature_stats.json",
        {
            "layer_id": int(selected_layer["layer_id"]),
            "full_selection_primary": full_primary,
            "full_selection_control_drop": full_control_drop,
            "correlation_threshold": correlation_threshold,
            "feature_stats": clustered_feature_stats,
        },
    )

    representative_feature_ids: list[int] = []
    seen_clusters: set[int] = set()
    for feature_stat in clustered_feature_stats:
        cluster_id = int(feature_stat["cluster_id"])
        if cluster_id in seen_clusters:
            continue
        representative_feature_ids.append(int(feature_stat["feature_id"]))
        seen_clusters.add(cluster_id)

    subset_evaluations: list[dict[str, Any]] = []
    current_subset: list[int] = []
    current_primary = float(
        base_selection_summary["task_eval"]["grouped_metrics"]["primary"][
            "strict_full_call_success"
        ]
    )
    current_control_drop = float(base_selection_summary["control_eval"]["control_drop"])
    current_objective = _subset_objective(
        primary_value=current_primary,
        control_drop=current_control_drop,
        control_drop_tolerance=float(selection_candidate_config["control_drop_tolerance"]),
        control_penalty_alpha=float(selection_candidate_config["control_penalty_alpha"]),
        feature_penalty_beta=feature_penalty_beta,
        feature_count=0,
    )
    for step in range(max_selected_features):
        best_eval = None
        for feature_id in representative_feature_ids:
            if feature_id in current_subset:
                continue
            candidate_subset = sorted([*current_subset, feature_id])
            candidate_summary = _run_candidate_summary(
                config={
                    "model": dict(config.get("model", {})),
                    "candidate_eval": _selection_eval_config(
                        base_candidate_config=selection_candidate_config,
                        feature_ids=candidate_subset,
                        max_examples=selection_max_examples,
                        control_max_examples=selection_control_max_examples,
                        base_control_reference=base_control_reference,
                    ),
                },
                output_dir=destination
                / "selection"
                / "greedy"
                / f"step_{step + 1:02d}"
                / ("_".join(f"{feature_id:03d}" for feature_id in candidate_subset)),
            )
            candidate_primary, candidate_control_drop = _summary_primary_and_control(
                candidate_summary
            )
            candidate_objective = _subset_objective(
                primary_value=candidate_primary,
                control_drop=candidate_control_drop,
                control_drop_tolerance=float(selection_candidate_config["control_drop_tolerance"]),
                control_penalty_alpha=float(selection_candidate_config["control_penalty_alpha"]),
                feature_penalty_beta=feature_penalty_beta,
                feature_count=len(candidate_subset),
            )
            eval_record = {
                "feature_ids": candidate_subset,
                "primary_val": candidate_primary,
                "control_drop": candidate_control_drop,
                "objective": candidate_objective,
                "summary_path": candidate_summary["summary_path"],
            }
            subset_evaluations.append(eval_record)
            if best_eval is None or (
                candidate_objective,
                candidate_primary,
                -candidate_control_drop,
                [-feature_id for feature_id in candidate_subset],
            ) > (
                best_eval["objective"],
                best_eval["primary_val"],
                -best_eval["control_drop"],
                [-feature_id for feature_id in best_eval["feature_ids"]],
            ):
                best_eval = eval_record

        if best_eval is None or best_eval["objective"] <= current_objective:
            break
        current_subset = list(best_eval["feature_ids"])
        current_primary = float(best_eval["primary_val"])
        current_control_drop = float(best_eval["control_drop"])
        current_objective = float(best_eval["objective"])

    if not current_subset:
        single_feature_evals = [
            entry for entry in subset_evaluations if len(entry["feature_ids"]) == 1
        ]
        if not single_feature_evals:
            raise ValueError("Pruning search did not evaluate any single-feature subsets.")
        best_single = max(
            single_feature_evals,
            key=lambda entry: (
                entry["objective"],
                entry["primary_val"],
                -entry["control_drop"],
                [-feature_id for feature_id in entry["feature_ids"]],
            ),
        )
        current_subset = list(best_single["feature_ids"])
        current_primary = float(best_single["primary_val"])
        current_control_drop = float(best_single["control_drop"])
        current_objective = float(best_single["objective"])

    selected_subset_path = _write_json(
        destination / "selected_subset_manifest.json",
        {
            "status": "passed",
            "layer_id": int(selected_layer["layer_id"]),
            "gain": float(selected_layer["gain"]),
            "position_policy": str(selected_layer.get("position_policy", "last_token_only")),
            "selected_feature_ids": current_subset,
            "selected_feature_count": len(current_subset),
            "shortlist_feature_ids": shortlist_feature_ids,
            "representative_feature_ids": representative_feature_ids,
            "feature_stats_path": str(feature_stats_path.resolve()),
            "selection_objective": {
                "primary_val": current_primary,
                "control_drop": current_control_drop,
                "score": current_objective,
                "feature_penalty_beta": feature_penalty_beta,
                "control_drop_tolerance": float(
                    selection_candidate_config["control_drop_tolerance"]
                ),
                "control_penalty_alpha": float(selection_candidate_config["control_penalty_alpha"]),
            },
            "subset_evaluations": subset_evaluations,
        },
    )

    frozen_summary_reference = {
        "base_control_predictions_path": selected_eval_summary["control_eval"][
            "base_predictions_path"
        ],
        "base_control_summary_path": selected_eval_summary["control_eval"]["base_summary_path"],
        "base_control_metrics_path": selected_eval_summary["control_eval"]["base_metrics_path"],
    }
    frozen_candidate_config = {
        "checkpoint_path": str(Path(str(selected_layer["checkpoint_path"])).resolve()),
        "layer_id": int(selected_layer["layer_id"]),
        "gain": float(selected_layer["gain"]),
        "position_policy": str(selected_layer.get("position_policy", "last_token_only")),
        "baseline_summary_path": str(Path(str(baseline_summary_path)).resolve()),
        "eval_manifest_path": str(Path(str(frozen_eval_manifest_path)).resolve()),
        "control_manifest_path": str(Path(str(frozen_control_manifest_path)).resolve()),
        "prediction_backend": "model_greedy",
        "control_prediction_backend": "model_greedy",
        "batch_size": batch_size,
        "control_batch_size": control_batch_size,
        "max_new_tokens": int(prune_config.get("max_new_tokens", 64)),
        "control_max_new_tokens": int(prune_config.get("control_max_new_tokens", 32)),
        "device": str(prune_config.get("eval_device", "auto")),
        "control_drop_tolerance": float(prune_config.get("control_drop_tolerance", 0.02)),
        "control_penalty_alpha": float(prune_config.get("control_penalty_alpha", 1.0)),
        "heartbeat_interval_seconds": float(prune_config.get("heartbeat_interval_seconds", 10.0)),
    }
    selected_subset_summary = _run_candidate_summary(
        config={
            "model": dict(config.get("model", {})),
            "candidate_eval": _selection_eval_config(
                base_candidate_config=frozen_candidate_config,
                feature_ids=current_subset,
                max_examples=None,
                control_max_examples=None,
                base_control_reference=frozen_summary_reference,
            ),
        },
        output_dir=destination / "frozen_eval" / "selected_subset",
    )

    rng = random.Random(random_seed)
    random_subset_summaries: list[dict[str, Any]] = []
    seen_random_subsets: set[tuple[int, ...]] = {tuple(current_subset)}
    while len(random_subset_summaries) < random_subset_count:
        candidate_subset = tuple(sorted(rng.sample(range(latent_width), k=len(current_subset))))
        if candidate_subset in seen_random_subsets:
            continue
        seen_random_subsets.add(candidate_subset)
        subset_summary = _run_candidate_summary(
            config={
                "model": dict(config.get("model", {})),
                "candidate_eval": _selection_eval_config(
                    base_candidate_config=frozen_candidate_config,
                    feature_ids=list(candidate_subset),
                    max_examples=None,
                    control_max_examples=None,
                    base_control_reference=frozen_summary_reference,
                ),
            },
            output_dir=destination
            / "frozen_eval"
            / "random_subsets"
            / ("_".join(f"{feature_id:03d}" for feature_id in candidate_subset)),
        )
        random_subset_summaries.append(
            {
                "feature_ids": list(candidate_subset),
                "primary_strict": float(
                    subset_summary["task_eval"]["grouped_metrics"]["primary"][
                        "strict_full_call_success"
                    ]
                ),
                "control_drop": float(subset_summary["control_eval"]["control_drop"]),
                "summary_path": subset_summary["summary_path"],
            }
        )

    full_frozen_primary = float(
        selected_eval_summary["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"]
    )
    base_primary = float(
        selected_eval_summary["baseline_reference"]["primary_metric"]["base_value"]
    )
    selected_subset_primary = float(
        selected_subset_summary["task_eval"]["grouped_metrics"]["primary"][
            "strict_full_call_success"
        ]
    )
    retained_gain_fraction = (
        None
        if (full_frozen_primary - base_primary) <= 0.0
        else (selected_subset_primary - base_primary) / (full_frozen_primary - base_primary)
    )
    random_primary_values = [entry["primary_strict"] for entry in random_subset_summaries]
    random_control_drops = [entry["control_drop"] for entry in random_subset_summaries]

    summary_path = destination / "summary.json"
    summary_payload = {
        "status": "passed",
        "claim_bearing": False,
        "selected_layer": {
            "layer_id": int(selected_layer["layer_id"]),
            "gain": float(selected_layer["gain"]),
            "checkpoint_path": str(Path(str(selected_layer["checkpoint_path"])).resolve()),
            "position_policy": str(selected_layer.get("position_policy", "last_token_only")),
        },
        "selection_manifests": {
            "primary_manifest_path": selection_candidate_config["eval_manifest_path"],
            "control_manifest_path": selection_candidate_config["control_manifest_path"],
            "max_examples": selection_max_examples,
            "control_max_examples": selection_control_max_examples,
        },
        "feature_stats_path": str(feature_stats_path.resolve()),
        "selected_subset_manifest_path": str(selected_subset_path.resolve()),
        "selected_subset": {
            "feature_ids": current_subset,
            "feature_count": len(current_subset),
            "selection_primary_val": current_primary,
            "selection_control_drop": current_control_drop,
            "selection_objective": current_objective,
            "frozen_primary_strict": selected_subset_primary,
            "frozen_control_drop": float(selected_subset_summary["control_eval"]["control_drop"]),
            "retained_gain_fraction_vs_full": retained_gain_fraction,
            "summary_path": selected_subset_summary["summary_path"],
        },
        "full_same_size_reference": {
            "primary_strict": full_frozen_primary,
            "control_drop": float(selected_eval_summary["control_eval"]["control_drop"]),
            "summary_path": str(Path(str(selected_eval_summary_path)).resolve()),
        },
        "random_subset_controls": {
            "count": len(random_subset_summaries),
            "seed": random_seed,
            "mean_primary_strict": (sum(random_primary_values) / len(random_primary_values)),
            "best_primary_strict": max(random_primary_values),
            "mean_control_drop": (sum(random_control_drops) / len(random_control_drops)),
            "results": random_subset_summaries,
        },
        "notes": [
            (
                "Subset selection used only the calibration bundle manifests, "
                "not the frozen M1 eval manifests."
            ),
            (
                "Random same-size subset controls were evaluated on the frozen manifests "
                "after the subset was frozen."
            ),
        ],
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path
