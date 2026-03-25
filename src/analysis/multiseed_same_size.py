from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

from src.analysis.eval_layer_candidate import write_layer_candidate_summary
from src.train.train_delta_module import fit_layer_delta_module
from src.utils.seed import PROJECT_SEEDS, set_seed


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_ci(
    values: list[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(bootstrap_seed)
    sample_means = []
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        sample_means.append(sum(sample) / len(sample))
    sample_means.sort()
    return (
        _quantile(sample_means, alpha / 2.0),
        _quantile(sample_means, 1.0 - alpha / 2.0),
    )


def _metric_summary(
    values: list[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    mean_value = sum(values) / len(values) if values else 0.0
    ci_lower, ci_upper = _bootstrap_ci(
        values,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "per_seed": values,
        "mean": mean_value,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "ci95": [ci_lower, ci_upper],
        "seed_count": len(values),
    }


def _control_reference_from_summary(summary: dict[str, Any]) -> dict[str, str]:
    return {
        "base_control_predictions_path": summary["control_eval"]["base_predictions_path"],
        "base_control_summary_path": summary["control_eval"]["base_summary_path"],
        "base_control_metrics_path": summary["control_eval"]["base_metrics_path"],
    }


def _reuse_seed_map(multiseed_config: dict[str, Any]) -> dict[int, str]:
    return {
        int(seed): str(path)
        for seed, path in dict(multiseed_config.get("reuse_existing_seed_summaries", {})).items()
    }


def _layer_scan_config(
    *,
    fit_summary: dict[str, Any],
    multiseed_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    layer_training_config = dict(multiseed_config.get("layer_training", {}))
    return {
        "seed": seed,
        "layer_scan": {
            "cache_manifest_path": str(Path(str(fit_summary["cache_manifest_path"])).resolve()),
            "device": str(multiseed_config.get("device", "auto")),
            "batch_size": int(layer_training_config.get("batch_size", 1024)),
            "epochs": int(layer_training_config.get("epochs", 4)),
            "learning_rate": float(layer_training_config.get("learning_rate", 5e-4)),
            "lambda_act": float(layer_training_config.get("lambda_act", 1e-4)),
            "lambda_dec": float(layer_training_config.get("lambda_dec", 1e-5)),
            "weight_decay": float(layer_training_config.get("weight_decay", 0.0)),
            "latent_width": int(
                layer_training_config.get("latent_width", fit_summary["latent_width"])
            ),
            "validation_fraction": float(layer_training_config.get("validation_fraction", 0.1)),
            "split_seed": int(layer_training_config.get("split_seed", seed)),
            "train_seed": int(layer_training_config.get("train_seed", seed)),
            "max_rows_per_layer": layer_training_config.get("max_rows_per_layer"),
            "max_feature_stats_rows": layer_training_config.get("max_feature_stats_rows"),
            "feature_report_limit": int(layer_training_config.get("feature_report_limit", 32)),
        },
    }


def _candidate_summary_config(
    *,
    model_config: dict[str, Any],
    fit_summary: dict[str, Any],
    multiseed_config: dict[str, Any],
    reference_control_summary: dict[str, Any],
    gain: float,
    position_policy: str,
) -> dict[str, Any]:
    candidate_eval = {
        "checkpoint_path": str(Path(str(fit_summary["checkpoint_path"])).resolve()),
        "layer_id": int(fit_summary["layer_id"]),
        "kind": "sparse",
        "gain": float(gain),
        "position_policy": position_policy,
        "baseline_summary_path": str(
            Path(str(multiseed_config["baseline_summary_path"])).resolve()
        ),
        "eval_manifest_path": str(Path(str(multiseed_config["eval_manifest_path"])).resolve()),
        "control_manifest_path": str(
            Path(str(multiseed_config["control_manifest_path"])).resolve()
        ),
        "prediction_backend": str(multiseed_config.get("prediction_backend", "model_greedy")),
        "control_prediction_backend": str(
            multiseed_config.get("control_prediction_backend", "model_greedy")
        ),
        "batch_size": int(multiseed_config.get("batch_size", 16)),
        "control_batch_size": int(
            multiseed_config.get("control_batch_size", multiseed_config.get("batch_size", 16))
        ),
        "max_new_tokens": int(multiseed_config.get("max_new_tokens", 64)),
        "control_max_new_tokens": int(multiseed_config.get("control_max_new_tokens", 32)),
        "device": str(multiseed_config.get("device", "auto")),
        "control_drop_tolerance": float(multiseed_config.get("control_drop_tolerance", 0.02)),
        "control_penalty_alpha": float(multiseed_config.get("control_penalty_alpha", 1.0)),
        "proceed_min_recovery": float(multiseed_config.get("proceed_min_recovery", 0.1)),
        "heartbeat_interval_seconds": float(
            multiseed_config.get("heartbeat_interval_seconds", 10.0)
        ),
        **reference_control_summary,
    }
    return {"model": dict(model_config), "candidate_eval": candidate_eval}


def _seed_result_from_summary(
    *,
    seed: int,
    source: str,
    summary: dict[str, Any],
    fit_summary_path: str | None,
) -> dict[str, Any]:
    primary_strict = float(
        summary["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"]
    )
    control_drop = float(summary["control_eval"]["control_drop"])
    donor_gap_recovery = (
        None if summary.get("donor_gap_recovery") is None else float(summary["donor_gap_recovery"])
    )
    base_value = float(summary["baseline_reference"]["primary_metric"]["base_value"])
    layer_candidate_summary_path = (
        str(Path(str(summary["summary_path"])).resolve()) if "summary_path" in summary else None
    )
    candidate_eval_summary_path = summary["task_eval"].get("summary_path")
    return {
        "seed": int(seed),
        "source": source,
        "fit_summary_path": fit_summary_path,
        "candidate_eval_summary_path": candidate_eval_summary_path,
        "layer_candidate_summary_path": layer_candidate_summary_path,
        "primary_strict": primary_strict,
        "control_drop": control_drop,
        "donor_gap_recovery": donor_gap_recovery,
        "improvement_over_base": primary_strict - base_value,
    }


def _confirmatory_decision(
    *,
    seed_results: list[dict[str, Any]],
    dense_primary_reference: float | None,
    control_drop_tolerance: float,
) -> dict[str, Any]:
    positive_across_seeds = all(
        float(result["improvement_over_base"]) > 0.0 for result in seed_results
    )
    acceptable_control_across_seeds = all(
        float(result["control_drop"]) <= control_drop_tolerance for result in seed_results
    )
    mean_primary = sum(float(result["primary_strict"]) for result in seed_results) / len(
        seed_results
    )
    if dense_primary_reference is None:
        beats_dense_reference = None
    else:
        beats_dense_reference = mean_primary > float(dense_primary_reference)

    status = (
        "pass"
        if positive_across_seeds
        and acceptable_control_across_seeds
        and beats_dense_reference is not False
        else "hold"
    )
    if not positive_across_seeds:
        reason = "sign_flipped_against_base_on_at_least_one_seed"
    elif not acceptable_control_across_seeds:
        reason = "control_drop_exceeded_tolerance_on_at_least_one_seed"
    elif beats_dense_reference is False:
        reason = "mean_sparse_primary_did_not_clear_dense_single_seed_reference"
    else:
        reason = "sign_and_main_qualitative_conclusion_preserved"
    return {
        "status": status,
        "reason": reason,
        "positive_across_seeds": positive_across_seeds,
        "acceptable_control_across_seeds": acceptable_control_across_seeds,
        "dense_single_seed_reference_available": dense_primary_reference is not None,
        "beats_dense_single_seed_reference": beats_dense_reference,
        "control_drop_tolerance": control_drop_tolerance,
    }


def write_same_size_multiseed_report(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    multiseed_config = dict(config.get("multiseed_same_size", {}))
    for required_key in (
        "fit_summary_path",
        "reference_same_size_summary_path",
        "reference_selected_eval_summary_path",
        "baseline_summary_path",
        "eval_manifest_path",
        "control_manifest_path",
    ):
        if not multiseed_config.get(required_key):
            raise ValueError(f"multiseed_same_size.{required_key} is required.")

    fit_summary_template = _load_json(multiseed_config["fit_summary_path"])
    same_size_summary = _load_json(multiseed_config["reference_same_size_summary_path"])
    reference_selected_eval_summary = _load_json(
        multiseed_config["reference_selected_eval_summary_path"]
    )
    shortcut_summary = (
        _load_json(multiseed_config["shortcut_summary_path"])
        if multiseed_config.get("shortcut_summary_path")
        else None
    )

    seeds = [int(seed) for seed in multiseed_config.get("seeds", PROJECT_SEEDS)]
    if not seeds:
        raise ValueError("multiseed_same_size.seeds must not be empty.")
    reuse_seed_summaries = _reuse_seed_map(multiseed_config)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    gain = float(multiseed_config.get("gain", same_size_summary["selected_result"]["gain"]))
    position_policy = str(
        multiseed_config.get(
            "position_policy",
            same_size_summary["candidate"]["position_policy"],
        )
    )
    control_reference = _control_reference_from_summary(reference_selected_eval_summary)
    bootstrap_samples = int(multiseed_config.get("bootstrap_samples", 2000))
    bootstrap_seed = int(multiseed_config.get("bootstrap_seed", int(config.get("seed", 17))))

    seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        existing_summary_path = reuse_seed_summaries.get(seed)
        if existing_summary_path is not None:
            summary = _load_json(existing_summary_path)
            summary["summary_path"] = str(Path(existing_summary_path).resolve())
            seed_results.append(
                _seed_result_from_summary(
                    seed=seed,
                    source="reused_existing",
                    summary=summary,
                    fit_summary_path=multiseed_config["fit_summary_path"],
                )
            )
            continue

        seed_dir = output_root / f"seed_{seed}"
        fit_dir = seed_dir / "fit"
        set_seed(seed)
        fit_summary = fit_layer_delta_module(
            config=_layer_scan_config(
                fit_summary=fit_summary_template,
                multiseed_config=multiseed_config,
                seed=seed,
            ),
            output_dir=fit_dir,
            layer_id=int(fit_summary_template["layer_id"]),
            topk=int(fit_summary_template["topk"]),
        )
        set_seed(seed)
        summary_path = write_layer_candidate_summary(
            config=_candidate_summary_config(
                model_config=dict(config.get("model", {})),
                fit_summary=fit_summary,
                multiseed_config=multiseed_config,
                reference_control_summary=control_reference,
                gain=gain,
                position_policy=position_policy,
            ),
            output_dir=seed_dir / "frozen_eval",
        )
        summary = _load_json(summary_path)
        summary["summary_path"] = str(summary_path.resolve())
        seed_results.append(
            _seed_result_from_summary(
                seed=seed,
                source="confirmatory_rerun",
                summary=summary,
                fit_summary_path=fit_summary["summary_path"],
            )
        )

    seed_results = sorted(seed_results, key=lambda result: int(result["seed"]))
    primary_values = [float(result["primary_strict"]) for result in seed_results]
    control_drop_values = [float(result["control_drop"]) for result in seed_results]
    donor_gap_values = [
        float(result["donor_gap_recovery"])
        for result in seed_results
        if result["donor_gap_recovery"] is not None
    ]
    improvement_values = [float(result["improvement_over_base"]) for result in seed_results]

    dense_reference = None
    steering_reference = None
    if shortcut_summary is not None:
        dense_reference = dict(shortcut_summary["dense_control"]["frozen_eval"])
        steering_reference = dict(shortcut_summary["steering_control"]["frozen_eval"])

    confirmatory_decision = _confirmatory_decision(
        seed_results=seed_results,
        dense_primary_reference=(
            None if dense_reference is None else float(dense_reference["primary_strict"])
        ),
        control_drop_tolerance=float(multiseed_config.get("control_drop_tolerance", 0.02)),
    )

    summary_payload = {
        "status": "passed",
        "claim_bearing": True,
        "variant": str(config.get("execution_variant", "V24")),
        "selected_setting": {
            "layer_id": int(fit_summary_template["layer_id"]),
            "topk": int(fit_summary_template["topk"]),
            "latent_width": int(fit_summary_template["latent_width"]),
            "gain": gain,
            "position_policy": position_policy,
            "fit_summary_template_path": str(Path(multiseed_config["fit_summary_path"]).resolve()),
            "reference_same_size_summary_path": str(
                Path(multiseed_config["reference_same_size_summary_path"]).resolve()
            ),
            "reference_selected_eval_summary_path": str(
                Path(multiseed_config["reference_selected_eval_summary_path"]).resolve()
            ),
        },
        "seed_policy": {
            "project_seeds": list(PROJECT_SEEDS),
            "evaluated_seeds": seeds,
            "reused_existing_seed_summaries": {
                str(seed): str(Path(path).resolve()) for seed, path in reuse_seed_summaries.items()
            },
        },
        "seed_results": seed_results,
        "aggregate": {
            "primary_strict": _metric_summary(
                primary_values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "control_drop": _metric_summary(
                control_drop_values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 1,
            ),
            "donor_gap_recovery": _metric_summary(
                donor_gap_values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 2,
            ),
            "improvement_over_base": _metric_summary(
                improvement_values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 3,
            ),
        },
        "single_seed_references": {
            "same_size_seed17": {
                "primary_strict": float(
                    reference_selected_eval_summary["task_eval"]["grouped_metrics"]["primary"][
                        "strict_full_call_success"
                    ]
                ),
                "control_drop": float(
                    reference_selected_eval_summary["control_eval"]["control_drop"]
                ),
                "donor_gap_recovery": float(reference_selected_eval_summary["donor_gap_recovery"]),
                "summary_path": str(
                    Path(multiseed_config["reference_selected_eval_summary_path"]).resolve()
                ),
            },
            "dense_shortcut_seed17": dense_reference,
            "steering_shortcut_seed17": steering_reference,
        },
        "confirmatory_decision": confirmatory_decision,
        "notes": [
            (
                "This V24-S6 slice reuses the discovery-seed sparse result for seed 17 "
                "and reruns the locked final same-size setting on confirmatory seeds only."
            ),
            (
                "Dense and steering shortcut controls remain fixed single-seed references "
                "here to keep the confirmatory slice within the V24-S6 budget."
            ),
        ],
    }
    return _write_json(output_root / "summary.json", summary_payload)
