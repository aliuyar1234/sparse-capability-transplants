from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis.multiseed_same_size import _metric_summary
from src.analysis.shortcut_controls import (
    _candidate_layer_config,
    _control_reference_from_summary,
    _fit_dense_shortcut_control,
    _gain_grid_from_same_size_summary,
    _load_json,
    _run_candidate_summary,
    _run_gain_sweep,
    _write_json,
)
from src.utils.seed import PROJECT_SEEDS, set_seed


def _seed_shortcut_config(
    shortcut_config: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    seeded_config = json.loads(json.dumps(shortcut_config))
    layer_training = dict(seeded_config.get("layer_training", {}))
    layer_training.setdefault("validation_fraction", 0.1)
    layer_training["split_seed"] = int(layer_training.get("split_seed", seed))
    seeded_config["layer_training"] = layer_training

    dense_control = dict(seeded_config.get("dense_control", {}))
    dense_control["train_seed"] = int(dense_control.get("train_seed", seed))
    seeded_config["dense_control"] = dense_control
    return seeded_config


def _seed_result_from_summary(
    *,
    seed: int,
    source: str,
    summary: dict[str, Any],
    fit_summary_path: str | None,
    selected_gain: float | None,
) -> dict[str, Any]:
    primary_strict = float(
        summary["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"]
    )
    control_drop = float(summary["control_eval"]["control_drop"])
    donor_gap_recovery = (
        None if summary.get("donor_gap_recovery") is None else float(summary["donor_gap_recovery"])
    )
    base_value = float(summary["baseline_reference"]["primary_metric"]["base_value"])
    return {
        "seed": int(seed),
        "source": source,
        "fit_summary_path": fit_summary_path,
        "selected_gain": selected_gain,
        "candidate_eval_summary_path": summary["task_eval"].get("summary_path"),
        "layer_candidate_summary_path": summary.get("summary_path"),
        "primary_strict": primary_strict,
        "control_drop": control_drop,
        "donor_gap_recovery": donor_gap_recovery,
        "improvement_over_base": primary_strict - base_value,
    }


def _comparison_to_sparse_multiseed(
    *,
    dense_seed_results: list[dict[str, Any]],
    sparse_summary: dict[str, Any],
    sparse_summary_path: str | Path,
) -> dict[str, Any]:
    dense_by_seed = {int(result["seed"]): result for result in dense_seed_results}
    sparse_by_seed = {
        int(result["seed"]): result
        for result in list(sparse_summary.get("seed_results", []))
        if "seed" in result
    }
    matched_seeds = sorted(set(dense_by_seed) & set(sparse_by_seed))
    primary_deltas = [
        float(sparse_by_seed[seed]["primary_strict"]) - float(dense_by_seed[seed]["primary_strict"])
        for seed in matched_seeds
    ]
    control_deltas = [
        float(sparse_by_seed[seed]["control_drop"]) - float(dense_by_seed[seed]["control_drop"])
        for seed in matched_seeds
    ]
    sparse_primary_mean = float(sparse_summary["aggregate"]["primary_strict"]["mean"])
    dense_primary_mean = sum(
        float(result["primary_strict"]) for result in dense_seed_results
    ) / len(dense_seed_results)
    return {
        "sparse_multiseed_summary_path": str(Path(sparse_summary_path).resolve()),
        "matched_seeds": matched_seeds,
        "sparse_primary_strict_mean": sparse_primary_mean,
        "dense_primary_strict_mean": dense_primary_mean,
        "mean_primary_delta_sparse_minus_dense": sparse_primary_mean - dense_primary_mean,
        "per_seed_primary_delta_sparse_minus_dense": primary_deltas,
        "per_seed_control_drop_delta_sparse_minus_dense": control_deltas,
        "sparse_beats_dense_mean": sparse_primary_mean > dense_primary_mean,
        "sparse_beats_dense_on_all_matched_seeds": all(delta > 0.0 for delta in primary_deltas),
    }


def _confirmatory_decision(
    *,
    comparison_to_sparse: dict[str, Any],
) -> dict[str, Any]:
    sparse_beats_dense_mean = bool(comparison_to_sparse["sparse_beats_dense_mean"])
    sparse_beats_dense_on_all = bool(
        comparison_to_sparse["sparse_beats_dense_on_all_matched_seeds"]
    )
    if sparse_beats_dense_mean and sparse_beats_dense_on_all:
        status = "pass"
        reason = "sparse_mean_and_all_matched_seeds_clear_dense_multiseed_control"
    elif sparse_beats_dense_mean:
        status = "pass"
        reason = "sparse_mean_clears_dense_multiseed_control"
    else:
        status = "hold"
        reason = "dense_multiseed_mean_matches_or_exceeds_sparse_mean"
    return {
        "status": status,
        "reason": reason,
        "matched_seed_count": len(comparison_to_sparse["matched_seeds"]),
        "sparse_beats_dense_mean": sparse_beats_dense_mean,
        "sparse_beats_dense_on_all_matched_seeds": sparse_beats_dense_on_all,
    }


def write_dense_control_multiseed_report(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    multiseed_config = dict(config.get("multiseed_dense_control", {}))
    for required_key in (
        "fit_summary_path",
        "same_size_summary_path",
        "reference_selected_eval_summary_path",
        "shortcut_summary_path",
        "sparse_multiseed_summary_path",
        "baseline_summary_path",
        "eval_manifest_path",
        "control_manifest_path",
    ):
        if not multiseed_config.get(required_key):
            raise ValueError(f"multiseed_dense_control.{required_key} is required.")

    fit_summary_template = _load_json(multiseed_config["fit_summary_path"])
    same_size_summary = _load_json(multiseed_config["same_size_summary_path"])
    reference_selected_eval_summary = _load_json(
        multiseed_config["reference_selected_eval_summary_path"]
    )
    shortcut_summary = _load_json(multiseed_config["shortcut_summary_path"])
    sparse_multiseed_summary = _load_json(multiseed_config["sparse_multiseed_summary_path"])

    seeds = [int(seed) for seed in multiseed_config.get("seeds", PROJECT_SEEDS)]
    if not seeds:
        raise ValueError("multiseed_dense_control.seeds must not be empty.")
    reuse_seed_summaries = {
        int(seed): str(path)
        for seed, path in dict(multiseed_config.get("reuse_existing_seed_summaries", {})).items()
    }
    reuse_training_summaries = {
        int(seed): str(path)
        for seed, path in dict(
            multiseed_config.get("reuse_existing_training_summaries", {})
        ).items()
    }

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    gain_grid = _gain_grid_from_same_size_summary(same_size_summary)
    layer_id = int(same_size_summary["candidate"]["layer_id"])
    position_policy = str(
        multiseed_config.get("position_policy", same_size_summary["candidate"]["position_policy"])
    )
    bootstrap_samples = int(multiseed_config.get("bootstrap_samples", 2000))
    bootstrap_seed = int(multiseed_config.get("bootstrap_seed", int(config.get("seed", 17))))
    frozen_control_reference = _control_reference_from_summary(reference_selected_eval_summary)

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
                    fit_summary_path=(
                        reuse_training_summaries.get(seed)
                        or shortcut_summary["dense_control"]["training_summary_path"]
                    ),
                    selected_gain=float(
                        shortcut_summary["dense_control"]["calibration_selected_gain"]
                    ),
                )
            )
            continue

        seeded_shortcut_config = _seed_shortcut_config(
            {
                "same_size_summary_path": multiseed_config["same_size_summary_path"],
                "fit_summary_path": multiseed_config["fit_summary_path"],
                "selected_eval_summary_path": multiseed_config[
                    "reference_selected_eval_summary_path"
                ],
                "baseline_summary_path": multiseed_config["baseline_summary_path"],
                "frozen_eval_manifest_path": multiseed_config["eval_manifest_path"],
                "frozen_control_manifest_path": multiseed_config["control_manifest_path"],
                "batch_size": int(multiseed_config.get("batch_size", 16)),
                "control_batch_size": int(
                    multiseed_config.get(
                        "control_batch_size", multiseed_config.get("batch_size", 16)
                    )
                ),
                "max_new_tokens": int(multiseed_config.get("max_new_tokens", 64)),
                "control_max_new_tokens": int(multiseed_config.get("control_max_new_tokens", 32)),
                "device": str(multiseed_config.get("device", "auto")),
                "control_drop_tolerance": float(
                    multiseed_config.get("control_drop_tolerance", 0.02)
                ),
                "control_penalty_alpha": float(multiseed_config.get("control_penalty_alpha", 1.0)),
                "proceed_min_recovery": float(multiseed_config.get("proceed_min_recovery", 0.1)),
                "heartbeat_interval_seconds": float(
                    multiseed_config.get("heartbeat_interval_seconds", 10.0)
                ),
                "layer_training": dict(multiseed_config.get("layer_training", {})),
                "dense_control": dict(multiseed_config.get("dense_control", {})),
            },
            seed=seed,
        )
        seed_dir = output_root / f"seed_{seed}"
        set_seed(seed)
        dense_summary = _fit_dense_shortcut_control(
            fit_summary=fit_summary_template,
            shortcut_config=seeded_shortcut_config,
            output_dir=seed_dir / "train",
        )
        _, selected_calibration = _run_gain_sweep(
            kind="dense_mlp",
            checkpoint_path=dense_summary["checkpoint_path"],
            layer_id=layer_id,
            position_policy=position_policy,
            gain_grid=gain_grid,
            model_config=dict(config.get("model", {})),
            same_size_summary=same_size_summary,
            baseline_summary_path=multiseed_config["baseline_summary_path"],
            shortcut_config=seeded_shortcut_config,
            output_dir=seed_dir / "calibration",
        )
        frozen_summary = _run_candidate_summary(
            model_config=dict(config.get("model", {})),
            output_dir=seed_dir / "frozen_eval",
            layer_config=_candidate_layer_config(
                kind="dense_mlp",
                checkpoint_path=dense_summary["checkpoint_path"],
                layer_id=layer_id,
                gain=float(selected_calibration["candidate"]["gain"]),
                position_policy=position_policy,
            ),
            eval_manifest_path=multiseed_config["eval_manifest_path"],
            control_manifest_path=multiseed_config["control_manifest_path"],
            baseline_summary_path=multiseed_config["baseline_summary_path"],
            base_control_reference=frozen_control_reference,
            shortcut_config=seeded_shortcut_config,
        )
        frozen_summary["summary_path"] = str(
            (seed_dir / "frozen_eval" / "layer_candidate_summary.json").resolve()
        )
        seed_results.append(
            _seed_result_from_summary(
                seed=seed,
                source="confirmatory_rerun",
                summary=frozen_summary,
                fit_summary_path=dense_summary["summary_path"],
                selected_gain=float(selected_calibration["candidate"]["gain"]),
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

    comparison_to_sparse = _comparison_to_sparse_multiseed(
        dense_seed_results=seed_results,
        sparse_summary=sparse_multiseed_summary,
        sparse_summary_path=multiseed_config["sparse_multiseed_summary_path"],
    )

    summary_payload = {
        "status": "passed",
        "claim_bearing": True,
        "variant": str(config.get("execution_variant", "V24")),
        "selected_layer": {
            "layer_id": layer_id,
            "topk": int(fit_summary_template["topk"]),
            "latent_width": int(fit_summary_template["latent_width"]),
            "position_policy": position_policy,
            "gain_grid": gain_grid,
            "same_size_summary_path": str(
                Path(multiseed_config["same_size_summary_path"]).resolve()
            ),
            "shortcut_summary_path": str(Path(multiseed_config["shortcut_summary_path"]).resolve()),
        },
        "seed_policy": {
            "project_seeds": list(PROJECT_SEEDS),
            "evaluated_seeds": seeds,
            "reused_existing_seed_summaries": {
                str(seed): str(Path(path).resolve()) for seed, path in reuse_seed_summaries.items()
            },
            "reused_existing_training_summaries": {
                str(seed): str(Path(path).resolve())
                for seed, path in reuse_training_summaries.items()
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
        "single_seed_reference": {
            "primary_strict": float(
                shortcut_summary["dense_control"]["frozen_eval"]["primary_strict"]
            ),
            "control_drop": float(shortcut_summary["dense_control"]["frozen_eval"]["control_drop"]),
            "donor_gap_recovery": float(
                shortcut_summary["dense_control"]["frozen_eval"]["donor_gap_recovery"]
            ),
            "summary_path": str(
                Path(shortcut_summary["dense_control"]["frozen_eval"]["summary_path"]).resolve()
            ),
        },
        "comparison_to_sparse_multiseed": comparison_to_sparse,
        "confirmatory_decision": _confirmatory_decision(
            comparison_to_sparse=comparison_to_sparse,
        ),
        "notes": [
            (
                "This bounded M5 follow-up reruns the dense parameter-matched "
                "same-size control across the same seeds and frozen manifests "
                "as the sparse V24-S6 confirmatory slice."
            ),
            (
                "The dense control keeps the same hook site, gain grid, calibration bundle, and "
                "frozen control reference so the multiseed comparison stays method-aligned."
            ),
        ],
    }
    return _write_json(output_root / "summary.json", summary_payload)
