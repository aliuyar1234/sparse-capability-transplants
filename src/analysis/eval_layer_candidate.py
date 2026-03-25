from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.analysis.summarize_baselines import PRIMARY_VARIANTS
from src.eval.control_metrics import ControlScore, aggregate_control_scores
from src.eval.metrics import ExampleScore, aggregate_scores
from src.eval.run_control_eval import run_control_eval_pipeline
from src.eval.run_eval import run_eval_pipeline


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_prediction_rows(predictions_path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(predictions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_fit_summary(candidate_config: dict[str, Any]) -> dict[str, Any]:
    fit_summary_path = candidate_config.get("fit_summary_path")
    if fit_summary_path:
        return _load_json(fit_summary_path)

    checkpoint_path = candidate_config.get("checkpoint_path")
    layer_id = candidate_config.get("layer_id")
    if not checkpoint_path or layer_id is None:
        raise ValueError(
            "candidate_eval requires fit_summary_path, or both checkpoint_path and layer_id."
        )
    return {
        "checkpoint_path": str(Path(str(checkpoint_path)).resolve()),
        "layer_id": int(layer_id),
        "topk": candidate_config.get("topk"),
        "latent_width": candidate_config.get("latent_width"),
    }


def _primary_metrics_from_predictions(predictions_path: str | Path) -> dict[str, Any]:
    rows = _load_prediction_rows(predictions_path)
    primary_scores = [
        ExampleScore(**row["score"]) for row in rows if row["variant"] in PRIMARY_VARIANTS
    ]
    if not primary_scores:
        raise ValueError("Candidate eval predictions did not contain any primary-metric rows.")
    grouped = {
        "all": [ExampleScore(**row["score"]) for row in rows],
        "primary": primary_scores,
    }
    return {name: asdict(aggregate_scores(scores)) for name, scores in grouped.items()}


def _control_exact_match_average(predictions_path: str | Path) -> float:
    rows = _load_prediction_rows(predictions_path)
    scores = [ControlScore(**row["score"]) for row in rows]
    return aggregate_control_scores(scores)


def _resolve_base_control_reference(
    candidate_config: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    predictions_path = candidate_config.get("base_control_predictions_path")
    summary_path = candidate_config.get("base_control_summary_path")
    metrics_path = candidate_config.get("base_control_metrics_path")
    return (
        None if predictions_path is None else str(Path(str(predictions_path)).resolve()),
        None if summary_path is None else str(Path(str(summary_path)).resolve()),
        None if metrics_path is None else str(Path(str(metrics_path)).resolve()),
    )


def build_layer_candidate_summary(
    *, config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    candidate_config = dict(config.get("candidate_eval", {}))
    fit_summary = _load_fit_summary(candidate_config)
    checkpoint_path = str(Path(fit_summary["checkpoint_path"]).resolve())
    layer_id = int(fit_summary["layer_id"])
    kind = str(candidate_config.get("kind", fit_summary.get("module_kind", "sparse"))).lower()
    gain = float(candidate_config.get("gain", 1.0))
    position_policy = str(candidate_config.get("position_policy", "last_token_only"))
    feature_ids = (
        [int(feature_id) for feature_id in candidate_config.get("feature_ids", [])]
        if candidate_config.get("feature_ids") is not None
        else None
    )
    eval_manifest_path = candidate_config.get("eval_manifest_path")
    control_manifest_path = candidate_config.get("control_manifest_path")
    if not eval_manifest_path:
        raise ValueError("candidate_eval.eval_manifest_path is required.")
    if not control_manifest_path:
        raise ValueError("candidate_eval.control_manifest_path is required.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    variant_filter = candidate_config.get("variant_filter", sorted(PRIMARY_VARIANTS))
    transplant_layers = [
        {
            "kind": kind,
            "checkpoint_path": checkpoint_path,
            "layer_id": layer_id,
            "gain": gain,
            "position_policy": position_policy,
            **({"feature_ids": feature_ids} if feature_ids is not None else {}),
        }
    ]

    candidate_eval_artifacts = run_eval_pipeline(
        config={
            "model": dict(config.get("model", {})),
            "eval": {
                "manifest_path": str(Path(str(eval_manifest_path)).resolve()),
                "prediction_backend": str(
                    candidate_config.get("prediction_backend", "model_greedy")
                ),
                "batch_size": int(candidate_config.get("batch_size", 1)),
                **(
                    {"max_examples": int(candidate_config["max_examples"])}
                    if candidate_config.get("max_examples") is not None
                    else {}
                ),
                "max_new_tokens": int(candidate_config.get("max_new_tokens", 64)),
                "device": str(candidate_config.get("device", "auto")),
                "heartbeat_interval_seconds": float(
                    candidate_config.get("heartbeat_interval_seconds", 10.0)
                ),
                "variant_filter": variant_filter,
                "transplant": {"layers": transplant_layers},
            },
        },
        output_dir=destination / "candidate_eval",
    )
    candidate_metrics = _primary_metrics_from_predictions(candidate_eval_artifacts.predictions_path)

    (
        base_control_predictions_path,
        base_control_summary_path,
        base_control_metrics_path,
    ) = _resolve_base_control_reference(candidate_config)
    if base_control_predictions_path is None:
        base_control_artifacts = run_control_eval_pipeline(
            config={
                "model": dict(config.get("model", {})),
                "control_eval": {
                    "manifest_path": str(Path(str(control_manifest_path)).resolve()),
                    "prediction_backend": str(
                        candidate_config.get("control_prediction_backend", "model_greedy")
                    ),
                    "batch_size": int(
                        candidate_config.get(
                            "control_batch_size",
                            candidate_config.get("batch_size", 1),
                        )
                    ),
                    **(
                        {"max_examples": int(candidate_config["control_max_examples"])}
                        if candidate_config.get("control_max_examples") is not None
                        else {}
                    ),
                    "max_new_tokens": int(candidate_config.get("control_max_new_tokens", 64)),
                    "device": str(candidate_config.get("device", "auto")),
                    "heartbeat_interval_seconds": float(
                        candidate_config.get("heartbeat_interval_seconds", 10.0)
                    ),
                },
            },
            output_dir=destination / "base_control_eval",
        )
        base_control_predictions_path = base_control_artifacts.predictions_path
        base_control_summary_path = base_control_artifacts.summary_path
        base_control_metrics_path = base_control_artifacts.metrics_path

    candidate_control_artifacts = run_control_eval_pipeline(
        config={
            "model": dict(config.get("model", {})),
            "control_eval": {
                "manifest_path": str(Path(str(control_manifest_path)).resolve()),
                "prediction_backend": str(
                    candidate_config.get("control_prediction_backend", "model_greedy")
                ),
                "batch_size": int(
                    candidate_config.get(
                        "control_batch_size",
                        candidate_config.get("batch_size", 1),
                    )
                ),
                **(
                    {"max_examples": int(candidate_config["control_max_examples"])}
                    if candidate_config.get("control_max_examples") is not None
                    else {}
                ),
                "max_new_tokens": int(candidate_config.get("control_max_new_tokens", 64)),
                "device": str(candidate_config.get("device", "auto")),
                "heartbeat_interval_seconds": float(
                    candidate_config.get("heartbeat_interval_seconds", 10.0)
                ),
                "transplant": {"layers": transplant_layers},
            },
        },
        output_dir=destination / "candidate_control_eval",
    )

    base_control_score = _control_exact_match_average(base_control_predictions_path)
    candidate_control_score = _control_exact_match_average(
        candidate_control_artifacts.predictions_path
    )
    control_drop = base_control_score - candidate_control_score

    baseline_summary_path = candidate_config.get("baseline_summary_path")
    donor_gap_recovery = None
    baseline_summary = None
    if baseline_summary_path:
        baseline_summary = _load_json(baseline_summary_path)
        base_value = float(baseline_summary["primary_metric"]["base_value"])
        donor_value = float(baseline_summary["primary_metric"]["donor_value"])
        candidate_value = float(candidate_metrics["primary"]["strict_full_call_success"])
        denominator = donor_value - base_value
        donor_gap_recovery = (
            None if denominator <= 0.0 else (candidate_value - base_value) / denominator
        )

    control_drop_tolerance = float(candidate_config.get("control_drop_tolerance", 0.02))
    control_penalty_alpha = float(candidate_config.get("control_penalty_alpha", 1.0))
    objective = float(candidate_metrics["primary"]["strict_full_call_success"]) - (
        control_penalty_alpha * max(0.0, control_drop - control_drop_tolerance)
    )
    min_recovery = float(candidate_config.get("proceed_min_recovery", 0.10))
    proceed = donor_gap_recovery is not None and donor_gap_recovery >= min_recovery

    return {
        "status": "passed",
        "claim_bearing": False,
        "candidate": {
            "kind": kind,
            "layer_id": layer_id,
            "checkpoint_path": checkpoint_path,
            "gain": gain,
            "position_policy": position_policy,
            "topk": fit_summary.get("topk"),
            "latent_width": fit_summary.get("latent_width"),
            **(
                {"feature_ids": feature_ids, "feature_count": len(feature_ids)}
                if feature_ids
                else {}
            ),
        },
        "task_eval": {
            "summary_path": candidate_eval_artifacts.summary_path,
            "metrics_path": candidate_eval_artifacts.metrics_path,
            "predictions_path": candidate_eval_artifacts.predictions_path,
            "grouped_metrics": candidate_metrics,
        },
        "control_eval": {
            "base_summary_path": base_control_summary_path,
            "base_metrics_path": base_control_metrics_path,
            "base_predictions_path": base_control_predictions_path,
            "candidate_summary_path": candidate_control_artifacts.summary_path,
            "candidate_metrics_path": candidate_control_artifacts.metrics_path,
            "candidate_predictions_path": candidate_control_artifacts.predictions_path,
            "base_exact_match_average": base_control_score,
            "candidate_exact_match_average": candidate_control_score,
            "control_drop": control_drop,
        },
        "baseline_reference": baseline_summary,
        "donor_gap_recovery": donor_gap_recovery,
        "validation_objective": {
            "score": objective,
            "control_drop_tolerance": control_drop_tolerance,
            "control_penalty_alpha": control_penalty_alpha,
        },
        "proceed_decision": {
            "status": "pass" if proceed else "hold",
            "reason": (
                "recovery_clears_min_threshold"
                if proceed
                else "recovery_missing_or_below_threshold"
            ),
            "min_recovery": min_recovery,
        },
        "notes": [
            (
                "This summary evaluates one single-layer intervention candidate with "
                "generation-time injection on the locked eval/control manifests."
            ),
            (
                "It is reusable for same-size shortcut-control and frozen selected-checkpoint "
                "analysis, not only M4 rough ranking."
            ),
        ],
    }


def write_layer_candidate_summary(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = build_layer_candidate_summary(config=config, output_dir=destination)
    summary_path = destination / "layer_candidate_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path
