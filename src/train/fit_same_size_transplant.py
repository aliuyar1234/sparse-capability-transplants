from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.analysis.param_budget import sparse_same_size_params
from src.analysis.summarize_baselines import PRIMARY_VARIANTS
from src.data.build_alias_bank import alias_banks_hash, freeze_alias_banks
from src.data.build_control_suite import (
    build_control_examples_from_canonical_examples,
    write_control_suite,
    write_control_suite_manifest,
)
from src.data.freeze_eval_artifacts import build_alias_candidates
from src.data.generate_nocall import (
    generate_missing_tool_nocall_examples,
    generate_unsupported_intent_nocall_examples,
)
from src.data.generate_schema_shift import generate_schema_shift_examples
from src.data.manifest import load_examples, load_manifest_payload, write_manifest
from src.eval.control_metrics import ControlScore, aggregate_control_scores
from src.eval.metrics import ExampleScore, aggregate_scores
from src.eval.run_control_eval import run_control_eval_pipeline
from src.eval.run_eval import run_eval_pipeline
from src.models.format_prompts import PROMPT_CONTRACT_VERSION


@dataclass(frozen=True)
class SameSizeTransplantArtifacts:
    summary_path: str
    checkpoint_path: str
    gain_sweep_path: str
    calibration_manifest_path: str
    calibration_control_manifest_path: str


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fit_summary_from_config(same_size_config: dict[str, Any]) -> dict[str, Any]:
    fit_summary_path = same_size_config.get("fit_summary_path")
    if not fit_summary_path:
        raise ValueError("same_size.fit_summary_path is required.")
    return _load_json(fit_summary_path)


def _prompt_contract_version(manifest_payload: dict[str, Any]) -> str:
    return str(manifest_payload.get("prompt_contract_version", PROMPT_CONTRACT_VERSION))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _freeze_primary_calibration_bundle(
    *,
    canonical_manifest_path: str | Path,
    output_dir: str | Path,
    alias_bank_id: str,
) -> dict[str, Any]:
    manifest_payload = load_manifest_payload(canonical_manifest_path)
    canonical_examples = load_examples(manifest_payload["dataset_path"])
    calib_examples = [example for example in canonical_examples if example.split == "calib"]
    if not calib_examples:
        raise ValueError("Canonical manifest does not contain any calib examples.")

    prompt_contract_version = _prompt_contract_version(manifest_payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    alias_banks = freeze_alias_banks(build_alias_candidates(canonical_examples))
    alias_banks_path = _write_json(
        destination / "alias_banks" / "banks.json",
        {
            "alias_banks_hash": alias_banks_hash(alias_banks),
            "banks": alias_banks.banks,
        },
    )

    schema_shift_examples = generate_schema_shift_examples(
        calib_examples,
        alias_banks,
        bank_id=alias_bank_id,
    )
    nocall_missing_examples = generate_missing_tool_nocall_examples(calib_examples)
    nocall_unsupported_examples = generate_unsupported_intent_nocall_examples(calib_examples)
    primary_examples = [
        *schema_shift_examples,
        *nocall_missing_examples,
        *nocall_unsupported_examples,
    ]
    primary_manifest = write_manifest(
        examples=primary_examples,
        output_dir=destination / "primary_calibration_manifest",
        manifest_id="manifest_m5_mobile_actions_real_primary_calib_v1",
        prompt_contract_version=prompt_contract_version,
        alias_banks=alias_banks.banks,
        metadata={
            "kind": "m5_primary_calibration_slices",
            "source_manifest_id": manifest_payload["manifest_id"],
            "source_manifest_hash": manifest_payload["manifest_hash"],
            "split_manifest_hash": manifest_payload["metadata"]["split_manifest_hash"],
            "source_split": "calib",
            "alias_bank_id": alias_bank_id,
            "variant_filter": sorted(PRIMARY_VARIANTS),
        },
    )

    control_examples = build_control_examples_from_canonical_examples(calib_examples)
    control_suite_path = write_control_suite(
        examples=control_examples,
        output_path=destination / "control_suite" / "controls.jsonl",
    )
    control_manifest = write_control_suite_manifest(
        examples=control_examples,
        dataset_path=control_suite_path,
        output_path=destination / "control_suite" / "manifest.json",
        manifest_id="manifest_m5_mobile_actions_real_control_calib_v1",
    )

    summary_payload = {
        "status": "passed",
        "source_manifest_id": manifest_payload["manifest_id"],
        "source_manifest_hash": manifest_payload["manifest_hash"],
        "split_manifest_hash": manifest_payload["metadata"]["split_manifest_hash"],
        "prompt_contract_version": prompt_contract_version,
        "alias_bank_id": alias_bank_id,
        "alias_banks_hash": alias_banks_hash(alias_banks),
        "alias_banks_path": str(alias_banks_path.resolve()),
        "primary_manifest": asdict(primary_manifest),
        "control_manifest": asdict(control_manifest),
        "control_suite_path": str(control_suite_path.resolve()),
        "counts": {
            "calib_examples": len(calib_examples),
            "schema_shift_examples": len(schema_shift_examples),
            "nocall_missing_examples": len(nocall_missing_examples),
            "nocall_unsupported_examples": len(nocall_unsupported_examples),
            "primary_examples": len(primary_examples),
            "control_examples": len(control_examples),
        },
        "notes": [
            (
                "This bundle freezes M5 calibration-only primary slices from the "
                "canonical calib split."
            ),
            (
                "It is separate from the frozen M1 eval manifests and avoids eval "
                "leakage during gain selection."
            ),
        ],
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_payload["summary_path"] = str(summary_path.resolve())
    return summary_payload


def _grouped_primary_metrics(predictions_path: str | Path) -> dict[str, Any]:
    rows = _prediction_rows(predictions_path)
    primary_rows = [row for row in rows if row["variant"] in PRIMARY_VARIANTS]
    if not primary_rows:
        raise ValueError("Expected primary-metric rows in same-size predictions.")
    grouped = {
        "all": [ExampleScore(**row["score"]) for row in rows],
        "primary": [ExampleScore(**row["score"]) for row in primary_rows],
    }
    return {name: asdict(aggregate_scores(scores)) for name, scores in grouped.items()}


def _control_exact_match_average(predictions_path: str | Path) -> float:
    rows = _prediction_rows(predictions_path)
    scores = [ControlScore(**row["score"]) for row in rows]
    return aggregate_control_scores(scores)


def _gain_dir_label(gain: float) -> str:
    return str(gain).replace("-", "neg_").replace(".", "_")


def _gain_grid_from_config(same_size_config: dict[str, Any]) -> list[float]:
    gain_grid = [
        float(gain) for gain in same_size_config.get("gain_grid", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ]
    if not gain_grid:
        raise ValueError("same_size.gain_grid must not be empty.")
    unique_sorted = sorted(set(gain_grid))
    if 0.0 not in unique_sorted:
        raise ValueError(
            "same_size.gain_grid must include 0.0 so calibration includes the base reference."
        )
    return unique_sorted


def _build_layer_config(
    *,
    fit_summary: dict[str, Any],
    gain: float,
    position_policy: str,
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(Path(str(fit_summary["checkpoint_path"])).resolve()),
        "layer_id": int(fit_summary["layer_id"]),
        "gain": float(gain),
        "position_policy": position_policy,
    }


def _selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    objective = float(result["validation_objective"]["score"])
    strict = float(result["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"])
    control_drop = float(result["control_eval"]["control_drop"])
    gain = float(result["gain"])
    return (objective, strict, -control_drop, -abs(gain))


def run_same_size_fit_pipeline(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> SameSizeTransplantArtifacts:
    same_size_config = dict(config.get("same_size", {}))
    fit_summary = _fit_summary_from_config(same_size_config)
    if fit_summary.get("cache_version") is None:
        raise ValueError("same_size.fit_summary_path must point to a summary with cache_version.")
    canonical_manifest_path = same_size_config.get("canonical_manifest_path")
    if not canonical_manifest_path:
        raise ValueError("same_size.canonical_manifest_path is required.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    calibration_bundle = _freeze_primary_calibration_bundle(
        canonical_manifest_path=canonical_manifest_path,
        output_dir=destination / "calibration_bundle",
        alias_bank_id=str(same_size_config.get("alias_bank_id", "val")),
    )
    gain_grid = _gain_grid_from_config(same_size_config)
    position_policy = str(same_size_config.get("position_policy", "last_token_only"))
    if position_policy not in {"all_tokens", "last_token_only"}:
        raise ValueError(
            "same_size.position_policy must be 'all_tokens' or 'last_token_only', "
            f"got {position_policy!r}."
        )

    donor_reference = None
    donor_model_config = dict(config.get("donor_model", {}))
    if donor_model_config:
        donor_eval_artifacts = run_eval_pipeline(
            config={
                "model": donor_model_config,
                "eval": {
                    "manifest_path": calibration_bundle["primary_manifest"]["manifest_path"],
                    "prediction_backend": str(
                        same_size_config.get("prediction_backend", "model_greedy")
                    ),
                    "batch_size": int(same_size_config.get("batch_size", 1)),
                    "max_new_tokens": int(same_size_config.get("max_new_tokens", 64)),
                    "device": str(same_size_config.get("device", "auto")),
                    "heartbeat_interval_seconds": float(
                        same_size_config.get("heartbeat_interval_seconds", 10.0)
                    ),
                    "variant_filter": sorted(PRIMARY_VARIANTS),
                },
            },
            output_dir=destination / "donor_reference_eval",
        )
        donor_reference = {
            "summary_path": donor_eval_artifacts.summary_path,
            "metrics_path": donor_eval_artifacts.metrics_path,
            "predictions_path": donor_eval_artifacts.predictions_path,
            "grouped_metrics": _grouped_primary_metrics(donor_eval_artifacts.predictions_path),
        }

    gain_results: list[dict[str, Any]] = []
    for gain in gain_grid:
        layer_config = _build_layer_config(
            fit_summary=fit_summary,
            gain=gain,
            position_policy=position_policy,
        )
        gain_dir = destination / "gain_sweep" / f"gain_{_gain_dir_label(gain)}"
        task_eval_artifacts = run_eval_pipeline(
            config={
                "model": dict(config.get("model", {})),
                "eval": {
                    "manifest_path": calibration_bundle["primary_manifest"]["manifest_path"],
                    "prediction_backend": str(
                        same_size_config.get("prediction_backend", "model_greedy")
                    ),
                    "batch_size": int(same_size_config.get("batch_size", 1)),
                    "max_new_tokens": int(same_size_config.get("max_new_tokens", 64)),
                    "device": str(same_size_config.get("device", "auto")),
                    "heartbeat_interval_seconds": float(
                        same_size_config.get("heartbeat_interval_seconds", 10.0)
                    ),
                    "variant_filter": sorted(PRIMARY_VARIANTS),
                    "transplant": {"layers": [layer_config]},
                },
            },
            output_dir=gain_dir / "task_eval",
        )
        control_eval_artifacts = run_control_eval_pipeline(
            config={
                "model": dict(config.get("model", {})),
                "control_eval": {
                    "manifest_path": calibration_bundle["control_manifest"]["manifest_path"],
                    "prediction_backend": str(
                        same_size_config.get("control_prediction_backend", "model_greedy")
                    ),
                    "batch_size": int(
                        same_size_config.get(
                            "control_batch_size",
                            same_size_config.get("batch_size", 1),
                        )
                    ),
                    "max_new_tokens": int(same_size_config.get("control_max_new_tokens", 32)),
                    "device": str(same_size_config.get("device", "auto")),
                    "heartbeat_interval_seconds": float(
                        same_size_config.get("heartbeat_interval_seconds", 10.0)
                    ),
                    "transplant": {"layers": [layer_config]},
                },
            },
            output_dir=gain_dir / "control_eval",
        )
        gain_results.append(
            {
                "gain": float(gain),
                "task_eval": {
                    "summary_path": task_eval_artifacts.summary_path,
                    "metrics_path": task_eval_artifacts.metrics_path,
                    "predictions_path": task_eval_artifacts.predictions_path,
                    "grouped_metrics": _grouped_primary_metrics(
                        task_eval_artifacts.predictions_path
                    ),
                },
                "control_eval": {
                    "summary_path": control_eval_artifacts.summary_path,
                    "metrics_path": control_eval_artifacts.metrics_path,
                    "predictions_path": control_eval_artifacts.predictions_path,
                    "exact_match_average": _control_exact_match_average(
                        control_eval_artifacts.predictions_path
                    ),
                },
            }
        )

    base_result = next(result for result in gain_results if float(result["gain"]) == 0.0)
    base_primary_strict = float(
        base_result["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"]
    )
    base_control_score = float(base_result["control_eval"]["exact_match_average"])
    donor_primary_strict = None
    if donor_reference is not None:
        donor_primary_strict = float(
            donor_reference["grouped_metrics"]["primary"]["strict_full_call_success"]
        )

    control_drop_tolerance = float(same_size_config.get("control_drop_tolerance", 0.02))
    control_penalty_alpha = float(same_size_config.get("control_penalty_alpha", 1.0))
    for result in gain_results:
        primary_strict = float(
            result["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"]
        )
        control_score = float(result["control_eval"]["exact_match_average"])
        control_drop = base_control_score - control_score
        objective = primary_strict - (
            control_penalty_alpha * max(0.0, control_drop - control_drop_tolerance)
        )
        donor_gap_recovery = None
        if donor_primary_strict is not None and donor_primary_strict > base_primary_strict:
            donor_gap_recovery = (primary_strict - base_primary_strict) / (
                donor_primary_strict - base_primary_strict
            )
        result["base_reference"] = {
            "primary_strict_full_call_success": base_primary_strict,
            "control_exact_match_average": base_control_score,
        }
        result["control_eval"]["control_drop"] = control_drop
        result["improvement_over_base"] = primary_strict - base_primary_strict
        result["donor_gap_recovery"] = donor_gap_recovery
        result["validation_objective"] = {
            "score": objective,
            "control_drop_tolerance": control_drop_tolerance,
            "control_penalty_alpha": control_penalty_alpha,
        }

    selected_result = max(gain_results, key=_selection_key)
    layer_order = [int(fit_summary["layer_id"])]
    added_params = sparse_same_size_params(
        hidden_size=int(fit_summary["input_dim"]),
        bottleneck_size=int(fit_summary["latent_width"]),
        layer_count=len(layer_order),
    )

    checkpoint_payload = {
        "status": "passed",
        "layer_order": layer_order,
        "selected_layers": [
            {
                "layer_id": int(fit_summary["layer_id"]),
                "checkpoint_path": str(Path(str(fit_summary["checkpoint_path"])).resolve()),
                "cache_version": str(fit_summary["cache_version"]),
                "gain": float(selected_result["gain"]),
                "position_policy": position_policy,
                "topk": int(fit_summary["topk"]),
                "latent_width": int(fit_summary["latent_width"]),
                "input_dim": int(fit_summary["input_dim"]),
            }
        ],
        "selected_gain": float(selected_result["gain"]),
        "parameter_budget": {
            "added_params": added_params,
            "formula": "2 * hidden_size * latent_width + latent_width + 1",
        },
        "calibration_bundle": {
            "primary_manifest_path": calibration_bundle["primary_manifest"]["manifest_path"],
            "control_manifest_path": calibration_bundle["control_manifest"]["manifest_path"],
        },
    }
    checkpoint_path = destination / "same_size_checkpoint.pt"
    torch.save(checkpoint_payload, checkpoint_path)

    gain_sweep_path = destination / "gain_sweep.json"
    gain_sweep_path.write_text(
        json.dumps({"results": gain_results}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary_payload = {
        "status": "passed",
        "claim_bearing": False,
        "candidate": {
            "layer_id": int(fit_summary["layer_id"]),
            "checkpoint_path": str(Path(str(fit_summary["checkpoint_path"])).resolve()),
            "cache_version": str(fit_summary["cache_version"]),
            "topk": int(fit_summary["topk"]),
            "latent_width": int(fit_summary["latent_width"]),
            "input_dim": int(fit_summary["input_dim"]),
            "position_policy": position_policy,
        },
        "progressive_fit": {
            "selected_layer_order": layer_order,
            "selected_layer_count": len(layer_order),
            "downstream_activation_refresh_performed": False,
            "notes": [
                "This first M5 slice supports one selected layer only.",
                (
                    "For a single selected layer, progressive fitting is vacuous "
                    "because there is no downstream transplanted layer to refit yet."
                ),
            ],
        },
        "calibration_bundle": calibration_bundle,
        "base_reference": {
            "gain": 0.0,
            "primary_strict_full_call_success": base_primary_strict,
            "control_exact_match_average": base_control_score,
            "task_eval_summary_path": base_result["task_eval"]["summary_path"],
            "control_eval_summary_path": base_result["control_eval"]["summary_path"],
        },
        "donor_reference": donor_reference,
        "selected_result": selected_result,
        "gain_sweep_path": str(gain_sweep_path.resolve()),
        "gain_results": gain_results,
        "same_size_checkpoint_path": str(checkpoint_path.resolve()),
        "parameter_budget": checkpoint_payload["parameter_budget"],
        "notes": [
            (
                "This is the first M5 same-size fitting slice: calibration-bundle "
                "freeze plus gain-only selection for the chosen layer12 candidate."
            ),
            (
                "It does not yet include feature pruning, dense/steering controls, "
                "or multi-layer progressive refits."
            ),
        ],
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return SameSizeTransplantArtifacts(
        summary_path=str(summary_path.resolve()),
        checkpoint_path=str(checkpoint_path.resolve()),
        gain_sweep_path=str(gain_sweep_path.resolve()),
        calibration_manifest_path=str(
            Path(calibration_bundle["primary_manifest"]["manifest_path"]).resolve()
        ),
        calibration_control_manifest_path=str(
            Path(calibration_bundle["control_manifest"]["manifest_path"]).resolve()
        ),
    )
