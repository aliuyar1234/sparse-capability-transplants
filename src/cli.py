from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analysis.donor_gap_gate import write_donor_gap_gate
from src.analysis.error_analysis import write_error_analysis_report
from src.analysis.eval_layer_candidate import write_layer_candidate_summary
from src.analysis.export_final_registry import export_final_registry
from src.analysis.multiseed_dense_control import write_dense_control_multiseed_report
from src.analysis.multiseed_same_size import write_same_size_multiseed_report
from src.analysis.param_budget import write_budget_report
from src.analysis.plot_recovery import write_recovery_artifacts
from src.analysis.plot_tradeoffs import write_tradeoff_artifacts
from src.analysis.prune_features import write_pruned_feature_report
from src.analysis.rank_layers import write_layer_ranking_report
from src.analysis.shortcut_controls import write_same_size_shortcut_control_report
from src.analysis.summarize_baselines import write_baseline_summary
from src.data.freeze_eval_artifacts import run_eval_freeze_pipeline_from_manifest
from src.data.parse_mobile_actions import MobileActionsParseError, import_mobile_actions_dataset
from src.data.smoke_data import run_smoke_data_pipeline
from src.eval.run_eval import run_eval_pipeline
from src.models.smoke import probe_model_loading
from src.train.cache_activations import collect_activation_caches
from src.train.fit_same_size_transplant import run_same_size_fit_pipeline
from src.train.train_donor_fullft import run_donor_training
from src.train.train_recipient_fullft import run_recipient_fullft_smoke_training
from src.train.train_recipient_lora import run_recipient_lora_smoke_training
from src.utils.config import (
    config_hash,
    dump_config_snapshot,
    ensure_execution_variant,
    load_config,
)
from src.utils.logging import configure_logging
from src.utils.run_manifest import create_run_manifest, update_run_manifest, write_run_manifest
from src.utils.seed import set_seed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sparse Capability Transplants task runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_smoke = subparsers.add_parser(
        "manifest-smoke",
        help="Load a config, seed the runtime, and write a run manifest snapshot.",
    )
    manifest_smoke.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    smoke_model = subparsers.add_parser(
        "smoke-model",
        help="Attempt a lightweight model probe or write a documented blocker.",
    )
    smoke_model.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )
    smoke_model.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when model probing is blocked.",
    )

    smoke_data = subparsers.add_parser(
        "smoke-data",
        help="Build a synthetic M1 data harness smoke artifact set.",
    )
    smoke_data.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    import_mobile_actions = subparsers.add_parser(
        "import-mobile-actions",
        help="Import Mobile Actions-style raw rows into a canonical manifest.",
    )
    import_mobile_actions.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    train_donor = subparsers.add_parser(
        "train-donor",
        help="Run the M2 donor smoke training pipeline.",
    )
    train_donor.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    train_recipient = subparsers.add_parser(
        "train-recipient-baselines",
        help="Run an M2 recipient baseline smoke training pipeline.",
    )
    train_recipient.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    cache_activations = subparsers.add_parser(
        "cache-activations",
        help="Collect teacher-forced donor/base activation caches at documented MLP hooks.",
    )
    cache_activations.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    rank_layers = subparsers.add_parser(
        "rank-layers",
        help="Fit rough sparse delta modules and rank candidate layers from cached activations.",
    )
    rank_layers.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    eval_layer_candidate = subparsers.add_parser(
        "eval-layer-candidate",
        help=(
            "Run the M4 end-to-end single-layer candidate eval on the frozen "
            "eval/control manifests."
        ),
    )
    eval_layer_candidate.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    eval_main = subparsers.add_parser(
        "eval-main",
        help="Run the deterministic evaluation wrapper.",
    )
    eval_main.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    fit_same_size = subparsers.add_parser(
        "fit-same-size-transplant",
        help=(
            "Run the first M5 same-size fitting slice: freeze calibration artifacts, "
            "sweep gain-only calibration, and write a reusable same-size artifact."
        ),
    )
    fit_same_size.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    prune_features = subparsers.add_parser(
        "prune-features",
        help=(
            "Run the M5 feature-pruning slice: freeze a causal subset on the calibration "
            "bundle and compare it against random same-size subsets on the frozen manifests."
        ),
    )
    prune_features.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    shortcut_controls = subparsers.add_parser(
        "same-size-shortcut-controls",
        help=(
            "Run the remaining M5 shortcut controls: dense parameter-matched and "
            "one-vector steering baselines on the selected same-size checkpoint."
        ),
    )
    shortcut_controls.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    same_size_multiseed = subparsers.add_parser(
        "same-size-multiseed",
        help=(
            "Run the V24-S6 confirmatory multi-seed reruns for the locked final same-size setting."
        ),
    )
    same_size_multiseed.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    dense_control_multiseed = subparsers.add_parser(
        "same-size-dense-multiseed-control",
        help=(
            "Run the bounded M5 matched multi-seed reruns for the dense parameter-matched "
            "same-size shortcut control."
        ),
    )
    dense_control_multiseed.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    error_analysis = subparsers.add_parser(
        "error-analysis",
        help="Generate M7 strict-vs-semantic, error-category, and appendix-example artifacts.",
    )
    error_analysis.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    plot_recovery = subparsers.add_parser(
        "plot-recovery",
        help="Generate M7 recovery, budget, and retained-gain tables/figures.",
    )
    plot_recovery.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    plot_tradeoffs = subparsers.add_parser(
        "plot-tradeoffs",
        help="Generate M7 tradeoff tables and plots for same-size vs controls.",
    )
    plot_tradeoffs.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    export_registry = subparsers.add_parser(
        "export-final-registry",
        help="Generate the M8 final claim audit, artifact inventory, and section map.",
    )
    export_registry.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    param_budget = subparsers.add_parser(
        "param-budget",
        help="Write a parameter-budget smoke artifact.",
    )
    param_budget.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    summarize_baselines = subparsers.add_parser(
        "summarize-baselines",
        help="Summarize base/donor prediction artifacts and compute the donor-gap metric.",
    )
    summarize_baselines.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    donor_gap_gate = subparsers.add_parser(
        "donor-gap-gate",
        help="Write the explicit R20 donor-gap gate decision artifact.",
    )
    donor_gap_gate.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )

    freeze_mobile_actions_eval = subparsers.add_parser(
        "freeze-mobile-actions-eval",
        help="Freeze M1 real-corpus evaluation artifacts from an imported canonical manifest.",
    )
    freeze_mobile_actions_eval.add_argument(
        "--config",
        required=True,
        help="Path to a JSON/YAML/TOML config file.",
    )
    return parser


def _run_dir_from_manifest(manifest: dict[str, Any]) -> Path:
    return Path(manifest["artifact_paths"]["run_dir"])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _initialize_run(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = load_config(config_path)
    ensure_execution_variant(config)
    set_seed(int(config.get("seed", 17)))

    manifest = create_run_manifest(
        config=config,
        config_path=config_path,
        command=["python", "-m", "src.cli"],
    )
    run_dir = write_run_manifest(manifest)
    snapshot_path = run_dir / "config_snapshot.json"
    dump_config_snapshot(config, snapshot_path)
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "config_snapshot": str(snapshot_path),
            "config_hash": config_hash(config),
        },
    )
    return config, manifest, run_dir


def _resolve_config_path(config_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def _resolve_model_local_path(config: dict[str, Any], config_path: Path) -> None:
    model_config = dict(config.get("model", {}))
    local_path_value = model_config.get("local_path")
    if local_path_value:
        model_config["local_path"] = _resolve_config_path(config_path, str(local_path_value))
        config["model"] = model_config


def _resolve_named_model_local_path(
    config: dict[str, Any], config_path: Path, model_key: str
) -> None:
    model_config = dict(config.get(model_key, {}))
    local_path_value = model_config.get("local_path")
    if local_path_value:
        model_config["local_path"] = _resolve_config_path(config_path, str(local_path_value))
        config[model_key] = model_config


def _resolve_analysis_paths(config: dict[str, Any], config_path: Path) -> None:
    analysis_config = dict(config.get("analysis", {}))
    for key in ("base_predictions_path", "donor_predictions_path", "summary_path"):
        value = analysis_config.get(key)
        if value:
            analysis_config[key] = _resolve_config_path(config_path, str(value))
    if analysis_config:
        config["analysis"] = analysis_config


def _resolve_cache_paths(config: dict[str, Any], config_path: Path) -> None:
    cache_config = dict(config.get("cache", {}))
    manifest_value = cache_config.get("manifest_path")
    if manifest_value:
        cache_config["manifest_path"] = _resolve_config_path(config_path, str(manifest_value))
    if cache_config:
        config["cache"] = cache_config


def _resolve_layer_scan_paths(config: dict[str, Any], config_path: Path) -> None:
    layer_scan_config = dict(config.get("layer_scan", {}))
    cache_manifest_value = layer_scan_config.get("cache_manifest_path")
    if cache_manifest_value:
        layer_scan_config["cache_manifest_path"] = _resolve_config_path(
            config_path,
            str(cache_manifest_value),
        )
    if layer_scan_config:
        config["layer_scan"] = layer_scan_config


def _resolve_eval_paths(config: dict[str, Any], config_path: Path) -> None:
    eval_config = dict(config.get("eval", {}))
    manifest_value = eval_config.get("manifest_path")
    if manifest_value:
        eval_config["manifest_path"] = _resolve_config_path(config_path, str(manifest_value))
    transplant_config = dict(eval_config.get("transplant", {}))
    layer_configs = transplant_config.get("layers", [])
    if isinstance(layer_configs, list):
        resolved_layers = []
        for layer_config in layer_configs:
            resolved_layer_config = dict(layer_config)
            checkpoint_value = resolved_layer_config.get("checkpoint_path")
            if checkpoint_value:
                resolved_layer_config["checkpoint_path"] = _resolve_config_path(
                    config_path, str(checkpoint_value)
                )
            resolved_layers.append(resolved_layer_config)
        if resolved_layers:
            transplant_config["layers"] = resolved_layers
            eval_config["transplant"] = transplant_config
    if eval_config:
        config["eval"] = eval_config


def _resolve_candidate_eval_paths(config: dict[str, Any], config_path: Path) -> None:
    candidate_config = dict(config.get("candidate_eval", {}))
    for key in (
        "fit_summary_path",
        "checkpoint_path",
        "eval_manifest_path",
        "control_manifest_path",
        "baseline_summary_path",
        "base_control_predictions_path",
        "base_control_summary_path",
        "base_control_metrics_path",
    ):
        value = candidate_config.get(key)
        if value:
            candidate_config[key] = _resolve_config_path(config_path, str(value))
    if candidate_config:
        config["candidate_eval"] = candidate_config


def _resolve_same_size_paths(config: dict[str, Any], config_path: Path) -> None:
    same_size_config = dict(config.get("same_size", {}))
    for key in ("fit_summary_path", "canonical_manifest_path"):
        value = same_size_config.get(key)
        if value:
            same_size_config[key] = _resolve_config_path(config_path, str(value))
    if same_size_config:
        config["same_size"] = same_size_config


def _resolve_prune_feature_paths(config: dict[str, Any], config_path: Path) -> None:
    prune_config = dict(config.get("prune_features", {}))
    for key in (
        "same_size_summary_path",
        "fit_summary_path",
        "selected_eval_summary_path",
        "baseline_summary_path",
        "frozen_eval_manifest_path",
        "frozen_control_manifest_path",
        "selection_eval_manifest_path",
        "selection_control_manifest_path",
    ):
        value = prune_config.get(key)
        if value:
            prune_config[key] = _resolve_config_path(config_path, str(value))
    if prune_config:
        config["prune_features"] = prune_config


def _resolve_shortcut_control_paths(config: dict[str, Any], config_path: Path) -> None:
    shortcut_config = dict(config.get("shortcut_controls", {}))
    for key in (
        "same_size_summary_path",
        "fit_summary_path",
        "selected_eval_summary_path",
        "baseline_summary_path",
        "frozen_eval_manifest_path",
        "frozen_control_manifest_path",
        "prune_summary_path",
    ):
        value = shortcut_config.get(key)
        if value:
            shortcut_config[key] = _resolve_config_path(config_path, str(value))
    if shortcut_config:
        config["shortcut_controls"] = shortcut_config


def _resolve_multiseed_same_size_paths(config: dict[str, Any], config_path: Path) -> None:
    multiseed_config = dict(config.get("multiseed_same_size", {}))
    for key in (
        "fit_summary_path",
        "reference_same_size_summary_path",
        "reference_selected_eval_summary_path",
        "baseline_summary_path",
        "eval_manifest_path",
        "control_manifest_path",
        "shortcut_summary_path",
    ):
        value = multiseed_config.get(key)
        if value:
            multiseed_config[key] = _resolve_config_path(config_path, str(value))
    reuse_existing = dict(multiseed_config.get("reuse_existing_seed_summaries", {}))
    if reuse_existing:
        multiseed_config["reuse_existing_seed_summaries"] = {
            str(seed): _resolve_config_path(config_path, str(path))
            for seed, path in reuse_existing.items()
        }
    if multiseed_config:
        config["multiseed_same_size"] = multiseed_config


def _resolve_multiseed_dense_control_paths(config: dict[str, Any], config_path: Path) -> None:
    multiseed_config = dict(config.get("multiseed_dense_control", {}))
    for key in (
        "fit_summary_path",
        "same_size_summary_path",
        "reference_selected_eval_summary_path",
        "shortcut_summary_path",
        "sparse_multiseed_summary_path",
        "baseline_summary_path",
        "eval_manifest_path",
        "control_manifest_path",
    ):
        value = multiseed_config.get(key)
        if value:
            multiseed_config[key] = _resolve_config_path(config_path, str(value))
    for mapping_key in (
        "reuse_existing_seed_summaries",
        "reuse_existing_training_summaries",
    ):
        mapping = dict(multiseed_config.get(mapping_key, {}))
        if mapping:
            multiseed_config[mapping_key] = {
                str(seed): _resolve_config_path(config_path, str(path))
                for seed, path in mapping.items()
            }
    if multiseed_config:
        config["multiseed_dense_control"] = multiseed_config


def _resolve_paper_artifact_paths(config: dict[str, Any], config_path: Path) -> None:
    paper_config = dict(config.get("paper_artifacts", {}))
    for key in (
        "baseline_summary_path",
        "eval_manifest_path",
        "control_manifest_path",
        "same_size_summary_path",
        "sparse_selected_eval_summary_path",
        "prune_summary_path",
        "shortcut_summary_path",
        "sparse_multiseed_summary_path",
        "dense_multiseed_summary_path",
    ):
        value = paper_config.get(key)
        if value:
            paper_config[key] = _resolve_config_path(config_path, str(value))
    if paper_config:
        config["paper_artifacts"] = paper_config


def _resolve_final_registry_paths(config: dict[str, Any], config_path: Path) -> None:
    registry_config = dict(config.get("final_registry", {}))
    for key in (
        "error_analysis_summary_path",
        "plot_recovery_summary_path",
        "plot_tradeoffs_summary_path",
        "claims_matrix_path",
    ):
        value = registry_config.get(key)
        if value:
            registry_config[key] = _resolve_config_path(config_path, str(value))
    if registry_config:
        config["final_registry"] = registry_config


def _register_progress_artifacts(
    *,
    run_dir: Path,
    artifact_root: Path,
    manifest_key_prefix: str,
    include_resume: bool = False,
) -> None:
    artifacts = {
        f"{manifest_key_prefix}_heartbeat": str((artifact_root / "heartbeat.json").resolve()),
        f"{manifest_key_prefix}_progress": str((artifact_root / "progress.json").resolve()),
    }
    if include_resume:
        artifacts[f"{manifest_key_prefix}_resume_state"] = str(
            (artifact_root / "resume_checkpoint" / "state.pt").resolve()
        )
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "artifacts": artifacts,
        },
    )


def _handle_manifest_smoke(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    logger.info(
        "Initialized run %s for %s/%s",
        manifest["run_id"],
        config["execution_variant"],
        config["milestone"],
    )
    logger.info("Wrote manifest to %s", run_dir)
    return 0


def _handle_smoke_model(config_path: Path, *, strict: bool) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)

    probe_result = probe_model_loading(config.get("model", {}))
    probe_path = run_dir / "smoke_model_probe.json"
    _write_json(probe_path, probe_result.to_dict())
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": probe_result.status,
            "notes": [probe_result.message],
            "artifacts": {
                "smoke_model_probe": str(probe_path),
            },
        },
    )

    logger.info("Run %s probe status: %s", manifest["run_id"], probe_result.status)
    logger.info("%s", probe_result.message)
    if probe_result.blocker_code:
        logger.info("Blocker code: %s", probe_result.blocker_code)

    if probe_result.status == "blocked" and strict:
        return 1
    return 0


def _handle_smoke_data(config_path: Path) -> int:
    logger = configure_logging()
    _, manifest, run_dir = _initialize_run(config_path)
    smoke_root = run_dir / "smoke_data"
    summary = run_smoke_data_pipeline(output_dir=smoke_root)
    summary_path = smoke_root / "summary.json"
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "passed",
            "notes": ["Synthetic M1 smoke-data pipeline completed successfully."],
            "artifacts": {
                "smoke_data_summary": str(summary_path),
            },
        },
    )
    logger.info("Run %s smoke-data status: passed", manifest["run_id"])
    logger.info("Canonical examples: %s", summary["counts"]["canonical_examples"])
    logger.info("Smoke-data summary written to %s", summary_path)
    return 0


def _handle_import_mobile_actions(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    data_config = config.get("data", {})

    raw_path_value = data_config.get("raw_path")
    if not raw_path_value:
        message = "Config is missing data.raw_path for import-mobile-actions."
        blocker_path = run_dir / "mobile_actions_import_blocker.json"
        _write_json(blocker_path, {"status": "blocked", "message": message})
        update_run_manifest(
            run_dir / "run_manifest.json",
            {
                "status": "blocked",
                "notes": [message],
                "artifacts": {"mobile_actions_import_blocker": str(blocker_path)},
            },
        )
        logger.info("%s", message)
        return 1

    raw_path = Path(str(raw_path_value))
    if not raw_path.is_absolute():
        raw_path = (config_path.parent / raw_path).resolve()

    try:
        result = import_mobile_actions_dataset(
            raw_path=raw_path,
            output_dir=run_dir / "mobile_actions_import",
            manifest_id=str(data_config.get("manifest_id", "manifest_mobile_actions_core_v1")),
            prompt_contract_version=str(config.get("prompt_contract_version", "unknown")),
            source=str(data_config.get("source", "mobile_actions")),
            skip_unsupported=bool(data_config.get("skip_unsupported", False)),
        )
    except (FileNotFoundError, MobileActionsParseError) as exc:
        blocker_path = run_dir / "mobile_actions_import_blocker.json"
        _write_json(blocker_path, {"status": "blocked", "message": str(exc)})
        update_run_manifest(
            run_dir / "run_manifest.json",
            {
                "status": "blocked",
                "notes": [str(exc)],
                "artifacts": {"mobile_actions_import_blocker": str(blocker_path)},
            },
        )
        logger.info("Run %s import status: blocked", manifest["run_id"])
        logger.info("%s", exc)
        return 1

    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "passed",
            "notes": ["Mobile Actions-style raw rows imported successfully."],
            "artifacts": {
                "mobile_actions_import_summary": result.summary_path,
                "canonical_manifest": result.canonical_manifest.manifest_path,
                "split_manifest": result.split_manifest_path,
            },
        },
    )

    logger.info("Run %s import status: passed", manifest["run_id"])
    logger.info(
        "Imported %s raw rows into %s canonical examples",
        result.row_count,
        result.canonical_manifest.example_count,
    )
    logger.info("Import summary written to %s", result.summary_path)
    return 0


def _handle_freeze_mobile_actions_eval(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    data_config = config.get("data", {})

    canonical_manifest_value = data_config.get("canonical_manifest_path")
    if not canonical_manifest_value:
        message = "Config is missing data.canonical_manifest_path for freeze-mobile-actions-eval."
        blocker_path = run_dir / "mobile_actions_eval_freeze_blocker.json"
        _write_json(blocker_path, {"status": "blocked", "message": message})
        update_run_manifest(
            run_dir / "run_manifest.json",
            {
                "status": "blocked",
                "notes": [message],
                "artifacts": {"mobile_actions_eval_freeze_blocker": str(blocker_path)},
            },
        )
        logger.info("%s", message)
        return 1

    canonical_manifest_path = Path(str(canonical_manifest_value))
    if not canonical_manifest_path.is_absolute():
        canonical_manifest_path = (config_path.parent / canonical_manifest_path).resolve()

    summary = run_eval_freeze_pipeline_from_manifest(
        canonical_manifest_path=canonical_manifest_path,
        output_dir=run_dir / "mobile_actions_eval_freeze",
    )
    summary_path = Path(summary["summary_path"])
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "passed",
            "notes": ["Frozen real-corpus M1 evaluation artifacts were generated successfully."],
            "artifacts": {
                "mobile_actions_eval_freeze_summary": str(summary_path),
                "evaluation_manifest": summary["evaluation_manifest"]["manifest_path"],
                "control_manifest": summary["control_manifest"]["manifest_path"],
                "leakage_audit": summary["leakage_audit_path"],
                "golden_fixture": summary["golden_fixture_path"],
            },
        },
    )

    logger.info("Run %s eval-freeze status: passed", manifest["run_id"])
    logger.info(
        "Frozen evaluation manifest written to %s", summary["evaluation_manifest"]["manifest_path"]
    )
    logger.info("Leakage audit written to %s", summary["leakage_audit_path"])
    return 0


def _handle_train_donor(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    data_config = dict(config.get("data", {}))
    manifest_value = data_config.get("train_manifest_path")
    if manifest_value:
        data_config["train_manifest_path"] = _resolve_config_path(config_path, str(manifest_value))
        config["data"] = data_config
    output_root = run_dir / "train_donor"
    _register_progress_artifacts(
        run_dir=run_dir,
        artifact_root=output_root,
        manifest_key_prefix="train_donor",
        include_resume=True,
    )
    summary = run_donor_training(config=config, output_dir=output_root)
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary["status"],
            "notes": [f"Donor {summary['train_profile']} training completed successfully."],
            "artifacts": {
                "train_donor_summary": summary["summary_path"],
                "train_donor_serialized_examples": summary["serialized_examples_path"],
                "train_donor_trace": summary["train_trace_path"],
                "train_donor_checkpoint": summary["checkpoint_dir"],
                "train_donor_resume_state": summary["resume_state_path"],
                "train_donor_heartbeat": summary["heartbeat_path"],
                "train_donor_progress": summary["progress_path"],
                "train_donor_post_eval_summary": summary["post_train_eval_summary_path"],
                "train_donor_post_eval_metrics": summary["post_train_eval_metrics_path"],
                "train_donor_post_eval_predictions": summary["post_train_eval_predictions_path"],
            },
        },
    )
    logger.info("Run %s donor-%s status: passed", manifest["run_id"], summary["train_profile"])
    logger.info("Donor %s summary written to %s", summary["train_profile"], summary["summary_path"])
    return 0


def _handle_eval_main(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_eval_paths(config, config_path)
    output_root = run_dir / "eval_main"
    _register_progress_artifacts(
        run_dir=run_dir,
        artifact_root=output_root,
        manifest_key_prefix="eval",
        include_resume=False,
    )
    artifacts = run_eval_pipeline(config=config, output_dir=output_root)
    summary_payload = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload.get("status", "passed"),
            "notes": ["Deterministic evaluation wrapper completed successfully."],
            "artifacts": {
                "eval_summary": artifacts.summary_path,
                "eval_metrics": artifacts.metrics_path,
                "eval_predictions": artifacts.predictions_path,
                "eval_heartbeat": summary_payload["heartbeat_path"],
                "eval_progress": summary_payload["progress_path"],
            },
        },
    )
    logger.info("Run %s eval-main status: passed", manifest["run_id"])
    logger.info("Eval summary written to %s", artifacts.summary_path)
    return 0


def _handle_eval_layer_candidate(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_candidate_eval_paths(config, config_path)
    summary_path = write_layer_candidate_summary(
        config=config,
        output_dir=run_dir / "eval_layer_candidate",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": ["End-to-end single-layer rough-candidate scoring completed successfully."],
            "artifacts": {
                "layer_candidate_summary": str(summary_path.resolve()),
                "candidate_eval_summary": summary_payload["task_eval"]["summary_path"],
                "candidate_control_eval_summary": summary_payload["control_eval"][
                    "candidate_summary_path"
                ],
                "base_control_eval_summary": summary_payload["control_eval"]["base_summary_path"],
            },
        },
    )
    logger.info("Run %s eval-layer-candidate status: passed", manifest["run_id"])
    logger.info("Layer candidate summary written to %s", summary_path)
    return 0


def _handle_fit_same_size_transplant(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_named_model_local_path(config, config_path, "donor_model")
    _resolve_same_size_paths(config, config_path)
    artifacts = run_same_size_fit_pipeline(
        config=config,
        output_dir=run_dir / "fit_same_size_transplant",
    )
    summary_payload = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": [
                (
                    "First M5 same-size fitting slice completed: the calibration bundle "
                    "was frozen and the single-layer gain sweep was written."
                )
            ],
            "artifacts": {
                "same_size_summary": artifacts.summary_path,
                "same_size_checkpoint": artifacts.checkpoint_path,
                "same_size_gain_sweep": artifacts.gain_sweep_path,
                "same_size_calibration_manifest": artifacts.calibration_manifest_path,
                "same_size_calibration_control_manifest": (
                    artifacts.calibration_control_manifest_path
                ),
            },
        },
    )
    logger.info("Run %s fit-same-size-transplant status: passed", manifest["run_id"])
    logger.info("Same-size summary written to %s", artifacts.summary_path)
    return 0


def _handle_prune_features(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_prune_feature_paths(config, config_path)
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "notes": [
                (
                    "M5 feature pruning started: the same-size subset is being frozen on "
                    "calibration manifests before frozen-manifest controls are opened."
                )
            ],
        },
    )
    summary_path = write_pruned_feature_report(
        config=config,
        output_dir=run_dir / "prune_features",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": [
                (
                    "M5 feature pruning completed: the same-size subset was frozen on "
                    "calibration manifests and compared against random same-size controls."
                )
            ],
            "artifacts": {
                "prune_features_summary": str(summary_path.resolve()),
                "prune_features_feature_stats": summary_payload["feature_stats_path"],
                "prune_features_subset_manifest": summary_payload["selected_subset_manifest_path"],
                "prune_features_selected_subset_eval": summary_payload["selected_subset"][
                    "summary_path"
                ],
            },
        },
    )
    logger.info("Run %s prune-features status: passed", manifest["run_id"])
    logger.info("Prune-features summary written to %s", summary_path)
    return 0


def _handle_same_size_shortcut_controls(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_shortcut_control_paths(config, config_path)
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "notes": [
                (
                    "M5 shortcut controls started: dense parameter-matched and "
                    "one-vector steering baselines are being calibrated on the "
                    "same bundle and evaluated on the frozen manifests."
                )
            ],
        },
    )
    summary_path = write_same_size_shortcut_control_report(
        config=config,
        output_dir=run_dir / "same_size_shortcut_controls",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": [
                (
                    "M5 shortcut controls completed: dense and steering baselines "
                    "were calibrated on the same bundle and compared against the "
                    "selected same-size checkpoint on the frozen manifests."
                )
            ],
            "artifacts": {
                "same_size_shortcut_controls_summary": str(summary_path.resolve()),
                "dense_control_frozen_eval": summary_payload["dense_control"]["frozen_eval"][
                    "summary_path"
                ],
                "steering_control_frozen_eval": summary_payload["steering_control"]["frozen_eval"][
                    "summary_path"
                ],
            },
        },
    )
    logger.info("Run %s same-size-shortcut-controls status: passed", manifest["run_id"])
    logger.info("Shortcut-control summary written to %s", summary_path)
    return 0


def _handle_same_size_multiseed(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_multiseed_same_size_paths(config, config_path)
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "notes": [
                (
                    "V24-S6 same-size confirmatory multi-seed reruns started: the locked "
                    "final sparse setting is being re-fit on confirmatory seeds and "
                    "evaluated on the frozen manifests."
                )
            ],
        },
    )
    summary_path = write_same_size_multiseed_report(
        config=config,
        output_dir=run_dir / "same_size_multiseed",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": [
                (
                    "V24-S6 same-size confirmatory multi-seed reruns completed: the locked "
                    "final sparse setting was evaluated across discovery and confirmatory seeds."
                )
            ],
            "artifacts": {
                "same_size_multiseed_summary": str(summary_path.resolve()),
            },
        },
    )
    logger.info("Run %s same-size-multiseed status: passed", manifest["run_id"])
    logger.info("Same-size multiseed summary written to %s", summary_path)
    return 0


def _handle_same_size_dense_multiseed_control(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_multiseed_dense_control_paths(config, config_path)
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "running",
            "notes": [
                (
                    "Bounded M5 dense-control multi-seed reruns started: the parameter-matched "
                    "dense shortcut is being refit on the same seeds and frozen manifests as "
                    "the sparse confirmatory slice."
                )
            ],
        },
    )
    summary_path = write_dense_control_multiseed_report(
        config=config,
        output_dir=run_dir / "dense_control_multiseed",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": [
                (
                    "Bounded M5 dense-control multi-seed reruns completed: the parameter-matched "
                    "dense shortcut was evaluated across the same seeds and frozen manifests as "
                    "the sparse confirmatory slice."
                )
            ],
            "artifacts": {
                "dense_control_multiseed_summary": str(summary_path.resolve()),
            },
        },
    )
    logger.info(
        "Run %s same-size-dense-multiseed-control status: passed",
        manifest["run_id"],
    )
    logger.info("Dense multiseed control summary written to %s", summary_path)
    return 0


def _handle_error_analysis(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_paper_artifact_paths(config, config_path)
    summary_path = write_error_analysis_report(
        config=config,
        output_dir=run_dir / "error_analysis",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": ["M7 error analysis artifacts were generated from saved prediction files."],
            "artifacts": {
                "error_analysis_summary": str(summary_path.resolve()),
                "error_category_table": summary_payload["error_category_table_path"],
                "strict_vs_semantic_table": summary_payload["strict_vs_semantic_table_path"],
                "appendix_examples": summary_payload["appendix_examples_path"],
            },
        },
    )
    logger.info("Run %s error-analysis status: passed", manifest["run_id"])
    logger.info("Error-analysis summary written to %s", summary_path)
    return 0


def _handle_plot_recovery(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_paper_artifact_paths(config, config_path)
    summary_path = write_recovery_artifacts(
        config=config,
        output_dir=run_dir / "plot_recovery",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": ["M7 recovery, budget, and retained-gain artifacts were generated."],
            "artifacts": {
                "plot_recovery_summary": str(summary_path.resolve()),
                "baseline_table": summary_payload["baseline_table_path"],
                "parameter_budget_table": summary_payload["parameter_budget_table_path"],
                "retained_gain_table": summary_payload["retained_gain_table_path"],
                "recovery_vs_parameters_figure": summary_payload[
                    "recovery_vs_parameters_figure_path"
                ],
                "retained_gain_figure": summary_payload["retained_gain_figure_path"],
            },
        },
    )
    logger.info("Run %s plot-recovery status: passed", manifest["run_id"])
    logger.info("Plot-recovery summary written to %s", summary_path)
    return 0


def _handle_plot_tradeoffs(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_paper_artifact_paths(config, config_path)
    summary_path = write_tradeoff_artifacts(
        config=config,
        output_dir=run_dir / "plot_tradeoffs",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": ["M7 same-size/control tradeoff tables and plots were generated."],
            "artifacts": {
                "plot_tradeoffs_summary": str(summary_path.resolve()),
                "same_size_vs_controls_table": summary_payload["same_size_vs_controls_table_path"],
                "calibration_sensitivity_table": summary_payload[
                    "calibration_sensitivity_table_path"
                ],
                "per_slice_metrics_table": summary_payload["per_slice_metrics_table_path"],
                "calibration_sensitivity_figure": summary_payload[
                    "calibration_sensitivity_figure_path"
                ],
                "control_tradeoff_figure": summary_payload["control_tradeoff_figure_path"],
                "per_slice_figure": summary_payload["per_slice_figure_path"],
            },
        },
    )
    logger.info("Run %s plot-tradeoffs status: passed", manifest["run_id"])
    logger.info("Plot-tradeoffs summary written to %s", summary_path)
    return 0


def _handle_export_final_registry(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_paper_artifact_paths(config, config_path)
    _resolve_final_registry_paths(config, config_path)
    summary_path = export_final_registry(
        config=config,
        output_dir=run_dir / "final_registry",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary_payload["status"],
            "notes": ["M8 final claim audit and artifact registry were generated."],
            "artifacts": {
                "final_registry_summary": str(summary_path.resolve()),
                "final_claim_audit": summary_payload["final_claim_audit_path"],
                "artifact_inventory": summary_payload["artifact_inventory_path"],
                "section_to_artifact_map": summary_payload["section_to_artifact_map_path"],
                "claim_audit_checklist": summary_payload["claim_audit_checklist_path"],
            },
        },
    )
    logger.info("Run %s export-final-registry status: passed", manifest["run_id"])
    logger.info("Final-registry summary written to %s", summary_path)
    return 0


def _handle_train_recipient_baselines(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    data_config = dict(config.get("data", {}))
    manifest_value = data_config.get("train_manifest_path")
    if manifest_value:
        data_config["train_manifest_path"] = _resolve_config_path(config_path, str(manifest_value))
        config["data"] = data_config
    output_root = run_dir / "train_recipient"
    _register_progress_artifacts(
        run_dir=run_dir,
        artifact_root=output_root,
        manifest_key_prefix="train_recipient",
        include_resume=True,
    )

    baseline_kind = str(config.get("train", {}).get("baseline_kind", "")).lower()
    if baseline_kind in {"small_data_lora", "full_data_lora"}:
        summary = run_recipient_lora_smoke_training(
            config=config,
            output_dir=output_root,
        )
        artifacts = {
            "train_recipient_summary": summary["summary_path"],
            "train_recipient_serialized_examples": summary["serialized_examples_path"],
            "train_recipient_trace": summary["train_trace_path"],
            "train_recipient_adapter_checkpoint": summary["adapter_checkpoint_dir"],
            "train_recipient_merged_checkpoint": summary["merged_checkpoint_dir"],
            "train_recipient_post_eval_summary": summary["post_train_eval_summary_path"],
            "train_recipient_post_eval_metrics": summary["post_train_eval_metrics_path"],
            "train_recipient_post_eval_predictions": summary["post_train_eval_predictions_path"],
        }
    elif baseline_kind == "full_data_fullft":
        summary = run_recipient_fullft_smoke_training(
            config=config,
            output_dir=output_root,
        )
        artifacts = {
            "train_recipient_summary": summary["summary_path"],
            "train_recipient_serialized_examples": summary["serialized_examples_path"],
            "train_recipient_trace": summary["train_trace_path"],
            "train_recipient_checkpoint": summary["checkpoint_dir"],
            "train_recipient_post_eval_summary": summary["post_train_eval_summary_path"],
            "train_recipient_post_eval_metrics": summary["post_train_eval_metrics_path"],
            "train_recipient_post_eval_predictions": summary["post_train_eval_predictions_path"],
        }
    else:
        raise ValueError(
            "train.baseline_kind must be one of "
            "'small_data_lora', 'full_data_lora', or 'full_data_fullft'."
        )

    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary["status"],
            "notes": [f"Recipient baseline smoke training completed for {baseline_kind}."],
            "artifacts": {
                **artifacts,
                "train_recipient_heartbeat": summary.get("heartbeat_path"),
                "train_recipient_progress": summary.get("progress_path"),
                "train_recipient_resume_state": summary.get("resume_state_path"),
            },
        },
    )
    logger.info("Run %s recipient-smoke status: passed", manifest["run_id"])
    logger.info("Recipient smoke summary written to %s", summary["summary_path"])
    return 0


def _handle_cache_activations(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_model_local_path(config, config_path)
    _resolve_named_model_local_path(config, config_path, "donor_model")
    _resolve_cache_paths(config, config_path)
    summary = collect_activation_caches(config=config, output_dir=run_dir / "cache_activations")
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": summary["status"],
            "notes": ["Teacher-forced donor/base activation caches were collected successfully."],
            "artifacts": {
                "cache_summary": summary["summary_path"],
                "cache_manifest": summary["cache_manifest_path"],
                "cache_hook_audit": summary["hook_audit_path"],
                "cache_heartbeat": summary["heartbeat_path"],
                "cache_progress": summary["progress_path"],
            },
        },
    )
    logger.info("Run %s cache-activations status: passed", manifest["run_id"])
    logger.info("Cache summary written to %s", summary["summary_path"])
    return 0


def _handle_rank_layers(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_layer_scan_paths(config, config_path)
    ranking_path = write_layer_ranking_report(config=config, output_dir=run_dir / "rank_layers")
    ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": ranking_payload["status"],
            "notes": [
                (
                    "Rough sparse delta modules were fit and the candidate layer proxy "
                    "ranking report was written."
                )
            ],
            "artifacts": {
                "layer_ranking": str(ranking_path.resolve()),
                "rank_layers_heartbeat": ranking_payload["heartbeat_path"],
                "rank_layers_progress": ranking_payload["progress_path"],
            },
        },
    )
    logger.info("Run %s rank-layers status: passed", manifest["run_id"])
    logger.info("Layer ranking report written to %s", ranking_path)
    return 0


def _handle_param_budget(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    report_path = write_budget_report(config=config, output_dir=run_dir / "param_budget")
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "passed",
            "notes": ["Parameter-budget smoke artifact written successfully."],
            "artifacts": {"param_budget_report": str(report_path.resolve())},
        },
    )
    logger.info("Run %s param-budget status: passed", manifest["run_id"])
    logger.info("Budget report written to %s", report_path)
    return 0


def _handle_summarize_baselines(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_analysis_paths(config, config_path)
    summary_path = write_baseline_summary(config=config, output_dir=run_dir / "baseline_summary")
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "passed",
            "notes": ["Baseline summary and donor-gap metric computed successfully."],
            "artifacts": {"baseline_summary": str(summary_path.resolve())},
        },
    )
    logger.info("Run %s summarize-baselines status: passed", manifest["run_id"])
    logger.info("Baseline summary written to %s", summary_path)
    return 0


def _handle_donor_gap_gate(config_path: Path) -> int:
    logger = configure_logging()
    config, manifest, run_dir = _initialize_run(config_path)
    _resolve_analysis_paths(config, config_path)
    gate_path = write_donor_gap_gate(config=config, output_dir=run_dir / "donor_gap_gate")
    update_run_manifest(
        run_dir / "run_manifest.json",
        {
            "status": "passed",
            "notes": ["Donor-gap R20 gate decision written successfully."],
            "artifacts": {"donor_gap_gate": str(gate_path.resolve())},
        },
    )
    logger.info("Run %s donor-gap-gate status: passed", manifest["run_id"])
    logger.info("Donor-gap gate written to %s", gate_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        if args.command == "manifest-smoke":
            return _handle_manifest_smoke(config_path)
        if args.command == "smoke-model":
            return _handle_smoke_model(config_path, strict=args.strict)
        if args.command == "smoke-data":
            return _handle_smoke_data(config_path)
        if args.command == "import-mobile-actions":
            return _handle_import_mobile_actions(config_path)
        if args.command == "train-donor":
            return _handle_train_donor(config_path)
        if args.command == "train-recipient-baselines":
            return _handle_train_recipient_baselines(config_path)
        if args.command == "cache-activations":
            return _handle_cache_activations(config_path)
        if args.command == "rank-layers":
            return _handle_rank_layers(config_path)
        if args.command == "eval-layer-candidate":
            return _handle_eval_layer_candidate(config_path)
        if args.command == "eval-main":
            return _handle_eval_main(config_path)
        if args.command == "fit-same-size-transplant":
            return _handle_fit_same_size_transplant(config_path)
        if args.command == "prune-features":
            return _handle_prune_features(config_path)
        if args.command == "same-size-shortcut-controls":
            return _handle_same_size_shortcut_controls(config_path)
        if args.command == "same-size-multiseed":
            return _handle_same_size_multiseed(config_path)
        if args.command == "same-size-dense-multiseed-control":
            return _handle_same_size_dense_multiseed_control(config_path)
        if args.command == "error-analysis":
            return _handle_error_analysis(config_path)
        if args.command == "plot-recovery":
            return _handle_plot_recovery(config_path)
        if args.command == "plot-tradeoffs":
            return _handle_plot_tradeoffs(config_path)
        if args.command == "export-final-registry":
            return _handle_export_final_registry(config_path)
        if args.command == "param-budget":
            return _handle_param_budget(config_path)
        if args.command == "summarize-baselines":
            return _handle_summarize_baselines(config_path)
        if args.command == "donor-gap-gate":
            return _handle_donor_gap_gate(config_path)
        if args.command == "freeze-mobile-actions-eval":
            return _handle_freeze_mobile_actions_eval(config_path)
        parser.error(f"Unsupported command: {args.command}")
        return 2
    except KeyboardInterrupt:
        print("Interrupted; resumable state and heartbeat were preserved where supported.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
