from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.train.train_delta_module import fit_layer_delta_module
from src.utils.config import load_config
from src.utils.progress import RunHeartbeat


def _load_cache_manifest(cache_manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))


def _fit_destination(output_dir: Path, *, layer_id: int, topk: int) -> Path:
    return output_dir / "fits" / f"layer_{layer_id:02d}_k_{topk:02d}"


def _score_fit_summary(summary: dict[str, Any], *, ranking_metric: str) -> float:
    val_metrics = dict(summary.get("val_metrics", {}))
    if ranking_metric == "negative_weighted_mse":
        return -float(val_metrics["weighted_mse"])
    if ranking_metric not in val_metrics:
        raise ValueError(
            f"Unsupported layer_scan.ranking_metric {ranking_metric!r}; "
            f"available val metrics: {sorted(val_metrics)}"
        )
    return float(val_metrics[ranking_metric])


def build_layer_ranking_report(*, config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    layer_scan_config = dict(config.get("layer_scan", {}))
    cache_manifest_path = layer_scan_config.get("cache_manifest_path")
    if not cache_manifest_path:
        raise ValueError("Config is missing layer_scan.cache_manifest_path.")

    cache_manifest = _load_cache_manifest(cache_manifest_path)
    layer_ids = [
        int(layer_id)
        for layer_id in layer_scan_config.get("layer_ids", cache_manifest["layer_ids"])
    ]
    if not layer_ids:
        raise ValueError("layer_scan.layer_ids resolved to an empty candidate set.")
    topk_values = [int(value) for value in layer_scan_config.get("topk_values", [8, 16])]
    if not topk_values:
        raise ValueError("layer_scan.topk_values must not be empty.")
    ranking_metric = str(layer_scan_config.get("ranking_metric", "explained_fraction_vs_zero"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    heartbeat = RunHeartbeat(
        output_dir=destination,
        phase="rank_layers",
        total_units=len(layer_ids) * len(topk_values),
        unit_name="fits",
        heartbeat_interval_seconds=float(layer_scan_config.get("heartbeat_interval_seconds", 10.0)),
    )
    heartbeat.start(
        message="Started M4 rough layer ranking.",
        extra={
            "ranking_mode": "reconstruction_proxy",
            "candidate_layers": layer_ids,
            "topk_values": topk_values,
        },
    )

    fit_summaries: list[dict[str, Any]] = []
    completed_fits = 0
    reuse_existing = bool(layer_scan_config.get("reuse_existing_fits", True))
    for layer_id in layer_ids:
        for topk in topk_values:
            fit_dir = _fit_destination(destination, layer_id=layer_id, topk=topk)
            fit_summary_path = fit_dir / "summary.json"
            if reuse_existing and fit_summary_path.exists():
                summary = json.loads(fit_summary_path.read_text(encoding="utf-8"))
                summary["summary_path"] = str(fit_summary_path.resolve())
            else:
                summary = fit_layer_delta_module(
                    config=config,
                    output_dir=fit_dir,
                    layer_id=layer_id,
                    topk=topk,
                )
            summary["ranking_score"] = _score_fit_summary(summary, ranking_metric=ranking_metric)
            fit_summaries.append(summary)
            completed_fits += 1
            heartbeat.maybe_update(
                completed_units=completed_fits,
                message=(
                    f"Processed rough fit {completed_fits} of {len(layer_ids) * len(topk_values)} "
                    f"(layer {layer_id}, TopK {topk})."
                ),
                metrics={"best_score_so_far": max(item["ranking_score"] for item in fit_summaries)},
                extra={"last_fit_summary_path": summary["summary_path"]},
            )

    layer_best_results = []
    for layer_id in layer_ids:
        candidates = [summary for summary in fit_summaries if int(summary["layer_id"]) == layer_id]
        best = max(
            candidates,
            key=lambda item: (item["ranking_score"], -item["val_metrics"]["weighted_mse"]),
        )
        layer_best_results.append(
            {
                "layer_id": layer_id,
                "best_topk": int(best["topk"]),
                "ranking_score": float(best["ranking_score"]),
                "val_metrics": dict(best["val_metrics"]),
                "shortcut_controls": dict(best["shortcut_controls"]),
                "summary_path": best["summary_path"],
                "checkpoint_path": best["checkpoint_path"],
            }
        )

    ranked_layers = sorted(
        layer_best_results,
        key=lambda item: (item["ranking_score"], -item["val_metrics"]["weighted_mse"]),
        reverse=True,
    )
    best_overall = ranked_layers[0]
    proceed_min_explained_fraction = float(
        layer_scan_config.get("proceed_min_explained_fraction", 0.1)
    )
    proceed_min_control_margin = float(layer_scan_config.get("proceed_min_control_margin", 0.0))
    best_val_metrics = dict(best_overall["val_metrics"])
    proxy_passes = (
        float(best_val_metrics["explained_fraction_vs_zero"]) >= proceed_min_explained_fraction
        and float(best_val_metrics["improvement_over_mean_delta_control"])
        > proceed_min_control_margin
    )
    proceed_reason = (
        "proxy_reconstruction_signal_clears_thresholds"
        if proxy_passes
        else "proxy_reconstruction_signal_did_not_clear_thresholds"
    )

    report = {
        "status": "passed",
        "ranking_mode": "reconstruction_proxy",
        "claim_bearing": False,
        "cache_manifest_path": str(Path(cache_manifest_path).resolve()),
        "cache_version": str(cache_manifest["cache_version"]),
        "candidate_layers": layer_ids,
        "topk_values": topk_values,
        "latent_width": int(layer_scan_config.get("latent_width", 256)),
        "ranking_metric": ranking_metric,
        "fit_count": len(fit_summaries),
        "layer_rankings": ranked_layers,
        "fit_summaries": [
            {
                "layer_id": int(summary["layer_id"]),
                "topk": int(summary["topk"]),
                "ranking_score": float(summary["ranking_score"]),
                "summary_path": summary["summary_path"],
            }
            for summary in sorted(
                fit_summaries,
                key=lambda item: (int(item["layer_id"]), int(item["topk"])),
            )
        ],
        "proxy_proceed_decision": {
            "status": "pass" if proxy_passes else "hold",
            "reason": proceed_reason,
            "thresholds": {
                "min_explained_fraction_vs_zero": proceed_min_explained_fraction,
                "min_improvement_over_mean_delta_control": proceed_min_control_margin,
            },
            "best_layer_id": int(best_overall["layer_id"]),
            "best_topk": int(best_overall["best_topk"]),
        },
        "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
        "progress_path": str(heartbeat.paths.progress_path.resolve()),
        "notes": [
            "This ranking report is an M4 reconstruction-proxy diagnostic only.",
            (
                "The locked paper path still requires end-to-end single-layer injection "
                "and validation scoring before M4 can count as claim-bearingly ranked."
            ),
        ],
    }
    report_path = destination / "layer_ranking.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    heartbeat.mark_completed(
        completed_units=len(layer_ids) * len(topk_values),
        message="M4 rough layer ranking completed successfully.",
        metrics={"best_score": float(best_overall["ranking_score"])},
        extra={"layer_ranking_path": str(report_path.resolve())},
    )
    report["summary_path"] = str(report_path.resolve())
    return report


def write_layer_ranking_report(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    report = build_layer_ranking_report(config=config, output_dir=output_dir)
    return Path(report["summary_path"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the M4 rough layer ranking pipeline.")
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML/TOML config file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the ranking artifact directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (Path("runs") / "adhoc_layer_scan").resolve()
    )
    report_path = write_layer_ranking_report(config=config, output_dir=output_dir)
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
