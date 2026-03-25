from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.analysis.paper_artifacts import (
    grouped_primary_metrics,
    layer_candidate_summary,
    prompt_contract_version,
    read_jsonl,
    write_csv,
    write_json,
)


def _gain_sweep_rows(gain_sweep_path: str | Path, *, label: str) -> list[dict[str, Any]]:
    payload = layer_candidate_summary(gain_sweep_path)
    rows: list[dict[str, Any]] = []
    for result in payload["results"]:
        gain_value = result.get("gain")
        if gain_value is None:
            gain_value = result.get("candidate", {}).get("gain")
        if gain_value is None:
            raise KeyError(f"{gain_sweep_path} result is missing gain/candidate.gain.")
        rows.append(
            {
                "system_label": label,
                "gain": float(gain_value),
                "primary_strict": float(
                    result["task_eval"]["grouped_metrics"]["primary"]["strict_full_call_success"]
                ),
                "semantic_success": float(
                    result["task_eval"]["grouped_metrics"]["primary"]["semantic_full_call_success"]
                ),
                "control_drop": float(result["control_eval"]["control_drop"]),
            }
        )
    return rows


def _primary_slice_rows(
    *,
    system_id: str,
    label: str,
    predictions_path: str | Path,
    prompt_version: str,
) -> list[dict[str, Any]]:
    metrics = grouped_primary_metrics(read_jsonl(predictions_path))
    return [
        {
            "system_id": system_id,
            "system_label": label,
            "slice": slice_name,
            "prompt_contract_version": prompt_version,
            "strict_full_call_success": values["strict_full_call_success"],
            "semantic_full_call_success": values["semantic_full_call_success"],
        }
        for slice_name, values in metrics.items()
    ]


def write_tradeoff_artifacts(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    paper_config = dict(config.get("paper_artifacts", {}))
    required_paths = (
        "baseline_summary_path",
        "eval_manifest_path",
        "same_size_summary_path",
        "sparse_selected_eval_summary_path",
        "prune_summary_path",
        "shortcut_summary_path",
        "sparse_multiseed_summary_path",
        "dense_multiseed_summary_path",
    )
    for key in required_paths:
        if not paper_config.get(key):
            raise ValueError(f"paper_artifacts.{key} is required.")

    prompt_version = prompt_contract_version(paper_config["eval_manifest_path"])
    baseline_summary = layer_candidate_summary(paper_config["baseline_summary_path"])
    same_size_summary = layer_candidate_summary(paper_config["same_size_summary_path"])
    sparse_selected_summary = layer_candidate_summary(
        paper_config["sparse_selected_eval_summary_path"]
    )
    prune_summary = layer_candidate_summary(paper_config["prune_summary_path"])
    shortcut_summary = layer_candidate_summary(paper_config["shortcut_summary_path"])
    sparse_multiseed = layer_candidate_summary(paper_config["sparse_multiseed_summary_path"])
    dense_multiseed = layer_candidate_summary(paper_config["dense_multiseed_summary_path"])
    dense_frozen_summary = layer_candidate_summary(
        shortcut_summary["dense_control"]["frozen_eval"]["summary_path"]
    )
    steering_frozen_summary = layer_candidate_summary(
        shortcut_summary["steering_control"]["frozen_eval"]["summary_path"]
    )

    destination = Path(output_dir)
    tables_dir = destination / "tables"
    figures_dir = destination / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    base_primary = float(baseline_summary["primary_metric"]["base_value"])
    controls_rows = [
        {
            "system_id": "base",
            "system_label": "Base",
            "primary_strict": base_primary,
            "donor_gap_recovery": 0.0,
            "control_drop": 0.0,
            "evidence_level": "single_seed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "donor",
            "system_label": "Donor",
            "primary_strict": float(baseline_summary["primary_metric"]["donor_value"]),
            "donor_gap_recovery": 1.0,
            "control_drop": 0.0,
            "evidence_level": "single_seed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "sparse",
            "system_label": "Sparse same-size (seed 17)",
            "primary_strict": float(
                sparse_selected_summary["task_eval"]["grouped_metrics"]["primary"][
                    "strict_full_call_success"
                ]
            ),
            "donor_gap_recovery": float(sparse_selected_summary["donor_gap_recovery"]),
            "control_drop": float(sparse_selected_summary["control_eval"]["control_drop"]),
            "evidence_level": "single_seed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "dense",
            "system_label": "Dense shortcut (seed 17)",
            "primary_strict": float(
                shortcut_summary["dense_control"]["frozen_eval"]["primary_strict"]
            ),
            "donor_gap_recovery": float(
                shortcut_summary["dense_control"]["frozen_eval"]["donor_gap_recovery"]
            ),
            "control_drop": float(shortcut_summary["dense_control"]["frozen_eval"]["control_drop"]),
            "evidence_level": "single_seed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "steering",
            "system_label": "Steering shortcut (seed 17)",
            "primary_strict": float(
                shortcut_summary["steering_control"]["frozen_eval"]["primary_strict"]
            ),
            "donor_gap_recovery": float(
                shortcut_summary["steering_control"]["frozen_eval"]["donor_gap_recovery"]
            ),
            "control_drop": float(
                shortcut_summary["steering_control"]["frozen_eval"]["control_drop"]
            ),
            "evidence_level": "single_seed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "subset",
            "system_label": "Selected 1-feature subset",
            "primary_strict": float(prune_summary["selected_subset"]["frozen_primary_strict"]),
            "donor_gap_recovery": None,
            "control_drop": float(prune_summary["selected_subset"]["frozen_control_drop"]),
            "evidence_level": "single_seed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "random_subset_mean",
            "system_label": "Random 1-feature mean",
            "primary_strict": float(prune_summary["random_subset_controls"]["mean_primary_strict"]),
            "donor_gap_recovery": None,
            "control_drop": float(prune_summary["random_subset_controls"]["mean_control_drop"]),
            "evidence_level": "single_seed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "sparse_multiseed",
            "system_label": "Sparse same-size multiseed",
            "primary_strict": float(sparse_multiseed["aggregate"]["primary_strict"]["mean"]),
            "donor_gap_recovery": float(
                sparse_multiseed["aggregate"]["donor_gap_recovery"]["mean"]
            ),
            "control_drop": float(sparse_multiseed["aggregate"]["control_drop"]["mean"]),
            "evidence_level": "multiseed",
            "prompt_contract_version": prompt_version,
        },
        {
            "system_id": "dense_multiseed",
            "system_label": "Dense shortcut multiseed",
            "primary_strict": float(dense_multiseed["aggregate"]["primary_strict"]["mean"]),
            "donor_gap_recovery": float(dense_multiseed["aggregate"]["donor_gap_recovery"]["mean"]),
            "control_drop": float(dense_multiseed["aggregate"]["control_drop"]["mean"]),
            "evidence_level": "multiseed",
            "prompt_contract_version": prompt_version,
        },
    ]
    controls_json = write_json(
        tables_dir / "same_size_vs_controls_table.json",
        {"rows": controls_rows},
    )
    controls_csv = write_csv(
        tables_dir / "same_size_vs_controls_table.csv",
        controls_rows,
    )

    sparse_gain_rows = _gain_sweep_rows(
        same_size_summary["gain_sweep_path"],
        label="Sparse same-size",
    )
    shortcut_root = Path(shortcut_summary["summary_path"]).parent
    dense_gain_rows = _gain_sweep_rows(
        shortcut_root / "dense_control" / "calibration" / "gain_sweep.json",
        label="Dense shortcut",
    )
    steering_gain_rows = _gain_sweep_rows(
        shortcut_root / "steering_control" / "calibration" / "gain_sweep.json",
        label="Steering shortcut",
    )
    sensitivity_rows = sparse_gain_rows + dense_gain_rows + steering_gain_rows
    sensitivity_json = write_json(
        tables_dir / "calibration_sensitivity_table.json",
        {"rows": sensitivity_rows},
    )
    sensitivity_csv = write_csv(
        tables_dir / "calibration_sensitivity_table.csv",
        sensitivity_rows,
    )

    sensitivity_figure = figures_dir / "calibration_sensitivity_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, rows in (
        ("Sparse same-size", sparse_gain_rows),
        ("Dense shortcut", dense_gain_rows),
        ("Steering shortcut", steering_gain_rows),
    ):
        axes[0].plot(
            [row["gain"] for row in rows],
            [row["primary_strict"] for row in rows],
            marker="o",
            label=label,
        )
        axes[1].plot(
            [row["gain"] for row in rows],
            [row["control_drop"] for row in rows],
            marker="o",
            label=label,
        )
    axes[0].set_title("Primary strict vs gain")
    axes[0].set_xlabel("Gain")
    axes[0].set_ylabel("Primary strict success")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Control drop vs gain")
    axes[1].set_xlabel("Gain")
    axes[1].set_ylabel("Control drop")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(sensitivity_figure, dpi=180)
    plt.close(fig)

    tradeoff_figure = figures_dir / "control_drop_vs_primary_gain.png"
    plt.figure(figsize=(7, 4.5))
    for row in controls_rows[2:]:
        if row["primary_strict"] is None:
            continue
        plt.scatter(
            float(row["control_drop"]),
            float(row["primary_strict"]) - base_primary,
            s=60,
            label=row["system_label"],
        )
    plt.axvline(0.02, color="tab:red", linestyle="--", linewidth=1)
    plt.xlabel("Control drop")
    plt.ylabel("Primary strict improvement over base")
    plt.title("Primary gain vs control drop")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(tradeoff_figure, dpi=180)
    plt.close()

    per_slice_rows = (
        _primary_slice_rows(
            system_id="base",
            label="Base",
            predictions_path=baseline_summary["base_predictions_path"],
            prompt_version=prompt_version,
        )
        + _primary_slice_rows(
            system_id="donor",
            label="Donor",
            predictions_path=baseline_summary["donor_predictions_path"],
            prompt_version=prompt_version,
        )
        + _primary_slice_rows(
            system_id="sparse",
            label="Sparse same-size (seed 17)",
            predictions_path=sparse_selected_summary["task_eval"]["predictions_path"],
            prompt_version=prompt_version,
        )
        + _primary_slice_rows(
            system_id="dense",
            label="Dense shortcut (seed 17)",
            predictions_path=dense_frozen_summary["task_eval"]["predictions_path"],
            prompt_version=prompt_version,
        )
        + _primary_slice_rows(
            system_id="steering",
            label="Steering shortcut (seed 17)",
            predictions_path=steering_frozen_summary["task_eval"]["predictions_path"],
            prompt_version=prompt_version,
        )
    )
    per_slice_json = write_json(
        tables_dir / "per_slice_metrics_table.json",
        {"rows": per_slice_rows},
    )
    per_slice_csv = write_csv(
        tables_dir / "per_slice_metrics_table.csv",
        per_slice_rows,
    )

    slices = ["schema_shift", "nocall_missing_tool", "nocall_unsupported", "primary"]
    systems = [
        ("base", "Base"),
        ("donor", "Donor"),
        ("sparse", "Sparse"),
        ("dense", "Dense"),
        ("steering", "Steering"),
    ]
    per_slice_figure = figures_dir / "per_slice_primary_vs_semantic.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x_positions = range(len(slices))
    for system_index, (system_id, label) in enumerate(systems):
        system_rows = {row["slice"]: row for row in per_slice_rows if row["system_id"] == system_id}
        offset = (system_index - 2) * 0.14
        axes[0].bar(
            [position + offset for position in x_positions],
            [system_rows[slice_name]["strict_full_call_success"] for slice_name in slices],
            width=0.12,
            label=label,
        )
        axes[1].bar(
            [position + offset for position in x_positions],
            [system_rows[slice_name]["semantic_full_call_success"] for slice_name in slices],
            width=0.12,
            label=label,
        )
    axes[0].set_ylabel("Strict success")
    axes[0].set_title("Per-slice strict vs semantic metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("Semantic success")
    axes[1].set_xticks(list(x_positions), slices)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(per_slice_figure, dpi=180)
    plt.close(fig)

    summary = {
        "status": "passed",
        "prompt_contract_version": prompt_version,
        "same_size_vs_controls_table_path": str(controls_json.resolve()),
        "same_size_vs_controls_table_csv_path": str(controls_csv.resolve()),
        "calibration_sensitivity_table_path": str(sensitivity_json.resolve()),
        "calibration_sensitivity_table_csv_path": str(sensitivity_csv.resolve()),
        "per_slice_metrics_table_path": str(per_slice_json.resolve()),
        "per_slice_metrics_table_csv_path": str(per_slice_csv.resolve()),
        "calibration_sensitivity_figure_path": str(sensitivity_figure.resolve()),
        "control_tradeoff_figure_path": str(tradeoff_figure.resolve()),
        "per_slice_figure_path": str(per_slice_figure.resolve()),
        "matched_dense_outperforms_sparse": bool(
            dense_multiseed["comparison_to_sparse_multiseed"]["sparse_beats_dense_mean"] is False
        ),
    }
    return write_json(destination / "summary.json", summary)
