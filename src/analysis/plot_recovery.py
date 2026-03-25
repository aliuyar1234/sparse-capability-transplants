from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.analysis.paper_artifacts import load_json, prompt_contract_version, write_csv, write_json


def _figure_path(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.png"


def write_recovery_artifacts(*, config: dict[str, Any], output_dir: str | Path) -> Path:
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

    baseline_summary = load_json(paper_config["baseline_summary_path"])
    same_size_summary = load_json(paper_config["same_size_summary_path"])
    sparse_selected_summary = load_json(paper_config["sparse_selected_eval_summary_path"])
    prune_summary = load_json(paper_config["prune_summary_path"])
    shortcut_summary = load_json(paper_config["shortcut_summary_path"])
    sparse_multiseed = load_json(paper_config["sparse_multiseed_summary_path"])
    dense_multiseed = load_json(paper_config["dense_multiseed_summary_path"])
    prompt_version = prompt_contract_version(paper_config["eval_manifest_path"])

    destination = Path(output_dir)
    tables_dir = destination / "tables"
    figures_dir = destination / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict[str, Any]] = []
    for system_id, label, metrics_key in (
        ("base", "Base", "base_metrics"),
        ("donor", "Donor", "donor_metrics"),
    ):
        system_metrics = baseline_summary[metrics_key]
        for slice_name, metrics in system_metrics.items():
            baseline_rows.append(
                {
                    "system_id": system_id,
                    "system_label": label,
                    "slice": slice_name,
                    "prompt_contract_version": prompt_version,
                    **metrics,
                }
            )
    baseline_rows.sort(key=lambda row: (row["system_id"], row["slice"]))
    baseline_json = write_json(tables_dir / "baseline_table.json", {"rows": baseline_rows})
    baseline_csv = write_csv(tables_dir / "baseline_table.csv", baseline_rows)

    steering_vector_summary = load_json(shortcut_summary["steering_control"]["vector_summary_path"])
    parameter_rows = [
        {
            "system_id": "sparse",
            "system_label": "Sparse same-size",
            "added_params": same_size_summary["parameter_budget"]["added_params"],
            "donor_gap_recovery": sparse_selected_summary["donor_gap_recovery"],
            "prompt_contract_version": prompt_version,
            "evidence_level": "single_seed",
        },
        {
            "system_id": "dense",
            "system_label": "Dense shortcut",
            "added_params": shortcut_summary["dense_control"]["budget"]["dense_params"],
            "donor_gap_recovery": shortcut_summary["dense_control"]["frozen_eval"][
                "donor_gap_recovery"
            ],
            "prompt_contract_version": prompt_version,
            "evidence_level": "single_seed",
        },
        {
            "system_id": "steering",
            "system_label": "Steering shortcut",
            "added_params": steering_vector_summary["input_dim"],
            "donor_gap_recovery": shortcut_summary["steering_control"]["frozen_eval"][
                "donor_gap_recovery"
            ],
            "prompt_contract_version": prompt_version,
            "evidence_level": "single_seed",
        },
        {
            "system_id": "sparse_multiseed",
            "system_label": "Sparse same-size multiseed",
            "added_params": same_size_summary["parameter_budget"]["added_params"],
            "donor_gap_recovery": sparse_multiseed["aggregate"]["donor_gap_recovery"]["mean"],
            "prompt_contract_version": prompt_version,
            "evidence_level": "multiseed",
        },
        {
            "system_id": "dense_multiseed",
            "system_label": "Dense shortcut multiseed",
            "added_params": shortcut_summary["dense_control"]["budget"]["dense_params"],
            "donor_gap_recovery": dense_multiseed["aggregate"]["donor_gap_recovery"]["mean"],
            "prompt_contract_version": prompt_version,
            "evidence_level": "multiseed",
        },
    ]
    parameter_json = write_json(
        tables_dir / "parameter_budget_table.json",
        {"rows": parameter_rows},
    )
    parameter_csv = write_csv(tables_dir / "parameter_budget_table.csv", parameter_rows)

    full_primary = float(prune_summary["full_same_size_reference"]["primary_strict"])
    base_primary = float(baseline_summary["primary_metric"]["base_value"])
    full_gain = max(full_primary - base_primary, 1e-9)
    random_mean_fraction = (
        float(prune_summary["random_subset_controls"]["mean_primary_strict"]) - base_primary
    ) / full_gain
    retained_gain_rows = [
        {
            "label": "Selected subset",
            "feature_count": prune_summary["selected_subset"]["feature_count"],
            "retained_gain_fraction": prune_summary["selected_subset"][
                "retained_gain_fraction_vs_full"
            ],
            "prompt_contract_version": prompt_version,
        },
        {
            "label": "Random subset mean",
            "feature_count": 1,
            "retained_gain_fraction": random_mean_fraction,
            "prompt_contract_version": prompt_version,
        },
        {
            "label": "Full sparse module",
            "feature_count": same_size_summary["candidate"]["latent_width"],
            "retained_gain_fraction": 1.0,
            "prompt_contract_version": prompt_version,
        },
    ]
    retained_gain_json = write_json(
        tables_dir / "retained_gain_table.json",
        {"rows": retained_gain_rows},
    )
    retained_gain_csv = write_csv(tables_dir / "retained_gain_table.csv", retained_gain_rows)

    recovery_figure = _figure_path(figures_dir, "donor_gap_recovery_vs_parameters")
    plt.figure(figsize=(7, 4.5))
    for row in parameter_rows:
        plt.scatter(
            float(row["added_params"]),
            float(row["donor_gap_recovery"]),
            s=60,
            label=row["system_label"],
        )
    plt.xlabel("Added parameters")
    plt.ylabel("Donor-gap recovery")
    plt.title("Recovery vs added parameters")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(recovery_figure, dpi=180)
    plt.close()

    retained_gain_figure = _figure_path(figures_dir, "retained_gain_vs_features")
    plt.figure(figsize=(7, 4.5))
    ordered_rows = sorted(retained_gain_rows, key=lambda row: float(row["feature_count"]))
    plt.plot(
        [float(row["feature_count"]) for row in ordered_rows],
        [float(row["retained_gain_fraction"]) for row in ordered_rows],
        marker="o",
    )
    for row in ordered_rows:
        plt.annotate(
            row["label"],
            (float(row["feature_count"]), float(row["retained_gain_fraction"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlabel("Kept features")
    plt.ylabel("Retained gain fraction")
    plt.title("Retained sparse gain vs selected features")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(retained_gain_figure, dpi=180)
    plt.close()

    summary = {
        "status": "passed",
        "prompt_contract_version": prompt_version,
        "baseline_table_path": str(baseline_json.resolve()),
        "baseline_table_csv_path": str(baseline_csv.resolve()),
        "parameter_budget_table_path": str(parameter_json.resolve()),
        "parameter_budget_table_csv_path": str(parameter_csv.resolve()),
        "retained_gain_table_path": str(retained_gain_json.resolve()),
        "retained_gain_table_csv_path": str(retained_gain_csv.resolve()),
        "recovery_vs_parameters_figure_path": str(recovery_figure.resolve()),
        "retained_gain_figure_path": str(retained_gain_figure.resolve()),
    }
    return write_json(destination / "summary.json", summary)
