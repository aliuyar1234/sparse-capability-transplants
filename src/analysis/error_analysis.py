from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis.paper_artifacts import (
    candidate_base_control_predictions_path,
    candidate_control_predictions_path,
    candidate_task_predictions_path,
    control_damage_examples,
    control_dataset_examples,
    error_category,
    eval_dataset_examples,
    grouped_primary_metrics,
    layer_candidate_summary,
    load_json,
    prompt_contract_version,
    read_jsonl,
    sorted_slice_key,
    sorted_system_key,
    summarize_error_categories,
    write_csv,
    write_json,
)


def _system_prediction_rows(
    *,
    system_id: str,
    label: str,
    predictions_path: str | Path,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(predictions_path)
    grouped_metrics = grouped_primary_metrics(rows)
    metric_rows = []
    for slice_name, metrics in grouped_metrics.items():
        metric_rows.append(
            {
                "system_id": system_id,
                "system_label": label,
                "slice": slice_name,
                "prompt_contract_version": prompt_version,
                **metrics,
            }
        )
    metric_rows.sort(
        key=lambda row: (sorted_system_key(row["system_id"]), sorted_slice_key(row["slice"]))
    )
    return rows, metric_rows


def _error_rows_for_system(
    *,
    system_id: str,
    label: str,
    rows: list[dict[str, Any]],
    prompt_version: str,
) -> list[dict[str, Any]]:
    primary_rows = [
        row
        for row in rows
        if row.get("variant") in {"schema_shift", "nocall_missing_tool", "nocall_unsupported"}
    ]
    counts = summarize_error_categories(primary_rows)
    return [
        {
            "system_id": system_id,
            "system_label": label,
            "prompt_contract_version": prompt_version,
            **payload,
        }
        for payload in counts
    ]


def _help_hurt_examples(
    *,
    eval_examples: dict[str, dict[str, Any]],
    base_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {
        "sparse_improves_over_base": [],
        "dense_beats_sparse": [],
        "sparse_beats_dense": [],
    }
    base_by_id = {row["example_id"]: row for row in base_rows}
    sparse_by_id = {row["example_id"]: row for row in sparse_rows}
    dense_by_id = {row["example_id"]: row for row in dense_rows}
    for example_id in sorted(set(base_by_id) & set(sparse_by_id) & set(dense_by_id)):
        base_row = base_by_id[example_id]
        sparse_row = sparse_by_id[example_id]
        dense_row = dense_by_id[example_id]
        example = eval_examples.get(example_id, {})
        sparse_help = not bool(base_row["score"]["strict_correct"]) and bool(
            sparse_row["score"]["strict_correct"]
        )
        if sparse_help and len(examples["sparse_improves_over_base"]) < limit:
            examples["sparse_improves_over_base"].append(
                {
                    "example_id": example_id,
                    "variant": sparse_row.get("variant"),
                    "user_request": example.get("user_request"),
                    "gold": example.get("gold"),
                    "base_raw_output": base_row.get("raw_output"),
                    "sparse_raw_output": sparse_row.get("raw_output"),
                    "dense_raw_output": dense_row.get("raw_output"),
                    "sparse_error_category": error_category(sparse_row),
                    "dense_error_category": error_category(dense_row),
                }
            )
        dense_strict = bool(dense_row["score"]["strict_correct"])
        sparse_strict = bool(sparse_row["score"]["strict_correct"])
        if dense_strict and not sparse_strict and len(examples["dense_beats_sparse"]) < limit:
            examples["dense_beats_sparse"].append(
                {
                    "example_id": example_id,
                    "variant": dense_row.get("variant"),
                    "user_request": example.get("user_request"),
                    "gold": example.get("gold"),
                    "sparse_raw_output": sparse_row.get("raw_output"),
                    "dense_raw_output": dense_row.get("raw_output"),
                    "sparse_error_category": error_category(sparse_row),
                    "dense_error_category": error_category(dense_row),
                }
            )
        if sparse_strict and not dense_strict and len(examples["sparse_beats_dense"]) < limit:
            examples["sparse_beats_dense"].append(
                {
                    "example_id": example_id,
                    "variant": sparse_row.get("variant"),
                    "user_request": example.get("user_request"),
                    "gold": example.get("gold"),
                    "sparse_raw_output": sparse_row.get("raw_output"),
                    "dense_raw_output": dense_row.get("raw_output"),
                    "sparse_error_category": error_category(sparse_row),
                    "dense_error_category": error_category(dense_row),
                }
            )
    return examples


def write_error_analysis_report(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    paper_config = dict(config.get("paper_artifacts", {}))
    required_paths = (
        "baseline_summary_path",
        "eval_manifest_path",
        "control_manifest_path",
        "sparse_selected_eval_summary_path",
        "shortcut_summary_path",
    )
    for key in required_paths:
        if not paper_config.get(key):
            raise ValueError(f"paper_artifacts.{key} is required.")

    error_config = dict(config.get("error_analysis", {}))
    example_limit = int(error_config.get("max_examples_per_bucket", 10))
    destination = Path(output_dir)
    tables_dir = destination / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    prompt_version = prompt_contract_version(paper_config["eval_manifest_path"])
    baseline_summary = load_json(paper_config["baseline_summary_path"])
    sparse_summary = layer_candidate_summary(paper_config["sparse_selected_eval_summary_path"])
    shortcut_summary = load_json(paper_config["shortcut_summary_path"])
    dense_summary = layer_candidate_summary(
        shortcut_summary["dense_control"]["frozen_eval"]["summary_path"]
    )
    steering_summary = layer_candidate_summary(
        shortcut_summary["steering_control"]["frozen_eval"]["summary_path"]
    )

    system_specs = [
        ("base", "Base", baseline_summary["base_predictions_path"]),
        ("donor", "Donor", baseline_summary["donor_predictions_path"]),
        ("sparse", "Sparse same-size (seed 17)", candidate_task_predictions_path(sparse_summary)),
        ("dense", "Dense shortcut (seed 17)", candidate_task_predictions_path(dense_summary)),
        (
            "steering",
            "Steering shortcut (seed 17)",
            candidate_task_predictions_path(steering_summary),
        ),
    ]

    strict_semantic_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    prediction_rows_by_system: dict[str, list[dict[str, Any]]] = {}
    for system_id, label, predictions_path in system_specs:
        prediction_rows, metric_rows = _system_prediction_rows(
            system_id=system_id,
            label=label,
            predictions_path=predictions_path,
            prompt_version=prompt_version,
        )
        strict_semantic_rows.extend(metric_rows)
        error_rows.extend(
            _error_rows_for_system(
                system_id=system_id,
                label=label,
                rows=prediction_rows,
                prompt_version=prompt_version,
            )
        )
        prediction_rows_by_system[system_id] = prediction_rows

    strict_semantic_rows.sort(
        key=lambda row: (sorted_system_key(row["system_id"]), sorted_slice_key(row["slice"]))
    )
    error_rows.sort(key=lambda row: (sorted_system_key(row["system_id"]), row["error_category"]))

    strict_semantic_json = write_json(
        tables_dir / "strict_vs_semantic_table.json",
        {"rows": strict_semantic_rows},
    )
    strict_semantic_csv = write_csv(
        tables_dir / "strict_vs_semantic_table.csv",
        strict_semantic_rows,
    )
    error_category_json = write_json(
        tables_dir / "error_category_table.json",
        {"rows": error_rows},
    )
    error_category_csv = write_csv(
        tables_dir / "error_category_table.csv",
        error_rows,
    )

    eval_examples = eval_dataset_examples(paper_config["eval_manifest_path"])
    control_examples = control_dataset_examples(paper_config["control_manifest_path"])
    appendix_examples = _help_hurt_examples(
        eval_examples=eval_examples,
        base_rows=[
            row
            for row in prediction_rows_by_system["base"]
            if row.get("variant") in {"schema_shift", "nocall_missing_tool", "nocall_unsupported"}
        ],
        sparse_rows=prediction_rows_by_system["sparse"],
        dense_rows=prediction_rows_by_system["dense"],
        limit=example_limit,
    )
    appendix_examples["control_damage_examples"] = {
        "sparse": control_damage_examples(
            base_predictions_path=candidate_base_control_predictions_path(sparse_summary),
            candidate_predictions_path=candidate_control_predictions_path(sparse_summary),
            control_examples=control_examples,
            limit=example_limit,
        ),
        "dense": control_damage_examples(
            base_predictions_path=candidate_base_control_predictions_path(dense_summary),
            candidate_predictions_path=candidate_control_predictions_path(dense_summary),
            control_examples=control_examples,
            limit=example_limit,
        ),
        "steering": control_damage_examples(
            base_predictions_path=candidate_base_control_predictions_path(steering_summary),
            candidate_predictions_path=candidate_control_predictions_path(steering_summary),
            control_examples=control_examples,
            limit=example_limit,
        ),
    }
    appendix_examples_path = write_json(
        destination / "appendix_examples.json",
        appendix_examples,
    )

    base_primary = next(
        row
        for row in strict_semantic_rows
        if row["system_id"] == "base" and row["slice"] == "primary"
    )
    sparse_primary = next(
        row
        for row in strict_semantic_rows
        if row["system_id"] == "sparse" and row["slice"] == "primary"
    )
    base_schema = next(
        row
        for row in strict_semantic_rows
        if row["system_id"] == "base" and row["slice"] == "schema_shift"
    )
    sparse_schema = next(
        row
        for row in strict_semantic_rows
        if row["system_id"] == "sparse" and row["slice"] == "schema_shift"
    )

    summary = {
        "status": "passed",
        "prompt_contract_version": prompt_version,
        "strict_vs_semantic_table_path": str(strict_semantic_json.resolve()),
        "strict_vs_semantic_csv_path": str(strict_semantic_csv.resolve()),
        "error_category_table_path": str(error_category_json.resolve()),
        "error_category_csv_path": str(error_category_csv.resolve()),
        "appendix_examples_path": str(appendix_examples_path.resolve()),
        "semantic_transfer_snapshot": {
            "base_primary_strict": base_primary["strict_full_call_success"],
            "sparse_primary_strict": sparse_primary["strict_full_call_success"],
            "base_primary_semantic": base_primary["semantic_full_call_success"],
            "sparse_primary_semantic": sparse_primary["semantic_full_call_success"],
            "base_schema_strict": base_schema["strict_full_call_success"],
            "sparse_schema_strict": sparse_schema["strict_full_call_success"],
            "base_schema_semantic": base_schema["semantic_full_call_success"],
            "sparse_schema_semantic": sparse_schema["semantic_full_call_success"],
            "schema_shift_strict_gain": sparse_schema["strict_full_call_success"]
            - base_schema["strict_full_call_success"],
            "primary_semantic_gain": sparse_primary["semantic_full_call_success"]
            - base_primary["semantic_full_call_success"],
        },
        "control_damage_counts": {
            system_id: len(rows)
            for system_id, rows in appendix_examples["control_damage_examples"].items()
        },
    }
    return write_json(destination / "summary.json", summary)
