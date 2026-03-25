from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from src.analysis.summarize_baselines import PRIMARY_VARIANTS
from src.eval.control_metrics import ControlScore, aggregate_control_scores
from src.eval.metrics import ExampleScore, aggregate_scores

PRIMARY_SLICE_ORDER = ["schema_shift", "nocall_missing_tool", "nocall_unsupported", "primary"]
PRIMARY_SYSTEM_ORDER = ["base", "donor", "sparse", "dense", "steering"]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    materialized_rows = list(rows)
    fieldnames = sorted({key for row in materialized_rows for key in row})
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized_rows:
            writer.writerow(row)
    return destination


def prompt_contract_version(manifest_path: str | Path) -> str:
    payload = load_json(manifest_path)
    return str(payload.get("prompt_contract_version", "unknown"))


def eval_dataset_examples(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_json(manifest_path)
    dataset_path = payload.get("dataset_path")
    if not dataset_path:
        raise ValueError(f"{manifest_path} is missing dataset_path.")
    return {
        row["example_id"]: row
        for row in read_jsonl(dataset_path)
        if isinstance(row.get("example_id"), str)
    }


def control_dataset_examples(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_json(manifest_path)
    dataset_path = payload.get("dataset_path")
    if not dataset_path:
        raise ValueError(f"{manifest_path} is missing dataset_path.")
    return {
        row["example_id"]: row
        for row in read_jsonl(dataset_path)
        if isinstance(row.get("example_id"), str)
    }


def layer_candidate_summary(summary_path: str | Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    summary["summary_path"] = str(Path(summary_path).resolve())
    return summary


def candidate_task_predictions_path(summary: dict[str, Any]) -> str:
    return str(summary["task_eval"]["predictions_path"])


def candidate_control_predictions_path(summary: dict[str, Any]) -> str:
    return str(summary["control_eval"]["candidate_predictions_path"])


def candidate_base_control_predictions_path(summary: dict[str, Any]) -> str:
    return str(summary["control_eval"]["base_predictions_path"])


def grouped_primary_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped_rows = {
        "schema_shift": [row for row in rows if row.get("variant") == "schema_shift"],
        "nocall_missing_tool": [row for row in rows if row.get("variant") == "nocall_missing_tool"],
        "nocall_unsupported": [row for row in rows if row.get("variant") == "nocall_unsupported"],
        "primary": [row for row in rows if row.get("variant") in PRIMARY_VARIANTS],
    }
    return {
        name: asdict(aggregate_scores([ExampleScore(**row["score"]) for row in slice_rows]))
        for name, slice_rows in grouped_rows.items()
        if slice_rows
    }


def error_category(row: dict[str, Any]) -> str:
    score = dict(row["score"])
    parse_status = str(score.get("parse_status", "missing"))
    if parse_status != "ok":
        return "parsing_failure"
    if bool(score.get("strict_correct")):
        return "strict_success"
    if bool(score.get("semantic_correct")) and not bool(score.get("strict_correct")):
        return "alias_or_surface_form"

    semantic_error = score.get("semantic_error") or score.get("strict_error")
    if semantic_error == "wrong_nocall_decision":
        return "wrong_call_vs_no_tool"
    if semantic_error == "wrong_tool":
        return "distractor_confusion" if row.get("variant") == "distractor" else "wrong_tool"
    if semantic_error in {
        "arguments_not_object",
        "missing_argument",
        "missing_required_argument",
        "unexpected_argument",
        "unknown_argument",
    }:
        return "wrong_argument_key"
    if semantic_error in {
        "argument_normalization_failed",
        "argument_value_mismatch",
        "gold_argument_normalization_failed",
    }:
        return "wrong_argument_value"
    if score.get("predicted_name") and score.get("semantic_predicted_name"):
        if score["predicted_name"] != score["semantic_predicted_name"]:
            return "alias_confusion"
    return "other_semantic_error"


def summarize_error_categories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = Counter(error_category(row) for row in rows)
    total = len(rows)
    return [
        {
            "error_category": category,
            "count": count,
            "rate": 0.0 if total == 0 else count / total,
        }
        for category, count in sorted(categories.items())
    ]


def aligned_rows_by_example_id(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_by_id = {row["example_id"]: row for row in left_rows}
    right_by_id = {row["example_id"]: row for row in right_rows}
    example_ids = sorted(set(left_by_id) & set(right_by_id))
    return [(left_by_id[example_id], right_by_id[example_id]) for example_id in example_ids]


def control_exact_match_average(predictions_path: str | Path) -> float:
    scores = [ControlScore(**row["score"]) for row in read_jsonl(predictions_path)]
    return aggregate_control_scores(scores)


def control_damage_examples(
    *,
    base_predictions_path: str | Path,
    candidate_predictions_path: str | Path,
    control_examples: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    base_rows = read_jsonl(base_predictions_path)
    candidate_rows = read_jsonl(candidate_predictions_path)
    examples: list[dict[str, Any]] = []
    for base_row, candidate_row in aligned_rows_by_example_id(base_rows, candidate_rows):
        base_score = dict(base_row["score"])
        candidate_score = dict(candidate_row["score"])
        if bool(base_score.get("exact_match")) and not bool(candidate_score.get("exact_match")):
            example_id = str(base_row["example_id"])
            control_example = control_examples.get(example_id, {})
            examples.append(
                {
                    "example_id": example_id,
                    "target_text": control_example.get("target_text"),
                    "base_prediction": base_score.get("normalized_prediction"),
                    "candidate_prediction": candidate_score.get("normalized_prediction"),
                }
            )
        if len(examples) >= limit:
            break
    return examples


def sorted_slice_key(slice_name: str) -> tuple[int, str]:
    order = {name: index for index, name in enumerate(PRIMARY_SLICE_ORDER)}
    return (order.get(slice_name, len(order)), slice_name)


def sorted_system_key(system_id: str) -> tuple[int, str]:
    order = {name: index for index, name in enumerate(PRIMARY_SYSTEM_ORDER)}
    return (order.get(system_id, len(order)), system_id)
