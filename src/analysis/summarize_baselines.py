from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.eval.metrics import ExampleScore, aggregate_scores

PRIMARY_VARIANTS = frozenset({"schema_shift", "nocall_missing_tool", "nocall_unsupported"})
NOCALL_VARIANTS = frozenset({"nocall_missing_tool", "nocall_unsupported"})


def _load_prediction_rows(predictions_path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(predictions_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _rows_to_scores(rows: list[dict[str, Any]]) -> list[ExampleScore]:
    return [ExampleScore(**row["score"]) for row in rows]


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


def _grouped_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "all": rows,
        "iid": [row for row in rows if row["variant"] == "canonical"],
        "schema_shift": [row for row in rows if row["variant"] == "schema_shift"],
        "distractor": [row for row in rows if row["variant"] == "distractor"],
        "nocall": [row for row in rows if row["variant"] in NOCALL_VARIANTS],
        "primary": [row for row in rows if row["variant"] in PRIMARY_VARIANTS],
    }
    return {
        name: asdict(aggregate_scores(_rows_to_scores(group_rows)))
        for name, group_rows in grouped.items()
        if group_rows
    }


def _align_system_rows(
    *,
    base_rows: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    base_by_id = {row["example_id"]: row for row in base_rows}
    donor_by_id = {row["example_id"]: row for row in donor_rows}
    if set(base_by_id) != set(donor_by_id):
        missing_in_base = sorted(set(donor_by_id) - set(base_by_id))
        missing_in_donor = sorted(set(base_by_id) - set(donor_by_id))
        raise ValueError(
            "Base/donor prediction files do not cover the same example IDs. "
            f"Missing in base: {missing_in_base[:5]}; missing in donor: {missing_in_donor[:5]}"
        )

    aligned = []
    for example_id in sorted(base_by_id):
        base_row = base_by_id[example_id]
        donor_row = donor_by_id[example_id]
        if base_row["variant"] != donor_row["variant"]:
            raise ValueError(f"Variant mismatch for example {example_id}.")
        aligned.append((base_row, donor_row))
    return aligned


def build_baseline_summary(config: dict[str, Any]) -> dict[str, Any]:
    analysis_config = dict(config.get("analysis", {}))
    base_predictions_path = analysis_config.get("base_predictions_path")
    donor_predictions_path = analysis_config.get("donor_predictions_path")
    if not base_predictions_path or not donor_predictions_path:
        raise ValueError(
            "Config is missing analysis.base_predictions_path or analysis.donor_predictions_path."
        )

    base_rows = _load_prediction_rows(base_predictions_path)
    donor_rows = _load_prediction_rows(donor_predictions_path)
    aligned_rows = _align_system_rows(base_rows=base_rows, donor_rows=donor_rows)
    primary_pairs = [
        (base_row, donor_row)
        for base_row, donor_row in aligned_rows
        if base_row["variant"] in PRIMARY_VARIANTS
    ]
    if not primary_pairs:
        raise ValueError("Primary metric subset is empty; expected SchemaShift and NoCall rows.")

    base_metrics = _grouped_metrics(base_rows)
    donor_metrics = _grouped_metrics(donor_rows)
    metric_names = sorted(set(base_metrics["primary"]) & set(donor_metrics["primary"]))
    primary_deltas = {
        metric_name: donor_metrics["primary"][metric_name] - base_metrics["primary"][metric_name]
        for metric_name in metric_names
        if metric_name != "total_examples"
    }

    strict_differences = [
        float(donor_row["score"]["strict_correct"]) - float(base_row["score"]["strict_correct"])
        for base_row, donor_row in primary_pairs
    ]
    ci_lower, ci_upper = _bootstrap_ci(
        strict_differences,
        bootstrap_samples=int(analysis_config.get("bootstrap_samples", 2000)),
        bootstrap_seed=int(analysis_config.get("bootstrap_seed", 17)),
    )
    primary_delta = sum(strict_differences) / len(strict_differences)
    if primary_delta >= 0.15:
        gate_status = "pass"
        gate_reason = "delta_at_least_15pp"
    elif primary_delta >= 0.05 and ci_lower > 0.0:
        gate_status = "pass"
        gate_reason = "delta_at_least_5pp_with_positive_ci"
    else:
        gate_status = "fail"
        gate_reason = "r20_not_met"

    return {
        "base_predictions_path": str(Path(base_predictions_path).resolve()),
        "donor_predictions_path": str(Path(donor_predictions_path).resolve()),
        "base_metrics": base_metrics,
        "donor_metrics": donor_metrics,
        "primary_metric": {
            "name": "strict_full_call_success_schema_shift_union_nocall",
            "variant_set": sorted(PRIMARY_VARIANTS),
            "base_value": base_metrics["primary"]["strict_full_call_success"],
            "donor_value": donor_metrics["primary"]["strict_full_call_success"],
            "delta": primary_delta,
            "delta_ci95": [ci_lower, ci_upper],
            "bootstrap_samples": int(analysis_config.get("bootstrap_samples", 2000)),
            "bootstrap_seed": int(analysis_config.get("bootstrap_seed", 17)),
        },
        "primary_metric_deltas": primary_deltas,
        "gate_decision": {
            "status": gate_status,
            "reason": gate_reason,
            "r20_rule": ("pass if donor delta >= 0.15, or delta >= 0.05 with CI lower bound > 0.0"),
        },
    }


def write_baseline_summary(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = build_baseline_summary(config)
    summary_path = destination / "baseline_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path
