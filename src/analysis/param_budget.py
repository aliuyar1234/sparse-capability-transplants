from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoraBudgetChoice:
    target_params: int
    selected_rank: int
    selected_params: int
    lower_rank: int | None
    lower_params: int | None
    upper_rank: int | None
    upper_params: int | None


def sparse_same_size_params(*, hidden_size: int, bottleneck_size: int, layer_count: int = 1) -> int:
    per_layer = 2 * hidden_size * bottleneck_size + bottleneck_size + 1
    return per_layer * layer_count


def dense_two_layer_params(*, hidden_size: int, mlp_hidden_size: int, layer_count: int = 1) -> int:
    per_layer = 2 * hidden_size * mlp_hidden_size + mlp_hidden_size + hidden_size
    return per_layer * layer_count


def stitch_pair_params(
    *, donor_dim: int, recipient_dim: int, rank: int, pair_count: int = 1
) -> int:
    per_pair = (
        rank * recipient_dim
        + donor_dim * rank
        + donor_dim
        + rank * donor_dim
        + recipient_dim * rank
        + recipient_dim
    )
    return per_pair * pair_count


def lora_total_params(
    *,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    rank: int,
    attention_projection_count: int = 4,
    mlp_projection_count: int = 3,
) -> int:
    attention_per_layer = attention_projection_count * rank * (hidden_size + hidden_size)
    mlp_per_layer = mlp_projection_count * rank * (hidden_size + intermediate_size)
    return num_layers * (attention_per_layer + mlp_per_layer)


def choose_lora_rank_for_budget(
    *,
    target_params: int,
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    rank_grid: list[int],
) -> LoraBudgetChoice:
    ranked = sorted(set(rank_grid))
    scored = [
        (
            rank,
            lora_total_params(
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rank=rank,
            ),
        )
        for rank in ranked
    ]
    selected_rank, selected_params = min(scored, key=lambda item: abs(item[1] - target_params))

    lower = [
        (rank, params)
        for rank, params in scored
        if params <= target_params and rank != selected_rank
    ]
    upper = [
        (rank, params)
        for rank, params in scored
        if params >= target_params and rank != selected_rank
    ]
    lower_rank, lower_params = lower[-1] if lower else (None, None)
    upper_rank, upper_params = upper[0] if upper else (None, None)
    return LoraBudgetChoice(
        target_params=target_params,
        selected_rank=selected_rank,
        selected_params=selected_params,
        lower_rank=lower_rank,
        lower_params=lower_params,
        upper_rank=upper_rank,
        upper_params=upper_params,
    )


def build_budget_report(config: dict[str, Any]) -> dict[str, Any]:
    budget_config = config.get("param_budget", {})
    sparse_config = budget_config.get("sparse_same_size", {})
    stitch_config = budget_config.get("stitch_pair", {})
    lora_config = budget_config.get("lora", {})

    sparse_params = sparse_same_size_params(
        hidden_size=int(sparse_config["hidden_size"]),
        bottleneck_size=int(sparse_config["bottleneck_size"]),
        layer_count=int(sparse_config.get("layer_count", 1)),
    )
    dense_params = dense_two_layer_params(
        hidden_size=int(sparse_config["hidden_size"]),
        mlp_hidden_size=int(
            budget_config.get("dense_match", {}).get(
                "mlp_hidden_size", sparse_config["bottleneck_size"]
            )
        ),
        layer_count=int(sparse_config.get("layer_count", 1)),
    )
    stitch_params = stitch_pair_params(
        donor_dim=int(stitch_config["donor_dim"]),
        recipient_dim=int(stitch_config["recipient_dim"]),
        rank=int(stitch_config["rank"]),
        pair_count=int(stitch_config.get("pair_count", 1)),
    )
    lora_choice = choose_lora_rank_for_budget(
        target_params=sparse_params,
        num_layers=int(lora_config["num_layers"]),
        hidden_size=int(lora_config["hidden_size"]),
        intermediate_size=int(lora_config["intermediate_size"]),
        rank_grid=[int(rank) for rank in lora_config.get("rank_grid", [1, 2, 4, 8, 16, 32])],
    )

    return {
        "sparse_same_size_params": sparse_params,
        "dense_two_layer_params": dense_params,
        "stitch_pair_params": stitch_params,
        "lora_budget_choice": asdict(lora_choice),
    }


def write_budget_report(*, config: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = build_budget_report(config)
    output_path = destination / "budget_report.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
