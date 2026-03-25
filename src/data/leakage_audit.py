from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.build_alias_bank import ALIAS_BANK_NAMES, AliasBanks
from src.data.build_control_suite import ControlExample
from src.data.canonical import CanonicalExample


@dataclass(frozen=True)
class LeakageAuditReport:
    alias_bank_disjoint: bool
    alias_bank_collisions: list[dict[str, str]]
    split_overlap_pairs: list[dict[str, str]]
    schema_shift_name_leaks: list[dict[str, str]]
    schema_shift_description_leaks: list[dict[str, str]]
    distractor_tool_collisions: list[dict[str, str]]
    derived_non_eval_sources: list[dict[str, str]]
    control_examples_with_tool_json: list[str]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _alias_bank_collisions(alias_banks: AliasBanks) -> list[dict[str, str]]:
    collisions: list[dict[str, str]] = []
    first_bank = next(iter(alias_banks.banks.values()), {})
    for domain_name in first_bank.keys():
        seen: dict[str, tuple[str, str]] = {}
        for bank_name in ALIAS_BANK_NAMES:
            for canonical_key, aliases in alias_banks.banks[bank_name][domain_name].items():
                for alias in aliases:
                    previous = seen.get(alias)
                    if previous is None:
                        seen[alias] = (bank_name, canonical_key)
                        continue
                    previous_bank, previous_key = previous
                    if previous_key != canonical_key:
                        collisions.append(
                            {
                                "domain": domain_name,
                                "alias": alias,
                                "first_bank": previous_bank,
                                "first_key": previous_key,
                                "second_bank": bank_name,
                                "second_key": canonical_key,
                            }
                        )
    return collisions


def _split_overlap_pairs(examples: list[CanonicalExample]) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    overlaps: list[dict[str, str]] = []
    for example in examples:
        previous_split = seen.get(example.example_id)
        if previous_split is None:
            seen[example.example_id] = example.split
            continue
        if previous_split != example.split:
            overlaps.append(
                {
                    "example_id": example.example_id,
                    "first_split": previous_split,
                    "second_split": example.split,
                }
            )
    return overlaps


def _schema_shift_leaks(
    canonical_examples: list[CanonicalExample],
    schema_shift_examples: list[CanonicalExample],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    canonical_by_id = {example.example_id: example for example in canonical_examples}
    canonical_by_source_row = {
        str(example.meta.get("source_row_id")): example
        for example in canonical_examples
        if example.meta.get("source_row_id") is not None
    }
    name_leaks: list[dict[str, str]] = []
    description_leaks: list[dict[str, str]] = []

    for shifted in schema_shift_examples:
        source_example = canonical_by_id.get(str(shifted.meta.get("source_example_id", "")))
        if source_example is None:
            source_id = str(shifted.meta.get("source_row_id", shifted.example_id))
            source_example = canonical_by_id.get(source_id) or canonical_by_source_row.get(
                source_id
            )
        if source_example is None:
            continue

        source_tools_by_id = {tool.tool_id: tool for tool in source_example.tools}
        for shifted_tool in shifted.tools:
            source_tool = source_tools_by_id.get(shifted_tool.tool_id)
            if source_tool is None:
                continue
            if shifted_tool.name == source_tool.name:
                name_leaks.append(
                    {
                        "example_id": shifted.example_id,
                        "tool_id": shifted_tool.tool_id,
                        "visible_name": shifted_tool.name,
                    }
                )
            if source_tool.name.lower() in shifted_tool.description.lower():
                description_leaks.append(
                    {
                        "example_id": shifted.example_id,
                        "tool_id": shifted_tool.tool_id,
                        "visible_description": shifted_tool.description,
                    }
                )

    return name_leaks, description_leaks


def _control_examples_with_tool_json(examples: list[ControlExample]) -> list[str]:
    offending_ids: list[str] = []
    for example in examples:
        if '"name"' in example.target_text and '"arguments"' in example.target_text:
            offending_ids.append(example.example_id)
    return offending_ids


def _distractor_tool_collisions(examples: list[CanonicalExample]) -> list[dict[str, str]]:
    collisions: list[dict[str, str]] = []
    for example in examples:
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for tool in example.tools:
            if tool.tool_id in seen_ids or tool.name in seen_names:
                collisions.append(
                    {
                        "example_id": example.example_id,
                        "tool_id": tool.tool_id,
                        "tool_name": tool.name,
                    }
                )
            seen_ids.add(tool.tool_id)
            seen_names.add(tool.name)
    return collisions


def _derived_non_eval_sources(
    canonical_examples: list[CanonicalExample],
    derived_examples: list[CanonicalExample],
) -> list[dict[str, str]]:
    canonical_by_id = {example.example_id: example for example in canonical_examples}
    canonical_by_source_row = {
        str(example.meta.get("source_row_id")): example
        for example in canonical_examples
        if example.meta.get("source_row_id") is not None
    }
    invalid_sources: list[dict[str, str]] = []

    for example in derived_examples:
        source_example = canonical_by_id.get(str(example.meta.get("source_example_id", "")))
        source_id = str(example.meta.get("source_row_id", example.example_id))
        if source_example is None:
            source_example = canonical_by_id.get(source_id) or canonical_by_source_row.get(
                source_id
            )
        if source_example is None:
            invalid_sources.append(
                {
                    "example_id": example.example_id,
                    "source_example_id": source_id,
                    "reason": "missing_source_example",
                }
            )
            continue
        if source_example.split != "eval":
            invalid_sources.append(
                {
                    "example_id": example.example_id,
                    "source_example_id": source_example.example_id,
                    "source_split": source_example.split,
                }
            )
    return invalid_sources


def run_leakage_audit(
    *,
    canonical_examples: list[CanonicalExample],
    schema_shift_examples: list[CanonicalExample],
    alias_banks: AliasBanks,
    control_examples: list[ControlExample],
    distractor_examples: list[CanonicalExample] | None = None,
    nocall_examples: list[CanonicalExample] | None = None,
) -> LeakageAuditReport:
    collisions = _alias_bank_collisions(alias_banks)
    split_overlaps = _split_overlap_pairs(canonical_examples)
    name_leaks, description_leaks = _schema_shift_leaks(canonical_examples, schema_shift_examples)
    distractor_collisions = _distractor_tool_collisions(distractor_examples or [])
    derived_non_eval_sources = _derived_non_eval_sources(
        canonical_examples,
        [*schema_shift_examples, *(distractor_examples or []), *(nocall_examples or [])],
    )
    control_json = _control_examples_with_tool_json(control_examples)

    summary = {
        "canonical_example_count": len(canonical_examples),
        "schema_shift_example_count": len(schema_shift_examples),
        "distractor_example_count": len(distractor_examples or []),
        "nocall_example_count": len(nocall_examples or []),
        "control_example_count": len(control_examples),
        "alias_bank_collision_count": len(collisions),
        "split_overlap_count": len(split_overlaps),
        "schema_shift_name_leak_count": len(name_leaks),
        "schema_shift_description_leak_count": len(description_leaks),
        "distractor_tool_collision_count": len(distractor_collisions),
        "derived_non_eval_source_count": len(derived_non_eval_sources),
        "control_tool_json_count": len(control_json),
    }
    return LeakageAuditReport(
        alias_bank_disjoint=len(collisions) == 0,
        alias_bank_collisions=collisions,
        split_overlap_pairs=split_overlaps,
        schema_shift_name_leaks=name_leaks,
        schema_shift_description_leaks=description_leaks,
        distractor_tool_collisions=distractor_collisions,
        derived_non_eval_sources=derived_non_eval_sources,
        control_examples_with_tool_json=control_json,
        summary=summary,
    )


def write_leakage_audit(report: LeakageAuditReport, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
