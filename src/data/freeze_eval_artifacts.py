from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.build_alias_bank import AliasBanks, alias_banks_hash, freeze_alias_banks
from src.data.build_control_suite import (
    build_control_examples_from_canonical_examples,
    write_control_suite,
    write_control_suite_manifest,
)
from src.data.canonical import CanonicalExample, ToolSpec
from src.data.generate_distractor import (
    build_distractor_tool_library,
    distractor_library_hash,
    generate_distractor_examples,
)
from src.data.generate_nocall import (
    generate_missing_tool_nocall_examples,
    generate_unsupported_intent_nocall_examples,
)
from src.data.generate_schema_shift import generate_schema_shift_examples
from src.data.leakage_audit import run_leakage_audit, write_leakage_audit
from src.data.manifest import (
    load_examples,
    load_manifest_payload,
    write_manifest,
)
from src.eval.control_metrics import aggregate_control_scores, score_control_prediction
from src.eval.golden_fixtures import write_golden_fixture
from src.eval.metrics import aggregate_scores, score_prediction
from src.models.format_prompts import PROMPT_CONTRACT_VERSION

_TOOL_ALIAS_OVERRIDES = {
    "send_email": ["dispatch_message", "compose_email", "transmit_mail"],
    "create_calendar_event": [
        "schedule_calendar_entry",
        "add_calendar_event",
        "create_agenda_item",
    ],
    "show_map": ["display_location_map", "open_map_view", "locate_on_map"],
    "create_contact": ["add_phone_contact", "save_contact_card", "create_address_entry"],
    "turn_on_flashlight": ["enable_flashlight", "switch_on_torch", "activate_flashlight"],
    "turn_off_flashlight": ["disable_flashlight", "switch_off_torch", "deactivate_flashlight"],
    "open_wifi_settings": ["launch_wifi_settings", "open_wireless_preferences", "show_wifi_panel"],
}

_ARGUMENT_ALIAS_OVERRIDES = {
    "send_email.to": ["recipient_email", "destination_address", "email_recipient"],
    "send_email.subject": ["message_subject", "email_topic", "subject_line"],
    "send_email.body": ["message_body", "email_body_text", "body_text"],
    "create_calendar_event.datetime": ["event_datetime", "scheduled_time", "start_datetime"],
    "create_calendar_event.title": ["event_title", "calendar_title", "agenda_title"],
    "show_map.query": ["location_query", "place_query", "map_search"],
    "create_contact.email": ["contact_email", "email_address", "contact_mail"],
    "create_contact.first_name": ["given_name", "contact_first", "first"],
    "create_contact.last_name": ["family_name", "contact_last", "surname"],
    "create_contact.phone_number": ["contact_phone", "mobile_number", "phone"],
}

_DESCRIPTION_ALIAS_OVERRIDES = {
    "send_email": [
        "Compose and send an email message.",
        "Dispatch an email to a recipient.",
        "Create and send outbound mail.",
    ],
    "create_calendar_event": [
        "Schedule a new event on the calendar.",
        "Add an event to the calendar with a title and time.",
        "Create a calendar entry for a scheduled event.",
    ],
    "show_map": [
        "Display a place or address on the map.",
        "Open a map view for a requested location.",
        "Show the selected place on a map.",
    ],
    "create_contact": [
        "Create a contact in the address book.",
        "Save a new contact card on the device.",
        "Add a person to the phone contact list.",
    ],
    "turn_on_flashlight": [
        "Turn the flashlight on.",
        "Enable the device torch.",
        "Activate the flashlight.",
    ],
    "turn_off_flashlight": [
        "Turn the flashlight off.",
        "Disable the device torch.",
        "Deactivate the flashlight.",
    ],
    "open_wifi_settings": [
        "Open the Wi-Fi settings screen.",
        "Launch the wireless settings panel.",
        "Show the Wi-Fi preferences page.",
    ],
}


def _manifest_prompt_contract_version(manifest_payload: dict[str, Any]) -> str:
    return str(manifest_payload.get("prompt_contract_version", PROMPT_CONTRACT_VERSION))


def _tool_inventory(examples: list[CanonicalExample]) -> dict[str, ToolSpec]:
    inventory: dict[str, ToolSpec] = {}
    for example in examples:
        for tool in example.tools:
            existing = inventory.get(tool.tool_id)
            if existing is None:
                inventory[tool.tool_id] = tool
                continue
            if existing.to_dict() != tool.to_dict():
                raise ValueError(f"Inconsistent tool schema detected for {tool.tool_id!r}.")
    return inventory


def _default_tool_aliases(tool: ToolSpec) -> list[str]:
    label = tool.name.replace("_", " ").strip()
    compact = "_".join(label.split())
    return [f"{compact}_alt_one", f"{compact}_alt_two", f"{compact}_alt_three"]


def _default_argument_aliases(tool_id: str, argument_name: str) -> list[str]:
    compact = "_".join(argument_name.replace("_", " ").split())
    return [
        f"{compact}_{tool_id}_alt_one",
        f"{compact}_{tool_id}_alt_two",
        f"{compact}_{tool_id}_alt_three",
    ]


def _default_description_aliases(tool: ToolSpec) -> list[str]:
    label = tool.name.replace("_", " ")
    return [
        f"Use this tool for {label}.",
        f"This operation handles {label}.",
        f"Select this tool when you need {label}.",
    ]


def build_alias_candidates(examples: list[CanonicalExample]) -> dict[str, dict[str, list[str]]]:
    inventory = _tool_inventory(examples)
    tool_names: dict[str, list[str]] = {}
    argument_names: dict[str, list[str]] = {}
    tool_descriptions: dict[str, list[str]] = {}

    for tool_id, tool in sorted(inventory.items()):
        tool_names[tool_id] = _TOOL_ALIAS_OVERRIDES.get(tool_id, _default_tool_aliases(tool))
        tool_descriptions[tool_id] = _DESCRIPTION_ALIAS_OVERRIDES.get(
            tool_id, _default_description_aliases(tool)
        )
        for argument in tool.arguments:
            key = f"{tool_id}.{argument.name}"
            argument_names[key] = _ARGUMENT_ALIAS_OVERRIDES.get(
                key, _default_argument_aliases(tool_id, argument.name)
            )

    return {
        "tool_names": tool_names,
        "argument_names": argument_names,
        "tool_descriptions": tool_descriptions,
    }


def _serialize_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "tool_id": tool.tool_id,
        "name": tool.name,
        "description": tool.description,
        "arguments": [argument.to_dict() for argument in tool.arguments],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_alias_banks(alias_banks: AliasBanks, output_path: Path) -> Path:
    payload = {
        "alias_banks_hash": alias_banks_hash(alias_banks),
        "banks": alias_banks.banks,
    }
    return _write_json(output_path, payload)


def _write_distractor_library(distractor_tools: list[ToolSpec], output_path: Path) -> Path:
    payload = {
        "distractor_library_hash": distractor_library_hash(distractor_tools),
        "tools": [_serialize_tool(tool) for tool in distractor_tools],
    }
    return _write_json(output_path, payload)


def _prediction_for_example(example: CanonicalExample) -> str:
    return json.dumps(example.gold, separators=(",", ":"), sort_keys=True)


def _write_golden_fixture_for_real_eval(
    *,
    output_dir: Path,
    prompt_contract_version: str,
    iid_examples: list[CanonicalExample],
    schema_shift_examples: list[CanonicalExample],
    distractor_examples: list[CanonicalExample],
    nocall_missing_examples: list[CanonicalExample],
    nocall_unsupported_examples: list[CanonicalExample],
    control_examples: list,
) -> Path:
    score_groups = [
        ("iid", iid_examples[0]),
        ("schema_shift", schema_shift_examples[0]),
        ("distractor", distractor_examples[0]),
        ("nocall_missing_tool", nocall_missing_examples[0]),
        ("nocall_unsupported", nocall_unsupported_examples[0]),
    ]
    scored_cases = []
    scores = []
    for slice_name, example in score_groups:
        raw_output = _prediction_for_example(example)
        score = score_prediction(raw_output=raw_output, example=example)
        scores.append(score)
        scored_cases.append(
            {
                "slice": slice_name,
                "example": example.to_dict(),
                "raw_output": raw_output,
                "score": score,
            }
        )

    control_subset = control_examples[:3]
    control_scores = [
        score_control_prediction(raw_output=example.target_text, example=example)
        for example in control_subset
    ]
    fixture_payload = {
        "prompt_contract_version": prompt_contract_version,
        "cases": scored_cases,
        "aggregate_metrics": aggregate_scores(scores),
        "control_scores": control_scores,
        "control_exact_match_average": aggregate_control_scores(control_scores),
    }
    return write_golden_fixture(
        fixture_payload=fixture_payload,
        output_path=output_dir / "golden_fixtures" / "scorer_fixture.json",
    )


def run_eval_freeze_pipeline(
    *,
    canonical_examples: list[CanonicalExample],
    canonical_manifest_payload: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    prompt_contract_version = _manifest_prompt_contract_version(canonical_manifest_payload)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    eval_examples = [example for example in canonical_examples if example.split == "eval"]
    if not eval_examples:
        raise ValueError("Canonical manifest does not contain any eval examples.")

    alias_banks = freeze_alias_banks(build_alias_candidates(canonical_examples))
    alias_banks_path = _write_alias_banks(alias_banks, destination / "alias_banks" / "banks.json")

    schema_shift_examples = generate_schema_shift_examples(
        eval_examples, alias_banks, bank_id="test"
    )
    distractor_tools = build_distractor_tool_library()
    distractor_library_path = _write_distractor_library(
        distractor_tools, destination / "distractor_library" / "tools.json"
    )
    distractor_examples = generate_distractor_examples(
        schema_shift_examples,
        distractor_tools=distractor_tools,
    )
    nocall_missing_examples = generate_missing_tool_nocall_examples(eval_examples)
    nocall_unsupported_examples = generate_unsupported_intent_nocall_examples(eval_examples)
    control_examples = build_control_examples_from_canonical_examples(eval_examples)

    iid_eval_manifest = write_manifest(
        examples=eval_examples,
        output_dir=destination / "iid_eval_manifest",
        manifest_id="manifest_m1_mobile_actions_real_iid_eval_v1",
        prompt_contract_version=prompt_contract_version,
        metadata={
            "kind": "iid_eval",
            "source_manifest_id": canonical_manifest_payload["manifest_id"],
            "source_manifest_hash": canonical_manifest_payload["manifest_hash"],
        },
    )
    evaluation_manifest = write_manifest(
        examples=[
            *eval_examples,
            *schema_shift_examples,
            *distractor_examples,
            *nocall_missing_examples,
            *nocall_unsupported_examples,
        ],
        output_dir=destination / "evaluation_slices",
        manifest_id="manifest_m1_mobile_actions_real_eval_slices_v1",
        prompt_contract_version=prompt_contract_version,
        alias_banks=alias_banks.banks,
        metadata={
            "kind": "eval_slices",
            "source_manifest_id": canonical_manifest_payload["manifest_id"],
            "source_manifest_hash": canonical_manifest_payload["manifest_hash"],
            "split_manifest_hash": canonical_manifest_payload["metadata"]["split_manifest_hash"],
            "distractor_library_hash": distractor_library_hash(distractor_tools),
        },
    )

    control_suite_path = write_control_suite(
        examples=control_examples,
        output_path=destination / "control_suite" / "controls.jsonl",
    )
    control_manifest = write_control_suite_manifest(
        examples=control_examples,
        dataset_path=control_suite_path,
        output_path=destination / "control_suite" / "manifest.json",
        manifest_id="manifest_m1_mobile_actions_real_control_v1",
    )

    golden_fixture_path = _write_golden_fixture_for_real_eval(
        output_dir=destination,
        prompt_contract_version=prompt_contract_version,
        iid_examples=eval_examples,
        schema_shift_examples=schema_shift_examples,
        distractor_examples=distractor_examples,
        nocall_missing_examples=nocall_missing_examples,
        nocall_unsupported_examples=nocall_unsupported_examples,
        control_examples=control_examples,
    )

    leakage_report = run_leakage_audit(
        canonical_examples=canonical_examples,
        schema_shift_examples=schema_shift_examples,
        alias_banks=alias_banks,
        control_examples=control_examples,
        distractor_examples=distractor_examples,
        nocall_examples=[*nocall_missing_examples, *nocall_unsupported_examples],
    )
    leakage_path = write_leakage_audit(
        leakage_report,
        destination / "leakage_audit" / "audit.json",
    )

    summary = {
        "source_manifest_id": canonical_manifest_payload["manifest_id"],
        "source_manifest_hash": canonical_manifest_payload["manifest_hash"],
        "split_manifest_hash": canonical_manifest_payload["metadata"]["split_manifest_hash"],
        "prompt_contract_version": prompt_contract_version,
        "alias_banks_hash": alias_banks_hash(alias_banks),
        "alias_banks_path": str(alias_banks_path.resolve()),
        "distractor_library_hash": distractor_library_hash(distractor_tools),
        "distractor_library_path": str(distractor_library_path.resolve()),
        "iid_eval_manifest": iid_eval_manifest.__dict__,
        "evaluation_manifest": evaluation_manifest.__dict__,
        "control_manifest": control_manifest.__dict__,
        "control_suite_path": str(control_suite_path.resolve()),
        "golden_fixture_path": str(golden_fixture_path.resolve()),
        "leakage_audit_path": str(leakage_path.resolve()),
        "counts": {
            "iid_examples": len(eval_examples),
            "schema_shift_examples": len(schema_shift_examples),
            "distractor_examples": len(distractor_examples),
            "nocall_missing_examples": len(nocall_missing_examples),
            "nocall_unsupported_examples": len(nocall_unsupported_examples),
            "control_examples": len(control_examples),
        },
        "leakage_summary": leakage_report.summary,
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def run_eval_freeze_pipeline_from_manifest(
    *,
    canonical_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest_payload = load_manifest_payload(canonical_manifest_path)
    canonical_examples = load_examples(manifest_payload["dataset_path"])
    return run_eval_freeze_pipeline(
        canonical_examples=canonical_examples,
        canonical_manifest_payload=manifest_payload,
        output_dir=output_dir,
    )
