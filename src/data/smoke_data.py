from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.build_alias_bank import alias_banks_hash, freeze_alias_banks
from src.data.build_control_suite import build_control_example, write_control_suite
from src.data.canonical import ArgSpec, CanonicalExample, ToolSpec, build_canonical_example
from src.data.generate_nocall import (
    build_unsupported_intent_nocall_example,
    generate_missing_tool_nocall_examples,
)
from src.data.generate_schema_shift import generate_schema_shift_examples
from src.data.leakage_audit import run_leakage_audit, write_leakage_audit
from src.data.manifest import write_manifest
from src.data.splits import assign_locked_splits
from src.eval.control_metrics import aggregate_control_scores, score_control_prediction
from src.eval.golden_fixtures import write_golden_fixture
from src.eval.metrics import aggregate_scores, score_prediction
from src.models.format_prompts import PROMPT_CONTRACT_VERSION


def _tool_catalog() -> dict[str, ToolSpec]:
    return {
        "send_email": ToolSpec(
            tool_id="send_email",
            name="send_email",
            description="Send an email to a contact.",
            arguments=[
                ArgSpec(
                    name="recipient", type="string", required=True, description="Target address"
                ),
                ArgSpec(name="subject", type="string", required=True, description="Subject line"),
            ],
        ),
        "set_alarm": ToolSpec(
            tool_id="set_alarm",
            name="set_alarm",
            description="Create an alarm for a time.",
            arguments=[
                ArgSpec(name="time", type="string", required=True, description="Alarm time"),
            ],
        ),
    }


def _raw_examples() -> list[CanonicalExample]:
    tools = _tool_catalog()
    return [
        build_canonical_example(
            source="synthetic",
            raw_split="train",
            user_request="Email Sam about tomorrow's meeting.",
            tools=[tools["send_email"], tools["set_alarm"]],
            gold={
                "name": "send_email",
                "arguments": {
                    "recipient": "sam@example.com",
                    "subject": "tomorrow meeting",
                },
            },
            source_row_id="raw-001",
        ),
        build_canonical_example(
            source="synthetic",
            raw_split="train",
            user_request="Set an alarm for 07:30.",
            tools=[tools["send_email"], tools["set_alarm"]],
            gold={"name": "set_alarm", "arguments": {"time": "07:30"}},
            source_row_id="raw-002",
        ),
        build_canonical_example(
            source="synthetic",
            raw_split="train",
            user_request="Email Dana the budget update.",
            tools=[tools["send_email"], tools["set_alarm"]],
            gold={
                "name": "send_email",
                "arguments": {
                    "recipient": "dana@example.com",
                    "subject": "budget update",
                },
            },
            source_row_id="raw-003",
        ),
        build_canonical_example(
            source="synthetic",
            raw_split="train",
            user_request="Set an alarm for 18:45.",
            tools=[tools["send_email"], tools["set_alarm"]],
            gold={"name": "set_alarm", "arguments": {"time": "18:45"}},
            source_row_id="raw-004",
        ),
        build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Email Priya the slides.",
            tools=[tools["send_email"], tools["set_alarm"]],
            gold={
                "name": "send_email",
                "arguments": {
                    "recipient": "priya@example.com",
                    "subject": "slides",
                },
            },
            source_row_id="raw-101",
        ),
        build_canonical_example(
            source="synthetic",
            raw_split="eval",
            user_request="Set an alarm for 06:15.",
            tools=[tools["send_email"], tools["set_alarm"]],
            gold={"name": "set_alarm", "arguments": {"time": "06:15"}},
            source_row_id="raw-102",
        ),
    ]


def _alias_bank_inputs() -> dict[str, dict[str, list[str]]]:
    return {
        "tool_names": {
            "send_email": ["dispatch_message", "transmit_mail", "compose_email"],
            "set_alarm": ["schedule_alarm", "wake_timer", "arm_clock"],
        },
        "argument_names": {
            "send_email.recipient": ["to", "addressee", "target"],
            "send_email.subject": ["topic", "headline", "subject_line"],
            "set_alarm.time": ["alarm_time", "time_of_alarm", "ring_at"],
        },
        "tool_descriptions": {
            "send_email": [
                "Compose and send an electronic mail message.",
                "Dispatch an email to a recipient.",
                "Create an outbound email message.",
            ],
            "set_alarm": [
                "Schedule a device alarm.",
                "Create a wake-up timer.",
                "Arm an alarm for a chosen time.",
            ],
        },
    }


def _control_examples() -> list:
    return [
        build_control_example(
            source="synthetic",
            prompt="Rewrite to lowercase: HELLO WORLD",
            target_text="hello world",
            source_row_id="ctrl-001",
            meta={"task_type": "rewrite_lower"},
        ),
        build_control_example(
            source="synthetic",
            prompt="Return the first number from: order 583 ships today",
            target_text="583",
            source_row_id="ctrl-002",
            meta={"task_type": "extract_number"},
        ),
    ]


def run_smoke_data_pipeline(*, output_dir: str | Path) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    canonical_examples, split_manifest = assign_locked_splits(_raw_examples())
    alias_banks = freeze_alias_banks(_alias_bank_inputs())

    eval_examples = [example for example in canonical_examples if example.split == "eval"]
    schema_shift_examples = generate_schema_shift_examples(
        eval_examples, alias_banks, bank_id="test"
    )
    nocall_missing = generate_missing_tool_nocall_examples(eval_examples)
    nocall_unsupported = [
        build_unsupported_intent_nocall_example(
            source="synthetic",
            raw_split="eval",
            user_request="Tell me a joke about clocks.",
            tools=_raw_examples()[0].tools,
            source_row_id="raw-201",
        )
    ]
    control_examples = _control_examples()

    canonical_record = write_manifest(
        examples=canonical_examples,
        output_dir=destination / "canonical_manifest",
        manifest_id="manifest_m1_smoke_core_v1",
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        alias_banks=alias_banks.banks,
        metadata={"kind": "canonical_core", "split_manifest_hash": split_manifest.manifest_hash},
    )
    eval_record = write_manifest(
        examples=[*schema_shift_examples, *nocall_missing, *nocall_unsupported],
        output_dir=destination / "evaluation_slices",
        manifest_id="manifest_m1_smoke_eval_v1",
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        alias_banks=alias_banks.banks,
        metadata={"kind": "eval_slices"},
    )

    control_path = write_control_suite(
        examples=control_examples,
        output_path=destination / "control_suite" / "controls.jsonl",
    )

    score_cases = [
        (
            schema_shift_examples[0],
            '{"name":"dispatch_message","arguments":{"to":"priya@example.com","topic":"slides"}}',
        ),
        (nocall_missing[0], '{"name":"NO_TOOL","arguments":{}}'),
        (nocall_unsupported[0], '{"name":"NO_TOOL","arguments":{}}'),
        (
            eval_examples[0],
            (
                '{"name":"send_email","arguments":{"recipient":"priya@example.com",'
                '"subject":"slides"}} '
                '{"name":"NO_TOOL","arguments":{}}'
            ),
        ),
    ]
    scored_examples = [
        score_prediction(raw_output=raw_output, example=example)
        for example, raw_output in score_cases
    ]
    aggregate = aggregate_scores(scored_examples)

    control_predictions = [
        score_control_prediction(raw_output="hello world", example=control_examples[0]),
        score_control_prediction(raw_output="583", example=control_examples[1]),
    ]
    control_average = aggregate_control_scores(control_predictions)

    golden_fixture_path = write_golden_fixture(
        fixture_payload={
            "prompt_contract_version": PROMPT_CONTRACT_VERSION,
            "cases": [
                {
                    "example": example.to_dict(),
                    "raw_output": raw_output,
                    "score": score,
                }
                for (example, raw_output), score in zip(score_cases, scored_examples, strict=False)
            ],
            "aggregate_metrics": aggregate,
            "control_scores": control_predictions,
            "control_exact_match_average": control_average,
        },
        output_path=destination / "golden_fixtures" / "scorer_fixture.json",
    )

    leakage_report = run_leakage_audit(
        canonical_examples=canonical_examples,
        schema_shift_examples=schema_shift_examples,
        alias_banks=alias_banks,
        control_examples=control_examples,
    )
    leakage_path = write_leakage_audit(
        leakage_report,
        destination / "leakage_audit" / "audit.json",
    )

    summary = {
        "canonical_manifest": canonical_record.__dict__,
        "evaluation_manifest": eval_record.__dict__,
        "control_suite_path": str(control_path.resolve()),
        "golden_fixture_path": str(golden_fixture_path.resolve()),
        "leakage_audit_path": str(leakage_path.resolve()),
        "alias_banks_hash": alias_banks_hash(alias_banks),
        "split_manifest_hash": split_manifest.manifest_hash,
        "score_aggregate": aggregate.__dict__,
        "control_exact_match_average": control_average,
        "counts": {
            "canonical_examples": len(canonical_examples),
            "schema_shift_examples": len(schema_shift_examples),
            "nocall_missing_examples": len(nocall_missing),
            "nocall_unsupported_examples": len(nocall_unsupported),
            "control_examples": len(control_examples),
        },
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
