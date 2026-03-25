from __future__ import annotations

import unittest

from src.data.build_alias_bank import freeze_alias_banks
from src.data.build_control_suite import build_control_example
from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.generate_schema_shift import generate_schema_shift_examples
from src.data.leakage_audit import run_leakage_audit


def _canonical_examples() -> list:
    tool = ToolSpec(
        tool_id="send_email",
        name="send_email",
        description="Send an email to a contact.",
        arguments=[
            ArgSpec(name="recipient", type="string", required=True, description="Target address"),
        ],
    )
    first = build_canonical_example(
        source="synthetic",
        raw_split="eval",
        user_request="Email Sam.",
        tools=[tool],
        gold={"name": "send_email", "arguments": {"recipient": "sam@example.com"}},
        source_row_id="raw-1",
    )
    return [first]


class LeakageAuditTests(unittest.TestCase):
    def test_leakage_audit_reports_clean_smoke_case(self) -> None:
        canonical_examples = _canonical_examples()
        alias_banks = freeze_alias_banks(
            {
                "tool_names": {
                    "send_email": ["dispatch_message", "transmit_mail", "compose_email"]
                },
                "argument_names": {"send_email.recipient": ["to", "target", "addressee"]},
                "tool_descriptions": {
                    "send_email": [
                        "Compose and send an electronic mail message.",
                        "Dispatch email.",
                        "Create outbound mail.",
                    ]
                },
            }
        )
        schema_shift = generate_schema_shift_examples(
            canonical_examples, alias_banks, bank_id="test"
        )
        controls = [
            build_control_example(
                source="synthetic",
                prompt="Rewrite: HELLO",
                target_text="hello",
            )
        ]

        report = run_leakage_audit(
            canonical_examples=canonical_examples,
            schema_shift_examples=schema_shift,
            alias_banks=alias_banks,
            control_examples=controls,
        )

        self.assertTrue(report.alias_bank_disjoint)
        self.assertEqual(report.summary["split_overlap_count"], 0)
        self.assertEqual(report.summary["control_tool_json_count"], 0)

    def test_control_examples_with_tool_json_are_flagged(self) -> None:
        canonical_examples = _canonical_examples()
        alias_banks = freeze_alias_banks(
            {
                "tool_names": {
                    "send_email": ["dispatch_message", "transmit_mail", "compose_email"]
                },
                "argument_names": {"send_email.recipient": ["to", "target", "addressee"]},
                "tool_descriptions": {
                    "send_email": [
                        "Compose and send an electronic mail message.",
                        "Dispatch email.",
                        "Create outbound mail.",
                    ]
                },
            }
        )
        schema_shift = generate_schema_shift_examples(
            canonical_examples, alias_banks, bank_id="test"
        )
        controls = [
            build_control_example(
                source="synthetic",
                prompt="Return this JSON",
                target_text='{"name":"send_email","arguments":{}}',
            )
        ]

        report = run_leakage_audit(
            canonical_examples=canonical_examples,
            schema_shift_examples=schema_shift,
            alias_banks=alias_banks,
            control_examples=controls,
        )

        self.assertEqual(report.summary["control_tool_json_count"], 1)


if __name__ == "__main__":
    unittest.main()
