from __future__ import annotations

import unittest

from src.data.build_alias_bank import freeze_alias_banks
from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.generate_nocall import (
    build_unsupported_intent_nocall_example,
    generate_missing_tool_nocall_examples,
)
from src.data.generate_schema_shift import generate_schema_shift_examples


def _base_example() -> object:
    return build_canonical_example(
        source="synthetic",
        raw_split="eval",
        user_request="Email Sam about the meeting.",
        tools=[
            ToolSpec(
                tool_id="send_email",
                name="send_email",
                description="Send an email to a contact.",
                arguments=[
                    ArgSpec(
                        name="recipient", type="string", required=True, description="Target address"
                    ),
                    ArgSpec(
                        name="subject", type="string", required=True, description="Subject line"
                    ),
                ],
            )
        ],
        gold={
            "name": "send_email",
            "arguments": {
                "recipient": "sam@example.com",
                "subject": "meeting",
            },
        },
    )


class SchemaShiftTests(unittest.TestCase):
    def test_schema_shift_rewrites_visible_tool_and_argument_names(self) -> None:
        example = _base_example()
        alias_banks = freeze_alias_banks(
            {
                "tool_names": {
                    "send_email": ["dispatch_message", "transmit_mail", "compose_email"],
                },
                "argument_names": {
                    "send_email.recipient": ["to", "target", "addressee"],
                    "send_email.subject": ["topic", "subject_line", "headline"],
                },
                "tool_descriptions": {
                    "send_email": [
                        "Compose and send an electronic mail message",
                        "Dispatch mail to a recipient",
                        "Create an outbound email",
                    ],
                },
            }
        )

        shifted = generate_schema_shift_examples([example], alias_banks, bank_id="test")[0]

        self.assertEqual(shifted.meta["variant"], "schema_shift")
        self.assertEqual(shifted.meta["alias_bank_id"], "test")
        self.assertEqual(shifted.meta["canonical_tool_id"], "send_email")
        self.assertEqual(
            set(shifted.meta["canonical_argument_map"].values()), {"recipient", "subject"}
        )
        self.assertEqual(
            set(shifted.gold["arguments"]),
            set(shifted.meta["canonical_argument_map"]),
        )

    def test_missing_tool_nocall_generation_removes_gold_tool(self) -> None:
        example = _base_example()
        missing_tool = generate_missing_tool_nocall_examples([example])[0]

        self.assertEqual(missing_tool.gold["name"], "NO_TOOL")
        self.assertEqual(missing_tool.split, "eval")
        self.assertEqual(missing_tool.meta["variant"], "nocall_missing_tool")
        self.assertEqual(missing_tool.meta["withheld_tool_name"], "send_email")
        self.assertEqual(len(missing_tool.tools), 0)

    def test_unsupported_intent_wrapper_sets_nocall_gold(self) -> None:
        example = _base_example()
        unsupported = build_unsupported_intent_nocall_example(
            source="synthetic",
            raw_split="eval",
            user_request="Tell me a joke.",
            tools=example.tools,
            source_row_id="unsupported-1",
        )

        self.assertEqual(unsupported.gold["name"], "NO_TOOL")
        self.assertEqual(unsupported.split, "eval")
        self.assertEqual(unsupported.meta["variant"], "nocall_unsupported")


if __name__ == "__main__":
    unittest.main()
