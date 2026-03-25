from __future__ import annotations

import unittest

from src.data.canonical import ArgSpec, ToolSpec, build_example_id


class ExampleIdTests(unittest.TestCase):
    def test_example_id_is_stable_across_argument_key_order(self) -> None:
        tool = ToolSpec(
            tool_id="send_email",
            name="send_email",
            description="Send an email.",
            arguments=[
                ArgSpec(
                    name="recipient",
                    type="string",
                    required=True,
                    description="Target address",
                ),
                ArgSpec(
                    name="subject",
                    type="string",
                    required=True,
                    description="Subject line",
                ),
            ],
        )
        first = build_example_id(
            source="mobile_actions",
            user_request="Email Sam about tomorrow's meeting.",
            tools=[tool],
            gold={
                "name": "send_email",
                "arguments": {
                    "recipient": "sam@example.com",
                    "subject": "meeting",
                },
            },
        )
        second = build_example_id(
            source="mobile_actions",
            user_request="Email Sam about tomorrow's meeting.",
            tools=[tool],
            gold={
                "name": "send_email",
                "arguments": {
                    "subject": "meeting",
                    "recipient": "sam@example.com",
                },
            },
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
