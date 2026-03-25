from __future__ import annotations

import unittest

from src.data.canonical import ArgSpec, ToolSpec
from src.models.format_prompts import (
    PROMPT_CONTRACT_VERSION,
    SYSTEM_MESSAGE,
    build_chat_messages,
    build_prompt_content,
    render_assistant_target,
    render_chat_prompt,
)


class PromptContractTests(unittest.TestCase):
    def test_prompt_contract_is_locked(self) -> None:
        tool = ToolSpec(
            tool_id="send_email",
            name="send_email",
            description="Send an email to a contact.",
            arguments=[
                ArgSpec(
                    name="recipient",
                    type="string",
                    required=True,
                    description="Target address",
                ),
            ],
        )
        prompt = build_prompt_content(
            user_request="Email Sam.",
            tools=[tool],
            target={"name": "send_email", "arguments": {"recipient": "sam@example.com"}},
        )

        expected_user_message = (
            "Available tools:\n"
            "[\n"
            "  {\n"
            '    "tool_id": "send_email",\n'
            '    "name": "send_email",\n'
            '    "description": "Send an email to a contact.",\n'
            '    "arguments": [\n'
            "      {\n"
            '        "name": "recipient",\n'
            '        "type": "string",\n'
            '        "required": true,\n'
            '        "description": "Target address"\n'
            "      }\n"
            "    ]\n"
            "  }\n"
            "]\n\n"
            "User request:\n"
            "Email Sam.\n\n"
            'Return exactly one JSON object with keys "name" and "arguments".\n'
            'If no tool applies, return {"name": "NO_TOOL","arguments": {}}.\n'
            "Do not return prose, Markdown, or multiple JSON objects."
        )

        self.assertEqual(prompt.prompt_contract_version, PROMPT_CONTRACT_VERSION)
        self.assertEqual(prompt.system_message, SYSTEM_MESSAGE)
        self.assertEqual(prompt.user_message, expected_user_message)

    def test_assistant_target_is_minified_and_sorted(self) -> None:
        target = render_assistant_target(
            {
                "name": "send_email",
                "arguments": {
                    "subject": "meeting",
                    "recipient": "sam@example.com",
                },
            }
        )
        self.assertEqual(
            target,
            '{"name":"send_email","arguments":{"recipient":"sam@example.com","subject":"meeting"}}',
        )

    def test_chat_template_serialization_uses_locked_messages(self) -> None:
        prompt = build_prompt_content(
            user_request="Email Sam.",
            tools=[],
            target={"name": "NO_TOOL", "arguments": {}},
        )

        class DummyTokenizer:
            chat_template = "{{ bos_token }}"

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return "serialized"

        tokenizer = DummyTokenizer()
        serialized = render_chat_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            add_generation_prompt=True,
        )

        self.assertEqual(serialized, "serialized")
        self.assertEqual(tokenizer.messages, build_chat_messages(prompt))
        self.assertFalse(tokenizer.tokenize)
        self.assertTrue(tokenizer.add_generation_prompt)


if __name__ == "__main__":
    unittest.main()
