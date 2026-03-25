from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.data.canonical import ToolSpec

PROMPT_CONTRACT_VERSION = "fc_v1"
SYSTEM_MESSAGE = "Return exactly one JSON object and no prose."
NO_TOOL_OBJECT = {"name": "NO_TOOL", "arguments": {}}


def _stable_tool_schema(tool: ToolSpec | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool, ToolSpec):
        return tool.to_dict()
    return tool


def _stable_json_payload(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, separators=(",", ": "))


def render_user_message(user_request: str, tools: list[ToolSpec | dict[str, Any]]) -> str:
    tool_inventory = [_stable_tool_schema(tool) for tool in tools]
    tool_json = _stable_json_payload(tool_inventory, indent=2)
    no_tool_json = _stable_json_payload(NO_TOOL_OBJECT)
    return (
        "Available tools:\n"
        f"{tool_json}\n\n"
        "User request:\n"
        f"{user_request}\n\n"
        'Return exactly one JSON object with keys "name" and "arguments".\n'
        f"If no tool applies, return {no_tool_json}.\n"
        "Do not return prose, Markdown, or multiple JSON objects."
    )


def render_assistant_target(target: dict[str, Any]) -> str:
    payload = {
        "name": target["name"],
        "arguments": dict(sorted(target.get("arguments", {}).items())),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class PromptContent:
    prompt_contract_version: str
    system_message: str
    user_message: str
    assistant_target: str


def build_chat_messages(prompt: PromptContent) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt.system_message},
        {"role": "user", "content": prompt.user_message},
    ]


def build_training_chat_messages(prompt: PromptContent) -> list[dict[str, str]]:
    return [
        *build_chat_messages(prompt),
        {"role": "assistant", "content": prompt.assistant_target},
    ]


def render_chat_prompt(
    *,
    prompt: PromptContent,
    tokenizer: Any,
    add_generation_prompt: bool,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not implement apply_chat_template.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer does not expose a chat_template.")
    return str(
        tokenizer.apply_chat_template(
            build_chat_messages(prompt),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    )


def build_prompt_content(
    *,
    user_request: str,
    tools: list[ToolSpec | dict[str, Any]],
    target: dict[str, Any],
) -> PromptContent:
    return PromptContent(
        prompt_contract_version=PROMPT_CONTRACT_VERSION,
        system_message=SYSTEM_MESSAGE,
        user_message=render_user_message(user_request=user_request, tools=tools),
        assistant_target=render_assistant_target(target),
    )
