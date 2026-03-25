from __future__ import annotations

import json
from hashlib import sha1

from src.data.canonical import ArgSpec, CanonicalExample, ToolSpec, build_example_id


def build_distractor_tool_library() -> list[ToolSpec]:
    return [
        ToolSpec(
            tool_id="open_bluetooth_settings",
            name="open_bluetooth_settings",
            description="Opens the Bluetooth settings screen.",
            arguments=[],
        ),
        ToolSpec(
            tool_id="create_note",
            name="create_note",
            description="Creates a note with a title and body.",
            arguments=[
                ArgSpec(name="title", type="string", required=True, description="Note title"),
                ArgSpec(name="body", type="string", required=True, description="Note content"),
            ],
        ),
        ToolSpec(
            tool_id="set_timer",
            name="set_timer",
            description="Starts a countdown timer.",
            arguments=[
                ArgSpec(
                    name="duration_minutes",
                    type="integer",
                    required=True,
                    description="Timer duration in minutes.",
                ),
            ],
        ),
    ]


def _stable_tool_order(example_id: str, tools: list[ToolSpec]) -> list[ToolSpec]:
    return sorted(
        tools,
        key=lambda tool: sha1(f"{example_id}::{tool.tool_id}".encode("utf-8")).hexdigest(),
    )


def _clone_tools(tools: list[ToolSpec]) -> list[ToolSpec]:
    return [
        ToolSpec(
            tool_id=tool.tool_id,
            name=tool.name,
            description=tool.description,
            arguments=[
                ArgSpec(
                    name=argument.name,
                    type=argument.type,
                    required=argument.required,
                    description=argument.description,
                )
                for argument in tool.arguments
            ],
        )
        for tool in tools
    ]


def distractor_library_hash(tools: list[ToolSpec]) -> str:
    payload = [
        {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "description": tool.description,
            "arguments": [argument.to_dict() for argument in tool.arguments],
        }
        for tool in tools
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha1(encoded).hexdigest()


def generate_distractor_examples(
    examples: list[CanonicalExample],
    *,
    distractor_tools: list[ToolSpec] | None = None,
) -> list[CanonicalExample]:
    library = distractor_tools or build_distractor_tool_library()
    library_ids = {tool.tool_id for tool in library}
    distractor_examples: list[CanonicalExample] = []

    for example in examples:
        existing_ids = {tool.tool_id for tool in example.tools}
        overlap = sorted(existing_ids & library_ids)
        if overlap:
            raise ValueError(f"Distractor tool ids collide with existing tools: {overlap}")

        combined_tools = _stable_tool_order(
            example.example_id,
            [*_clone_tools(example.tools), *_clone_tools(library)],
        )
        meta = dict(example.meta)
        meta.update(
            {
                "variant": "distractor",
                "distractor_tool_ids": [tool.tool_id for tool in library],
                "source_example_id": str(meta.get("source_example_id", example.example_id)),
            }
        )
        source_row_id = str(meta.get("source_row_id", example.example_id))
        distractor_examples.append(
            CanonicalExample(
                example_id=build_example_id(
                    source=str(meta.get("source", "synthetic")),
                    user_request=example.user_request,
                    tools=combined_tools,
                    gold=dict(example.gold),
                    variant="distractor",
                    source_row_id=source_row_id,
                ),
                split=example.split,
                user_request=example.user_request,
                tools=combined_tools,
                gold=dict(example.gold),
                meta=meta,
            )
        )
    return distractor_examples
