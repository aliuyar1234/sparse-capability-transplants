from __future__ import annotations

from dataclasses import replace

from src.data.build_alias_bank import AliasBanks
from src.data.canonical import ArgSpec, CanonicalExample, ToolSpec, build_example_id


def _argument_bank_key(tool_id: str, argument_name: str) -> str:
    return f"{tool_id}.{argument_name}"


def _first_alias(
    alias_banks: AliasBanks, bank_id: str, domain: str, key: str, fallback: str
) -> str:
    candidates = alias_banks.banks[bank_id][domain].get(key, [])
    return candidates[0] if candidates else fallback


def generate_schema_shift_examples(
    examples: list[CanonicalExample],
    alias_banks: AliasBanks,
    *,
    bank_id: str = "test",
) -> list[CanonicalExample]:
    shifted_examples: list[CanonicalExample] = []
    for example in examples:
        canonical_argument_map: dict[str, str] = {}
        shifted_tools: list[ToolSpec] = []
        for tool in example.tools:
            shifted_arguments: list[ArgSpec] = []
            for argument in tool.arguments:
                shifted_name = _first_alias(
                    alias_banks,
                    bank_id,
                    "argument_names",
                    _argument_bank_key(tool.tool_id, argument.name),
                    argument.name,
                )
                canonical_argument_map[shifted_name] = argument.name
                shifted_arguments.append(replace(argument, name=shifted_name))

            shifted_tools.append(
                ToolSpec(
                    tool_id=tool.tool_id,
                    name=_first_alias(alias_banks, bank_id, "tool_names", tool.tool_id, tool.name),
                    description=_first_alias(
                        alias_banks,
                        bank_id,
                        "tool_descriptions",
                        tool.tool_id,
                        tool.description,
                    ),
                    arguments=shifted_arguments,
                )
            )

        shifted_gold = dict(example.gold)
        shifted_gold_arguments = dict(example.gold.get("arguments", {}))
        canonical_tool_id = None
        if shifted_gold["name"] != "NO_TOOL":
            original_tool = next(
                tool for tool in example.tools if tool.name == shifted_gold["name"]
            )
            canonical_tool_id = original_tool.tool_id
            shifted_gold["name"] = next(
                tool.name for tool in shifted_tools if tool.tool_id == original_tool.tool_id
            )
            shifted_gold_arguments = {
                canonical_to_shifted_name(canonical_argument_map, argument_name): value
                for argument_name, value in shifted_gold_arguments.items()
            }
        shifted_gold["arguments"] = shifted_gold_arguments

        shifted_meta = dict(example.meta)
        shifted_meta.update(
            {
                "variant": "schema_shift",
                "alias_bank_id": bank_id,
                "canonical_tool_id": canonical_tool_id,
                "canonical_argument_map": canonical_argument_map,
                "source_example_id": example.example_id,
            }
        )
        shifted_example_id = build_example_id(
            source=str(shifted_meta.get("source", "synthetic")),
            user_request=example.user_request,
            tools=shifted_tools,
            gold=shifted_gold,
            variant="schema_shift",
            source_row_id=example.example_id,
        )
        shifted_examples.append(
            CanonicalExample(
                example_id=shifted_example_id,
                split=example.split,
                user_request=example.user_request,
                tools=shifted_tools,
                gold=shifted_gold,
                meta=shifted_meta,
            )
        )

    return shifted_examples


def canonical_to_shifted_name(canonical_argument_map: dict[str, str], canonical_name: str) -> str:
    for shifted_name, mapped_canonical in canonical_argument_map.items():
        if mapped_canonical == canonical_name:
            return shifted_name
    return canonical_name
