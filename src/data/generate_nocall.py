from __future__ import annotations

from dataclasses import replace
from hashlib import sha1

from src.data.canonical import CanonicalExample, ToolSpec, build_canonical_example

_FROZEN_SPLITS = {"train", "val", "calib", "eval", "control"}


def _resolved_split(*, example: CanonicalExample | None = None, raw_split: str) -> str:
    if example is not None and example.split in _FROZEN_SPLITS:
        return example.split
    if raw_split in _FROZEN_SPLITS:
        return raw_split
    return "unassigned"


def generate_missing_tool_nocall_examples(
    examples: list[CanonicalExample],
) -> list[CanonicalExample]:
    nocall_examples: list[CanonicalExample] = []
    for example in examples:
        gold_name = str(example.gold["name"])
        if gold_name == "NO_TOOL":
            continue

        remaining_tools = [tool for tool in example.tools if tool.name != gold_name]
        if len(remaining_tools) == len(example.tools):
            continue

        meta = dict(example.meta)
        meta.update(
            {
                "variant": "nocall_missing_tool",
                "canonical_tool_id": None,
                "canonical_argument_map": {},
                "withheld_tool_name": gold_name,
                "source_example_id": example.example_id,
            }
        )
        nocall_examples.append(
            replace(
                build_canonical_example(
                    source=str(meta.get("source", "synthetic")),
                    raw_split=str(meta.get("raw_split", "eval")),
                    user_request=example.user_request,
                    tools=remaining_tools,
                    gold={"name": "NO_TOOL", "arguments": {}},
                    variant="nocall_missing_tool",
                    source_row_id=example.example_id,
                    meta=meta,
                ),
                split=_resolved_split(
                    example=example,
                    raw_split=str(meta.get("raw_split", "eval")),
                ),
            )
        )
    return nocall_examples


def build_unsupported_intent_nocall_example(
    *,
    source: str,
    raw_split: str,
    user_request: str,
    tools: list[ToolSpec],
    source_row_id: str | None = None,
    meta: dict[str, object] | None = None,
) -> CanonicalExample:
    merged_meta = {"canonical_tool_id": None, "canonical_argument_map": {}, "alias_bank_id": "none"}
    if meta:
        merged_meta.update(meta)
    return replace(
        build_canonical_example(
            source=source,
            raw_split=raw_split,
            user_request=user_request,
            tools=tools,
            gold={"name": "NO_TOOL", "arguments": {}},
            variant="nocall_unsupported",
            source_row_id=source_row_id,
            meta=merged_meta,
        ),
        split=_resolved_split(raw_split=raw_split),
    )


_UNSUPPORTED_INTENT_TEMPLATES = (
    (
        "translate_to_german",
        'Translate the following sentence into German and return only the translation: "{topic}"',
    ),
    (
        "write_haiku",
        'Write a short haiku about the following topic: "{topic}"',
    ),
    (
        "five_word_summary",
        'Summarize the following sentence in exactly five words: "{topic}"',
    ),
)


def _topic_snippet(text: str, *, max_chars: int = 96) -> str:
    stripped = " ".join(text.split()).strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 3].rstrip() + "..."


def generate_unsupported_intent_nocall_examples(
    examples: list[CanonicalExample],
) -> list[CanonicalExample]:
    unsupported_examples: list[CanonicalExample] = []
    for example in examples:
        selector = int(
            sha1(f"{example.example_id}::unsupported".encode("utf-8")).hexdigest()[:8], 16
        ) % len(_UNSUPPORTED_INTENT_TEMPLATES)
        template_id, template = _UNSUPPORTED_INTENT_TEMPLATES[selector]
        unsupported_examples.append(
            build_unsupported_intent_nocall_example(
                source=str(example.meta.get("source", "synthetic")),
                raw_split=str(example.meta.get("raw_split", example.split)),
                user_request=template.format(topic=_topic_snippet(example.user_request)),
                tools=example.tools,
                source_row_id=example.example_id,
                meta={
                    "source_example_id": example.example_id,
                    "source_split": example.split,
                    "template_id": template_id,
                },
            )
        )
    return unsupported_examples
