from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import Any


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sorted_arguments(arguments: list["ArgSpec"]) -> list[dict[str, Any]]:
    return [argument.to_dict() for argument in arguments]


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str
    required: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    name: str
    description: str
    arguments: list[ArgSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "arguments": _sorted_arguments(self.arguments),
        }


@dataclass(frozen=True)
class CanonicalExample:
    example_id: str
    split: str
    user_request: str
    tools: list[ToolSpec]
    gold: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "user_request": self.user_request,
            "tools": [tool.to_dict() for tool in self.tools],
            "gold": self.gold,
            "meta": self.meta,
        }


def build_example_id(
    *,
    source: str,
    user_request: str,
    tools: list[ToolSpec],
    gold: dict[str, Any],
    variant: str = "canonical",
    source_row_id: str | None = None,
) -> str:
    payload = {
        "source": source,
        "source_row_id": source_row_id,
        "user_request": user_request,
        "tools": [tool.to_dict() for tool in tools],
        "gold": gold,
        "variant": variant,
    }
    digest = sha1(_stable_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"ex_{digest}"


def build_canonical_example(
    *,
    source: str,
    raw_split: str,
    user_request: str,
    tools: list[ToolSpec],
    gold: dict[str, Any],
    variant: str = "canonical",
    source_row_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> CanonicalExample:
    example_id = build_example_id(
        source=source,
        user_request=user_request,
        tools=tools,
        gold=gold,
        variant=variant,
        source_row_id=source_row_id,
    )
    merged_meta = {
        "source": source,
        "variant": variant,
        "raw_split": raw_split,
    }
    if meta:
        merged_meta.update(meta)
    return CanonicalExample(
        example_id=example_id,
        split="unassigned",
        user_request=user_request,
        tools=tools,
        gold=gold,
        meta=merged_meta,
    )
