from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonExtractionResult:
    status: str
    parsed: dict[str, Any] | None
    raw_json: str | None
    valid_object_count: int


def _find_balanced_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1

    return None


def extract_first_valid_json_object(text: str) -> JsonExtractionResult:
    valid_objects: list[tuple[dict[str, Any], str]] = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue

        end = _find_balanced_object_end(text, index)
        if end is None:
            index += 1
            continue

        candidate = text[index:end]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            index += 1
            continue

        if isinstance(parsed, dict):
            valid_objects.append((parsed, candidate))
            index = end
            continue

        index += 1

    if not valid_objects:
        return JsonExtractionResult(
            status="no_valid_object",
            parsed=None,
            raw_json=None,
            valid_object_count=0,
        )

    parsed, raw_json = valid_objects[0]
    status = "ok" if len(valid_objects) == 1 else "multiple_objects"
    return JsonExtractionResult(
        status=status,
        parsed=parsed,
        raw_json=raw_json,
        valid_object_count=len(valid_objects),
    )
