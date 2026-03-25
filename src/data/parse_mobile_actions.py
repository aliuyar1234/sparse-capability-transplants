from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.canonical import ArgSpec, CanonicalExample, ToolSpec, build_canonical_example
from src.data.manifest import ManifestRecord, write_manifest
from src.data.splits import SplitManifest, assign_locked_splits, write_split_manifest

_NO_TOOL_NAMES = {"NO_TOOL", "NO_CALL", "NO_FUNCTION"}
_RAW_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "eval": "eval",
    "evaluation": "eval",
    "test": "eval",
    "val": "eval",
    "validation": "eval",
    "dev": "eval",
}
_TOP_LEVEL_ROW_KEYS = ("rows", "data", "examples", "records", "items")
_TOOL_KEYS = ("tools", "available_tools", "functions", "tool_schemas", "actions")
_TEXT_KEYS = ("user_request", "request", "instruction", "query", "utterance", "prompt")
_SOURCE_ROW_ID_KEYS = ("id", "row_id", "source_row_id", "uid", "example_id")


class MobileActionsParseError(ValueError):
    """Raised when a raw Mobile Actions-style row cannot be canonicalized."""


@dataclass(frozen=True)
class MobileActionsImportResult:
    raw_path: str
    row_count: int
    retained_row_count: int
    skipped_row_count: int
    skipped_reasons: dict[str, int]
    canonical_manifest: ManifestRecord
    split_manifest: SplitManifest
    split_manifest_path: str
    summary_path: str
    positive_example_count: int
    no_tool_example_count: int
    unique_tool_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_path": self.raw_path,
            "row_count": self.row_count,
            "retained_row_count": self.retained_row_count,
            "skipped_row_count": self.skipped_row_count,
            "skipped_reasons": self.skipped_reasons,
            "canonical_manifest": asdict(self.canonical_manifest),
            "split_manifest": {
                "counts": self.split_manifest.counts,
                "example_ids_by_split": self.split_manifest.example_ids_by_split,
                "manifest_hash": self.split_manifest.manifest_hash,
            },
            "split_manifest_path": self.split_manifest_path,
            "summary_path": self.summary_path,
            "positive_example_count": self.positive_example_count,
            "no_tool_example_count": self.no_tool_example_count,
            "unique_tool_count": self.unique_tool_count,
        }


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _skip_reason_for_error(exc: MobileActionsParseError) -> str | None:
    message = str(exc)
    if "at most one tool call" in message:
        return "multi_tool_call_out_of_scope"
    return None


def _normalize_raw_split(raw_split: Any) -> tuple[str, str]:
    if raw_split is None:
        raise MobileActionsParseError("Missing raw split metadata.")

    original = str(raw_split).strip()
    normalized = _RAW_SPLIT_ALIASES.get(original.lower())
    if normalized is None:
        raise MobileActionsParseError(f"Unsupported raw split label: {original!r}")
    return normalized, original


def _extract_text_content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return _extract_text_content(value[key])
        return None
    if isinstance(value, list):
        parts = [part for item in value if (part := _extract_text_content(item))]
        if not parts:
            return None
        return "\n".join(parts)
    return None


def _extract_user_messages(row: dict[str, Any]) -> list[str]:
    for key in ("messages", "conversation", "chat"):
        payload = row.get(key)
        if not isinstance(payload, list):
            continue

        messages = []
        for message in payload:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() != "user":
                continue
            text = _extract_text_content(message.get("content"))
            if text:
                messages.append(text)
        if messages:
            return messages
    return []


def _extract_assistant_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("messages", "conversation", "chat"):
        payload = row.get(key)
        if not isinstance(payload, list):
            continue

        messages = []
        for message in payload:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() == "assistant":
                messages.append(message)
        if messages:
            return messages
    return []


def _extract_user_request(row: dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        text = _extract_text_content(row.get(key))
        if text:
            return text

    input_payload = row.get("input")
    if isinstance(input_payload, str):
        return input_payload.strip()
    if isinstance(input_payload, dict):
        for key in _TEXT_KEYS:
            text = _extract_text_content(input_payload.get(key))
            if text:
                return text

    user_messages = _extract_user_messages(row)
    if not user_messages:
        raise MobileActionsParseError("Could not extract a user request from the raw row.")

    unique_messages = {message.strip() for message in user_messages}
    if len(unique_messages) > 1:
        raise MobileActionsParseError(
            "Expected a single-turn user request but found multiple distinct user messages."
        )
    return user_messages[-1]


def _decode_json_string(value: str, *, context: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - covered by caller-specific tests
        raise MobileActionsParseError(f"Invalid JSON in {context}.") from exc


def _as_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = _decode_json_string(value, context=context)
        if isinstance(decoded, dict):
            return decoded
    raise MobileActionsParseError(f"Expected an object for {context}.")


def _schema_type(value: Any) -> str:
    if isinstance(value, str) and value:
        return value.lower()
    if isinstance(value, list):
        for item in value:
            resolved = _schema_type(item)
            if resolved != "string":
                return resolved
        return "string"
    if isinstance(value, dict):
        for key in ("type",):
            if key in value:
                return _schema_type(value[key])
        for key in ("anyOf", "oneOf", "allOf"):
            options = value.get(key)
            if isinstance(options, list):
                for option in options:
                    resolved = _schema_type(option)
                    if resolved != "string":
                        return resolved
    return "string"


def _parse_argument_specs_from_list(payload: list[Any]) -> list[ArgSpec]:
    arguments: list[ArgSpec] = []
    for raw_argument in payload:
        if not isinstance(raw_argument, dict):
            raise MobileActionsParseError("Tool arguments must be objects.")

        name = str(
            raw_argument.get("name")
            or raw_argument.get("arg_name")
            or raw_argument.get("key")
            or ""
        ).strip()
        if not name:
            raise MobileActionsParseError("Tool argument is missing a name.")

        arguments.append(
            ArgSpec(
                name=name,
                type=_schema_type(raw_argument.get("type")),
                required=bool(raw_argument.get("required", False)),
                description=str(raw_argument.get("description") or "").strip(),
            )
        )
    return arguments


def _parse_argument_specs_from_schema(payload: dict[str, Any]) -> list[ArgSpec]:
    schema = payload
    if "properties" in schema and isinstance(schema["properties"], dict):
        properties = schema["properties"]
        required_names = {str(name) for name in schema.get("required", []) if isinstance(name, str)}
    else:
        properties = {
            key: value
            for key, value in schema.items()
            if isinstance(value, dict) and key not in {"required", "type", "description"}
        }
        required_names = set()

    arguments = []
    for name in sorted(properties):
        raw_spec = properties[name]
        if not isinstance(raw_spec, dict):
            raise MobileActionsParseError("Tool argument schema entries must be objects.")

        arguments.append(
            ArgSpec(
                name=name,
                type=_schema_type(raw_spec.get("type")),
                required=name in required_names or bool(raw_spec.get("required", False)),
                description=str(raw_spec.get("description") or "").strip(),
            )
        )
    return arguments


def _extract_argument_specs(raw_tool: dict[str, Any]) -> list[ArgSpec]:
    direct_arguments = raw_tool.get("arguments")
    if isinstance(direct_arguments, list):
        return _parse_argument_specs_from_list(direct_arguments)
    if isinstance(direct_arguments, dict):
        return _parse_argument_specs_from_schema(direct_arguments)

    for key in ("parameters", "input_schema", "json_schema", "args_schema", "args"):
        if key not in raw_tool:
            continue
        payload = raw_tool[key]
        if isinstance(payload, list):
            return _parse_argument_specs_from_list(payload)
        if isinstance(payload, (dict, str)):
            return _parse_argument_specs_from_schema(_as_mapping(payload, context=f"tool {key}"))

    return []


def _unwrap_tool(raw_tool: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_tool.get("function"), dict):
        return raw_tool["function"]
    if isinstance(raw_tool.get("tool"), dict):
        return raw_tool["tool"]
    return raw_tool


def _parse_tool_spec(raw_tool: Any) -> ToolSpec:
    if not isinstance(raw_tool, dict):
        raise MobileActionsParseError("Tool entries must be objects.")

    payload = _unwrap_tool(raw_tool)
    name = str(payload.get("name") or payload.get("tool_name") or "").strip()
    if not name:
        raise MobileActionsParseError("Tool entry is missing a visible name.")

    tool_id = str(payload.get("tool_id") or payload.get("id") or name).strip()
    if not tool_id:
        raise MobileActionsParseError("Tool entry is missing a canonical identifier.")

    return ToolSpec(
        tool_id=tool_id,
        name=name,
        description=str(payload.get("description") or payload.get("summary") or "").strip(),
        arguments=_extract_argument_specs(payload),
    )


def _extract_tools(row: dict[str, Any]) -> list[ToolSpec]:
    candidate_sources = [row]
    if isinstance(row.get("input"), dict):
        candidate_sources.append(row["input"])

    for source in candidate_sources:
        for key in _TOOL_KEYS:
            payload = source.get(key)
            if not isinstance(payload, list):
                continue
            tools = [_parse_tool_spec(item) for item in payload]
            if not tools:
                continue
            tool_ids = [tool.tool_id for tool in tools]
            if len(set(tool_ids)) != len(tool_ids):
                raise MobileActionsParseError("Duplicate tool identifiers found in one row.")
            return tools

    raise MobileActionsParseError("Could not extract a tool inventory from the raw row.")


def _normalize_arguments_payload(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            return {}
        decoded = _decode_json_string(stripped, context="tool-call arguments")
        if not isinstance(decoded, dict):
            raise MobileActionsParseError("Tool-call arguments must decode to an object.")
        return decoded
    raise MobileActionsParseError("Tool-call arguments must be a mapping or JSON object string.")


def _normalize_gold_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        stripped = payload.strip()
        if not stripped:
            raise MobileActionsParseError("Gold output string is empty.")
        if stripped.upper() in _NO_TOOL_NAMES:
            return {"name": "NO_TOOL", "arguments": {}}
        return _normalize_gold_payload(_decode_json_string(stripped, context="gold output"))

    if isinstance(payload, list):
        if not payload:
            return {"name": "NO_TOOL", "arguments": {}}
        if len(payload) != 1:
            raise MobileActionsParseError("Single-turn rows must contain at most one tool call.")
        return _normalize_gold_payload(payload[0])

    if not isinstance(payload, dict):
        raise MobileActionsParseError("Gold output must be a JSON object, string, or list.")

    if payload.get("no_tool") is True or payload.get("should_call") is False:
        return {"name": "NO_TOOL", "arguments": {}}
    if "tool_calls" in payload:
        return _normalize_gold_payload(payload["tool_calls"])
    if "tool_call" in payload:
        return _normalize_gold_payload(payload["tool_call"])
    if isinstance(payload.get("function"), dict):
        return _normalize_gold_payload(payload["function"])
    if "content" in payload:
        text = _extract_text_content(payload["content"])
        if text:
            return _normalize_gold_payload(text)

    name = payload.get("name") or payload.get("tool_name") or payload.get("function_name")
    if name is None:
        raise MobileActionsParseError("Could not extract the gold tool name.")

    name_text = str(name).strip()
    if name_text.upper() in _NO_TOOL_NAMES:
        return {"name": "NO_TOOL", "arguments": {}}

    arguments = payload.get("arguments", payload.get("args"))
    return {"name": name_text, "arguments": _normalize_arguments_payload(arguments)}


def _extract_gold(row: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "gold",
        "target",
        "expected_output",
        "output",
        "response",
        "answer",
        "assistant",
        "tool_call",
        "tool_calls",
    ):
        if key in row:
            return _normalize_gold_payload(row[key])

    assistant_messages = _extract_assistant_messages(row)
    for message in reversed(assistant_messages):
        if "tool_calls" in message:
            return _normalize_gold_payload(message["tool_calls"])
        if "content" in message:
            text = _extract_text_content(message["content"])
            if text:
                return _normalize_gold_payload(text)

    if row.get("no_tool") is True or row.get("should_call") is False:
        return {"name": "NO_TOOL", "arguments": {}}

    raise MobileActionsParseError("Could not extract a gold tool call from the raw row.")


def _source_row_id(row: dict[str, Any], *, row_index: int) -> str:
    for key in _SOURCE_ROW_ID_KEYS:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row-{row_index:06d}"


def parse_mobile_actions_row(
    row: dict[str, Any],
    *,
    row_index: int,
    source: str = "mobile_actions",
) -> CanonicalExample:
    if not isinstance(row, dict):
        raise MobileActionsParseError("Raw rows must be JSON objects.")

    raw_split, original_raw_split = _normalize_raw_split(
        row.get("split") or row.get("raw_split") or row.get("metadata")
    )
    user_request = _extract_user_request(row)
    tools = _extract_tools(row)
    gold = _extract_gold(row)
    source_row_id = _source_row_id(row, row_index=row_index)

    canonical_tool_id: str | None = None
    canonical_argument_map: dict[str, str] = {}
    if gold["name"] != "NO_TOOL":
        matching_tool = next((tool for tool in tools if tool.name == gold["name"]), None)
        if matching_tool is None:
            raise MobileActionsParseError(
                f"Gold tool {gold['name']!r} was not found in the visible tool inventory."
            )
        canonical_tool_id = matching_tool.tool_id
        canonical_argument_map = {
            argument.name: argument.name for argument in matching_tool.arguments
        }
        unknown_arguments = sorted(set(gold["arguments"]) - set(canonical_argument_map.values()))
        if unknown_arguments:
            raise MobileActionsParseError(
                "Gold arguments "
                f"{unknown_arguments!r} are not defined by tool "
                f"{matching_tool.name!r}."
            )
    elif gold.get("arguments"):
        raise MobileActionsParseError("NO_TOOL gold outputs must not include arguments.")

    meta = {
        "source": source,
        "source_row_id": source_row_id,
        "canonical_tool_id": canonical_tool_id,
        "canonical_argument_map": canonical_argument_map,
        "alias_bank_id": "none",
    }
    if original_raw_split != raw_split:
        meta["original_raw_split"] = original_raw_split

    return build_canonical_example(
        source=source,
        raw_split=raw_split,
        user_request=user_request,
        tools=tools,
        gold=gold,
        source_row_id=source_row_id,
        meta=meta,
    )


def _rows_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise MobileActionsParseError("Top-level JSON arrays must contain objects only.")
        return payload

    if not isinstance(payload, dict):
        raise MobileActionsParseError("Top-level JSON payload must be a list or object.")

    for key in _TOP_LEVEL_ROW_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list) and all(isinstance(item, dict) for item in rows):
            return rows

    split_rows: list[dict[str, Any]] = []
    for split_name in ("train", "eval", "val", "validation", "dev", "test"):
        rows = payload.get(split_name)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                raise MobileActionsParseError("Split payload entries must be objects.")
            lifted = dict(row)
            lifted.setdefault("split", split_name)
            split_rows.append(lifted)
    if split_rows:
        return split_rows

    raise MobileActionsParseError(
        "Unsupported raw dataset container; expected rows/examples or split lists."
    )


def load_mobile_actions_rows(path: str | Path) -> list[dict[str, Any]]:
    raw_path = Path(path)
    suffix = raw_path.suffix.lower()
    text = raw_path.read_text(encoding="utf-8")

    if suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise MobileActionsParseError(
                    f"Invalid JSON on line {line_number} of {raw_path.name!r}."
                ) from exc
            if not isinstance(decoded, dict):
                raise MobileActionsParseError("JSONL rows must decode to objects.")
            rows.append(decoded)
        return rows

    if suffix == ".json":
        return _rows_from_json_payload(json.loads(text))

    raise MobileActionsParseError(f"Unsupported raw dataset format: {raw_path.suffix!r}")


def import_mobile_actions_dataset(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    manifest_id: str,
    prompt_contract_version: str,
    source: str = "mobile_actions",
    skip_unsupported: bool = False,
) -> MobileActionsImportResult:
    input_path = Path(raw_path).resolve()
    rows = load_mobile_actions_rows(input_path)
    examples: list[CanonicalExample] = []
    skipped_reasons: dict[str, int] = {}
    for index, row in enumerate(rows):
        try:
            examples.append(parse_mobile_actions_row(row, row_index=index, source=source))
        except MobileActionsParseError as exc:
            skip_reason = _skip_reason_for_error(exc)
            if not skip_unsupported or skip_reason is None:
                raise
            skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1

    if not examples:
        raise MobileActionsParseError("No supported single-call rows remained after import.")

    assigned_examples, split_manifest = assign_locked_splits(examples)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    canonical_manifest = write_manifest(
        examples=assigned_examples,
        output_dir=destination / "canonical_manifest",
        manifest_id=manifest_id,
        prompt_contract_version=prompt_contract_version,
        metadata={
            "kind": "canonical_core",
            "source": source,
            "raw_path": str(input_path),
            "raw_row_count": len(rows),
            "retained_row_count": len(examples),
            "skipped_row_count": len(rows) - len(examples),
            "skipped_reasons": skipped_reasons,
            "split_manifest_hash": split_manifest.manifest_hash,
        },
    )
    split_manifest_path = write_split_manifest(
        split_manifest,
        destination / "canonical_manifest" / "split_manifest.json",
    )

    positive_example_count = sum(
        1 for example in assigned_examples if example.gold["name"] != "NO_TOOL"
    )
    no_tool_example_count = len(assigned_examples) - positive_example_count
    unique_tool_count = len(
        {tool.tool_id for example in assigned_examples for tool in example.tools}
    )

    result = MobileActionsImportResult(
        raw_path=str(input_path),
        row_count=len(rows),
        retained_row_count=len(examples),
        skipped_row_count=len(rows) - len(examples),
        skipped_reasons=skipped_reasons,
        canonical_manifest=canonical_manifest,
        split_manifest=split_manifest,
        split_manifest_path=str(split_manifest_path.resolve()),
        summary_path=str((destination / "summary.json").resolve()),
        positive_example_count=positive_example_count,
        no_tool_example_count=no_tool_example_count,
        unique_tool_count=unique_tool_count,
    )
    Path(result.summary_path).write_text(
        _stable_json(result.to_dict()) + "\n",
        encoding="utf-8",
    )
    return result
