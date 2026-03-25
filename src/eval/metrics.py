from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.canonical import ArgSpec, CanonicalExample, ToolSpec
from src.eval.normalize_args import normalize_argument_dict
from src.eval.parse_json import JsonExtractionResult, extract_first_valid_json_object

NO_TOOL_NAME = "NO_TOOL"


@dataclass(frozen=True)
class ExampleScore:
    example_id: str
    parse_status: str
    json_valid: bool
    strict_correct: bool
    semantic_correct: bool
    strict_error: str | None
    semantic_error: str | None
    predicted_name: str | None
    semantic_predicted_name: str | None
    gold_name: str
    semantic_gold_name: str
    arg_exact_match: bool
    is_nocall_example: bool
    predicted_is_nocall: bool | None


@dataclass(frozen=True)
class AggregateMetrics:
    total_examples: int
    strict_full_call_success: float
    semantic_full_call_success: float
    json_valid_rate: float
    call_no_call_f1: float
    tool_selection_accuracy_when_call: float
    argument_exact_match: float


def _tool_by_visible_name(example: CanonicalExample, name: str) -> ToolSpec | None:
    for tool in example.tools:
        if tool.name == name:
            return tool
    return None


def _tool_by_canonical_id(example: CanonicalExample, tool_id: str) -> ToolSpec | None:
    for tool in example.tools:
        if tool.tool_id == tool_id:
            return tool
    return None


def _tool_name_to_id(example: CanonicalExample) -> dict[str, str]:
    return {tool.name: tool.tool_id for tool in example.tools}


def _canonical_argument_map(example: CanonicalExample) -> dict[str, str]:
    return dict(example.meta.get("canonical_argument_map", {}))


def _semantic_gold_name(example: CanonicalExample) -> str:
    gold_name = str(example.gold["name"])
    if gold_name == NO_TOOL_NAME:
        return NO_TOOL_NAME
    canonical_tool_id = example.meta.get("canonical_tool_id")
    if canonical_tool_id is not None:
        return str(canonical_tool_id)
    return _tool_name_to_id(example).get(gold_name, gold_name)


def _semantic_predicted_name(example: CanonicalExample, predicted_name: str | None) -> str | None:
    if predicted_name is None:
        return None
    if predicted_name == NO_TOOL_NAME:
        return NO_TOOL_NAME
    return _tool_name_to_id(example).get(predicted_name, predicted_name)


def _arg_specs_by_name(
    tool: ToolSpec, *, canonicalize: dict[str, str] | None = None
) -> dict[str, ArgSpec]:
    specs: dict[str, ArgSpec] = {}
    for argument in tool.arguments:
        key = (
            canonicalize.get(argument.name, argument.name)
            if canonicalize is not None
            else argument.name
        )
        specs[key] = argument
    return specs


def _canonicalize_argument_keys(
    arguments: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    canonicalized: dict[str, Any] = {}
    for name, value in arguments.items():
        canonical_name = mapping.get(name, name)
        canonicalized[canonical_name] = value
    return canonicalized


def _score_arguments(
    *,
    predicted_arguments: Any,
    gold_arguments: dict[str, Any],
    arg_specs: dict[str, ArgSpec],
) -> tuple[bool, str | None]:
    if not isinstance(predicted_arguments, dict):
        return False, "arguments_not_object"

    allowed_names = set(arg_specs)
    required_names = {name for name, spec in arg_specs.items() if spec.required}
    predicted_names = set(predicted_arguments)
    gold_names = set(gold_arguments)

    unknown_names = predicted_names - allowed_names
    if unknown_names:
        return False, "unknown_argument"

    missing_required = required_names - predicted_names
    if missing_required:
        return False, "missing_required_argument"

    unexpected_names = predicted_names - gold_names
    if unexpected_names:
        return False, "unexpected_argument"

    normalized_predicted, predicted_errors = normalize_argument_dict(predicted_arguments, arg_specs)
    if predicted_errors:
        return False, "argument_normalization_failed"

    normalized_gold, gold_errors = normalize_argument_dict(gold_arguments, arg_specs)
    if gold_errors:
        return False, "gold_argument_normalization_failed"

    for key, gold_value in normalized_gold.items():
        if key not in normalized_predicted:
            return False, "missing_argument"
        if normalized_predicted[key] != gold_value:
            return False, "argument_value_mismatch"

    return True, None


def _score_strict(
    *,
    example: CanonicalExample,
    extraction: JsonExtractionResult,
) -> tuple[bool, str | None, bool]:
    if extraction.status != "ok":
        return False, extraction.status, False

    payload = extraction.parsed
    if payload is None or not isinstance(payload.get("name"), str):
        return False, "missing_name", False

    predicted_name = payload["name"]
    gold_name = str(example.gold["name"])

    if predicted_name != gold_name:
        error = (
            "wrong_nocall_decision"
            if gold_name == NO_TOOL_NAME or predicted_name == NO_TOOL_NAME
            else "wrong_tool"
        )
        return False, error, False

    if gold_name == NO_TOOL_NAME:
        return True, None, True

    tool = _tool_by_visible_name(example, gold_name)
    if tool is None:
        return False, "gold_tool_missing_from_inventory", False

    arg_match, arg_error = _score_arguments(
        predicted_arguments=payload.get("arguments"),
        gold_arguments=dict(example.gold.get("arguments", {})),
        arg_specs=_arg_specs_by_name(tool),
    )
    return arg_match, arg_error, arg_match


def _score_semantic(
    *,
    example: CanonicalExample,
    extraction: JsonExtractionResult,
) -> tuple[bool, str | None, str | None, bool]:
    if extraction.status != "ok":
        return False, extraction.status, None, False

    payload = extraction.parsed
    if payload is None or not isinstance(payload.get("name"), str):
        return False, "missing_name", None, False

    predicted_name = payload["name"]
    semantic_predicted_name = _semantic_predicted_name(example, predicted_name)
    semantic_gold_name = _semantic_gold_name(example)
    if semantic_predicted_name != semantic_gold_name:
        error = (
            "wrong_nocall_decision"
            if semantic_gold_name == NO_TOOL_NAME or semantic_predicted_name == NO_TOOL_NAME
            else "wrong_tool"
        )
        return False, error, semantic_predicted_name, False

    if semantic_gold_name == NO_TOOL_NAME:
        return True, None, semantic_predicted_name, True

    tool = _tool_by_canonical_id(example, semantic_gold_name)
    if tool is None:
        return False, "gold_tool_missing_from_inventory", semantic_predicted_name, False

    predicted_arguments = payload.get("arguments")
    if not isinstance(predicted_arguments, dict):
        return False, "arguments_not_object", semantic_predicted_name, False

    canonical_arg_map = _canonical_argument_map(example)
    semantic_predicted_args = _canonicalize_argument_keys(predicted_arguments, canonical_arg_map)
    semantic_gold_args = _canonicalize_argument_keys(
        dict(example.gold.get("arguments", {})), canonical_arg_map
    )
    arg_match, arg_error = _score_arguments(
        predicted_arguments=semantic_predicted_args,
        gold_arguments=semantic_gold_args,
        arg_specs=_arg_specs_by_name(tool, canonicalize=canonical_arg_map),
    )
    return arg_match, arg_error, semantic_predicted_name, arg_match


def score_prediction(
    *,
    raw_output: str,
    example: CanonicalExample,
) -> ExampleScore:
    extraction = extract_first_valid_json_object(raw_output)
    payload = extraction.parsed or {}
    predicted_name = payload.get("name") if isinstance(payload.get("name"), str) else None

    strict_correct, strict_error, strict_arg_match = _score_strict(
        example=example, extraction=extraction
    )
    semantic_correct, semantic_error, semantic_predicted_name, semantic_arg_match = _score_semantic(
        example=example,
        extraction=extraction,
    )

    predicted_is_nocall = None if predicted_name is None else predicted_name == NO_TOOL_NAME
    return ExampleScore(
        example_id=example.example_id,
        parse_status=extraction.status,
        json_valid=extraction.valid_object_count > 0,
        strict_correct=strict_correct,
        semantic_correct=semantic_correct,
        strict_error=strict_error,
        semantic_error=semantic_error,
        predicted_name=predicted_name,
        semantic_predicted_name=semantic_predicted_name,
        gold_name=str(example.gold["name"]),
        semantic_gold_name=_semantic_gold_name(example),
        arg_exact_match=strict_arg_match and semantic_arg_match,
        is_nocall_example=str(example.gold["name"]) == NO_TOOL_NAME,
        predicted_is_nocall=predicted_is_nocall,
    )


def aggregate_scores(scores: list[ExampleScore]) -> AggregateMetrics:
    total = len(scores)
    if total == 0:
        return AggregateMetrics(
            total_examples=0,
            strict_full_call_success=0.0,
            semantic_full_call_success=0.0,
            json_valid_rate=0.0,
            call_no_call_f1=0.0,
            tool_selection_accuracy_when_call=0.0,
            argument_exact_match=0.0,
        )

    strict_successes = sum(score.strict_correct for score in scores)
    semantic_successes = sum(score.semantic_correct for score in scores)
    valid_json = sum(score.json_valid for score in scores)
    arg_exact = sum(score.arg_exact_match for score in scores)

    tp = fp = fn = 0
    tool_hits = 0
    tool_total = 0
    for score in scores:
        predicted_call = (
            score.semantic_predicted_name is not None
            and score.semantic_predicted_name != NO_TOOL_NAME
        )
        gold_call = score.semantic_gold_name != NO_TOOL_NAME
        if predicted_call and gold_call:
            tp += 1
        elif predicted_call and not gold_call:
            fp += 1
        elif not predicted_call and gold_call:
            fn += 1

        if gold_call:
            tool_total += 1
            if score.semantic_predicted_name == score.semantic_gold_name:
                tool_hits += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return AggregateMetrics(
        total_examples=total,
        strict_full_call_success=strict_successes / total,
        semantic_full_call_success=semantic_successes / total,
        json_valid_rate=valid_json / total,
        call_no_call_f1=f1,
        tool_selection_accuracy_when_call=tool_hits / tool_total if tool_total else 0.0,
        argument_exact_match=arg_exact / total,
    )
