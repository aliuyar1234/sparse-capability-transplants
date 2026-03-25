from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.data.canonical import ArgSpec


@dataclass(frozen=True)
class NormalizedValue:
    value: Any
    error: str | None = None


def _coerce_arg_spec(arg_spec: ArgSpec | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arg_spec, ArgSpec):
        payload = asdict(arg_spec)
    else:
        payload = dict(arg_spec)
    payload.setdefault("type", "string")
    payload.setdefault("required", False)
    payload.setdefault("case_sensitive", False)
    payload.setdefault("normalizer", None)
    return payload


def _normalize_phone(value: Any) -> str:
    digits = re.sub(r"\D+", "", str(value))
    return digits


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in {0, 1}:
            return bool(value)
        raise ValueError("numeric_bool_out_of_range")

    normalized = str(value).strip().lower()
    truthy = {"true", "1", "yes", "y", "on"}
    falsy = {"false", "0", "no", "n", "off"}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError("invalid_bool")


def _normalize_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("bool_is_not_int")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("non_integral_float")
        return int(value)

    normalized = str(value).strip()
    return int(normalized)


def _normalize_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("bool_is_not_float")
    numeric = float(str(value).strip()) if not isinstance(value, (int, float)) else float(value)
    if not math.isfinite(numeric):
        raise ValueError("non_finite_float")
    return numeric


def _normalize_timestamp(value: Any) -> str:
    text = str(value).strip()
    normalized = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_string(
    value: Any,
    *,
    case_sensitive: bool,
    normalizer: str | None,
) -> str:
    text = str(value).strip()
    if normalizer == "phone":
        return _normalize_phone(text)
    if normalizer == "timestamp":
        return _normalize_timestamp(text)
    if case_sensitive:
        return text
    return text.lower()


def normalize_argument_value(
    value: Any,
    arg_spec: ArgSpec | dict[str, Any],
) -> NormalizedValue:
    spec = _coerce_arg_spec(arg_spec)
    arg_type = str(spec["type"]).lower()
    case_sensitive = bool(spec.get("case_sensitive", False))
    normalizer = spec.get("normalizer")

    try:
        if arg_type in {"string", "str"}:
            normalized = _normalize_string(
                value,
                case_sensitive=case_sensitive,
                normalizer=normalizer,
            )
        elif arg_type in {"bool", "boolean"}:
            normalized = _normalize_bool(value)
        elif arg_type in {"int", "integer"}:
            normalized = _normalize_int(value)
        elif arg_type in {"float", "number"}:
            normalized = _normalize_float(value)
        elif arg_type == "phone":
            normalized = _normalize_phone(value)
        elif arg_type == "timestamp":
            normalized = _normalize_timestamp(value)
        else:
            normalized = _normalize_string(
                value,
                case_sensitive=case_sensitive,
                normalizer=normalizer,
            )
    except (TypeError, ValueError) as exc:
        return NormalizedValue(value=None, error=str(exc))

    return NormalizedValue(value=normalized, error=None)


def normalize_argument_dict(
    arguments: dict[str, Any],
    arg_specs_by_name: dict[str, ArgSpec | dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    normalized: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, value in arguments.items():
        arg_spec = arg_specs_by_name.get(name)
        if arg_spec is None:
            errors[name] = "unknown_argument"
            continue
        result = normalize_argument_value(value, arg_spec)
        if result.error is not None:
            errors[name] = result.error
            continue
        normalized[name] = result.value
    return normalized, errors
