from __future__ import annotations

import json
import tomllib
from copy import deepcopy
from hashlib import sha1
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only when PyYAML is absent
    yaml = None

SUPPORTED_VARIANTS = {"V24", "V48"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_single_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs.")
        data = yaml.safe_load(text)
    elif suffix == ".toml":
        data = tomllib.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path}")

    if not isinstance(data, dict):
        raise TypeError(f"Config must deserialize to a mapping: {path}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = _load_single_config(config_path)
    extends = raw.pop("extends", None)

    if not extends:
        return raw

    parent_paths = [extends] if isinstance(extends, str) else list(extends)
    merged: dict[str, Any] = {}
    for parent in parent_paths:
        parent_path = (config_path.parent / parent).resolve()
        merged = _deep_merge(merged, load_config(parent_path))
    return _deep_merge(merged, raw)


def dump_config_snapshot(config: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha1(payload).hexdigest()[:12]


def ensure_execution_variant(config: dict[str, Any], default: str = "V24") -> str:
    variant = str(config.get("execution_variant", default))
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported execution_variant {variant!r}; "
            f"expected one of {sorted(SUPPORTED_VARIANTS)}"
        )
    config["execution_variant"] = variant
    return variant
