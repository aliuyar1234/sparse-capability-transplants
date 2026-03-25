from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from src.data.canonical import ArgSpec, CanonicalExample, ToolSpec


@dataclass(frozen=True)
class ManifestRecord:
    manifest_id: str
    manifest_hash: str
    example_count: int
    split_counts: dict[str, int]
    prompt_contract_version: str
    dataset_path: str
    manifest_path: str


def canonical_example_from_dict(payload: dict[str, Any]) -> CanonicalExample:
    return CanonicalExample(
        example_id=str(payload["example_id"]),
        split=str(payload["split"]),
        user_request=str(payload["user_request"]),
        tools=[
            ToolSpec(
                tool_id=str(tool["tool_id"]),
                name=str(tool["name"]),
                description=str(tool.get("description", "")),
                arguments=[
                    ArgSpec(
                        name=str(argument["name"]),
                        type=str(argument["type"]),
                        required=bool(argument["required"]),
                        description=str(argument.get("description", "")),
                    )
                    for argument in tool.get("arguments", [])
                ],
            )
            for tool in payload.get("tools", [])
        ],
        gold=dict(payload["gold"]),
        meta=dict(payload.get("meta", {})),
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stable_examples(examples: list[CanonicalExample]) -> list[dict[str, Any]]:
    return sorted((example.to_dict() for example in examples), key=lambda row: row["example_id"])


def build_manifest_hash(
    *,
    examples: list[CanonicalExample],
    prompt_contract_version: str,
    alias_banks: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    payload = {
        "prompt_contract_version": prompt_contract_version,
        "examples": _stable_examples(examples),
        "alias_banks": alias_banks or {},
        "metadata": metadata or {},
    }
    return sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def _split_counts(examples: list[CanonicalExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.split] = counts.get(example.split, 0) + 1
    return counts


def write_manifest(
    *,
    examples: list[CanonicalExample],
    output_dir: str | Path,
    manifest_id: str,
    prompt_contract_version: str,
    alias_banks: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ManifestRecord:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    stable_examples = _stable_examples(examples)
    manifest_hash = build_manifest_hash(
        examples=examples,
        prompt_contract_version=prompt_contract_version,
        alias_banks=alias_banks,
        metadata=metadata,
    )

    dataset_path = destination / "examples.jsonl"
    dataset_lines = [_stable_json(example) for example in stable_examples]
    dataset_path.write_text("\n".join(dataset_lines) + "\n", encoding="utf-8")

    manifest_payload = {
        "manifest_id": manifest_id,
        "manifest_hash": manifest_hash,
        "example_count": len(examples),
        "split_counts": _split_counts(examples),
        "prompt_contract_version": prompt_contract_version,
        "alias_banks": alias_banks or {},
        "metadata": metadata or {},
        "dataset_path": str(dataset_path.resolve()),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return ManifestRecord(
        manifest_id=manifest_id,
        manifest_hash=manifest_hash,
        example_count=len(examples),
        split_counts=manifest_payload["split_counts"],
        prompt_contract_version=prompt_contract_version,
        dataset_path=str(dataset_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
    )


def load_examples(dataset_path: str | Path) -> list[CanonicalExample]:
    path = Path(dataset_path)
    examples: list[CanonicalExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        examples.append(canonical_example_from_dict(json.loads(stripped)))
    return examples


def load_manifest_payload(manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
