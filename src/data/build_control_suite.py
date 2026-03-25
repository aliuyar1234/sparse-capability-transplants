from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

from src.data.canonical import CanonicalExample


@dataclass(frozen=True)
class ControlExample:
    example_id: str
    split: str
    prompt: str
    target_text: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlSuiteRecord:
    manifest_id: str
    manifest_hash: str
    example_count: int
    dataset_path: str
    manifest_path: str


def build_control_example(
    *,
    source: str,
    prompt: str,
    target_text: str,
    source_row_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> ControlExample:
    payload = {
        "source": source,
        "source_row_id": source_row_id,
        "prompt": prompt,
        "target_text": target_text,
        "variant": "control",
    }
    digest = sha1(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    merged_meta = {
        "source": source,
        "variant": "control",
        "raw_split": "control",
    }
    if meta:
        merged_meta.update(meta)
    return ControlExample(
        example_id=f"ctrl_{digest}",
        split="control",
        prompt=prompt,
        target_text=target_text,
        meta=merged_meta,
    )


def _normalize_words(text: str) -> list[str]:
    return [token for token in re.split(r"\s+", text.strip()) if token]


def build_control_examples_from_canonical_examples(
    examples: Iterable[CanonicalExample],
) -> list[ControlExample]:
    control_examples: list[ControlExample] = []
    for example in examples:
        selector = (
            int(sha1(f"{example.example_id}::control".encode("utf-8")).hexdigest()[:8], 16) % 3
        )
        if selector == 0:
            prompt = (
                "Rewrite the following text in lowercase and return only the rewritten text.\n"
                f"Text: {example.user_request}"
            )
            target_text = example.user_request.lower()
            task_type = "rewrite_lower"
        elif selector == 1:
            prompt = (
                "Return the first five whitespace-separated words from the following text, "
                "joined by single spaces.\n"
                f"Text: {example.user_request}"
            )
            target_text = " ".join(_normalize_words(example.user_request)[:5])
            task_type = "extract_first_five_words"
        else:
            prompt = (
                "Count the words in the following text and return only the integer.\n"
                f"Text: {example.user_request}"
            )
            target_text = str(len(_normalize_words(example.user_request)))
            task_type = "word_count"

        control_examples.append(
            build_control_example(
                source=str(example.meta.get("source", "synthetic")),
                prompt=prompt,
                target_text=target_text,
                source_row_id=example.example_id,
                meta={
                    "task_type": task_type,
                    "source_example_id": example.example_id,
                    "source_split": example.split,
                },
            )
        )
    return control_examples


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_control_suite_hash(examples: list[ControlExample]) -> str:
    payload = [example.to_dict() for example in sorted(examples, key=lambda item: item.example_id)]
    return sha1(_stable_json(payload).encode("utf-8")).hexdigest()


def write_control_suite(
    *,
    examples: list[ControlExample],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True)
        for example in sorted(examples, key=lambda item: item.example_id)
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def write_control_suite_manifest(
    *,
    examples: list[ControlExample],
    dataset_path: str | Path,
    output_path: str | Path,
    manifest_id: str,
) -> ControlSuiteRecord:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = Path(dataset_path).resolve()
    manifest_hash = build_control_suite_hash(examples)
    payload = {
        "manifest_id": manifest_id,
        "manifest_hash": manifest_hash,
        "example_count": len(examples),
        "dataset_path": str(dataset),
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ControlSuiteRecord(
        manifest_id=manifest_id,
        manifest_hash=manifest_hash,
        example_count=len(examples),
        dataset_path=str(dataset),
        manifest_path=str(destination.resolve()),
    )
