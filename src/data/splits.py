from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path

from src.data.canonical import CanonicalExample

DEFAULT_VAL_ROWS = 512
DEFAULT_CALIB_ROWS = 1024
FALLBACK_THRESHOLD = 4096


@dataclass(frozen=True)
class SplitManifest:
    counts: dict[str, int]
    example_ids_by_split: dict[str, list[str]]
    manifest_hash: str


def _rank_key(example: CanonicalExample) -> str:
    return sha1(example.example_id.encode("utf-8")).hexdigest()


def _fallback_counts(train_pool_size: int) -> tuple[int, int]:
    if train_pool_size <= 0:
        return 0, 0

    val_rows = max(1, int(train_pool_size * 0.10))
    remaining = max(0, train_pool_size - val_rows)
    calib_rows = max(1, int(train_pool_size * 0.15))
    calib_rows = min(calib_rows, remaining)
    return val_rows, calib_rows


def assign_locked_splits(
    examples: list[CanonicalExample],
) -> tuple[list[CanonicalExample], SplitManifest]:
    eval_examples = [example for example in examples if example.meta.get("raw_split") == "eval"]
    train_pool = [example for example in examples if example.meta.get("raw_split") == "train"]
    unsupported = sorted(
        {
            str(example.meta.get("raw_split"))
            for example in examples
            if example.meta.get("raw_split") not in {"train", "eval"}
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported raw split labels: {unsupported}")

    ranked_train_pool = sorted(train_pool, key=_rank_key)
    if len(ranked_train_pool) < FALLBACK_THRESHOLD:
        val_rows, calib_rows = _fallback_counts(len(ranked_train_pool))
    else:
        val_rows, calib_rows = DEFAULT_VAL_ROWS, DEFAULT_CALIB_ROWS

    val_examples = ranked_train_pool[:val_rows]
    calib_examples = ranked_train_pool[val_rows : val_rows + calib_rows]
    train_examples = ranked_train_pool[val_rows + calib_rows :]

    assigned_examples = [
        *(replace(example, split="eval") for example in eval_examples),
        *(replace(example, split="val") for example in val_examples),
        *(replace(example, split="calib") for example in calib_examples),
        *(replace(example, split="train") for example in train_examples),
    ]

    example_ids_by_split = {
        "eval": [example.example_id for example in eval_examples],
        "val": [example.example_id for example in val_examples],
        "calib": [example.example_id for example in calib_examples],
        "train": [example.example_id for example in train_examples],
    }
    manifest_payload = json.dumps(example_ids_by_split, sort_keys=True, separators=(",", ":"))
    manifest_hash = sha1(manifest_payload.encode("utf-8")).hexdigest()
    manifest = SplitManifest(
        counts={name: len(ids) for name, ids in example_ids_by_split.items()},
        example_ids_by_split=example_ids_by_split,
        manifest_hash=manifest_hash,
    )
    return assigned_examples, manifest


def write_split_manifest(manifest: SplitManifest, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "counts": manifest.counts,
                "example_ids_by_split": manifest.example_ids_by_split,
                "manifest_hash": manifest.manifest_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination
