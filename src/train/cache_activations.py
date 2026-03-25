from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from src.data.canonical import CanonicalExample
from src.data.manifest import load_examples, load_manifest_payload
from src.models.format_prompts import (
    build_prompt_content,
    render_assistant_target,
    render_chat_prompt,
)
from src.models.hooks import HOOK_LIBRARY, candidate_layer_ids, capture_mlp_io
from src.models.load_gemma import (
    load_gemma_causal_lm,
    load_gemma_tokenizer,
    probe_gemma_loading,
    resolve_model_source,
)
from src.utils.progress import RunHeartbeat

TOKEN_CLASS_DECISION = "decision"
TOKEN_CLASS_TOOL = "tool"
TOKEN_CLASS_ARGKEY = "argkey"
TOKEN_CLASS_ARGVAL = "argval"
TOKEN_CLASS_OTHER = "other"
TASK_RELEVANT_TOKEN_CLASSES = (
    TOKEN_CLASS_DECISION,
    TOKEN_CLASS_TOOL,
    TOKEN_CLASS_ARGKEY,
    TOKEN_CLASS_ARGVAL,
)


@dataclass(frozen=True)
class CharSpan:
    start: int
    end: int
    token_class: str


@dataclass(frozen=True)
class CacheChunkRecord:
    layer_id: int
    chunk_index: int
    path: str
    row_count: int
    token_class_counts: dict[str, int]


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_target_char_spans(target: dict[str, Any]) -> list[CharSpan]:
    name_json = json.dumps(target["name"], ensure_ascii=False, separators=(",", ":"))
    sorted_arguments = dict(sorted(target.get("arguments", {}).items()))

    parts = ['{"name":']
    spans = [CharSpan(start=0, end=1, token_class=TOKEN_CLASS_DECISION)]
    cursor = len(parts[0])

    parts.append(name_json)
    spans.append(CharSpan(start=cursor, end=cursor + len(name_json), token_class=TOKEN_CLASS_TOOL))
    cursor += len(name_json)

    parts.append(',"arguments":{')
    cursor += len(parts[-1])

    for index, (key, value) in enumerate(sorted_arguments.items()):
        if index > 0:
            parts.append(",")
            cursor += 1
        key_json = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        parts.append(key_json)
        spans.append(
            CharSpan(start=cursor, end=cursor + len(key_json), token_class=TOKEN_CLASS_ARGKEY)
        )
        cursor += len(key_json)

        parts.append(":")
        cursor += 1

        value_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        parts.append(value_json)
        spans.append(
            CharSpan(start=cursor, end=cursor + len(value_json), token_class=TOKEN_CLASS_ARGVAL)
        )
        cursor += len(value_json)

    parts.append("}}")
    rendered = "".join(parts)
    if rendered != render_assistant_target(target):
        raise ValueError(
            "Gold target span builder drifted from the locked assistant target format."
        )
    return spans


def label_output_token_classes(
    *,
    offset_mapping: Iterable[tuple[int, int]],
    output_start_char: int,
    target: dict[str, Any],
) -> list[str | None]:
    spans = build_target_char_spans(target)
    labels: list[str | None] = []
    for start, end in offset_mapping:
        if end <= start or end <= output_start_char:
            labels.append(None)
            continue
        relative_start = max(start - output_start_char, 0)
        relative_end = end - output_start_char
        label = TOKEN_CLASS_OTHER
        for span in spans:
            if relative_start < span.end and relative_end > span.start:
                label = span.token_class
                break
        labels.append(label)
    return labels


def selected_token_positions(
    labels: Iterable[str | None],
    selected_token_classes: Iterable[str],
) -> list[int]:
    selected = set(selected_token_classes)
    return [index for index, label in enumerate(labels) if label in selected]


def _device_from_cache_config(config: dict[str, Any]) -> torch.device:
    cache_config = dict(config.get("cache", {}))
    device_name = str(cache_config.get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _offset_pairs(encoded: dict[str, Any]) -> list[tuple[int, int]]:
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "Tokenizer output did not include offset_mapping; fast tokenizer support is required."
        )
    return _offset_pairs_for_row(encoded, row_index=0)


def _offset_pairs_for_row(
    encoded: dict[str, Any],
    *,
    row_index: int,
) -> list[tuple[int, int]]:
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError(
            "Tokenizer output did not include offset_mapping; fast tokenizer support is required."
        )
    offset_row = offsets[row_index]
    offset_values = offset_row.tolist() if hasattr(offset_row, "tolist") else offset_row
    return [(int(start), int(end)) for start, end in offset_values]


def _resolve_cache_examples(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[CanonicalExample]]:
    cache_config = dict(config.get("cache", {}))
    manifest_path = cache_config.get("manifest_path")
    if not manifest_path:
        raise ValueError("Config is missing cache.manifest_path.")

    manifest_payload = load_manifest_payload(manifest_path)
    examples = load_examples(manifest_payload["dataset_path"])

    split_filter_raw = cache_config.get("split_filter")
    if split_filter_raw is not None:
        split_filter = (
            {str(split_filter_raw)}
            if isinstance(split_filter_raw, str)
            else {str(split) for split in split_filter_raw}
        )
        examples = [example for example in examples if example.split in split_filter]

    max_examples = cache_config.get("max_examples")
    if max_examples is not None:
        examples = examples[: int(max_examples)]

    if not examples:
        raise ValueError("Cache example set is empty after applying split_filter/max_examples.")
    return manifest_payload, examples


def _resolve_layer_ids(model: Any, cache_config: dict[str, Any]) -> list[int]:
    if "layer_ids" in cache_config:
        layer_ids = [int(layer_id) for layer_id in cache_config["layer_ids"]]
        if not layer_ids:
            raise ValueError("cache.layer_ids must not be empty when provided.")
        return layer_ids
    fractions = cache_config.get("candidate_layer_fractions", (0.25, 0.50, 0.65, 0.85))
    return candidate_layer_ids(model, [float(value) for value in fractions])


class CacheChunkWriter:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        layer_ids: list[int],
        chunk_size: int,
        cache_version: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, int(chunk_size))
        self.cache_version = cache_version
        self.buffers = {
            layer_id: {"x_b": [], "u_b": [], "u_d": [], "metadata": []} for layer_id in layer_ids
        }
        self.chunk_indices = {layer_id: 0 for layer_id in layer_ids}
        self.chunk_records: list[CacheChunkRecord] = []
        self.per_layer_row_counts = Counter()

    def append(
        self,
        *,
        layer_id: int,
        x_b: torch.Tensor,
        u_b: torch.Tensor,
        u_d: torch.Tensor,
        metadata_rows: list[dict[str, Any]],
    ) -> None:
        row_count = len(metadata_rows)
        if row_count == 0:
            return
        if x_b.shape[0] != row_count or u_b.shape[0] != row_count or u_d.shape[0] != row_count:
            raise ValueError("Cached tensor row counts must match metadata row counts.")

        buffer = self.buffers[layer_id]
        buffer["x_b"].append(x_b.cpu())
        buffer["u_b"].append(u_b.cpu())
        buffer["u_d"].append(u_d.cpu())
        buffer["metadata"].extend(metadata_rows)
        self.per_layer_row_counts[layer_id] += row_count

        if len(buffer["metadata"]) >= self.chunk_size:
            self.flush_layer(layer_id)

    def flush_layer(self, layer_id: int) -> None:
        buffer = self.buffers[layer_id]
        if not buffer["metadata"]:
            return

        chunk_index = self.chunk_indices[layer_id]
        chunk_path = self.output_dir / f"layer_{layer_id:02d}_chunk_{chunk_index:04d}.pt"
        token_class_counts = Counter(
            str(row["token_class"]) for row in buffer["metadata"] if row["token_class"] is not None
        )
        payload = {
            "layer_id": layer_id,
            "cache_version": self.cache_version,
            "row_count": len(buffer["metadata"]),
            "metadata": list(buffer["metadata"]),
            "token_class_counts": dict(sorted(token_class_counts.items())),
            "x_b": torch.cat(buffer["x_b"], dim=0),
            "u_b": torch.cat(buffer["u_b"], dim=0),
            "u_d": torch.cat(buffer["u_d"], dim=0),
        }
        torch.save(payload, chunk_path)
        self.chunk_records.append(
            CacheChunkRecord(
                layer_id=layer_id,
                chunk_index=chunk_index,
                path=str(chunk_path.resolve()),
                row_count=payload["row_count"],
                token_class_counts=dict(sorted(token_class_counts.items())),
            )
        )
        self.chunk_indices[layer_id] += 1
        self.buffers[layer_id] = {"x_b": [], "u_b": [], "u_d": [], "metadata": []}

    def finalize(self) -> tuple[list[CacheChunkRecord], dict[int, int]]:
        for layer_id in sorted(self.buffers):
            self.flush_layer(layer_id)
        return self.chunk_records, dict(sorted(self.per_layer_row_counts.items()))


def collect_activation_caches(config: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    cache_config = dict(config.get("cache", {}))
    selected_token_classes = tuple(
        str(value)
        for value in cache_config.get("selected_token_classes", TASK_RELEVANT_TOKEN_CLASSES)
    )
    batch_size = max(1, int(cache_config.get("batch_size", 1)))
    manifest_payload, examples = _resolve_cache_examples(config)

    base_model_config = dict(config.get("model", {}))
    donor_model_config = dict(config.get("donor_model", {}))
    if not donor_model_config:
        raise ValueError("Config is missing donor_model for cache collection.")

    base_report = probe_gemma_loading(base_model_config, require_chat_template=True)
    donor_report = probe_gemma_loading(donor_model_config, require_chat_template=False)
    if base_report.status != "passed":
        raise RuntimeError(
            f"Base model could not be prepared for cache collection: {base_report.message}"
        )
    if donor_report.status != "passed":
        raise RuntimeError(
            f"Donor model could not be prepared for cache collection: {donor_report.message}"
        )

    tokenizer = load_gemma_tokenizer(base_model_config)
    if hasattr(tokenizer, "is_fast") and not bool(tokenizer.is_fast):
        raise RuntimeError("Cache collection requires a fast tokenizer with offset mappings.")

    device = _device_from_cache_config(config)
    base_model = load_gemma_causal_lm(base_model_config).to(device)
    donor_model = load_gemma_causal_lm(donor_model_config).to(device)
    base_model.eval()
    donor_model.eval()

    layer_ids = _resolve_layer_ids(base_model, cache_config)
    cache_version = str(cache_config.get("cache_version", "untouched_base_v1"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    writer = CacheChunkWriter(
        output_dir=destination / "chunks",
        layer_ids=layer_ids,
        chunk_size=int(cache_config.get("chunk_size", 4096)),
        cache_version=cache_version,
    )
    heartbeat = RunHeartbeat(
        output_dir=destination,
        phase="cache_activations",
        total_units=len(examples),
        unit_name="examples",
        heartbeat_interval_seconds=float(cache_config.get("heartbeat_interval_seconds", 10.0)),
    )
    heartbeat.start(
        message="Activation cache collection started.",
        extra={"hook_library": HOOK_LIBRARY, "layer_ids": layer_ids, "batch_size": batch_size},
    )

    hook_shapes: dict[str, Any] | None = None
    token_class_counts = Counter()
    cached_example_count = 0

    for batch_start in range(0, len(examples), batch_size):
        batch_examples = examples[batch_start : batch_start + batch_size]
        full_texts: list[str] = []
        output_start_chars: list[int] = []
        prepared_examples: list[dict[str, Any]] = []

        for example in batch_examples:
            prompt = build_prompt_content(
                user_request=example.user_request,
                tools=example.tools,
                target=example.gold,
            )
            prompt_text = render_chat_prompt(
                prompt=prompt,
                tokenizer=tokenizer,
                add_generation_prompt=True,
            )
            target_text = render_assistant_target(example.gold)
            full_texts.append(prompt_text + target_text)
            output_start_chars.append(len(prompt_text))
            prepared_examples.append({"example": example})

        encoded = tokenizer(
            full_texts,
            add_special_tokens=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        for batch_index, prepared in enumerate(prepared_examples):
            example = prepared["example"]
            offset_mapping = _offset_pairs_for_row(encoded, row_index=batch_index)
            labels = label_output_token_classes(
                offset_mapping=offset_mapping,
                output_start_char=output_start_chars[batch_index],
                target=example.gold,
            )
            prepared["labels"] = labels
            prepared["positions"] = selected_token_positions(labels, selected_token_classes)

        base_captures: dict[int, Any] | None = None
        donor_captures: dict[int, Any] | None = None
        if any(prepared["positions"] for prepared in prepared_examples):
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.inference_mode():
                with capture_mlp_io(base_model, layer_ids) as batch_base_captures:
                    base_model(input_ids=input_ids, attention_mask=attention_mask)
                with capture_mlp_io(donor_model, layer_ids) as batch_donor_captures:
                    donor_model(input_ids=input_ids, attention_mask=attention_mask)
            base_captures = batch_base_captures
            donor_captures = batch_donor_captures

        for batch_index, prepared in enumerate(prepared_examples):
            example = prepared["example"]
            labels = prepared["labels"]
            positions = prepared["positions"]
            if positions and (base_captures is None or donor_captures is None):
                raise RuntimeError(
                    "Cache captures were not produced for a batch with selected rows."
                )

            if positions:
                for layer_id in layer_ids:
                    assert base_captures is not None
                    assert donor_captures is not None
                    base_capture = base_captures[layer_id]
                    donor_capture = donor_captures[layer_id]
                    if base_capture.mlp_input is None or base_capture.mlp_output is None:
                        raise RuntimeError(
                            f"Base hook capture for layer {layer_id} was incomplete."
                        )
                    if donor_capture.mlp_output is None:
                        raise RuntimeError(
                            f"Donor hook capture for layer {layer_id} was incomplete."
                        )

                    metadata_rows = []
                    for position in positions:
                        token_class = labels[position]
                        if token_class is None:
                            continue
                        metadata_rows.append(
                            {
                                "example_id": example.example_id,
                                "token_index": int(position),
                                "token_class": token_class,
                                "layer_id": layer_id,
                                "split": example.split,
                                "variant": str(example.meta.get("variant", "unknown")),
                                "cache_version": cache_version,
                            }
                        )
                        token_class_counts[token_class] += 1

                    writer.append(
                        layer_id=layer_id,
                        x_b=base_capture.mlp_input[batch_index, positions, :],
                        u_b=base_capture.mlp_output[batch_index, positions, :],
                        u_d=donor_capture.mlp_output[batch_index, positions, :],
                        metadata_rows=metadata_rows,
                    )

                    if hook_shapes is None:
                        hook_shapes = {
                            str(layer_id): {
                                "base_input_shape": list(base_capture.mlp_input.shape),
                                "base_output_shape": list(base_capture.mlp_output.shape),
                                "donor_output_shape": list(donor_capture.mlp_output.shape),
                            }
                            for layer_id in layer_ids
                        }

                cached_example_count += 1

        completed_units = batch_start + len(batch_examples)
        heartbeat.maybe_update(
            completed_units=completed_units,
            message=f"Processed cache example {completed_units} of {len(examples)}.",
            metrics={"cached_examples_so_far": cached_example_count},
            extra={"selected_rows_so_far": int(sum(token_class_counts.values()))},
        )

    chunk_records, per_layer_row_counts = writer.finalize()
    hook_audit_payload = {
        "hook_library": HOOK_LIBRARY,
        "layer_ids": layer_ids,
        "batch_size": batch_size,
        "cache_version": cache_version,
        "tokenizer_class": type(tokenizer).__name__,
        "base_model_type": type(base_model).__name__,
        "donor_model_type": type(donor_model).__name__,
        "shapes_by_layer": hook_shapes or {},
    }
    hook_audit_path = destination / "hook_audit.json"
    hook_audit_path.write_text(
        json.dumps(hook_audit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    chunk_payload = [asdict(record) for record in chunk_records]
    summary_payload = {
        "status": "passed",
        "source_manifest_id": manifest_payload["manifest_id"],
        "source_manifest_hash": manifest_payload["manifest_hash"],
        "cache_version": cache_version,
        "hook_library": HOOK_LIBRARY,
        "layer_ids": layer_ids,
        "batch_size": batch_size,
        "selected_token_classes": list(selected_token_classes),
        "source_base": resolve_model_source(base_model_config),
        "donor_source": resolve_model_source(donor_model_config),
        "cached_example_count": cached_example_count,
        "requested_example_count": len(examples),
        "per_layer_row_counts": per_layer_row_counts,
        "token_class_counts": dict(sorted(token_class_counts.items())),
        "chunk_count": len(chunk_records),
        "chunk_records": chunk_payload,
        "hook_audit_path": str(hook_audit_path.resolve()),
        "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
        "progress_path": str(heartbeat.paths.progress_path.resolve()),
        "base_loader_report": base_report.to_dict(),
        "donor_loader_report": donor_report.to_dict(),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = destination / "cache_manifest.json"
    manifest_payload_out = {
        "cache_version": cache_version,
        "hook_library": HOOK_LIBRARY,
        "source_manifest_id": manifest_payload["manifest_id"],
        "source_manifest_hash": manifest_payload["manifest_hash"],
        "layer_ids": layer_ids,
        "batch_size": batch_size,
        "selected_token_classes": list(selected_token_classes),
        "per_layer_row_counts": per_layer_row_counts,
        "chunk_records": chunk_payload,
        "summary_path": str(summary_path.resolve()),
        "hook_audit_path": str(hook_audit_path.resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload_out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    heartbeat.mark_completed(
        completed_units=len(examples),
        message="Activation cache collection completed successfully.",
        metrics={"cached_examples": cached_example_count},
        extra={"cache_manifest_path": str(manifest_path.resolve())},
    )
    summary_payload["summary_path"] = str(summary_path.resolve())
    summary_payload["cache_manifest_path"] = str(manifest_path.resolve())
    return summary_payload
