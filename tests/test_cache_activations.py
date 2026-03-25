from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.manifest import write_manifest
from src.train.cache_activations import collect_activation_caches


def _write_cache_fixture_manifest(output_dir: Path) -> Path:
    tool = ToolSpec(
        tool_id="send_email",
        name="send_email",
        description="Sends an email.",
        arguments=[
            ArgSpec(name="to", type="string", required=True, description="Recipient"),
            ArgSpec(name="subject", type="string", required=True, description="Subject"),
        ],
    )
    examples = [
        build_canonical_example(
            source="synthetic",
            raw_split="train",
            user_request="Email Sam about the report.",
            tools=[tool],
            gold={
                "name": "send_email",
                "arguments": {"to": "sam@example.com", "subject": "report"},
            },
            source_row_id="row-1",
            meta={"variant": "schema_shift"},
        ),
        build_canonical_example(
            source="synthetic",
            raw_split="train",
            user_request="Send Dana the lunch plan.",
            tools=[tool],
            gold={
                "name": "send_email",
                "arguments": {"to": "dana@example.com", "subject": "lunch plan"},
            },
            source_row_id="row-2",
            meta={"variant": "nocall"},
        ),
        build_canonical_example(
            source="synthetic",
            raw_split="train",
            user_request="Let Alex know the budget is approved today.",
            tools=[tool],
            gold={
                "name": "send_email",
                "arguments": {"to": "alex@example.com", "subject": "budget approved"},
            },
            source_row_id="row-3",
            meta={"variant": "iid"},
        ),
    ]
    record = write_manifest(
        examples=[replace(example, split="train") for example in examples],
        output_dir=output_dir / "manifest",
        manifest_id="manifest_cache_fixture_v1",
        prompt_contract_version="fc_v1",
        metadata={"kind": "cache_fixture"},
    )
    return Path(record.manifest_path)


class DummyProbeReport:
    status = "passed"
    message = "ok"

    def to_dict(self) -> dict[str, str]:
        return {"status": "passed"}


class DummyCacheTokenizer:
    chat_template = "{{ bos_token }}"
    pad_token_id = 0
    eos_token_id = 1
    is_fast = True

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        parts = [f"{message['role']}:{message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        return_offsets_mapping: bool = False,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else list(text)
        token_rows: list[list[int]] = []
        offset_rows: list[list[tuple[int, int]]] = []
        max_length = 0
        for value in texts:
            token_ids = [(ord(character) % 251) + 2 for character in value]
            token_rows.append(token_ids)
            offset_rows.append([(index, index + 1) for index, _ in enumerate(value)])
            max_length = max(max_length, len(token_ids))

        if not padding:
            max_length = len(token_rows[0])

        padded_ids: list[list[int]] = []
        padded_masks: list[list[int]] = []
        padded_offsets: list[list[tuple[int, int]]] = []
        for token_ids, offsets in zip(token_rows, offset_rows, strict=True):
            pad_count = max_length - len(token_ids)
            padded_ids.append(token_ids + ([self.pad_token_id] * pad_count))
            padded_masks.append(([1] * len(token_ids)) + ([0] * pad_count))
            padded_offsets.append(offsets + ([(0, 0)] * pad_count))

        payload: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = torch.tensor(padded_offsets, dtype=torch.long)
        if return_tensors != "pt":
            raise ValueError("DummyCacheTokenizer only supports return_tensors='pt'.")
        return payload


class DummyCacheLayer(nn.Module):
    def __init__(self, width: int, *, scale: float) -> None:
        super().__init__()
        self.mlp = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.mlp.weight.copy_(torch.eye(width, dtype=torch.float32) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class DummyCacheModel(nn.Module):
    def __init__(self, *, width: int, depth: int, scale: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [DummyCacheLayer(width, scale=scale + (0.1 * layer_id)) for layer_id in range(depth)]
        )
        with torch.no_grad():
            embedding_weights = torch.arange(256 * width, dtype=torch.float32).reshape(256, width)
            self.embedding.weight.copy_(embedding_weights / 1000.0)

    def forward(self, input_ids, attention_mask=None):  # type: ignore[override]
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def _load_dummy_model(model_config: dict[str, object]) -> DummyCacheModel:
    source_id = str(model_config.get("id", ""))
    scale = 0.6 if "donor" in source_id else 0.3
    return DummyCacheModel(width=4, depth=2, scale=scale)


def _chunk_payloads(summary: dict[str, object]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for record in summary["chunk_records"]:
        chunk = torch.load(record["path"], map_location="cpu")
        payloads.append(
            {
                "layer_id": int(chunk["layer_id"]),
                "cache_version": str(chunk["cache_version"]),
                "row_count": int(chunk["row_count"]),
                "metadata": chunk["metadata"],
                "token_class_counts": chunk["token_class_counts"],
                "x_b": chunk["x_b"],
                "u_b": chunk["u_b"],
                "u_d": chunk["u_d"],
            }
        )
    return payloads


class CacheActivationBatchingTests(unittest.TestCase):
    def test_batched_cache_collection_matches_single_example_cache_rows(self) -> None:
        output_root = Path("tests/_tmp/cache_activations_equivalence")
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_cache_fixture_manifest(output_root)

        base_config = {
            "model": {"loader": "transformers", "id": "dummy-base"},
            "donor_model": {"loader": "transformers", "id": "dummy-donor"},
            "cache": {
                "manifest_path": str(manifest_path),
                "split_filter": ["train"],
                "layer_ids": [0, 1],
                "chunk_size": 5,
                "cache_version": "batch_equivalence_v1",
                "heartbeat_interval_seconds": 0.0,
                "device": "cpu",
            },
        }
        single_config = {
            **base_config,
            "cache": {**base_config["cache"], "batch_size": 1},
        }
        batched_config = {
            **base_config,
            "cache": {**base_config["cache"], "batch_size": 2},
        }

        with patch(
            "src.train.cache_activations.probe_gemma_loading", return_value=DummyProbeReport()
        ):
            with patch(
                "src.train.cache_activations.load_gemma_tokenizer",
                return_value=DummyCacheTokenizer(),
            ):
                with patch(
                    "src.train.cache_activations.load_gemma_causal_lm",
                    side_effect=_load_dummy_model,
                ):
                    single_summary = collect_activation_caches(
                        config=single_config,
                        output_dir=output_root / "single",
                    )

        with patch(
            "src.train.cache_activations.probe_gemma_loading", return_value=DummyProbeReport()
        ):
            with patch(
                "src.train.cache_activations.load_gemma_tokenizer",
                return_value=DummyCacheTokenizer(),
            ):
                with patch(
                    "src.train.cache_activations.load_gemma_causal_lm",
                    side_effect=_load_dummy_model,
                ):
                    batched_summary = collect_activation_caches(
                        config=batched_config,
                        output_dir=output_root / "batched",
                    )

        self.assertEqual(single_summary["status"], "passed")
        self.assertEqual(batched_summary["status"], "passed")
        self.assertEqual(
            single_summary["requested_example_count"], batched_summary["requested_example_count"]
        )
        self.assertEqual(
            single_summary["cached_example_count"], batched_summary["cached_example_count"]
        )
        self.assertEqual(
            single_summary["per_layer_row_counts"], batched_summary["per_layer_row_counts"]
        )
        self.assertEqual(
            single_summary["token_class_counts"], batched_summary["token_class_counts"]
        )
        self.assertEqual(
            single_summary["selected_token_classes"], batched_summary["selected_token_classes"]
        )
        self.assertEqual(single_summary["layer_ids"], batched_summary["layer_ids"])
        self.assertEqual(single_summary["chunk_count"], batched_summary["chunk_count"])
        self.assertEqual(single_summary["batch_size"], 1)
        self.assertEqual(batched_summary["batch_size"], 2)

        single_chunks = _chunk_payloads(single_summary)
        batched_chunks = _chunk_payloads(batched_summary)
        self.assertEqual(len(single_chunks), len(batched_chunks))
        for single_chunk, batched_chunk in zip(single_chunks, batched_chunks, strict=True):
            self.assertEqual(single_chunk["layer_id"], batched_chunk["layer_id"])
            self.assertEqual(single_chunk["cache_version"], batched_chunk["cache_version"])
            self.assertEqual(single_chunk["row_count"], batched_chunk["row_count"])
            self.assertEqual(single_chunk["metadata"], batched_chunk["metadata"])
            self.assertEqual(
                single_chunk["token_class_counts"], batched_chunk["token_class_counts"]
            )
            self.assertTrue(torch.equal(single_chunk["x_b"], batched_chunk["x_b"]))
            self.assertTrue(torch.equal(single_chunk["u_b"], batched_chunk["u_b"]))
            self.assertTrue(torch.equal(single_chunk["u_d"], batched_chunk["u_d"]))

        single_manifest = json.loads(
            Path(single_summary["cache_manifest_path"]).read_text(encoding="utf-8")
        )
        batched_manifest = json.loads(
            Path(batched_summary["cache_manifest_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(single_manifest["layer_ids"], batched_manifest["layer_ids"])
        self.assertEqual(
            single_manifest["per_layer_row_counts"], batched_manifest["per_layer_row_counts"]
        )
        self.assertEqual(
            single_manifest["selected_token_classes"], batched_manifest["selected_token_classes"]
        )
        self.assertEqual(single_manifest["batch_size"], 1)
        self.assertEqual(batched_manifest["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
