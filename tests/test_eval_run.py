from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.manifest import write_manifest
from src.eval.run_eval import run_eval_pipeline
from src.models.format_prompts import render_assistant_target
from src.train.train_delta_module import SparseDeltaModule


def _write_fixture_manifest(output_dir: Path) -> Path:
    tool = ToolSpec(
        tool_id="show_map",
        name="show_map",
        description="Shows a location on the map.",
        arguments=[ArgSpec(name="query", type="string", required=True, description="Place")],
    )
    example = build_canonical_example(
        source="synthetic",
        raw_split="eval",
        user_request="Show me the cafe on Market Street.",
        tools=[tool],
        gold={"name": "show_map", "arguments": {"query": "cafe on Market Street"}},
        source_row_id="row-1",
        meta={
            "source_row_id": "row-1",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    record = write_manifest(
        examples=[replace(example, split="eval")],
        output_dir=output_dir / "manifest",
        manifest_id="manifest_eval_fixture_v1",
        prompt_contract_version="fc_v1",
        metadata={"kind": "eval_fixture"},
    )
    return Path(record.manifest_path)


def _write_split_fixture_manifest(output_dir: Path) -> Path:
    tool = ToolSpec(
        tool_id="show_map",
        name="show_map",
        description="Shows a location on the map.",
        arguments=[ArgSpec(name="query", type="string", required=True, description="Place")],
    )
    train_example = build_canonical_example(
        source="synthetic",
        raw_split="train",
        user_request="Show me the bakery on Pine Street.",
        tools=[tool],
        gold={"name": "show_map", "arguments": {"query": "bakery on Pine Street"}},
        source_row_id="row-train",
        meta={
            "source_row_id": "row-train",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    val_example = build_canonical_example(
        source="synthetic",
        raw_split="train",
        user_request="Show me the library on Oak Street.",
        tools=[tool],
        gold={"name": "show_map", "arguments": {"query": "library on Oak Street"}},
        source_row_id="row-val",
        meta={
            "source_row_id": "row-val",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    record = write_manifest(
        examples=[replace(train_example, split="train"), replace(val_example, split="val")],
        output_dir=output_dir / "split_manifest",
        manifest_id="manifest_eval_split_fixture_v1",
        prompt_contract_version="fc_v1",
        metadata={"kind": "eval_split_fixture"},
    )
    return Path(record.manifest_path)


def _write_two_eval_fixture_manifest(output_dir: Path) -> Path:
    tool = ToolSpec(
        tool_id="show_map",
        name="show_map",
        description="Shows a location on the map.",
        arguments=[ArgSpec(name="query", type="string", required=True, description="Place")],
    )
    first = build_canonical_example(
        source="synthetic",
        raw_split="eval",
        user_request="Show me the cafe on Market Street.",
        tools=[tool],
        gold={"name": "show_map", "arguments": {"query": "cafe on Market Street"}},
        source_row_id="row-1",
        meta={
            "source_row_id": "row-1",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    second = build_canonical_example(
        source="synthetic",
        raw_split="eval",
        user_request="Show me the museum on Elm Street.",
        tools=[tool],
        gold={"name": "show_map", "arguments": {"query": "museum on Elm Street"}},
        source_row_id="row-2",
        meta={
            "source_row_id": "row-2",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    record = write_manifest(
        examples=[replace(first, split="eval"), replace(second, split="eval")],
        output_dir=output_dir / "two_eval_manifest",
        manifest_id="manifest_eval_two_fixture_v1",
        prompt_contract_version="fc_v1",
        metadata={"kind": "eval_two_fixture"},
    )
    return Path(record.manifest_path)


def _write_variant_fixture_manifest(output_dir: Path) -> Path:
    tool = ToolSpec(
        tool_id="show_map",
        name="show_map",
        description="Shows a location on the map.",
        arguments=[ArgSpec(name="query", type="string", required=True, description="Place")],
    )
    schema_example = build_canonical_example(
        source="synthetic",
        raw_split="eval",
        user_request="Show me the cafe on Market Street.",
        tools=[tool],
        gold={"name": "show_map", "arguments": {"query": "cafe on Market Street"}},
        variant="schema_shift",
        source_row_id="row-schema",
        meta={
            "source_row_id": "row-schema",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    iid_example = build_canonical_example(
        source="synthetic",
        raw_split="eval",
        user_request="Show me the library on Oak Street.",
        tools=[tool],
        gold={"name": "show_map", "arguments": {"query": "library on Oak Street"}},
        variant="canonical",
        source_row_id="row-iid",
        meta={
            "source_row_id": "row-iid",
            "canonical_tool_id": "show_map",
            "canonical_argument_map": {"query": "query"},
            "alias_bank_id": "none",
        },
    )
    record = write_manifest(
        examples=[replace(schema_example, split="eval"), replace(iid_example, split="eval")],
        output_dir=output_dir / "variant_manifest",
        manifest_id="manifest_eval_variant_fixture_v1",
        prompt_contract_version="fc_v1",
        metadata={"kind": "eval_variant_fixture"},
    )
    return Path(record.manifest_path)


class EvalRunTests(unittest.TestCase):
    def test_eval_pipeline_scores_oracle_predictions(self) -> None:
        output_dir = Path("tests/_tmp/eval_run")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_fixture_manifest(output_dir)

        class DummyTokenizer:
            chat_template = "{{ bos_token }}"

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                return "serialized"

        with patch("src.eval.run_eval.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch("src.eval.run_eval.load_gemma_tokenizer", return_value=DummyTokenizer()):
                artifacts = run_eval_pipeline(
                    config={
                        "model": {"loader": "transformers", "id": "dummy"},
                        "eval": {
                            "manifest_path": str(manifest_path),
                            "prediction_backend": "oracle",
                            "serialize_with_chat_template": True,
                        },
                    },
                    output_dir=output_dir / "artifacts",
                )

        metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        self.assertEqual(metrics["aggregate"]["strict_full_call_success"], 1.0)
        self.assertEqual(summary["serialized_prompt_preview"], "serialized")

    def test_eval_pipeline_scores_model_greedy_predictions(self) -> None:
        output_dir = Path("tests/_tmp/eval_run_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_fixture_manifest(output_dir)
        target_output = render_assistant_target(
            {"name": "show_map", "arguments": {"query": "cafe on Market Street"}}
        )

        class DummyTokenizer:
            chat_template = "{{ bos_token }}"
            eos_token_id = 0
            pad_token_id = 0

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                return "serialized"

            def __call__(
                self,
                text: str,
                *,
                add_special_tokens: bool,
                return_tensors: str,
            ) -> dict[str, torch.Tensor]:
                ids = torch.tensor([[ord(char) for char in text]], dtype=torch.long)
                mask = torch.ones_like(ids)
                return {"input_ids": ids, "attention_mask": mask}

            def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
                values = token_ids.tolist()
                if values and isinstance(values[0], list):
                    values = values[0]
                filtered = [
                    value
                    for value in values
                    if not skip_special_tokens or value != self.eos_token_id
                ]
                return "".join(chr(value) for value in filtered)

        class DummyModel:
            def __init__(self, continuation: str, prompt_length: int) -> None:
                self._continuation_ids = [ord(char) for char in continuation]
                self._prompt_length = prompt_length
                self._vocab_size = 128

            def to(self, device: torch.device | str) -> "DummyModel":
                return self

            def eval(self) -> "DummyModel":
                return self

            def __call__(
                self,
                *,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> SimpleNamespace:
                del attention_mask
                next_index = input_ids.shape[1] - self._prompt_length
                next_token = (
                    self._continuation_ids[next_index]
                    if next_index < len(self._continuation_ids)
                    else 0
                )
                logits = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], self._vocab_size),
                    dtype=torch.float32,
                )
                logits[:, -1, next_token] = 1.0
                return SimpleNamespace(logits=logits)

        tokenizer = DummyTokenizer()
        prompt_length = len(
            tokenizer.apply_chat_template(
                [],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        model = DummyModel(target_output, prompt_length)

        with patch("src.eval.run_eval.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch("src.eval.run_eval.load_gemma_tokenizer", return_value=tokenizer):
                with patch("src.eval.run_eval.load_gemma_causal_lm", return_value=model):
                    artifacts = run_eval_pipeline(
                        config={
                            "model": {"loader": "transformers", "id": "dummy"},
                            "eval": {
                                "manifest_path": str(manifest_path),
                                "prediction_backend": "model_greedy",
                                "max_new_tokens": len(target_output) + 1,
                                "device": "cpu",
                            },
                        },
                        output_dir=output_dir / "artifacts",
                    )

        metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        predictions = [
            json.loads(line)
            for line in Path(artifacts.predictions_path)
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        ]
        self.assertEqual(metrics["aggregate"]["strict_full_call_success"], 1.0)
        self.assertEqual(summary["prediction_backend"], "model_greedy")
        self.assertEqual(summary["generation_device"], "cpu")
        self.assertEqual(predictions[0]["raw_output"], target_output)

    def test_eval_pipeline_scores_model_greedy_predictions_with_sparse_transplant(self) -> None:
        output_dir = Path("tests/_tmp/eval_run_model_transplant")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_fixture_manifest(output_dir)
        correct_output = render_assistant_target(
            {"name": "show_map", "arguments": {"query": "cafe on Market Street"}}
        )
        wrong_output = render_assistant_target({"name": "NO_TOOL", "arguments": {}})

        class DummyTokenizer:
            chat_template = "{{ bos_token }}"
            eos_token_id = 0
            pad_token_id = 0

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                del messages, tokenize, add_generation_prompt
                return "serialized"

            def __call__(
                self,
                text: str,
                *,
                add_special_tokens: bool,
                return_tensors: str,
            ) -> dict[str, torch.Tensor]:
                del add_special_tokens, return_tensors
                ids = torch.tensor([[ord(char) for char in text]], dtype=torch.long)
                mask = torch.ones_like(ids)
                return {"input_ids": ids, "attention_mask": mask}

            def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str:
                values = token_ids.tolist()
                if values and isinstance(values[0], list):
                    values = values[0]
                filtered = [
                    value
                    for value in values
                    if not skip_special_tokens or value != self.eos_token_id
                ]
                return "".join(chr(value) for value in filtered)

        class HookedLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.mlp.weight.fill_(1.0)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.mlp(x)

        class HookAwareDummyModel(nn.Module):
            def __init__(
                self,
                *,
                prompt_length: int,
                positive_continuation: str,
                negative_continuation: str,
            ) -> None:
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([HookedLayer()])
                self.prompt_length = prompt_length
                self.positive_ids = [ord(char) for char in positive_continuation]
                self.negative_ids = [ord(char) for char in negative_continuation]
                self.vocab_size = 128

            def to(self, device: torch.device | str) -> "HookAwareDummyModel":
                del device
                return self

            def eval(self) -> "HookAwareDummyModel":
                return self

            def forward(
                self,
                *,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
            ) -> SimpleNamespace:
                del attention_mask
                x = torch.full(
                    (input_ids.shape[0], input_ids.shape[1], 1), -1.0, dtype=torch.float32
                )
                for layer in self.model.layers:
                    x = layer(x)
                next_index = input_ids.shape[1] - self.prompt_length
                continuation = self.positive_ids if float(x[0, -1, 0]) > 0.0 else self.negative_ids
                next_token = continuation[next_index] if next_index < len(continuation) else 0
                logits = torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], self.vocab_size),
                    dtype=torch.float32,
                )
                logits[:, -1, next_token] = 1.0
                return SimpleNamespace(logits=logits)

            def generate(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                *,
                max_new_tokens: int,
                do_sample: bool,
                num_beams: int,
                use_cache: bool,
                return_dict_in_generate: bool,
                pad_token_id: int,
                eos_token_id: int,
            ) -> torch.Tensor:
                del do_sample, num_beams, use_cache, return_dict_in_generate, pad_token_id
                generated_ids = input_ids.clone()
                generated_mask = attention_mask.clone()
                for _ in range(max_new_tokens):
                    outputs = self.forward(input_ids=generated_ids, attention_mask=generated_mask)
                    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    generated_mask = torch.cat(
                        [generated_mask, torch.ones_like(next_token, dtype=generated_mask.dtype)],
                        dim=1,
                    )
                    if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                        break
                return generated_ids

        checkpoint_path = output_dir / "transplant_checkpoint.pt"
        module = SparseDeltaModule(input_dim=1, latent_width=1, topk=1)
        with torch.no_grad():
            module.encoder.weight.zero_()
            module.encoder.bias.fill_(1.0)
            module.decoder.weight.fill_(4.0)
        torch.save(
            {
                "layer_id": 0,
                "input_dim": 1,
                "latent_width": 1,
                "topk": 1,
                "state_dict": {key: value.cpu() for key, value in module.state_dict().items()},
            },
            checkpoint_path,
        )

        tokenizer = DummyTokenizer()
        prompt_length = len(
            tokenizer.apply_chat_template([], tokenize=False, add_generation_prompt=True)
        )
        model = HookAwareDummyModel(
            prompt_length=prompt_length,
            positive_continuation=correct_output,
            negative_continuation=wrong_output,
        )

        with patch("src.eval.run_eval.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch("src.eval.run_eval.load_gemma_tokenizer", return_value=tokenizer):
                with patch("src.eval.run_eval.load_gemma_causal_lm", return_value=model):
                    artifacts = run_eval_pipeline(
                        config={
                            "model": {"loader": "transformers", "id": "dummy"},
                            "eval": {
                                "manifest_path": str(manifest_path),
                                "prediction_backend": "model_greedy",
                                "max_new_tokens": len(correct_output) + 1,
                                "device": "cpu",
                                "transplant": {
                                    "layers": [
                                        {
                                            "checkpoint_path": str(checkpoint_path),
                                            "layer_id": 0,
                                            "gain": 1.0,
                                        }
                                    ]
                                },
                            },
                        },
                        output_dir=output_dir / "artifacts",
                    )

        metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        predictions = [
            json.loads(line)
            for line in Path(artifacts.predictions_path)
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        ]
        self.assertEqual(metrics["aggregate"]["strict_full_call_success"], 1.0)
        self.assertEqual(summary["transplant_layers"][0]["layer_id"], 0)
        self.assertEqual(predictions[0]["raw_output"], correct_output)

    def test_eval_pipeline_filters_examples_by_split(self) -> None:
        output_dir = Path("tests/_tmp/eval_run_split")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_split_fixture_manifest(output_dir)

        artifacts = run_eval_pipeline(
            config={
                "eval": {
                    "manifest_path": str(manifest_path),
                    "prediction_backend": "oracle",
                    "split_filter": "val",
                }
            },
            output_dir=output_dir / "artifacts",
        )

        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        predictions = [
            json.loads(line)
            for line in Path(artifacts.predictions_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary["example_count"], 1)
        self.assertEqual(summary["split_filter"], ["val"])
        self.assertEqual(predictions[0]["split"], "val")

    def test_eval_pipeline_filters_examples_by_variant(self) -> None:
        output_dir = Path("tests/_tmp/eval_run_variant")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_variant_fixture_manifest(output_dir)

        artifacts = run_eval_pipeline(
            config={
                "eval": {
                    "manifest_path": str(manifest_path),
                    "prediction_backend": "oracle",
                    "variant_filter": "schema_shift",
                }
            },
            output_dir=output_dir / "artifacts",
        )

        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        predictions = [
            json.loads(line)
            for line in Path(artifacts.predictions_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(summary["example_count"], 1)
        self.assertEqual(summary["variant_filter"], ["schema_shift"])
        self.assertEqual(predictions[0]["variant"], "schema_shift")

    def test_eval_pipeline_resumes_from_partial_predictions(self) -> None:
        output_dir = Path("tests/_tmp/eval_run_resume")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_two_eval_fixture_manifest(output_dir)
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        from src.data.manifest import load_examples
        from src.eval.metrics import score_prediction

        canonical_first = load_examples(manifest_payload["dataset_path"])[0]
        first_output = render_assistant_target(canonical_first.gold)
        partial_row = {
            "example_id": canonical_first.example_id,
            "split": canonical_first.split,
            "variant": str(canonical_first.meta.get("variant", "canonical")),
            "raw_output": first_output,
            "score": score_prediction(raw_output=first_output, example=canonical_first).__dict__,
        }
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "predictions.jsonl").write_text(
            json.dumps(partial_row, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifacts = run_eval_pipeline(
            config={
                "eval": {
                    "manifest_path": str(manifest_path),
                    "prediction_backend": "oracle",
                    "heartbeat_interval_seconds": 0.0,
                }
            },
            output_dir=artifacts_dir,
        )

        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        predictions = [
            json.loads(line)
            for line in Path(artifacts.predictions_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        heartbeat = json.loads((artifacts_dir / "heartbeat.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["resumed_from_partial"])
        self.assertEqual(len(predictions), 2)
        self.assertEqual(heartbeat["status"], "passed")

    def test_eval_pipeline_dedupes_existing_prediction_rows_on_resume(self) -> None:
        output_dir = Path("tests/_tmp/eval_run_resume_dedupe")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_two_eval_fixture_manifest(output_dir)
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        from src.data.manifest import load_examples
        from src.eval.metrics import score_prediction

        canonical_examples = load_examples(manifest_payload["dataset_path"])
        first_example = canonical_examples[0]
        first_output = render_assistant_target(first_example.gold)
        first_row = {
            "example_id": first_example.example_id,
            "split": first_example.split,
            "variant": str(first_example.meta.get("variant", "canonical")),
            "raw_output": first_output,
            "score": score_prediction(raw_output=first_output, example=first_example).__dict__,
        }
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "predictions.jsonl").write_text(
            "".join(
                [
                    json.dumps(first_row, sort_keys=True) + "\n",
                    json.dumps(first_row, sort_keys=True) + "\n",
                ]
            ),
            encoding="utf-8",
        )

        artifacts = run_eval_pipeline(
            config={
                "eval": {
                    "manifest_path": str(manifest_path),
                    "prediction_backend": "oracle",
                    "heartbeat_interval_seconds": 0.0,
                }
            },
            output_dir=artifacts_dir,
        )

        predictions = [
            json.loads(line)
            for line in Path(artifacts.predictions_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        heartbeat = json.loads((artifacts_dir / "heartbeat.json").read_text(encoding="utf-8"))
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len({row["example_id"] for row in predictions}), 2)
        self.assertEqual(heartbeat["extra"]["deduped_existing_prediction_rows"], 1)

    def test_eval_pipeline_rejects_concurrent_writer_lock(self) -> None:
        output_dir = Path("tests/_tmp/eval_run_lock")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_fixture_manifest(output_dir)
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "eval.lock").write_text(
            json.dumps({"pid": os.getpid(), "output_dir": str(artifacts_dir.resolve())}) + "\n",
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            run_eval_pipeline(
                config={
                    "eval": {
                        "manifest_path": str(manifest_path),
                        "prediction_backend": "oracle",
                    }
                },
                output_dir=artifacts_dir,
            )


if __name__ == "__main__":
    unittest.main()
