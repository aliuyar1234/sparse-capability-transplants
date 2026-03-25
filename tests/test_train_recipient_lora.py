from __future__ import annotations

import json
import re
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.data.canonical import ArgSpec, ToolSpec, build_canonical_example
from src.data.manifest import write_manifest
from src.train.train_recipient_lora import run_recipient_lora_smoke_training


def _write_train_fixture_manifest(output_dir: Path) -> Path:
    tool = ToolSpec(
        tool_id="send_email",
        name="send_email",
        description="Sends an email.",
        arguments=[
            ArgSpec(name="to", type="string", required=True, description="Recipient"),
            ArgSpec(name="subject", type="string", required=True, description="Subject"),
        ],
    )
    train_example = build_canonical_example(
        source="synthetic",
        raw_split="train",
        user_request="Email Sam about the report.",
        tools=[tool],
        gold={"name": "send_email", "arguments": {"to": "sam@example.com", "subject": "report"}},
        source_row_id="row-train",
        meta={
            "source_row_id": "row-train",
            "canonical_tool_id": "send_email",
            "canonical_argument_map": {"to": "to", "subject": "subject"},
            "alias_bank_id": "none",
        },
    )
    val_example = build_canonical_example(
        source="synthetic",
        raw_split="train",
        user_request="Email Dana about lunch.",
        tools=[tool],
        gold={"name": "send_email", "arguments": {"to": "dana@example.com", "subject": "lunch"}},
        source_row_id="row-val",
        meta={
            "source_row_id": "row-val",
            "canonical_tool_id": "send_email",
            "canonical_argument_map": {"to": "to", "subject": "subject"},
            "alias_bank_id": "none",
        },
    )
    calib_example = build_canonical_example(
        source="synthetic",
        raw_split="train",
        user_request="Email Lee about the agenda.",
        tools=[tool],
        gold={"name": "send_email", "arguments": {"to": "lee@example.com", "subject": "agenda"}},
        source_row_id="row-calib",
        meta={
            "source_row_id": "row-calib",
            "canonical_tool_id": "send_email",
            "canonical_argument_map": {"to": "to", "subject": "subject"},
            "alias_bank_id": "none",
        },
    )
    record = write_manifest(
        examples=[
            replace(train_example, split="train"),
            replace(val_example, split="val"),
            replace(calib_example, split="calib"),
        ],
        output_dir=output_dir / "manifest",
        manifest_id="manifest_recipient_lora_fixture_v1",
        prompt_contract_version="fc_v1",
        metadata={"kind": "recipient_lora_fixture"},
    )
    return Path(record.manifest_path)


class DummyTokenizer:
    chat_template = "{{ bos_token }}"
    pad_token_id = 0
    eos_token_id = 7

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        text = " ".join(str(message["content"]) for message in messages)
        if add_generation_prompt:
            return f"{text} <assistant>"
        return text

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        pieces = re.findall(r"\w+|[^\w\s]", text)
        tokens = [index + 1 for index, _ in enumerate(pieces)]
        if max_length is not None:
            tokens = tokens[:max_length]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, output_dir: Path | str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyMergedModel:
    def save_pretrained(
        self,
        output_dir: Path | str,
        *,
        safe_serialization: bool,
        max_shard_size: str,
    ) -> None:
        del safe_serialization, max_shard_size
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_text("dummy", encoding="utf-8")


class DummyLoraModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.lora_weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, labels=None):  # type: ignore[override]
        del attention_mask, labels
        loss = (self.lora_weight - 0.5) ** 2 + input_ids.float().mean() * 0.0
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8, device=input_ids.device)
        logits[:, -1, 1] = 1.0
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, output_dir: Path | str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_model.safetensors").write_text("dummy", encoding="utf-8")

    def merge_and_unload(self) -> DummyMergedModel:
        return DummyMergedModel()


class DummyLoraConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class RecipientLoraSmokeTests(unittest.TestCase):
    def test_recipient_lora_smoke_training_runs_and_writes_artifacts(self) -> None:
        output_dir = Path("tests/_tmp/train_recipient_lora")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_train_fixture_manifest(output_dir)

        with patch("src.train.train_recipient_lora.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch(
                "src.train.train_recipient_lora.load_gemma_tokenizer",
                return_value=DummyTokenizer(),
            ):
                with patch(
                    "src.train.train_recipient_lora.load_gemma_causal_lm",
                    return_value=torch.nn.Linear(1, 1, bias=False),
                ):
                    with patch("src.train.train_recipient_lora.run_eval_pipeline") as eval_mock:
                        eval_mock.return_value.summary_path = str(
                            (
                                output_dir / "artifacts" / "post_train_eval" / "summary.json"
                            ).resolve()
                        )
                        eval_mock.return_value.metrics_path = str(
                            (
                                output_dir / "artifacts" / "post_train_eval" / "metrics.json"
                            ).resolve()
                        )
                        eval_mock.return_value.predictions_path = str(
                            (
                                output_dir / "artifacts" / "post_train_eval" / "predictions.jsonl"
                            ).resolve()
                        )
                        with patch.dict(
                            "sys.modules",
                            {
                                "peft": SimpleNamespace(
                                    LoraConfig=DummyLoraConfig,
                                    get_peft_model=lambda model, config: DummyLoraModel(),
                                )
                            },
                        ):
                            summary = run_recipient_lora_smoke_training(
                                config={
                                    "model": {"loader": "transformers", "id": "dummy"},
                                    "data": {
                                        "train_manifest_path": str(manifest_path),
                                        "train_split": "calib",
                                        "eval_split": "val",
                                    },
                                    "train": {
                                        "baseline_kind": "small_data_lora",
                                        "smoke": {
                                            "max_examples": 1,
                                            "max_eval_examples": 1,
                                            "batch_size": 4,
                                            "eval_batch_size": 3,
                                            "max_steps": 1,
                                            "eval_sample_size": 1,
                                            "device": "cpu",
                                        },
                                    },
                                    "lora": {
                                        "rank": 2,
                                        "alpha": 4,
                                        "target_modules": ["q_proj"],
                                    },
                                },
                                output_dir=output_dir / "artifacts",
                            )

        trace_payload = json.loads(Path(summary["train_trace_path"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["baseline_kind"], "small_data_lora")
        self.assertEqual(summary["train_split"], "calib")
        self.assertEqual(summary["eval_split"], "val")
        self.assertEqual(summary["global_step"], 1)
        self.assertTrue(Path(summary["adapter_checkpoint_dir"]).exists())
        self.assertTrue(Path(summary["merged_checkpoint_dir"]).exists())
        self.assertGreater(summary["trainable_parameter_count"], 0)
        self.assertEqual(trace_payload["global_step"], 1)
        eval_call = eval_mock.call_args.kwargs
        self.assertEqual(eval_call["config"]["eval"]["split_filter"], "val")
        self.assertEqual(eval_call["config"]["eval"]["max_examples"], 1)
        self.assertEqual(eval_call["config"]["eval"]["batch_size"], 3)
        self.assertEqual(
            eval_call["config"]["model"]["local_path"],
            summary["merged_checkpoint_dir"],
        )


if __name__ == "__main__":
    unittest.main()
