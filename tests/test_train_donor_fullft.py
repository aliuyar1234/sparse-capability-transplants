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
from src.train.train_donor_fullft import run_donor_smoke_training


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
    record = write_manifest(
        examples=[replace(train_example, split="train"), replace(val_example, split="val")],
        output_dir=output_dir / "manifest",
        manifest_id="manifest_train_fixture_v1",
        prompt_contract_version="fc_v1",
        metadata={"kind": "train_fixture"},
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

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        return '{"name":"NO_TOOL","arguments":{}}'

    def save_pretrained(self, output_dir: Path | str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, labels=None):  # type: ignore[override]
        loss = (self.weight - 0.5) ** 2 + input_ids.float().mean() * 0.0
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8, device=input_ids.device)
        logits[:, -1, 1] = 1.0
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(
        self,
        output_dir: Path | str,
        *,
        safe_serialization: bool,
        max_shard_size: str,
    ) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_text("dummy", encoding="utf-8")


class TrainDonorSmokeTests(unittest.TestCase):
    def test_donor_smoke_training_runs_and_writes_artifacts(self) -> None:
        output_dir = Path("tests/_tmp/train_donor")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_train_fixture_manifest(output_dir)

        with patch("src.train.train_donor_fullft.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch(
                "src.train.train_donor_fullft.load_gemma_tokenizer",
                return_value=DummyTokenizer(),
            ):
                with patch(
                    "src.train.train_donor_fullft.load_gemma_causal_lm",
                    return_value=DummyModel(),
                ):
                    with patch("src.train.train_donor_fullft.run_eval_pipeline") as eval_mock:
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
                        summary = run_donor_smoke_training(
                            config={
                                "model": {"loader": "transformers", "id": "dummy"},
                                "data": {"train_manifest_path": str(manifest_path)},
                                "train": {
                                    "smoke": {
                                        "max_examples": 1,
                                        "max_val_examples": 1,
                                        "batch_size": 4,
                                        "eval_batch_size": 3,
                                        "max_steps": 1,
                                        "eval_sample_size": 1,
                                        "device": "cpu",
                                    }
                                },
                            },
                            output_dir=output_dir / "artifacts",
                        )

        trace_payload = json.loads(Path(summary["train_trace_path"]).read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["global_step"], 1)
        self.assertEqual(summary["smoke_serialized_example_count"], 2)
        self.assertTrue(Path(summary["checkpoint_dir"]).exists())
        self.assertEqual(summary["post_train_eval_split"], "val")
        self.assertTrue(summary["post_train_eval_summary_path"].endswith("summary.json"))
        self.assertTrue(summary["post_train_eval_metrics_path"].endswith("metrics.json"))
        self.assertTrue(summary["post_train_eval_predictions_path"].endswith("predictions.jsonl"))
        self.assertEqual(trace_payload["global_step"], 1)
        eval_call = eval_mock.call_args.kwargs
        self.assertEqual(eval_call["config"]["eval"]["prediction_backend"], "model_greedy")
        self.assertEqual(eval_call["config"]["eval"]["split_filter"], "val")
        self.assertEqual(eval_call["config"]["eval"]["max_examples"], 1)
        self.assertEqual(eval_call["config"]["eval"]["batch_size"], 3)
        self.assertEqual(eval_call["config"]["model"]["local_path"], summary["checkpoint_dir"])

    def test_donor_main_profile_is_selected_when_present(self) -> None:
        output_dir = Path("tests/_tmp/train_donor_main")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_train_fixture_manifest(output_dir)

        with patch("src.train.train_donor_fullft.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch(
                "src.train.train_donor_fullft.load_gemma_tokenizer",
                return_value=DummyTokenizer(),
            ):
                with patch(
                    "src.train.train_donor_fullft.load_gemma_causal_lm",
                    return_value=DummyModel(),
                ):
                    with patch("src.train.train_donor_fullft.run_eval_pipeline") as eval_mock:
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
                        summary = run_donor_smoke_training(
                            config={
                                "model": {"loader": "transformers", "id": "dummy"},
                                "data": {"train_manifest_path": str(manifest_path)},
                                "train": {
                                    "main": {
                                        "max_examples": 1,
                                        "max_val_examples": 1,
                                        "max_steps": 1,
                                        "eval_sample_size": 1,
                                        "device": "cpu",
                                    }
                                },
                            },
                            output_dir=output_dir / "artifacts",
                        )

        self.assertEqual(summary["train_profile"], "main")
        self.assertIn("M3 donor full-FT", summary["notes"][0])

    def test_donor_training_resumes_from_saved_checkpoint_state(self) -> None:
        output_dir = Path("tests/_tmp/train_donor_resume")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_train_fixture_manifest(output_dir)
        artifacts_dir = output_dir / "artifacts"
        resume_model_dir = artifacts_dir / "resume_checkpoint" / "model"
        resume_model_dir.mkdir(parents=True, exist_ok=True)
        optimizer = torch.optim.AdamW(DummyModel().parameters(), lr=1e-5)
        torch.save(
            {
                "global_step": 1,
                "train_step_losses": [0.25],
                "initial_val_loss": 0.5,
                "optimizer_state": optimizer.state_dict(),
                "train_profile": "main",
            },
            artifacts_dir / "resume_checkpoint" / "state.pt",
        )

        with patch("src.train.train_donor_fullft.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch(
                "src.train.train_donor_fullft.load_gemma_tokenizer",
                return_value=DummyTokenizer(),
            ):
                with patch(
                    "src.train.train_donor_fullft.load_gemma_causal_lm",
                    return_value=DummyModel(),
                ):
                    with patch("src.train.train_donor_fullft.run_eval_pipeline") as eval_mock:
                        eval_mock.return_value.summary_path = str(
                            (artifacts_dir / "post_train_eval" / "summary.json").resolve()
                        )
                        eval_mock.return_value.metrics_path = str(
                            (artifacts_dir / "post_train_eval" / "metrics.json").resolve()
                        )
                        eval_mock.return_value.predictions_path = str(
                            (artifacts_dir / "post_train_eval" / "predictions.jsonl").resolve()
                        )
                        summary = run_donor_smoke_training(
                            config={
                                "model": {"loader": "transformers", "id": "dummy"},
                                "data": {"train_manifest_path": str(manifest_path)},
                                "train": {
                                    "main": {
                                        "max_examples": 1,
                                        "max_val_examples": 1,
                                        "max_steps": 2,
                                        "epochs": 2,
                                        "eval_sample_size": 1,
                                        "device": "cpu",
                                        "checkpoint_interval_steps": 1,
                                        "heartbeat_interval_seconds": 0.0,
                                    }
                                },
                            },
                            output_dir=artifacts_dir,
                        )

        self.assertTrue(summary["resumed_from_checkpoint"])
        self.assertEqual(summary["global_step"], 2)
        self.assertTrue(Path(summary["heartbeat_path"]).exists())
        self.assertTrue(Path(summary["resume_state_path"]).exists())


if __name__ == "__main__":
    unittest.main()
