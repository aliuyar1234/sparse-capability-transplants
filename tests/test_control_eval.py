from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.data.build_control_suite import (
    build_control_example,
    write_control_suite,
    write_control_suite_manifest,
)
from src.eval.control_metrics import score_control_prediction
from src.eval.run_control_eval import run_control_eval_pipeline


def _write_control_manifest(output_dir: Path) -> Path:
    example = build_control_example(
        source="synthetic",
        prompt="Return only the text hello.",
        target_text="hello",
        source_row_id="ctrl-row-1",
    )
    dataset_path = write_control_suite(
        examples=[example],
        output_path=output_dir / "control_suite" / "controls.jsonl",
    )
    record = write_control_suite_manifest(
        examples=[example],
        dataset_path=dataset_path,
        output_path=output_dir / "control_suite" / "manifest.json",
        manifest_id="manifest_control_eval_fixture_v1",
    )
    return Path(record.manifest_path)


def _write_two_control_manifest(output_dir: Path) -> Path:
    examples = [
        build_control_example(
            source="synthetic",
            prompt="Return only the text hello.",
            target_text="hello",
            source_row_id="ctrl-row-1",
        ),
        build_control_example(
            source="synthetic",
            prompt="Return only the text hello again.",
            target_text="hello",
            source_row_id="ctrl-row-2",
        ),
    ]
    dataset_path = write_control_suite(
        examples=examples,
        output_path=output_dir / "control_suite" / "controls.jsonl",
    )
    record = write_control_suite_manifest(
        examples=examples,
        dataset_path=dataset_path,
        output_path=output_dir / "control_suite" / "manifest.json",
        manifest_id="manifest_control_eval_two_fixture_v1",
    )
    return Path(record.manifest_path)


class ControlEvalTests(unittest.TestCase):
    def test_run_control_eval_pipeline_scores_oracle_predictions(self) -> None:
        output_dir = Path("tests/_tmp/control_eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_control_manifest(output_dir)

        artifacts = run_control_eval_pipeline(
            config={
                "control_eval": {
                    "manifest_path": str(manifest_path),
                    "prediction_backend": "oracle",
                    "heartbeat_interval_seconds": 0.0,
                }
            },
            output_dir=output_dir / "artifacts",
        )

        metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
        summary = json.loads(Path(artifacts.summary_path).read_text(encoding="utf-8"))
        self.assertEqual(metrics["exact_match_average"], 1.0)
        self.assertEqual(summary["example_count"], 1)

    def test_run_control_eval_pipeline_resumes_from_partial_predictions(self) -> None:
        output_dir = Path("tests/_tmp/control_eval_resume")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_two_control_manifest(output_dir)
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_path = Path(manifest_payload["dataset_path"])
        first_example = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
        partial_score = score_control_prediction(
            raw_output=first_example["target_text"],
            example=build_control_example(
                source="synthetic",
                prompt=first_example["prompt"],
                target_text=first_example["target_text"],
                source_row_id="ctrl-resume-row",
            ),
        )
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "predictions.jsonl").write_text(
            json.dumps(
                {
                    "example_id": first_example["example_id"],
                    "split": first_example["split"],
                    "variant": "control",
                    "raw_output": first_example["target_text"],
                    "score": partial_score.__dict__,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        artifacts = run_control_eval_pipeline(
            config={
                "control_eval": {
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
        self.assertTrue(summary["resumed_from_partial"])
        self.assertEqual(len(predictions), 2)

    def test_run_control_eval_pipeline_scores_batched_model_predictions(self) -> None:
        output_dir = Path("tests/_tmp/control_eval_batched_model")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = _write_two_control_manifest(output_dir)

        class DummyTokenizer:
            chat_template = "{{ bos_token }}"
            eos_token_id = 0
            pad_token_id = 0
            padding_side = "right"

            def apply_chat_template(
                self,
                messages,
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                del tokenize, add_generation_prompt
                return messages[0]["content"]

            def __call__(
                self,
                text,
                *,
                add_special_tokens: bool,
                return_tensors: str,
                padding: bool = False,
            ) -> dict[str, torch.Tensor]:
                del add_special_tokens, return_tensors, padding
                if isinstance(text, str):
                    texts = [text]
                else:
                    texts = list(text)
                max_len = max(len(item) for item in texts)
                rows = []
                masks = []
                for item in texts:
                    values = [ord(char) for char in item]
                    if self.padding_side == "left":
                        pad = [self.pad_token_id] * (max_len - len(values))
                        rows.append(pad + values)
                        masks.append([0] * len(pad) + [1] * len(values))
                    else:
                        pad = [self.pad_token_id] * (max_len - len(values))
                        rows.append(values + pad)
                        masks.append([1] * len(values) + [0] * len(pad))
                return {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }

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

            def batch_decode(
                self, token_ids: torch.Tensor, *, skip_special_tokens: bool
            ) -> list[str]:
                return [
                    self.decode(row, skip_special_tokens=skip_special_tokens) for row in token_ids
                ]

        class DummyModel:
            def __init__(self, continuation: str) -> None:
                self._continuation_ids = [ord(char) for char in continuation]

            def to(self, device: torch.device | str) -> "DummyModel":
                del device
                return self

            def eval(self) -> "DummyModel":
                return self

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
                del attention_mask, do_sample, num_beams, use_cache, return_dict_in_generate
                del pad_token_id, eos_token_id
                batch_size = input_ids.shape[0]
                continuation = torch.tensor(
                    [self._continuation_ids[:max_new_tokens] for _ in range(batch_size)],
                    dtype=input_ids.dtype,
                )
                return torch.cat([input_ids, continuation], dim=1)

        tokenizer = DummyTokenizer()
        model = DummyModel("hello")
        with patch("src.eval.run_control_eval.probe_gemma_loading") as probe_mock:
            probe_mock.return_value.status = "passed"
            probe_mock.return_value.to_dict.return_value = {"status": "passed"}
            with patch("src.eval.run_control_eval.load_gemma_tokenizer", return_value=tokenizer):
                with patch("src.eval.run_control_eval.load_gemma_causal_lm", return_value=model):
                    artifacts = run_control_eval_pipeline(
                        config={
                            "model": {"loader": "transformers", "id": "dummy"},
                            "control_eval": {
                                "manifest_path": str(manifest_path),
                                "prediction_backend": "model_greedy",
                                "batch_size": 2,
                                "device": "cpu",
                                "max_new_tokens": 5,
                                "heartbeat_interval_seconds": 0.0,
                            },
                        },
                        output_dir=output_dir / "artifacts",
                    )

        metrics = json.loads(Path(artifacts.metrics_path).read_text(encoding="utf-8"))
        self.assertEqual(metrics["exact_match_average"], 1.0)


if __name__ == "__main__":
    unittest.main()
