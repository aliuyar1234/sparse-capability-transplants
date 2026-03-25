from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.build_control_suite import ControlExample
from src.eval.control_metrics import (
    ControlScore,
    aggregate_control_scores,
    score_control_prediction,
)
from src.eval.run_eval import (
    _acquire_eval_lock,
    _dedupe_rows_by_example_id,
    _greedy_model_outputs,
    _load_existing_rows,
    _release_eval_lock,
    _rewrite_predictions_file,
)
from src.models.load_gemma import load_gemma_causal_lm, load_gemma_tokenizer, probe_gemma_loading
from src.models.transplant import inject_sparse_delta_modules, load_sparse_transplant_layers
from src.utils.progress import RunHeartbeat


@dataclass(frozen=True)
class ControlEvalArtifacts:
    summary_path: str
    metrics_path: str
    predictions_path: str


def _load_control_examples(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], list[ControlExample]]:
    manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    dataset_path = Path(manifest_payload["dataset_path"])
    examples = [
        ControlExample(**json.loads(line))
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest_payload, examples


def _device_from_control_config(config: dict[str, Any]) -> torch.device:
    device_name = str(config.get("control_eval", {}).get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _oracle_output(example: ControlExample) -> str:
    return example.target_text


def _render_control_chat_prompt(*, example: ControlExample, tokenizer: Any) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not implement apply_chat_template.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer does not expose a chat_template.")
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def run_control_eval_pipeline(
    *, config: dict[str, Any], output_dir: str | Path
) -> ControlEvalArtifacts:
    control_config = dict(config.get("control_eval", {}))
    manifest_path = control_config.get("manifest_path")
    if not manifest_path:
        raise ValueError("Config is missing control_eval.manifest_path.")

    manifest_payload, examples = _load_control_examples(manifest_path)
    max_examples = control_config.get("max_examples")
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    if not examples:
        raise ValueError("Control evaluation example set is empty.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_eval_lock(destination)
    heartbeat = RunHeartbeat(
        output_dir=destination,
        phase="control_eval",
        total_units=len(examples),
        unit_name="examples",
        heartbeat_interval_seconds=float(control_config.get("heartbeat_interval_seconds", 10.0)),
    )

    backend = str(control_config.get("prediction_backend", "oracle"))
    requires_model = backend == "model_greedy"
    if control_config.get("transplant") and not requires_model:
        raise ValueError(
            "control_eval.transplant is only supported with prediction_backend=model_greedy."
        )
    predictions_path = destination / "predictions.jsonl"
    summary_path = destination / "summary.json"
    metrics_path = destination / "metrics.json"
    try:
        existing_rows = _load_existing_rows(predictions_path)
        expected_example_ids = {example.example_id for example in examples}
        for row in existing_rows:
            if row["example_id"] not in expected_example_ids:
                raise ValueError(
                    "Existing control predictions file contains example "
                    f"{row['example_id']} not present in this eval."
                )
        existing_rows, duplicate_rows = _dedupe_rows_by_example_id(existing_rows)
        if duplicate_rows > 0:
            _rewrite_predictions_file(predictions_path, existing_rows)
        completed_ids = {row["example_id"] for row in existing_rows}
        resumed_from_partial = bool(existing_rows)
        if (
            len(completed_ids) == len(expected_example_ids)
            and summary_path.exists()
            and metrics_path.exists()
        ):
            return ControlEvalArtifacts(
                summary_path=str(summary_path.resolve()),
                metrics_path=str(metrics_path.resolve()),
                predictions_path=str(predictions_path.resolve()),
            )

        tokenizer = None
        model = None
        device = None
        load_report = None
        loaded_transplant_layers = []
        if requires_model or bool(control_config.get("serialize_with_chat_template", False)):
            load_report = probe_gemma_loading(
                config.get("model", {}),
                require_tokenizer=True,
                require_chat_template=True,
            )
            if load_report.status != "passed":
                raise RuntimeError(load_report.message)
            tokenizer = load_gemma_tokenizer(config.get("model", {}))
        if requires_model:
            device = _device_from_control_config(config)
            model = load_gemma_causal_lm(config.get("model", {})).to(device)
            model.eval()
            loaded_transplant_layers = load_sparse_transplant_layers(
                transplant_config=dict(control_config.get("transplant", {})),
                device=device,
            )

        rows = list(existing_rows)
        exact_match_count = sum(int(row["score"]["exact_match"]) for row in rows)
        heartbeat.start(
            completed_units=len(rows),
            message=(
                "Resumed control-suite evaluation from partial predictions."
                if resumed_from_partial
                else "Started control-suite evaluation."
            ),
            extra={
                "prediction_backend": backend,
                "deduped_existing_prediction_rows": duplicate_rows,
            },
        )
        pending_examples = [
            example for example in examples if example.example_id not in completed_ids
        ]
        if not pending_examples:
            control_scores = [ControlScore(**row["score"]) for row in rows]
            exact_match_average = aggregate_control_scores(control_scores)
            metrics_payload = {
                "exact_match_average": exact_match_average,
                "total_examples": len(rows),
            }
            metrics_path.write_text(
                json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary_payload = {
                "status": "passed",
                "manifest_id": manifest_payload["manifest_id"],
                "manifest_hash": manifest_payload["manifest_hash"],
                "example_count": len(rows),
                "prediction_backend": backend,
                "metrics_path": str(metrics_path.resolve()),
                "predictions_path": str(predictions_path.resolve()),
                "loader_report": None if load_report is None else load_report.to_dict(),
                "generation_device": None if device is None else str(device),
                "transplant_layers": [layer.to_summary() for layer in loaded_transplant_layers],
                "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
                "progress_path": str(heartbeat.paths.progress_path.resolve()),
                "resumed_from_partial": resumed_from_partial,
            }
            summary_path.write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            heartbeat.mark_completed(
                completed_units=len(rows),
                message="Control-suite artifacts already complete; reused existing predictions.",
                metrics={"exact_match_average": exact_match_average},
                extra={"summary_path": str(summary_path.resolve())},
            )
            return ControlEvalArtifacts(
                summary_path=str(summary_path.resolve()),
                metrics_path=str(metrics_path.resolve()),
                predictions_path=str(predictions_path.resolve()),
            )

        batch_size = max(1, int(control_config.get("batch_size", 1)))
        file_mode = "a" if resumed_from_partial else "w"
        try:
            transplant_context = (
                inject_sparse_delta_modules(model, loaded_transplant_layers)
                if model is not None and loaded_transplant_layers
                else nullcontext()
            )
            with transplant_context:
                with predictions_path.open(file_mode, encoding="utf-8") as handle:
                    if backend == "model_greedy":
                        if model is None or tokenizer is None or device is None:
                            raise RuntimeError(
                                "Model-backed control eval requires model, tokenizer, and device."
                            )
                        prompt_rows = [
                            (
                                example,
                                _render_control_chat_prompt(example=example, tokenizer=tokenizer),
                            )
                            for example in pending_examples
                        ]
                        for start in range(0, len(prompt_rows), batch_size):
                            batch = prompt_rows[start : start + batch_size]
                            batch_examples = [example for example, _ in batch]
                            batch_prompts = [prompt_text for _, prompt_text in batch]
                            raw_outputs = _greedy_model_outputs(
                                prompt_texts=batch_prompts,
                                tokenizer=tokenizer,
                                model=model,
                                device=device,
                                max_new_tokens=int(control_config.get("max_new_tokens", 64)),
                            )
                            for example, raw_output in zip(
                                batch_examples, raw_outputs, strict=True
                            ):
                                score = score_control_prediction(
                                    raw_output=raw_output, example=example
                                )
                                row = {
                                    "example_id": example.example_id,
                                    "split": example.split,
                                    "variant": str(example.meta.get("variant", "control")),
                                    "raw_output": raw_output,
                                    "score": asdict(score),
                                }
                                rows.append(row)
                                exact_match_count += int(score.exact_match)
                                handle.write(json.dumps(row, sort_keys=True) + "\n")
                            heartbeat_payload = heartbeat.maybe_update(
                                completed_units=len(rows),
                                message=(
                                    f"Processed control example {len(rows)} of {len(examples)}."
                                ),
                                metrics={
                                    "exact_match_average_so_far": (
                                        exact_match_count / len(rows) if rows else 0.0
                                    ),
                                },
                                extra={
                                    "last_example_id": batch_examples[-1].example_id,
                                    "deduped_existing_prediction_rows": duplicate_rows,
                                },
                            )
                            if heartbeat_payload:
                                handle.flush()
                    elif backend == "oracle":
                        for example in pending_examples:
                            raw_output = _oracle_output(example)
                            score = score_control_prediction(raw_output=raw_output, example=example)
                            row = {
                                "example_id": example.example_id,
                                "split": example.split,
                                "variant": str(example.meta.get("variant", "control")),
                                "raw_output": raw_output,
                                "score": asdict(score),
                            }
                            rows.append(row)
                            exact_match_count += int(score.exact_match)
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            heartbeat_payload = heartbeat.maybe_update(
                                completed_units=len(rows),
                                message=(
                                    f"Processed control example {len(rows)} of {len(examples)}."
                                ),
                                metrics={
                                    "exact_match_average_so_far": (
                                        exact_match_count / len(rows) if rows else 0.0
                                    )
                                },
                                extra={
                                    "last_example_id": example.example_id,
                                    "deduped_existing_prediction_rows": duplicate_rows,
                                },
                            )
                            if heartbeat_payload:
                                handle.flush()
                    else:
                        raise ValueError(
                            "control_eval.prediction_backend must be 'oracle' or 'model_greedy', "
                            f"got {backend!r}."
                        )
        except KeyboardInterrupt:
            heartbeat.mark_interrupted(
                completed_units=len(rows),
                message="Control-suite evaluation interrupted; partial predictions are resumable.",
                metrics={
                    "exact_match_average_so_far": exact_match_count / len(rows) if rows else 0.0
                },
                extra={
                    "prediction_backend": backend,
                    "deduped_existing_prediction_rows": duplicate_rows,
                },
            )
            if model is not None:
                model.to("cpu")
                del model
                if device is not None and device.type == "cuda":
                    torch.cuda.empty_cache()
            raise

        control_scores = [ControlScore(**row["score"]) for row in rows]
        exact_match_average = aggregate_control_scores(control_scores)
        metrics_payload = {
            "exact_match_average": exact_match_average,
            "total_examples": len(rows),
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary_payload = {
            "status": "passed",
            "manifest_id": manifest_payload["manifest_id"],
            "manifest_hash": manifest_payload["manifest_hash"],
            "example_count": len(rows),
            "prediction_backend": backend,
            "metrics_path": str(metrics_path.resolve()),
            "predictions_path": str(predictions_path.resolve()),
            "loader_report": None if load_report is None else load_report.to_dict(),
            "generation_device": None if device is None else str(device),
            "transplant_layers": [layer.to_summary() for layer in loaded_transplant_layers],
            "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
            "progress_path": str(heartbeat.paths.progress_path.resolve()),
            "resumed_from_partial": resumed_from_partial,
        }
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        heartbeat.mark_completed(
            completed_units=len(rows),
            message="Control-suite evaluation completed successfully.",
            metrics={"exact_match_average": exact_match_average},
            extra={"summary_path": str(summary_path.resolve())},
        )

        if model is not None:
            model.to("cpu")
            del model
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()

        return ControlEvalArtifacts(
            summary_path=str(summary_path.resolve()),
            metrics_path=str(metrics_path.resolve()),
            predictions_path=str(predictions_path.resolve()),
        )
    finally:
        _release_eval_lock(lock_path)
