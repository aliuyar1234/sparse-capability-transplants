from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.manifest import load_examples, load_manifest_payload
from src.eval.metrics import ExampleScore, aggregate_scores, score_prediction
from src.models.format_prompts import (
    NO_TOOL_OBJECT,
    build_prompt_content,
    render_assistant_target,
    render_chat_prompt,
)
from src.models.load_gemma import (
    load_gemma_causal_lm,
    load_gemma_tokenizer,
    probe_gemma_loading,
)
from src.models.transplant import inject_sparse_delta_modules, load_sparse_transplant_layers
from src.utils.progress import RunHeartbeat


@dataclass(frozen=True)
class EvalArtifacts:
    summary_path: str
    metrics_path: str
    predictions_path: str


def _prediction_backend_output(example: Any, backend: str) -> str:
    if backend == "oracle":
        return render_assistant_target(example.gold)
    if backend == "no_tool":
        return render_assistant_target(NO_TOOL_OBJECT)
    raise ValueError(f"Unsupported eval prediction_backend {backend!r}")


def _device_from_eval_config(config: dict[str, Any]) -> torch.device:
    device_name = str(config.get("eval", {}).get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _greedy_model_output(
    *,
    prompt_text: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = input_ids.shape[1]

    model.eval()
    with torch.inference_mode():
        if hasattr(model, "generate"):
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                return_dict_in_generate=False,
                pad_token_id=pad_token_id if pad_token_id is not None else eos_token_id,
                eos_token_id=eos_token_id,
            )
        else:
            generated_ids = input_ids
            generated_mask = attention_mask
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=generated_mask,
                )
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                generated_mask = torch.cat(
                    [generated_mask, torch.ones_like(next_token, dtype=generated_mask.dtype)],
                    dim=1,
                )
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break

    continuation = generated_ids[0][prompt_length:]
    return str(tokenizer.decode(continuation, skip_special_tokens=True)).strip()


def _greedy_model_outputs(
    *,
    prompt_texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    if not prompt_texts:
        return []
    if len(prompt_texts) == 1:
        return [
            _greedy_model_output(
                prompt_text=prompt_texts[0],
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        ]

    original_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        encoded = tokenizer(
            prompt_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
    finally:
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = original_padding_side

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_length = input_ids.shape[1]

    model.eval()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=False,
            pad_token_id=(
                getattr(tokenizer, "pad_token_id", None)
                if getattr(tokenizer, "pad_token_id", None) is not None
                else getattr(tokenizer, "eos_token_id", None)
            ),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )

    continuations = generated_ids[:, prompt_length:]
    if hasattr(tokenizer, "batch_decode"):
        decoded = tokenizer.batch_decode(continuations, skip_special_tokens=True)
    else:
        decoded = [
            tokenizer.decode(continuation, skip_special_tokens=True)
            for continuation in continuations
        ]
    return [str(text).strip() for text in decoded]


def _variant_metrics(scores_by_variant: dict[str, list[Any]]) -> dict[str, Any]:
    return {
        variant: asdict(aggregate_scores(scores))
        for variant, scores in sorted(scores_by_variant.items())
    }


def _split_filter(eval_config: dict[str, Any]) -> set[str] | None:
    split_filter = eval_config.get("split_filter")
    if split_filter is None:
        return None
    if isinstance(split_filter, str):
        return {split_filter}
    if isinstance(split_filter, list):
        return {str(split) for split in split_filter}
    raise ValueError("eval.split_filter must be either a string or a list of strings.")


def _variant_filter(eval_config: dict[str, Any]) -> set[str] | None:
    variant_filter = eval_config.get("variant_filter")
    if variant_filter is None:
        return None
    if isinstance(variant_filter, str):
        return {variant_filter}
    if isinstance(variant_filter, list):
        return {str(variant) for variant in variant_filter}
    raise ValueError("eval.variant_filter must be either a string or a list of strings.")


def _load_existing_rows(predictions_path: Path) -> list[dict[str, Any]]:
    if not predictions_path.exists():
        return []
    return [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _dedupe_rows_by_example_id(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped_by_example_id: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in deduped_by_example_id:
            duplicate_rows += 1
        deduped_by_example_id[example_id] = row
    return list(deduped_by_example_id.values()), duplicate_rows


def _rewrite_predictions_file(predictions_path: Path, rows: list[dict[str, Any]]) -> None:
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _eval_lock_path(destination: Path) -> Path:
    return destination / "eval.lock"


def _acquire_eval_lock(destination: Path) -> Path:
    lock_path = _eval_lock_path(destination)
    payload = {
        "pid": os.getpid(),
        "output_dir": str(destination.resolve()),
    }
    while True:
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            break
        except FileExistsError as exc:
            if not lock_path.exists():
                continue
            try:
                existing_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_payload = {}
            existing_pid = int(existing_payload.get("pid", -1))
            if _process_is_running(existing_pid):
                raise RuntimeError(
                    "Another eval-main process is already writing to this run directory "
                    f"(pid={existing_pid}). Wait for it to finish or remove the stale lock if "
                    "the process has exited."
                ) from exc
            lock_path.unlink(missing_ok=True)
    return lock_path


def _release_eval_lock(lock_path: Path) -> None:
    lock_path.unlink(missing_ok=True)


def _write_eval_outputs(
    *,
    destination: Path,
    manifest_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    backend: str,
    serialize_prompts: bool,
    load_report: Any,
    serialized_preview: str | None,
    device: torch.device | None,
    max_new_tokens: int,
    split_filter: set[str] | None,
    variant_filter: set[str] | None,
    transplant_layers: list[dict[str, Any]],
    status: str,
    resumed_from_partial: bool,
    heartbeat: RunHeartbeat,
) -> tuple[Path, Path]:
    metrics_scores = [ExampleScore(**row["score"]) for row in rows]
    metrics_payload = {
        "aggregate": asdict(aggregate_scores(metrics_scores)),
        "by_variant": _variant_metrics(
            {
                variant: [ExampleScore(**row["score"]) for row in variant_rows]
                for variant, variant_rows in {
                    variant: [row for row in rows if row["variant"] == variant]
                    for variant in sorted({row["variant"] for row in rows})
                }.items()
            }
        ),
    }
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_payload = {
        "status": status,
        "source_manifest_id": manifest_payload["manifest_id"],
        "source_manifest_hash": manifest_payload["manifest_hash"],
        "prediction_backend": backend,
        "example_count": len(rows),
        "metrics_path": str(metrics_path.resolve()),
        "predictions_path": str((destination / "predictions.jsonl").resolve()),
        "serialize_with_chat_template": serialize_prompts,
        "loader_report": None if load_report is None else load_report.to_dict(),
        "serialized_prompt_preview": serialized_preview,
        "generation_device": None if device is None else str(device),
        "max_new_tokens": max_new_tokens if backend == "model_greedy" else None,
        "split_filter": None if split_filter is None else sorted(split_filter),
        "variant_filter": None if variant_filter is None else sorted(variant_filter),
        "transplant_layers": transplant_layers,
        "resumed_from_partial": resumed_from_partial,
        "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
        "progress_path": str(heartbeat.paths.progress_path.resolve()),
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path, metrics_path


def run_eval_pipeline(*, config: dict[str, Any], output_dir: str | Path) -> EvalArtifacts:
    eval_config = config.get("eval", {})
    manifest_path = eval_config.get("manifest_path")
    if not manifest_path:
        raise ValueError("Config is missing eval.manifest_path.")

    manifest_payload = load_manifest_payload(manifest_path)
    examples = load_examples(manifest_payload["dataset_path"])
    split_filter = _split_filter(eval_config)
    if split_filter is not None:
        examples = [example for example in examples if example.split in split_filter]
        if not examples:
            raise ValueError(
                f"Eval split filter {sorted(split_filter)!r} matched no examples in the manifest."
            )
    variant_filter = _variant_filter(eval_config)
    if variant_filter is not None:
        examples = [
            example
            for example in examples
            if str(example.meta.get("variant", "canonical")) in variant_filter
        ]
        if not examples:
            raise ValueError(
                "Eval variant filter "
                f"{sorted(variant_filter)!r} matched no examples in the manifest."
            )
    limit = int(eval_config.get("max_examples", 0))
    if limit > 0:
        examples = examples[:limit]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_eval_lock(destination)
    heartbeat = RunHeartbeat(
        output_dir=destination,
        phase="eval_main",
        total_units=len(examples),
        unit_name="examples",
        heartbeat_interval_seconds=float(eval_config.get("heartbeat_interval_seconds", 10.0)),
    )
    predictions_path = destination / "predictions.jsonl"
    try:
        existing_rows = _load_existing_rows(predictions_path)
        expected_example_ids = {example.example_id for example in examples}
        for row in existing_rows:
            if row["example_id"] not in expected_example_ids:
                raise ValueError(
                    "Existing predictions file contains example "
                    f"{row['example_id']} not present in this eval."
                )
        existing_rows, duplicate_rows = _dedupe_rows_by_example_id(existing_rows)
        if duplicate_rows > 0:
            _rewrite_predictions_file(predictions_path, existing_rows)
        completed_ids = {row["example_id"] for row in existing_rows}
        resumed_from_partial = bool(existing_rows)
        summary_path = destination / "summary.json"
        metrics_path = destination / "metrics.json"
        if (
            len(completed_ids) == len(expected_example_ids)
            and summary_path.exists()
            and metrics_path.exists()
        ):
            return EvalArtifacts(
                summary_path=str(summary_path.resolve()),
                metrics_path=str(metrics_path.resolve()),
                predictions_path=str(predictions_path.resolve()),
            )

        serialize_prompts = bool(eval_config.get("serialize_with_chat_template", False))
        backend = str(eval_config.get("prediction_backend", "oracle"))
        requires_model = backend == "model_greedy"
        if eval_config.get("transplant") and not requires_model:
            raise ValueError(
                "eval.transplant is only supported with prediction_backend=model_greedy."
            )
        max_new_tokens = int(eval_config.get("max_new_tokens", 64))
        batch_size = max(1, int(eval_config.get("batch_size", 1)))
        serialized_preview: str | None = None
        load_report = None
        tokenizer = None
        model = None
        device = None
        loaded_transplant_layers = []
        if serialize_prompts or requires_model:
            load_report = probe_gemma_loading(
                config.get("model", {}),
                require_tokenizer=True,
                require_chat_template=True,
            )
            if load_report.status != "passed":
                raise RuntimeError(load_report.message)
            tokenizer = load_gemma_tokenizer(config.get("model", {}))
        if requires_model:
            device = _device_from_eval_config(config)
            model = load_gemma_causal_lm(config.get("model", {}))
            model.to(device)
            loaded_transplant_layers = load_sparse_transplant_layers(
                transplant_config=dict(eval_config.get("transplant", {})),
                device=device,
            )

        if tokenizer is not None and serialized_preview is None and examples:
            preview_prompt = build_prompt_content(
                user_request=examples[0].user_request,
                tools=examples[0].tools,
                target=examples[0].gold,
            )
            serialized_preview = render_chat_prompt(
                prompt=preview_prompt,
                tokenizer=tokenizer,
                add_generation_prompt=True,
            )

        rows = list(existing_rows)
        strict_success_count = sum(1 for row in rows if row["score"]["strict_correct"])
        heartbeat.start(
            completed_units=len(rows),
            message=(
                "Resumed evaluation from partial predictions."
                if resumed_from_partial
                else "Started evaluation."
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
            heartbeat.mark_completed(
                completed_units=len(rows),
                message="Evaluation artifacts already complete; reused existing predictions.",
                extra={
                    "prediction_backend": backend,
                    "deduped_existing_prediction_rows": duplicate_rows,
                },
            )
            summary_path, metrics_path = _write_eval_outputs(
                destination=destination,
                manifest_payload=manifest_payload,
                rows=rows,
                backend=backend,
                serialize_prompts=serialize_prompts,
                load_report=load_report,
                serialized_preview=serialized_preview,
                device=device,
                max_new_tokens=max_new_tokens,
                split_filter=split_filter,
                variant_filter=variant_filter,
                transplant_layers=[layer.to_summary() for layer in loaded_transplant_layers],
                status="passed",
                resumed_from_partial=resumed_from_partial,
                heartbeat=heartbeat,
            )
            return EvalArtifacts(
                summary_path=str(summary_path.resolve()),
                metrics_path=str(metrics_path.resolve()),
                predictions_path=str(predictions_path.resolve()),
            )
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
                                "Model-backed eval requires tokenizer, model, and device."
                            )

                        prompt_rows: list[tuple[Any, str]] = []
                        for example in pending_examples:
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
                            if serialized_preview is None:
                                serialized_preview = prompt_text
                            prompt_rows.append((example, prompt_text))

                        for start in range(0, len(prompt_rows), batch_size):
                            batch = prompt_rows[start : start + batch_size]
                            batch_examples = [example for example, _ in batch]
                            batch_prompt_texts = [prompt_text for _, prompt_text in batch]
                            raw_outputs = _greedy_model_outputs(
                                prompt_texts=batch_prompt_texts,
                                tokenizer=tokenizer,
                                model=model,
                                device=device,
                                max_new_tokens=max_new_tokens,
                            )
                            for example, raw_output in zip(
                                batch_examples, raw_outputs, strict=True
                            ):
                                score = score_prediction(raw_output=raw_output, example=example)
                                row = {
                                    "example_id": example.example_id,
                                    "split": example.split,
                                    "variant": str(example.meta.get("variant", "canonical")),
                                    "raw_output": raw_output,
                                    "score": asdict(score),
                                }
                                rows.append(row)
                                strict_success_count += int(row["score"]["strict_correct"])
                                handle.write(json.dumps(row, sort_keys=True) + "\n")
                            heartbeat_payload = heartbeat.maybe_update(
                                completed_units=len(rows),
                                message=f"Processed eval example {len(rows)} of {len(examples)}.",
                                metrics={
                                    "strict_full_call_success_so_far": (
                                        strict_success_count / len(rows) if rows else 0.0
                                    )
                                },
                                extra={
                                    "last_example_id": batch_examples[-1].example_id,
                                    "deduped_existing_prediction_rows": duplicate_rows,
                                },
                            )
                            if heartbeat_payload:
                                handle.flush()
                    else:
                        for example in pending_examples:
                            raw_output = _prediction_backend_output(example, backend)
                            score = score_prediction(raw_output=raw_output, example=example)
                            row = {
                                "example_id": example.example_id,
                                "split": example.split,
                                "variant": str(example.meta.get("variant", "canonical")),
                                "raw_output": raw_output,
                                "score": asdict(score),
                            }
                            rows.append(row)
                            strict_success_count += int(row["score"]["strict_correct"])
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            heartbeat_payload = heartbeat.maybe_update(
                                completed_units=len(rows),
                                message=f"Processed eval example {len(rows)} of {len(examples)}.",
                                metrics={
                                    "strict_full_call_success_so_far": (
                                        strict_success_count / len(rows) if rows else 0.0
                                    )
                                },
                                extra={
                                    "last_example_id": example.example_id,
                                    "deduped_existing_prediction_rows": duplicate_rows,
                                },
                            )
                            if heartbeat_payload:
                                handle.flush()
        except KeyboardInterrupt:
            summary_path, metrics_path = _write_eval_outputs(
                destination=destination,
                manifest_payload=manifest_payload,
                rows=rows,
                backend=backend,
                serialize_prompts=serialize_prompts,
                load_report=load_report,
                serialized_preview=serialized_preview,
                device=device,
                max_new_tokens=max_new_tokens,
                split_filter=split_filter,
                variant_filter=variant_filter,
                transplant_layers=[layer.to_summary() for layer in loaded_transplant_layers],
                status="interrupted",
                resumed_from_partial=resumed_from_partial,
                heartbeat=heartbeat,
            )
            heartbeat.mark_interrupted(
                completed_units=len(rows),
                message="Evaluation interrupted; partial predictions are resumable.",
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

        if model is not None:
            model.to("cpu")
            del model
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()
        heartbeat.mark_completed(
            completed_units=len(rows),
            message="Evaluation completed successfully.",
            extra={
                "prediction_backend": backend,
                "deduped_existing_prediction_rows": duplicate_rows,
            },
        )
        summary_path, metrics_path = _write_eval_outputs(
            destination=destination,
            manifest_payload=manifest_payload,
            rows=rows,
            backend=backend,
            serialize_prompts=serialize_prompts,
            load_report=load_report,
            serialized_preview=serialized_preview,
            device=device,
            max_new_tokens=max_new_tokens,
            split_filter=split_filter,
            variant_filter=variant_filter,
            transplant_layers=[layer.to_summary() for layer in loaded_transplant_layers],
            status="passed",
            resumed_from_partial=resumed_from_partial,
            heartbeat=heartbeat,
        )

        return EvalArtifacts(
            summary_path=str(summary_path.resolve()),
            metrics_path=str(metrics_path.resolve()),
            predictions_path=str(predictions_path.resolve()),
        )
    finally:
        _release_eval_lock(lock_path)
