from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.canonical import CanonicalExample
from src.data.manifest import load_examples, load_manifest_payload
from src.eval.run_eval import run_eval_pipeline
from src.models.format_prompts import (
    build_chat_messages,
    build_prompt_content,
    build_training_chat_messages,
)
from src.models.load_gemma import (
    load_gemma_causal_lm,
    load_gemma_tokenizer,
    probe_gemma_loading,
)
from src.utils.progress import RunHeartbeat


def _train_profile_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    train_config = dict(config.get("train", {}))
    if "main" in train_config:
        return "main", dict(train_config.get("main", {}))
    return "smoke", dict(train_config.get("smoke", {}))


def _device_from_config(config: dict[str, Any], *, profile_config: dict[str, Any]) -> torch.device:
    device_name = str(profile_config.get("device", "auto")).lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _encode_training_example(
    *,
    example: CanonicalExample,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    prompt = build_prompt_content(
        user_request=example.user_request,
        tools=example.tools,
        target=example.gold,
    )
    prefix_text = str(
        tokenizer.apply_chat_template(
            build_chat_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    full_text = str(
        tokenizer.apply_chat_template(
            build_training_chat_messages(prompt),
            tokenize=False,
            add_generation_prompt=False,
        )
    )

    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"][0]
    full_encoding = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = full_encoding["input_ids"][0]
    attention_mask = full_encoding["attention_mask"][0]
    prefix_len = min(prefix_ids.numel(), input_ids.numel())

    labels = input_ids.clone()
    labels[:prefix_len] = -100
    if torch.all(labels == -100):
        raise ValueError(
            f"Training example {example.example_id} lost the assistant target after truncation."
        )

    return {
        "example_id": example.example_id,
        "variant": str(example.meta.get("variant", "canonical")),
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prefix_text": prefix_text,
        "full_text": full_text,
    }


def _collate_batch(
    batch: list[dict[str, Any]],
    *,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pad_token_id = int(tokenizer.pad_token_id)
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - item["input_ids"].numel()
        input_ids.append(
            torch.cat(
                [item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)],
                dim=0,
            )
        )
        attention_mask.append(
            torch.cat(
                [item["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)],
                dim=0,
            )
        )
        labels.append(
            torch.cat(
                [item["labels"], torch.full((pad_len,), -100, dtype=torch.long)],
                dim=0,
            )
        )

    return {
        "input_ids": torch.stack(input_ids).to(device),
        "attention_mask": torch.stack(attention_mask).to(device),
        "labels": torch.stack(labels).to(device),
    }


def _mean_loss(
    *,
    model: Any,
    encoded_examples: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> float | None:
    if not encoded_examples:
        return None
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(encoded_examples), batch_size):
            batch = _collate_batch(
                encoded_examples[start : start + batch_size],
                tokenizer=tokenizer,
                device=device,
            )
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    return sum(losses) / len(losses)


def _save_smoke_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    max_shard_size: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    return output_dir.resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _train_batches(
    encoded_examples: list[dict[str, Any]],
    *,
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    return [
        encoded_examples[start : start + batch_size]
        for start in range(0, len(encoded_examples), batch_size)
    ]


def _resume_checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / "resume_checkpoint"


def _resume_model_dir(output_dir: Path) -> Path:
    return _resume_checkpoint_dir(output_dir) / "model"


def _resume_state_path(output_dir: Path) -> Path:
    return _resume_checkpoint_dir(output_dir) / "state.pt"


def _load_fullft_resume_state(
    *,
    config: dict[str, Any],
    output_dir: Path,
    learning_rate: float,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    resume_state_path = _resume_state_path(output_dir)
    resume_model_dir = _resume_model_dir(output_dir)
    if resume_state_path.exists():
        state_payload = torch.load(resume_state_path, map_location="cpu")
        loader_config = dict(config.get("model", {}))
        tokenizer = load_gemma_tokenizer(loader_config)
        model = load_gemma_causal_lm(loader_config)
        if "model_state" in state_payload:
            model.load_state_dict(state_payload["model_state"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(state_payload["optimizer_state"])
            return loader_config, tokenizer, model, state_payload
        if resume_model_dir.exists():
            legacy_loader_config = {
                **dict(config.get("model", {})),
                "local_path": str(resume_model_dir),
                "local_files_only": True,
            }
            tokenizer = load_gemma_tokenizer(legacy_loader_config)
            model = load_gemma_causal_lm(legacy_loader_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            optimizer.load_state_dict(state_payload["optimizer_state"])
            return legacy_loader_config, tokenizer, model, state_payload

    loader_config = dict(config.get("model", {}))
    tokenizer = load_gemma_tokenizer(loader_config)
    model = load_gemma_causal_lm(loader_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return (
        loader_config,
        tokenizer,
        model,
        {
            "global_step": 0,
            "train_step_losses": [],
            "initial_val_loss": None,
        },
    )


def _save_fullft_resume_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    output_dir: Path,
    max_shard_size: str,
    global_step: int,
    train_step_losses: list[float],
    initial_val_loss: float | None,
    train_profile: str,
) -> dict[str, str]:
    resume_dir = _resume_checkpoint_dir(output_dir)
    resume_dir.mkdir(parents=True, exist_ok=True)
    state_path = _resume_state_path(output_dir)
    torch.save(
        {
            "model_state": model.state_dict(),
            "global_step": global_step,
            "train_step_losses": train_step_losses,
            "initial_val_loss": initial_val_loss,
            "optimizer_state": optimizer.state_dict(),
            "train_profile": train_profile,
        },
        state_path,
    )
    return {
        "resume_checkpoint_dir": str(resume_dir.resolve()),
        "resume_model_dir": str(_resume_model_dir(output_dir).resolve()),
        "resume_state_path": str(state_path.resolve()),
    }


def _train_summary(
    *,
    status: str,
    train_profile: str,
    manifest_payload: dict[str, Any],
    train_examples: list[CanonicalExample],
    val_examples: list[CanonicalExample],
    preview_rows: list[dict[str, Any]],
    load_report: Any,
    device: torch.device,
    global_step: int,
    target_global_steps: int,
    initial_val_loss: float | None,
    final_val_loss: float | None,
    train_step_losses: list[float],
    preview_path: Path,
    train_trace_path: Path,
    heartbeat: RunHeartbeat,
    resumed_from_checkpoint: bool,
    checkpoint_dir: Path | None,
    resume_artifacts: dict[str, str] | None,
    post_train_eval_artifacts: Any,
    eval_split: str | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "train_profile": train_profile,
        "source_manifest_id": manifest_payload["manifest_id"],
        "source_manifest_hash": manifest_payload["manifest_hash"],
        "train_example_count": len(train_examples),
        "val_example_count": len(val_examples),
        "smoke_serialized_example_count": len(preview_rows),
        "loader_report": load_report.to_dict(),
        "device": str(device),
        "global_step": global_step,
        "target_global_steps": target_global_steps,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "train_step_losses": train_step_losses,
        "serialized_examples_path": str(preview_path.resolve()),
        "train_trace_path": str(train_trace_path.resolve()),
        "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
        "progress_path": str(heartbeat.paths.progress_path.resolve()),
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir.resolve()),
        "resume_checkpoint_dir": (
            None if resume_artifacts is None else resume_artifacts["resume_checkpoint_dir"]
        ),
        "resume_state_path": (
            None if resume_artifacts is None else resume_artifacts["resume_state_path"]
        ),
        "post_train_eval_split": eval_split if checkpoint_dir is not None else None,
        "post_train_eval_summary_path": (
            None if post_train_eval_artifacts is None else post_train_eval_artifacts.summary_path
        ),
        "post_train_eval_metrics_path": (
            None if post_train_eval_artifacts is None else post_train_eval_artifacts.metrics_path
        ),
        "post_train_eval_predictions_path": (
            None
            if post_train_eval_artifacts is None
            else post_train_eval_artifacts.predictions_path
        ),
        "notes": [
            (
                "This is an M2 donor smoke training run only."
                if train_profile == "smoke"
                else "This is an M3 donor full-FT training run."
            ),
            (
                "No donor-gap or baseline claim is supported by this run."
                if train_profile == "smoke"
                else "Training alone does not support any donor-gap or transplant claim."
            ),
        ],
    }


def run_donor_training(*, config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    data_config = config.get("data", {})
    manifest_path = data_config.get("train_manifest_path")
    if not manifest_path:
        raise ValueError("Config is missing data.train_manifest_path.")

    manifest_payload = load_manifest_payload(manifest_path)
    examples = load_examples(manifest_payload["dataset_path"])
    train_examples = [example for example in examples if example.split == "train"]
    val_examples = [example for example in examples if example.split == "val"]
    if not train_examples:
        raise ValueError("Canonical manifest does not contain any train examples.")

    train_profile, profile_config = _train_profile_config(config)
    max_examples = int(profile_config.get("max_examples", len(train_examples)))
    max_val_examples = int(profile_config.get("max_val_examples", len(val_examples)))
    max_length = int(profile_config.get("max_length", 1024))
    batch_size = int(profile_config.get("batch_size", 1))
    learning_rate = float(profile_config.get("learning_rate", 1e-5))
    max_steps = int(profile_config.get("max_steps", 2))
    epochs = int(profile_config.get("epochs", 1))
    grad_clip = float(profile_config.get("grad_clip", 1.0))
    eval_sample_size = int(profile_config.get("eval_sample_size", 3))
    eval_batch_size = int(profile_config.get("eval_batch_size", batch_size))
    max_new_tokens = int(profile_config.get("max_new_tokens", 64))
    save_checkpoint = bool(profile_config.get("save_checkpoint", True))
    checkpoint_max_shard_size = str(profile_config.get("checkpoint_max_shard_size", "2GB"))
    checkpoint_interval_steps = int(profile_config.get("checkpoint_interval_steps", 50))
    heartbeat_interval_seconds = float(profile_config.get("heartbeat_interval_seconds", 10.0))

    selected_train = train_examples[:max_examples]
    selected_val = val_examples[:max_val_examples]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    loader_config, tokenizer, model, resume_state = _load_fullft_resume_state(
        config=config,
        output_dir=destination,
        learning_rate=learning_rate,
    )
    resumed_from_checkpoint = bool(resume_state.get("global_step", 0))
    load_report = probe_gemma_loading(
        loader_config,
        require_tokenizer=True,
        require_chat_template=True,
    )
    if load_report.status != "passed":
        raise RuntimeError(load_report.message)

    device = _device_from_config(config, profile_config=profile_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if resumed_from_checkpoint:
        optimizer.load_state_dict(resume_state["optimizer_state"])
    model.to(device)
    original_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        original_use_cache = bool(model.config.use_cache)
        model.config.use_cache = False

    encoded_train = [
        _encode_training_example(example=example, tokenizer=tokenizer, max_length=max_length)
        for example in selected_train
    ]
    encoded_val = [
        _encode_training_example(example=example, tokenizer=tokenizer, max_length=max_length)
        for example in selected_val
    ]

    preview_rows = [
        {
            "example_id": item["example_id"],
            "variant": item["variant"],
            "serialized_prompt_preview": item["full_text"][:400],
            "serialized_prompt_length": len(item["full_text"]),
            "label_token_count": int((item["labels"] != -100).sum().item()),
        }
        for item in [*encoded_train, *encoded_val]
    ]
    train_batches = _train_batches(encoded_train, batch_size=batch_size)
    if not train_batches:
        raise ValueError("Training selection produced no batches.")
    target_global_steps = min(max_steps, len(train_batches) * max(epochs, 1))
    heartbeat = RunHeartbeat(
        output_dir=destination,
        phase="train_donor",
        total_units=target_global_steps,
        unit_name="steps",
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    initial_val_loss = (
        resume_state.get("initial_val_loss")
        if resumed_from_checkpoint
        else _mean_loss(
            model=model,
            encoded_examples=encoded_val,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
    )
    train_step_losses = list(resume_state.get("train_step_losses", []))
    global_step = int(resume_state.get("global_step", 0))

    preview_path = destination / "serialized_examples.json"
    preview_path.write_text(
        json.dumps(preview_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_trace_path = destination / "train_trace.json"
    heartbeat.start(
        completed_units=global_step,
        message=(
            "Resumed donor training from checkpoint."
            if resumed_from_checkpoint
            else "Started donor training."
        ),
        metrics={"initial_val_loss": initial_val_loss},
        extra={"train_profile": train_profile, "target_global_steps": target_global_steps},
    )

    checkpoint_dir = None
    post_train_eval_artifacts = None
    eval_split = "val" if selected_val else "train"
    final_val_loss = None
    resume_artifacts: dict[str, str] | None = None
    try:
        model.train()
        while global_step < target_global_steps:
            batch_examples = train_batches[global_step % len(train_batches)]
            batch = _collate_batch(
                batch_examples,
                tokenizer=tokenizer,
                device=device,
            )
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_value = float(loss.detach().cpu())
            train_step_losses.append(loss_value)
            heartbeat.maybe_update(
                completed_units=global_step,
                message=f"Completed donor step {global_step} of {target_global_steps}.",
                metrics={"train_loss": loss_value, "initial_val_loss": initial_val_loss},
                extra={"train_profile": train_profile},
            )
            if save_checkpoint and (
                global_step % max(1, checkpoint_interval_steps) == 0
                or global_step == target_global_steps
            ):
                resume_artifacts = _save_fullft_resume_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    output_dir=destination,
                    max_shard_size=checkpoint_max_shard_size,
                    global_step=global_step,
                    train_step_losses=train_step_losses,
                    initial_val_loss=initial_val_loss,
                    train_profile=train_profile,
                )

        final_val_loss = _mean_loss(
            model=model,
            encoded_examples=encoded_val,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
        train_trace_payload = {
            "initial_val_loss": initial_val_loss,
            "final_val_loss": final_val_loss,
            "train_step_losses": train_step_losses,
            "global_step": global_step,
            "target_global_steps": target_global_steps,
            "resumed_from_checkpoint": resumed_from_checkpoint,
        }
        _write_json(train_trace_path, train_trace_payload)
    except KeyboardInterrupt:
        if save_checkpoint:
            resume_artifacts = _save_fullft_resume_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                output_dir=destination,
                max_shard_size=checkpoint_max_shard_size,
                global_step=global_step,
                train_step_losses=train_step_losses,
                initial_val_loss=initial_val_loss,
                train_profile=train_profile,
            )
        interrupted_trace_payload = {
            "initial_val_loss": initial_val_loss,
            "final_val_loss": None,
            "train_step_losses": train_step_losses,
            "global_step": global_step,
            "target_global_steps": target_global_steps,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "status": "interrupted",
        }
        _write_json(train_trace_path, interrupted_trace_payload)
        heartbeat.mark_interrupted(
            completed_units=global_step,
            message="Donor training interrupted; resumable checkpoint preserved.",
            metrics={"initial_val_loss": initial_val_loss},
            extra={"train_profile": train_profile},
        )
        interrupted_summary = _train_summary(
            status="interrupted",
            train_profile=train_profile,
            manifest_payload=manifest_payload,
            train_examples=train_examples,
            val_examples=val_examples,
            preview_rows=preview_rows,
            load_report=load_report,
            device=device,
            global_step=global_step,
            target_global_steps=target_global_steps,
            initial_val_loss=initial_val_loss,
            final_val_loss=None,
            train_step_losses=train_step_losses,
            preview_path=preview_path,
            train_trace_path=train_trace_path,
            heartbeat=heartbeat,
            resumed_from_checkpoint=resumed_from_checkpoint,
            checkpoint_dir=None,
            resume_artifacts=resume_artifacts,
            post_train_eval_artifacts=None,
            eval_split=None,
        )
        summary_path = destination / "summary.json"
        _write_json(summary_path, interrupted_summary)
        raise

    checkpoint_dir = None
    if save_checkpoint:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        checkpoint_dir = _save_smoke_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=destination / "checkpoint",
            max_shard_size=checkpoint_max_shard_size,
        )

    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if checkpoint_dir is not None:
        post_train_eval_artifacts = run_eval_pipeline(
            config={
                "model": {
                    **loader_config,
                    "local_path": str(checkpoint_dir),
                },
                "eval": {
                    "manifest_path": str(manifest_path),
                    "prediction_backend": "model_greedy",
                    "max_examples": eval_sample_size,
                    "batch_size": eval_batch_size,
                    "max_new_tokens": max_new_tokens,
                    "device": str(device),
                    "split_filter": eval_split,
                },
            },
            output_dir=destination / "post_train_eval",
        )

    heartbeat.mark_completed(
        completed_units=global_step,
        message="Donor training completed successfully.",
        metrics={"final_val_loss": final_val_loss, "initial_val_loss": initial_val_loss},
        extra={"train_profile": train_profile},
    )
    summary = _train_summary(
        status="passed",
        train_profile=train_profile,
        manifest_payload=manifest_payload,
        train_examples=train_examples,
        val_examples=val_examples,
        preview_rows=preview_rows,
        load_report=load_report,
        device=device,
        global_step=global_step,
        target_global_steps=target_global_steps,
        initial_val_loss=initial_val_loss,
        final_val_loss=final_val_loss,
        train_step_losses=train_step_losses,
        preview_path=preview_path,
        train_trace_path=train_trace_path,
        heartbeat=heartbeat,
        resumed_from_checkpoint=resumed_from_checkpoint,
        checkpoint_dir=checkpoint_dir,
        resume_artifacts=resume_artifacts,
        post_train_eval_artifacts=post_train_eval_artifacts,
        eval_split=eval_split,
    )
    summary_path = destination / "summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.resolve())

    return summary


def run_donor_smoke_training(*, config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    return run_donor_training(config=config, output_dir=output_dir)
