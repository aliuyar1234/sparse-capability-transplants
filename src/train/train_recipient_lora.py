from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data.manifest import load_examples, load_manifest_payload
from src.eval.run_eval import run_eval_pipeline
from src.models.load_gemma import (
    load_gemma_causal_lm,
    load_gemma_tokenizer,
    probe_gemma_loading,
)
from src.train.train_donor_fullft import (
    _collate_batch,
    _device_from_config,
    _encode_training_example,
    _mean_loss,
    _train_batches,
    _write_json,
)
from src.utils.progress import RunHeartbeat


def _count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _resume_state_path(output_dir: Path) -> Path:
    return output_dir / "resume_checkpoint" / "state.pt"


def _save_lora_resume_state(
    *,
    model: Any,
    optimizer: Any,
    output_dir: Path,
    global_step: int,
    train_step_losses: list[float],
    initial_eval_loss: float | None,
) -> str:
    resume_dir = output_dir / "resume_checkpoint"
    resume_dir.mkdir(parents=True, exist_ok=True)
    state_path = _resume_state_path(output_dir)
    torch.save(
        {
            "global_step": global_step,
            "train_step_losses": train_step_losses,
            "initial_eval_loss": initial_eval_loss,
            "optimizer_state": optimizer.state_dict(),
            "model_state": model.state_dict(),
        },
        state_path,
    )
    return str(state_path.resolve())


def run_recipient_lora_smoke_training(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model

    data_config = config.get("data", {})
    manifest_path = data_config.get("train_manifest_path")
    if not manifest_path:
        raise ValueError("Config is missing data.train_manifest_path.")

    train_split = str(data_config.get("train_split", "calib"))
    eval_split = str(data_config.get("eval_split", "val"))
    manifest_payload = load_manifest_payload(manifest_path)
    examples = load_examples(manifest_payload["dataset_path"])
    train_examples = [example for example in examples if example.split == train_split]
    eval_examples = [example for example in examples if example.split == eval_split]
    if not train_examples:
        raise ValueError(f"Canonical manifest does not contain any {train_split!r} examples.")

    smoke_config = config.get("train", {}).get("smoke", {})
    max_examples = int(smoke_config.get("max_examples", 4))
    max_eval_examples = int(smoke_config.get("max_eval_examples", max_examples))
    max_length = int(smoke_config.get("max_length", 1024))
    batch_size = int(smoke_config.get("batch_size", 1))
    learning_rate = float(smoke_config.get("learning_rate", 1e-4))
    max_steps = int(smoke_config.get("max_steps", 2))
    epochs = int(smoke_config.get("epochs", 1))
    grad_clip = float(smoke_config.get("grad_clip", 1.0))
    eval_sample_size = int(smoke_config.get("eval_sample_size", 3))
    eval_batch_size = int(smoke_config.get("eval_batch_size", batch_size))
    max_new_tokens = int(smoke_config.get("max_new_tokens", 64))
    save_checkpoint = bool(smoke_config.get("save_checkpoint", True))
    checkpoint_max_shard_size = str(smoke_config.get("checkpoint_max_shard_size", "2GB"))
    checkpoint_interval_steps = int(smoke_config.get("checkpoint_interval_steps", 10))
    heartbeat_interval_seconds = float(smoke_config.get("heartbeat_interval_seconds", 10.0))

    selected_train = train_examples[:max_examples]
    selected_eval = eval_examples[:max_eval_examples]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    load_report = probe_gemma_loading(
        config.get("model", {}),
        require_tokenizer=True,
        require_chat_template=True,
    )
    if load_report.status != "passed":
        raise RuntimeError(load_report.message)

    device = _device_from_config(config, profile_config=smoke_config)
    tokenizer = load_gemma_tokenizer(config.get("model", {}))
    base_model = load_gemma_causal_lm(config.get("model", {}))
    lora_config = config.get("lora", {})
    model = get_peft_model(
        base_model,
        LoraConfig(
            r=int(lora_config.get("rank", 2)),
            lora_alpha=int(lora_config.get("alpha", 4)),
            lora_dropout=float(lora_config.get("dropout", 0.0)),
            bias=str(lora_config.get("bias", "none")),
            task_type="CAUSAL_LM",
            target_modules=list(
                lora_config.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
        ),
    )
    resume_state_path = _resume_state_path(destination)
    resume_payload = None
    if resume_state_path.exists():
        resume_payload = torch.load(resume_state_path, map_location="cpu")
        model.load_state_dict(resume_payload["model_state"], strict=False)
    model.to(device)
    original_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        original_use_cache = bool(model.config.use_cache)
        model.config.use_cache = False
    trainable_parameter_count = _count_trainable_parameters(model)

    encoded_train = [
        _encode_training_example(example=example, tokenizer=tokenizer, max_length=max_length)
        for example in selected_train
    ]
    encoded_eval = [
        _encode_training_example(example=example, tokenizer=tokenizer, max_length=max_length)
        for example in selected_eval
    ]
    preview_rows = [
        {
            "example_id": item["example_id"],
            "variant": item["variant"],
            "serialized_prompt_preview": item["full_text"][:400],
            "serialized_prompt_length": len(item["full_text"]),
            "label_token_count": int((item["labels"] != -100).sum().item()),
        }
        for item in [*encoded_train, *encoded_eval]
    ]

    train_batches = _train_batches(encoded_train, batch_size=batch_size)
    if not train_batches:
        raise ValueError("Recipient LoRA selection produced no batches.")
    target_global_steps = min(max_steps, len(train_batches) * max(epochs, 1))
    heartbeat = RunHeartbeat(
        output_dir=destination,
        phase="train_recipient_lora",
        total_units=target_global_steps,
        unit_name="steps",
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    resumed_from_checkpoint = resume_payload is not None
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
    initial_eval_loss = (
        resume_payload.get("initial_eval_loss")
        if resume_payload is not None
        else _mean_loss(
            model=model,
            encoded_examples=encoded_eval,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
    )
    train_step_losses = (
        list(resume_payload.get("train_step_losses", [])) if resume_payload is not None else []
    )
    global_step = int(resume_payload.get("global_step", 0)) if resume_payload is not None else 0
    preview_path = destination / "serialized_examples.json"
    preview_path.write_text(
        json.dumps(preview_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    train_trace_path = destination / "train_trace.json"
    heartbeat.start(
        completed_units=global_step,
        message=(
            "Resumed recipient LoRA baseline from checkpoint."
            if resumed_from_checkpoint
            else "Started recipient LoRA baseline."
        ),
        metrics={"initial_eval_loss": initial_eval_loss},
    )

    adapter_checkpoint_dir = None
    merged_checkpoint_dir = None
    post_train_eval_artifacts = None
    final_eval_loss = None
    resume_state_value = None
    try:
        model.train()
        while global_step < target_global_steps:
            batch = _collate_batch(
                train_batches[global_step % len(train_batches)],
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
                message=f"Completed recipient LoRA step {global_step} of {target_global_steps}.",
                metrics={"train_loss": loss_value, "initial_eval_loss": initial_eval_loss},
            )
            if global_step % max(1, checkpoint_interval_steps) == 0:
                _save_lora_resume_state(
                    model=model,
                    optimizer=optimizer,
                    output_dir=destination,
                    global_step=global_step,
                    train_step_losses=train_step_losses,
                    initial_eval_loss=initial_eval_loss,
                )

        final_eval_loss = _mean_loss(
            model=model,
            encoded_examples=encoded_eval,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
        )
        _write_json(
            train_trace_path,
            {
                "initial_eval_loss": initial_eval_loss,
                "final_eval_loss": final_eval_loss,
                "train_step_losses": train_step_losses,
                "global_step": global_step,
                "target_global_steps": target_global_steps,
                "trainable_parameter_count": trainable_parameter_count,
                "resumed_from_checkpoint": resumed_from_checkpoint,
            },
        )
        resume_state_value = _save_lora_resume_state(
            model=model,
            optimizer=optimizer,
            output_dir=destination,
            global_step=global_step,
            train_step_losses=train_step_losses,
            initial_eval_loss=initial_eval_loss,
        )
    except KeyboardInterrupt:
        resume_state_value = _save_lora_resume_state(
            model=model,
            optimizer=optimizer,
            output_dir=destination,
            global_step=global_step,
            train_step_losses=train_step_losses,
            initial_eval_loss=initial_eval_loss,
        )
        _write_json(
            train_trace_path,
            {
                "initial_eval_loss": initial_eval_loss,
                "final_eval_loss": None,
                "train_step_losses": train_step_losses,
                "global_step": global_step,
                "target_global_steps": target_global_steps,
                "trainable_parameter_count": trainable_parameter_count,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "status": "interrupted",
            },
        )
        heartbeat.mark_interrupted(
            completed_units=global_step,
            message="Recipient LoRA baseline interrupted; resumable state preserved.",
        )
        interrupted_summary = {
            "status": "interrupted",
            "baseline_kind": str(config.get("train", {}).get("baseline_kind", "recipient_lora")),
            "source_manifest_id": manifest_payload["manifest_id"],
            "source_manifest_hash": manifest_payload["manifest_hash"],
            "train_split": train_split,
            "eval_split": eval_split,
            "train_example_count": len(train_examples),
            "eval_example_count": len(eval_examples),
            "smoke_serialized_example_count": len(preview_rows),
            "loader_report": load_report.to_dict(),
            "device": str(device),
            "global_step": global_step,
            "target_global_steps": target_global_steps,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "initial_eval_loss": initial_eval_loss,
            "final_eval_loss": None,
            "train_step_losses": train_step_losses,
            "trainable_parameter_count": trainable_parameter_count,
            "serialized_examples_path": str(preview_path.resolve()),
            "train_trace_path": str(train_trace_path.resolve()),
            "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
            "progress_path": str(heartbeat.paths.progress_path.resolve()),
            "resume_state_path": resume_state_value,
            "adapter_checkpoint_dir": None,
            "merged_checkpoint_dir": None,
            "post_train_eval_summary_path": None,
            "post_train_eval_metrics_path": None,
            "post_train_eval_predictions_path": None,
            "notes": [
                "This is an M2 recipient LoRA smoke run only.",
                "No recipient-baseline or transplant claim is supported by this run.",
            ],
        }
        summary_path = destination / "summary.json"
        _write_json(summary_path, interrupted_summary)
        raise

    if save_checkpoint:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        adapter_checkpoint_dir = destination / "adapter_checkpoint"
        adapter_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_checkpoint_dir)
        tokenizer.save_pretrained(adapter_checkpoint_dir)

        merged_checkpoint_dir = destination / "merged_checkpoint"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(
            merged_checkpoint_dir,
            safe_serialization=True,
            max_shard_size=checkpoint_max_shard_size,
        )
        tokenizer.save_pretrained(merged_checkpoint_dir)

    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if merged_checkpoint_dir is not None:
        post_train_eval_artifacts = run_eval_pipeline(
            config={
                "model": {
                    **dict(config.get("model", {})),
                    "local_path": str(merged_checkpoint_dir.resolve()),
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
        message="Recipient LoRA baseline completed successfully.",
        metrics={"final_eval_loss": final_eval_loss, "initial_eval_loss": initial_eval_loss},
    )

    summary = {
        "status": "passed",
        "baseline_kind": str(config.get("train", {}).get("baseline_kind", "recipient_lora")),
        "source_manifest_id": manifest_payload["manifest_id"],
        "source_manifest_hash": manifest_payload["manifest_hash"],
        "train_split": train_split,
        "eval_split": eval_split,
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "smoke_serialized_example_count": len(preview_rows),
        "loader_report": load_report.to_dict(),
        "device": str(device),
        "global_step": global_step,
        "target_global_steps": target_global_steps,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "train_step_losses": train_step_losses,
        "trainable_parameter_count": trainable_parameter_count,
        "serialized_examples_path": str(preview_path.resolve()),
        "train_trace_path": str(train_trace_path.resolve()),
        "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
        "progress_path": str(heartbeat.paths.progress_path.resolve()),
        "resume_state_path": resume_state_value,
        "adapter_checkpoint_dir": (
            None if adapter_checkpoint_dir is None else str(adapter_checkpoint_dir.resolve())
        ),
        "merged_checkpoint_dir": (
            None if merged_checkpoint_dir is None else str(merged_checkpoint_dir.resolve())
        ),
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
            "This is an M2 recipient LoRA smoke run only.",
            "No recipient-baseline or transplant claim is supported by this run.",
        ],
    }
    summary_path = destination / "summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.resolve())
    return summary
