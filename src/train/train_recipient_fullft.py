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
    _load_fullft_resume_state,
    _mean_loss,
    _save_fullft_resume_checkpoint,
    _save_smoke_checkpoint,
    _train_batches,
    _write_json,
)
from src.utils.progress import RunHeartbeat


def run_recipient_fullft_smoke_training(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    data_config = config.get("data", {})
    manifest_path = data_config.get("train_manifest_path")
    if not manifest_path:
        raise ValueError("Config is missing data.train_manifest_path.")

    train_split = str(data_config.get("train_split", "train"))
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
    learning_rate = float(smoke_config.get("learning_rate", 1e-5))
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

    if (destination / "resume_checkpoint" / "state.pt").exists():
        loader_config, tokenizer, model, resume_state = _load_fullft_resume_state(
            config=config,
            output_dir=destination,
            learning_rate=learning_rate,
        )
    else:
        loader_config = dict(config.get("model", {}))
        tokenizer = load_gemma_tokenizer(loader_config)
        model = load_gemma_causal_lm(loader_config)
        resume_state = {
            "global_step": 0,
            "train_step_losses": [],
            "initial_val_loss": None,
        }
    resumed_from_checkpoint = bool(resume_state.get("global_step", 0))

    load_report = probe_gemma_loading(
        loader_config,
        require_tokenizer=True,
        require_chat_template=True,
    )
    if load_report.status != "passed":
        raise RuntimeError(load_report.message)

    device = _device_from_config(config, profile_config=smoke_config)
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
        raise ValueError("Recipient full-FT selection produced no batches.")
    target_global_steps = min(max_steps, len(train_batches) * max(epochs, 1))
    heartbeat = RunHeartbeat(
        output_dir=destination,
        phase="train_recipient_fullft",
        total_units=target_global_steps,
        unit_name="steps",
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    initial_eval_loss = (
        resume_state.get("initial_val_loss")
        if resumed_from_checkpoint
        else _mean_loss(
            model=model,
            encoded_examples=encoded_eval,
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
            "Resumed recipient full-FT baseline from checkpoint."
            if resumed_from_checkpoint
            else "Started recipient full-FT baseline."
        ),
        metrics={"initial_eval_loss": initial_eval_loss},
    )

    checkpoint_dir = None
    post_train_eval_artifacts = None
    resume_artifacts: dict[str, str] | None = None
    final_eval_loss = None
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
                message=(
                    f"Completed recipient full-FT step {global_step} of {target_global_steps}."
                ),
                metrics={"train_loss": loss_value, "initial_eval_loss": initial_eval_loss},
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
                    initial_val_loss=initial_eval_loss,
                    train_profile="recipient_fullft",
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
                "resumed_from_checkpoint": resumed_from_checkpoint,
            },
        )
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
                initial_val_loss=initial_eval_loss,
                train_profile="recipient_fullft",
            )
        _write_json(
            train_trace_path,
            {
                "initial_eval_loss": initial_eval_loss,
                "final_eval_loss": None,
                "train_step_losses": train_step_losses,
                "global_step": global_step,
                "target_global_steps": target_global_steps,
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "status": "interrupted",
            },
        )
        heartbeat.mark_interrupted(
            completed_units=global_step,
            message="Recipient full-FT baseline interrupted; resumable checkpoint preserved.",
        )
        interrupted_summary = {
            "status": "interrupted",
            "baseline_kind": str(config.get("train", {}).get("baseline_kind", "recipient_fullft")),
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
            "serialized_examples_path": str(preview_path.resolve()),
            "train_trace_path": str(train_trace_path.resolve()),
            "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
            "progress_path": str(heartbeat.paths.progress_path.resolve()),
            "resume_state_path": (
                None if resume_artifacts is None else resume_artifacts["resume_state_path"]
            ),
            "checkpoint_dir": None,
            "post_train_eval_summary_path": None,
            "post_train_eval_metrics_path": None,
            "post_train_eval_predictions_path": None,
            "notes": [
                "This is an M2 recipient full-FT smoke run only.",
                "No recipient-baseline or transplant claim is supported by this run.",
            ],
        }
        summary_path = destination / "summary.json"
        _write_json(summary_path, interrupted_summary)
        raise

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
        message="Recipient full-FT baseline completed successfully.",
        metrics={"final_eval_loss": final_eval_loss, "initial_eval_loss": initial_eval_loss},
    )

    summary = {
        "status": "passed",
        "baseline_kind": str(config.get("train", {}).get("baseline_kind", "recipient_fullft")),
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
        "serialized_examples_path": str(preview_path.resolve()),
        "train_trace_path": str(train_trace_path.resolve()),
        "heartbeat_path": str(heartbeat.paths.heartbeat_path.resolve()),
        "progress_path": str(heartbeat.paths.progress_path.resolve()),
        "resume_state_path": (
            None if resume_artifacts is None else resume_artifacts["resume_state_path"]
        ),
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
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
            "This is an M2 recipient full-FT smoke run only.",
            "No recipient-baseline or transplant claim is supported by this run.",
        ],
    }
    summary_path = destination / "summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path.resolve())
    return summary
