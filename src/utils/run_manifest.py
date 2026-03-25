from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import config_hash, ensure_execution_variant


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def build_run_id(
    *,
    execution_variant: str,
    slot_id: str,
    milestone: str,
    experiment_name: str,
    seed: int,
    timestamp: datetime | None = None,
) -> str:
    moment = timestamp or datetime.now(timezone.utc)
    date_prefix = moment.strftime("%Y%m%d")
    slot_slug = _slugify(slot_id)
    milestone_slug = _slugify(milestone)
    experiment_slug = _slugify(experiment_name)
    return (
        f"{date_prefix}_{execution_variant}_{slot_slug}_{milestone_slug}_{experiment_slug}_s{seed}"
    )


def _git_commit_or_none(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _run_suffix(
    *,
    execution_variant: str,
    slot_id: str,
    milestone: str,
    experiment_name: str,
    seed: int,
) -> str:
    return (
        f"_{execution_variant}_{_slugify(slot_id)}_{_slugify(milestone)}"
        f"_{_slugify(experiment_name)}_s{seed}"
    )


def _find_resumable_manifest(
    *,
    output_root: Path,
    execution_variant: str,
    slot_id: str,
    milestone: str,
    experiment_name: str,
    seed: int,
    expected_config_hash: str,
) -> dict[str, Any] | None:
    if not output_root.exists():
        return None

    suffix = _run_suffix(
        execution_variant=execution_variant,
        slot_id=slot_id,
        milestone=milestone,
        experiment_name=experiment_name,
        seed=seed,
    )
    for candidate in sorted(output_root.iterdir(), reverse=True):
        if not candidate.is_dir() or not candidate.name.endswith(suffix):
            continue
        manifest_path = candidate / "run_manifest.json"
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("config_hash") != expected_config_hash:
            continue
        if payload.get("status") not in {"initialized", "running", "interrupted"}:
            continue
        return payload
    return None


def create_run_manifest(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    command: list[str],
) -> dict[str, Any]:
    ensure_execution_variant(config)
    config_file = Path(config_path).resolve()
    repo_root = Path.cwd().resolve()
    timestamp = datetime.now(timezone.utc)
    current_config_hash = config_hash(config)
    output_root = Path(config.get("run", {}).get("output_root", "runs")).resolve()

    resumable_manifest = _find_resumable_manifest(
        output_root=output_root,
        execution_variant=config["execution_variant"],
        slot_id=str(config.get("slot_id", "bootstrap")),
        milestone=str(config.get("milestone", "M0")),
        experiment_name=str(config.get("experiment_name", "run")),
        seed=int(config.get("seed", 17)),
        expected_config_hash=current_config_hash,
    )
    if resumable_manifest is not None:
        resume_count = int(resumable_manifest.get("resume_count", 0)) + 1
        resumable_manifest.update(
            {
                "status": "running",
                "command": command,
                "config_path": str(config_file),
                "config_hash": current_config_hash,
                "git_commit": _git_commit_or_none(repo_root),
                "hostname": socket.gethostname(),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "cwd": str(repo_root),
                "argv": sys.argv,
                "resumed_at_utc": timestamp.isoformat(),
                "resume_count": resume_count,
            }
        )
        return resumable_manifest

    run_id = build_run_id(
        execution_variant=config["execution_variant"],
        slot_id=str(config.get("slot_id", "bootstrap")),
        milestone=str(config.get("milestone", "M0")),
        experiment_name=str(config.get("experiment_name", "run")),
        seed=int(config.get("seed", 17)),
        timestamp=timestamp,
    )
    run_dir = output_root / run_id

    return {
        "run_id": run_id,
        "milestone": str(config.get("milestone", "M0")),
        "execution_variant": config["execution_variant"],
        "slot_id": str(config.get("slot_id", "bootstrap")),
        "experiment_name": str(config.get("experiment_name", "run")),
        "seed": int(config.get("seed", 17)),
        "planned_gpuh": float(config.get("planned_gpuh", 0.0)),
        "actual_gpuh": float(config.get("actual_gpuh", 0.0)),
        "prompt_contract_version": str(config.get("prompt_contract_version", "unknown")),
        "status": "initialized",
        "command": command,
        "config_path": str(config_file),
        "config_hash": current_config_hash,
        "git_commit": _git_commit_or_none(repo_root),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "started_at_utc": timestamp.isoformat(),
        "resume_count": 0,
        "artifact_paths": {
            "run_dir": str(run_dir),
            "manifest": str(run_dir / "run_manifest.json"),
        },
        "notes": [],
        "artifacts": {},
        "cwd": str(repo_root),
        "argv": sys.argv,
    }


def write_run_manifest(manifest: dict[str, Any]) -> Path:
    run_dir = Path(manifest["artifact_paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def update_run_manifest(manifest_path: str | Path, updates: dict[str, Any]) -> dict[str, Any]:
    path = Path(manifest_path)
    current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return current
