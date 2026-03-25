from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProbeResult:
    status: str
    message: str
    blocker_code: str | None = None
    loader: str = "transformers"
    model_id: str | None = None
    revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_model_loading(model_config: dict[str, Any]) -> ModelProbeResult:
    loader = str(model_config.get("loader", "transformers"))
    model_id = model_config.get("id")
    revision = model_config.get("revision")
    local_files_only = bool(model_config.get("local_files_only", True))
    local_path_raw = model_config.get("local_path")
    local_path = Path(local_path_raw) if local_path_raw else None

    if loader != "transformers":
        return ModelProbeResult(
            status="blocked",
            loader=loader,
            model_id=model_id,
            revision=revision,
            blocker_code="unsupported_loader",
            message=(
                f"Unsupported smoke loader {loader!r}; "
                "M0 only supports a lightweight transformers probe."
            ),
        )

    if not model_id and local_path is None:
        return ModelProbeResult(
            status="blocked",
            loader=loader,
            blocker_code="missing_model_source",
            message="Neither model.id nor model.local_path was provided for the smoke probe.",
        )

    if local_path is not None and not local_path.exists():
        return ModelProbeResult(
            status="blocked",
            loader=loader,
            model_id=model_id,
            revision=revision,
            blocker_code="local_path_missing",
            message="The configured local model path does not exist.",
            metadata={
                "local_path": str(local_path),
            },
        )

    try:
        from transformers import AutoConfig
    except ImportError:
        return ModelProbeResult(
            status="blocked",
            loader=loader,
            model_id=model_id,
            revision=revision,
            blocker_code="missing_dependency",
            message="Transformers is not installed yet, so checkpoint access cannot be validated.",
        )

    source_ref = str(local_path) if local_path is not None else model_id
    try:
        resolved = AutoConfig.from_pretrained(
            source_ref,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except Exception as exc:  # pragma: no cover - exercised in live environments
        return ModelProbeResult(
            status="blocked",
            loader=loader,
            model_id=model_id,
            revision=revision,
            blocker_code="checkpoint_unavailable",
            message=(
                "Checkpoint access is not validated yet; "
                "the smoke probe recorded the blocker cleanly."
            ),
            metadata={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "local_files_only": local_files_only,
                "local_path": str(local_path) if local_path is not None else None,
                "resolved_source": source_ref,
            },
        )

    return ModelProbeResult(
        status="passed",
        loader=loader,
        model_id=model_id,
        revision=revision,
        message="Model config resolved successfully for the smoke probe.",
        metadata={
            "local_files_only": local_files_only,
            "local_path": str(local_path) if local_path is not None else None,
            "resolved_source": source_ref,
            "used_local_path": local_path is not None,
            "model_type": getattr(resolved, "model_type", None),
            "architectures": getattr(resolved, "architectures", None),
        },
    )
