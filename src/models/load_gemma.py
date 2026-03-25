from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GemmaLoadReport:
    status: str
    message: str
    blocker_code: str | None = None
    loader: str = "transformers"
    model_id: str | None = None
    revision: str | None = None
    resolved_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_source(model_config: dict[str, Any]) -> str:
    model_id = model_config.get("id")
    local_path_raw = model_config.get("local_path")
    local_path = Path(str(local_path_raw)) if local_path_raw else None

    if local_path is not None:
        if not local_path.exists():
            raise FileNotFoundError(f"Configured local model path does not exist: {local_path}")
        return str(local_path)

    if model_id:
        return str(model_id)

    raise ValueError("Neither model.id nor model.local_path was provided.")


def load_gemma_tokenizer(model_config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    resolved_source = resolve_model_source(model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_source,
        revision=model_config.get("revision"),
        local_files_only=bool(model_config.get("local_files_only", True)),
        trust_remote_code=False,
    )
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_gemma_causal_lm(model_config: dict[str, Any]) -> Any:
    from transformers import AutoModelForCausalLM

    resolved_source = resolve_model_source(model_config)
    dtype_name = str(model_config.get("dtype", "bfloat16")).lower()
    import torch

    dtype = None
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "float32":
        dtype = torch.float32

    return AutoModelForCausalLM.from_pretrained(
        resolved_source,
        revision=model_config.get("revision"),
        local_files_only=bool(model_config.get("local_files_only", True)),
        trust_remote_code=False,
        dtype=dtype,
    )


def probe_gemma_loading(
    model_config: dict[str, Any],
    *,
    require_tokenizer: bool = True,
    require_chat_template: bool = False,
) -> GemmaLoadReport:
    loader = str(model_config.get("loader", "transformers"))
    model_id = model_config.get("id")
    revision = model_config.get("revision")

    if loader != "transformers":
        return GemmaLoadReport(
            status="blocked",
            message=f"Unsupported Gemma loader {loader!r}; only transformers is supported.",
            blocker_code="unsupported_loader",
            loader=loader,
            model_id=model_id,
            revision=revision,
        )

    try:
        resolved_source = resolve_model_source(model_config)
    except FileNotFoundError as exc:
        return GemmaLoadReport(
            status="blocked",
            message=str(exc),
            blocker_code="local_path_missing",
            loader=loader,
            model_id=model_id,
            revision=revision,
        )
    except ValueError as exc:
        return GemmaLoadReport(
            status="blocked",
            message=str(exc),
            blocker_code="missing_model_source",
            loader=loader,
            model_id=model_id,
            revision=revision,
        )

    try:
        from transformers import AutoConfig
    except ImportError:
        return GemmaLoadReport(
            status="blocked",
            message="Transformers is not installed yet, so Gemma loading cannot be validated.",
            blocker_code="missing_dependency",
            loader=loader,
            model_id=model_id,
            revision=revision,
            resolved_source=resolved_source,
        )

    local_files_only = bool(model_config.get("local_files_only", True))
    metadata: dict[str, Any] = {"local_files_only": local_files_only}
    try:
        config = AutoConfig.from_pretrained(
            resolved_source,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        metadata.update(
            {
                "model_type": getattr(config, "model_type", None),
                "architectures": getattr(config, "architectures", None),
            }
        )
    except Exception as exc:  # pragma: no cover - exercised in live environments
        return GemmaLoadReport(
            status="blocked",
            message="Gemma config loading is not validated yet; the blocker was recorded cleanly.",
            blocker_code="config_unavailable",
            loader=loader,
            model_id=model_id,
            revision=revision,
            resolved_source=resolved_source,
            metadata={"exception_type": type(exc).__name__, "exception_message": str(exc)},
        )

    if require_tokenizer or require_chat_template:
        try:
            tokenizer = load_gemma_tokenizer(model_config)
        except Exception as exc:  # pragma: no cover - exercised in live environments
            return GemmaLoadReport(
                status="blocked",
                message=(
                    "Gemma tokenizer loading is not validated yet; "
                    "the blocker was recorded cleanly."
                ),
                blocker_code="tokenizer_unavailable",
                loader=loader,
                model_id=model_id,
                revision=revision,
                resolved_source=resolved_source,
                metadata={
                    **metadata,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )

        chat_template = getattr(tokenizer, "chat_template", None)
        chat_template_available = bool(chat_template)
        metadata.update(
            {
                "tokenizer_class": type(tokenizer).__name__,
                "chat_template_available": chat_template_available,
            }
        )
        if require_chat_template and not chat_template_available:
            return GemmaLoadReport(
                status="blocked",
                message="Tokenizer loaded but did not expose an official chat template.",
                blocker_code="chat_template_unavailable",
                loader=loader,
                model_id=model_id,
                revision=revision,
                resolved_source=resolved_source,
                metadata=metadata,
            )

    return GemmaLoadReport(
        status="passed",
        message="Gemma config and tokenizer loading validated successfully.",
        loader=loader,
        model_id=model_id,
        revision=revision,
        resolved_source=resolved_source,
        metadata=metadata,
    )
