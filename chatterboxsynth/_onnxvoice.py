from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType

from .errors import OptionalDependencyError

_MODEL_ALIASES = {
    "nano": "nano-v2",
    "nano-v2": "nano-v2",
    "nano-110m": "nano-v2",
    "turbo": "turbo",
    "turbo-350m": "turbo",
}


def normalize_model_ref(model: str) -> tuple[str, str]:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty Chatterbox model name or reference")
    value = model.strip()
    if ":" in value:
        system, model_id = value.split(":", 1)
        if system.casefold() != "chatterbox" or not model_id:
            raise ValueError(f"Not a Chatterbox model reference: {model!r}")
        value = model_id
    canonical_id = _MODEL_ALIASES.get(value.casefold())
    if canonical_id is None:
        raise ValueError(f"Unsupported Chatterbox model: {model!r}; choose nano or turbo")
    return f"chatterbox:{canonical_id}", canonical_id


def onnxvoice_module() -> ModuleType:
    try:
        module = importlib.import_module("onnxvoice")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "OnnxVoice is required for model installation and ONNX sessions; install chatterboxsynth[cpu]."
        ) from exc
    return module


def normalize_providers(
    provider: str | Sequence[str] | None,
) -> str | Sequence[str] | None:
    """Translate CLI-friendly provider names to OnnxVoice provider identifiers."""
    if provider is None:
        return None
    if isinstance(provider, str):
        value = provider.casefold()
        if value == "auto":
            return None
        if value == "cpu":
            return "CPUExecutionProvider"
        if value == "cuda":
            return "CUDAExecutionProvider"
        return provider
    names = tuple(provider)
    if not names:
        raise ValueError("providers must not be empty")
    normalized_names: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError("provider sequences must contain strings")
        value = name.casefold()
        if value == "auto":
            raise ValueError("provider sequences must contain explicit execution-provider names")
        if value == "cpu":
            normalized_names.append("CPUExecutionProvider")
        elif value == "cuda":
            normalized_names.append("CUDAExecutionProvider")
        else:
            normalized_names.append(name)
    return tuple(normalized_names)
