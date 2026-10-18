from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

_ENGLISH_LANGUAGE_RE = re.compile(r"^en(?:-[a-z]{2})?$", re.IGNORECASE)


def _validate_request_language(language: str) -> str:
    if not isinstance(language, str):
        raise TypeError("language must be a string")
    normalized = language.strip()
    if not _ENGLISH_LANGUAGE_RE.fullmatch(normalized):
        raise ValueError("ChatterboxSynth currently supports English language requests only")
    if "-" not in normalized:
        return "en"
    base, region = normalized.split("-", 1)
    return f"{base.lower()}-{region.upper()}"


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """Language-tagged text request for orchestration and request measurement."""

    id: str
    text: str
    language: str = "en"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        object.__setattr__(self, "language", _validate_request_language(self.language))


@dataclass(frozen=True, slots=True)
class RequestMeasure:
    """Exact tokenizer-level request size reported by ChatterboxSynth."""

    fits: bool
    amount: int
    maximum: int | None
    unit: Literal["model_tokens"] = "model_tokens"
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fits, bool):
            raise TypeError("fits must be a boolean")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount < 0:
            raise ValueError("amount must be a non-negative integer")
        if self.maximum is not None and (
            isinstance(self.maximum, bool) or not isinstance(self.maximum, int) or self.maximum < 0
        ):
            raise ValueError("maximum must be a non-negative integer or None")
        if self.unit != "model_tokens":
            raise ValueError("unit must be 'model_tokens'")
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
