from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .errors import InvalidGenerationConfigError


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Strict deterministic generation controls; unset values use bundle metadata."""

    max_new_tokens: int | None = None
    repetition_penalty: float | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens is not None and (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise InvalidGenerationConfigError("max_new_tokens must be a positive integer or None")
        if self.repetition_penalty is not None and (
            isinstance(self.repetition_penalty, bool)
            or not isinstance(self.repetition_penalty, Real)
            or not math.isfinite(float(self.repetition_penalty))
            or self.repetition_penalty <= 0
        ):
            raise InvalidGenerationConfigError(
                "repetition_penalty must be a positive finite number or None"
            )
