from __future__ import annotations

from dataclasses import dataclass, field
from io import BufferedIOBase
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .audio import write_wav
from .errors import AudioPromptError, SynthesisError


@dataclass(frozen=True, slots=True, init=False)
class AudioPrompt:
    """Reference samples with their source rate; 2-D input is frames by channels."""

    samples: np.ndarray
    sample_rate: int

    def __init__(self, samples: Any, sample_rate: int) -> None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise AudioPromptError("sample_rate must be a positive integer")
        try:
            value = np.array(samples, dtype=np.float32, copy=True, order="C")
        except (TypeError, ValueError, OverflowError) as exc:
            raise AudioPromptError("samples must be numeric audio") from exc
        if value.ndim not in {1, 2}:
            raise AudioPromptError("samples must be mono or frames-by-channels audio")
        if value.size == 0 or (value.ndim == 2 and (value.shape[0] == 0 or value.shape[1] == 0)):
            raise AudioPromptError("samples must not be empty")
        if not np.isfinite(value).all():
            raise AudioPromptError("samples must be finite")
        value.setflags(write=False)
        object.__setattr__(self, "samples", value)
        object.__setattr__(self, "sample_rate", sample_rate)


@dataclass(frozen=True, slots=True)
class PreparedVoice:
    """Reusable encoder state tied to a specific Chatterbox model/profile revision."""

    state: Any = field(repr=False, compare=False)
    sample_rate: int
    model: str
    profile: str
    source_revision: str
    runtime_fingerprint: str
    source: str | None = None


@dataclass(frozen=True, slots=True, init=False)
class SynthesisResult:
    """Validated mono waveform and stable generation metadata."""

    audio: np.ndarray
    sample_rate: int
    model: str
    profile: str
    generated_tokens: int
    stopped_on_eos: bool
    text: str

    def __init__(
        self,
        audio: Any,
        sample_rate: int,
        model: str,
        profile: str,
        generated_tokens: int,
        stopped_on_eos: bool,
        text: str,
    ) -> None:
        try:
            value = np.array(audio, dtype=np.float32, copy=True, order="C")
        except (TypeError, ValueError, OverflowError) as exc:
            raise SynthesisError("output audio must be numeric") from exc
        if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
            raise SynthesisError("output audio must be non-empty, finite, mono float32")
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate != 24_000
        ):
            raise SynthesisError("Chatterbox output sample_rate must be 24,000 Hz")
        if not isinstance(model, str) or not model or not isinstance(profile, str) or not profile:
            raise SynthesisError("model and profile must be non-empty strings")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens < 0
        ):
            raise SynthesisError("generated_tokens must be a non-negative integer")
        if not isinstance(stopped_on_eos, bool):
            raise SynthesisError("stopped_on_eos must be a boolean")
        if not isinstance(text, str):
            raise SynthesisError("text must be a string")
        value.setflags(write=False)
        object.__setattr__(self, "audio", value)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "generated_tokens", generated_tokens)
        object.__setattr__(self, "stopped_on_eos", stopped_on_eos)
        object.__setattr__(self, "text", text)

    @property
    def duration_seconds(self) -> float:
        return self.audio.size / self.sample_rate

    def save_wav(
        self, target: str | Path | BinaryIO | BufferedIOBase
    ) -> str | Path | BinaryIO | BufferedIOBase:
        """Write mono PCM16 WAV, clipping only at this explicitly lossy boundary."""
        if isinstance(target, (str, Path)):
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_wav(path, self.audio, self.sample_rate)
            return path
        write_wav(target, self.audio, self.sample_rate)
        return target
