from __future__ import annotations

from io import BufferedIOBase
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

from .errors import AudioPromptError, OptionalDependencyError, SynthesisError


def _validate_rate(sample_rate: int, *, name: str) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioPromptError(f"{name} must be a positive integer")


def normalize_reference_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int,
) -> np.ndarray:
    """Downmix frames-by-channels audio, validate it, and resample with soxr HQ."""
    _validate_rate(sample_rate, name="source sample_rate")
    _validate_rate(target_sample_rate, name="target sample_rate")
    try:
        audio = np.asarray(samples, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioPromptError("reference audio must be numeric") from exc
    if audio.ndim == 2:
        if audio.shape[0] == 0 or audio.shape[1] == 0:
            raise AudioPromptError("reference audio must not be empty")
        if not np.isfinite(audio).all():
            raise AudioPromptError("reference audio must contain only finite samples")
        audio = audio.mean(axis=1, dtype=np.float32)
    elif audio.ndim != 1:
        raise AudioPromptError("reference audio must be mono or frames-by-channels")
    if audio.size == 0:
        raise AudioPromptError("reference audio must not be empty")
    if not np.isfinite(audio).all():
        raise AudioPromptError("reference audio must contain only finite samples")
    if np.any(audio < -1.0) or np.any(audio > 1.0):
        raise AudioPromptError(
            "reference audio samples must be within [-1, 1]; clipping is not implicit"
        )
    if sample_rate != target_sample_rate:
        try:
            import soxr
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "Audio resampling requires soxr; install chatterboxsynth."
            ) from exc
        audio = np.asarray(
            soxr.resample(audio, sample_rate, target_sample_rate, quality="HQ"), dtype=np.float32
        )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise AudioPromptError("audio conversion produced empty or non-finite samples")
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_reference_audio(source: str | Path, *, target_sample_rate: int) -> np.ndarray:
    """Read a reference file through SoundFile and normalize it to model rate."""
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "Reference audio loading requires soundfile; install chatterboxsynth."
        ) from exc
    path = Path(source).expanduser()
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AudioPromptError(f"Could not read reference audio {path}: {exc}") from exc
    return normalize_reference_audio(audio, int(sample_rate), target_sample_rate=target_sample_rate)


def write_wav(
    target: str | Path | BinaryIO | BufferedIOBase, audio: np.ndarray, sample_rate: int
) -> None:
    """Write mono PCM16 WAV; clipping is explicit and limited to this output conversion."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise SynthesisError("WAV sample_rate must be a positive integer")
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise SynthesisError("WAV output must be non-empty, finite, mono audio")
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "WAV writing requires soundfile; install chatterboxsynth."
        ) from exc
    if isinstance(target, (str, Path)):
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        destination: str | Path | BinaryIO = path
    else:
        destination = cast(BinaryIO, target)
    pcm_range = np.clip(value, -1.0, 1.0)
    try:
        sf.write(destination, pcm_range, sample_rate, format="WAV", subtype="PCM_16")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SynthesisError(f"Could not write WAV output: {exc}") from exc
