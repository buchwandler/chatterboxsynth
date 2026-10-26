from __future__ import annotations

import io
from importlib.metadata import version

import numpy as np
import pytest
import soundfile as sf

from chatterboxsynth.audio import load_reference_audio, normalize_reference_audio, write_wav
from chatterboxsynth.errors import AudioPromptError


def test_stereo_is_downmixed_and_resampled_to_native_rate() -> None:
    stereo = np.array([[0.2, -0.2], [0.5, 0.1], [-0.3, 0.3]], dtype=np.float32)
    result = normalize_reference_audio(stereo, 12_000, target_sample_rate=24_000)
    assert result.dtype == np.float32
    assert result.ndim == 1
    assert result.flags.c_contiguous
    assert result.size == 6
    assert np.allclose(result[::2], [0.0, 0.3, 0.0], atol=0.05)


def test_non_finite_empty_and_out_of_range_audio_are_rejected() -> None:
    with pytest.raises(AudioPromptError, match="finite"):
        normalize_reference_audio(np.array([0.0, np.nan]), 24_000, target_sample_rate=24_000)
    with pytest.raises(AudioPromptError, match="empty"):
        normalize_reference_audio(np.array([], dtype=np.float32), 24_000, target_sample_rate=24_000)
    with pytest.raises(AudioPromptError, match=r"\[-1, 1\]"):
        normalize_reference_audio(
            np.array([1.1], dtype=np.float32), 24_000, target_sample_rate=24_000
        )


def test_soundfile_input_and_pcm16_output(tmp_path) -> None:
    source = tmp_path / "stereo.wav"
    samples = np.column_stack((np.linspace(-0.5, 0.5, 240, dtype=np.float32),) * 2)
    sf.write(source, samples, 48_000, subtype="PCM_16")
    normalized = load_reference_audio(source, target_sample_rate=24_000)
    assert normalized.size == 120
    assert normalized.dtype == np.float32

    output = io.BytesIO()
    write_wav(output, np.array([1.2, -1.2], dtype=np.float32), 24_000)
    output.seek(0)
    decoded, rate = sf.read(output, dtype="float32")
    assert rate == 24_000
    assert decoded.shape == (2,)
    assert np.max(np.abs(decoded)) <= 1.0


def test_soxr_110_dependency_and_hq_resampling_path() -> None:
    assert version("soxr") == "1.1.0"

    source = np.linspace(-0.5, 0.5, 480, dtype=np.float32)
    result = normalize_reference_audio(source, 48_000, target_sample_rate=24_000)

    assert result.dtype == np.float32
    assert result.ndim == 1
    assert result.flags.c_contiguous
    assert result.size == 240
    assert np.isfinite(result).all()
