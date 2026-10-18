from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from chatterboxsynth import AudioPrompt, GenerationConfig, SynthesisResult
from chatterboxsynth.errors import AudioPromptError, InvalidGenerationConfigError, SynthesisError


def test_audio_prompt_copies_and_freezes_samples() -> None:
    source = np.array([0.1, 0.2], dtype=np.float64)
    prompt = AudioPrompt(source, 24_000)
    source[0] = 0.8
    assert prompt.samples.dtype == np.float32
    assert prompt.samples[0] == pytest.approx(0.1)
    assert not prompt.samples.flags.writeable
    with pytest.raises(FrozenInstanceError):
        prompt.sample_rate = 16_000  # type: ignore[misc]


def test_audio_prompt_rejects_invalid_rate_and_non_finite_samples() -> None:
    with pytest.raises(AudioPromptError, match="sample_rate"):
        AudioPrompt([0.0], 0)
    with pytest.raises(AudioPromptError, match="finite"):
        AudioPrompt([float("inf")], 24_000)


def test_generation_config_is_strict() -> None:
    assert GenerationConfig().max_new_tokens is None
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(max_new_tokens=True)
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(repetition_penalty=float("nan"))


def test_synthesis_result_validation_immutability_and_wav(tmp_path) -> None:
    result = SynthesisResult(
        audio=np.array([0.0, 0.25, -0.25], dtype=np.float64),
        sample_rate=24_000,
        model="nano-v2",
        profile="cpu",
        generated_tokens=2,
        stopped_on_eos=True,
        text="hello",
    )
    assert result.audio.dtype == np.float32
    assert not result.audio.flags.writeable
    assert result.duration_seconds == pytest.approx(3 / 24_000)
    with pytest.raises(SynthesisError, match="24,000 Hz"):
        SynthesisResult([0.0], 16_000, "nano-v2", "cpu", 1, False, "hello")
    destination = result.save_wav(tmp_path / "nested" / "voice.wav")
    assert destination.is_file()
    with pytest.raises(SynthesisError, match="finite"):
        SynthesisResult([float("nan")], 24_000, "turbo", "fp32", 1, False, "hi")
