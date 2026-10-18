"""Cleanroom Chatterbox Nano v2 and Turbo ONNX synthesis API."""

from .config import GenerationConfig
from .errors import (
    AudioPromptError,
    ChatterboxSynthError,
    EmptyTextError,
    IncompatibleVoiceError,
    InvalidGenerationConfigError,
    ModelLoadError,
    OptionalDependencyError,
    RuntimeClosedError,
    SynthesisError,
)
from .runtime import ChatterboxRuntime
from .types import AudioPrompt, PreparedVoice, SynthesisResult

try:
    from ._version import __version__
except ImportError:  # setuptools_scm generates this file during package builds
    __version__ = "0.0.0"

__all__ = [
    "AudioPrompt",
    "AudioPromptError",
    "ChatterboxRuntime",
    "ChatterboxSynthError",
    "EmptyTextError",
    "GenerationConfig",
    "IncompatibleVoiceError",
    "InvalidGenerationConfigError",
    "ModelLoadError",
    "OptionalDependencyError",
    "PreparedVoice",
    "RuntimeClosedError",
    "SynthesisError",
    "SynthesisResult",
    "__version__",
]
