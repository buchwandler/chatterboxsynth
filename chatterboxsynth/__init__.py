"""Cleanroom Chatterbox Nano v2 and Turbo ONNX synthesis API."""

from .api import REQUEST_API_VERSION, request_api_contract, runtime_identity
from .config import GenerationConfig
from .discovery import discover_bundles
from .errors import (
    AudioPromptError,
    ChatterboxSynthError,
    EmptyTextError,
    IncompatibleVoiceError,
    InvalidGenerationConfigError,
    ModelLoadError,
    ModelNotInstalledError,
    OptionalDependencyError,
    RuntimeClosedError,
    SynthesisError,
)
from .request import RequestMeasure, SynthesisRequest
from .runtime import ChatterboxRuntime
from .types import (
    AudioPrompt,
    DiscoveredBundle,
    PreparedVoice,
    RequestApiContract,
    SynthesisResult,
)

try:
    from ._version import __version__
except ImportError:  # setuptools_scm generates this file during package builds
    __version__ = "0.0.0"

__all__ = [
    "AudioPrompt",
    "AudioPromptError",
    "ChatterboxRuntime",
    "ChatterboxSynthError",
    "DiscoveredBundle",
    "discover_bundles",
    "EmptyTextError",
    "GenerationConfig",
    "IncompatibleVoiceError",
    "InvalidGenerationConfigError",
    "ModelLoadError",
    "ModelNotInstalledError",
    "OptionalDependencyError",
    "PreparedVoice",
    "REQUEST_API_VERSION",
    "RequestApiContract",
    "RequestMeasure",
    "RuntimeClosedError",
    "SynthesisError",
    "SynthesisRequest",
    "SynthesisResult",
    "__version__",
    "request_api_contract",
    "runtime_identity",
]
