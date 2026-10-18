class ChatterboxSynthError(Exception):
    """Base error for the ChatterboxSynth public API."""


class OptionalDependencyError(ChatterboxSynthError):
    """A required runtime dependency is missing."""


class ModelLoadError(ChatterboxSynthError):
    """A model installation or local bundle could not be opened."""


class AudioPromptError(ChatterboxSynthError, ValueError):
    """Reference audio does not satisfy the input contract."""


class EmptyTextError(ChatterboxSynthError, ValueError):
    """Synthesis text is empty or has no tokenizer input."""


class InvalidGenerationConfigError(ChatterboxSynthError, ValueError):
    """Generation settings are invalid."""


class IncompatibleVoiceError(ChatterboxSynthError, ValueError):
    """A prepared voice belongs to a different model/profile contract."""


class RuntimeClosedError(ChatterboxSynthError, RuntimeError):
    """Synthesis was requested after the runtime was closed."""


class SynthesisError(ChatterboxSynthError):
    """Model output or a synthesis request violated the public contract."""
