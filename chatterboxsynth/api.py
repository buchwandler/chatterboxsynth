from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .types import RequestApiContract

try:
    from ._version import __version__ as _engine_version
except ImportError:  # setuptools_scm generates this file during package builds
    _engine_version = "0.0.0"

REQUEST_API_VERSION = 1


def request_api_contract() -> RequestApiContract:
    """Return ChatterboxSynth's stable public request API capabilities."""
    return RequestApiContract(
        version=REQUEST_API_VERSION,
        supports_reference_voice=True,
        supports_named_voices=False,
        supports_request_measurement=True,
    )


def runtime_identity() -> dict[str, Any]:
    """Return stable package/backend identity fields for cache manifests."""
    try:
        onnxvoice_version = version("onnxvoice")
    except PackageNotFoundError:
        onnxvoice_version = "unknown"
    return {
        "engine": "chatterboxsynth",
        "engine_version": _engine_version,
        "backend": "onnxvoice",
        "onnxvoice_version": onnxvoice_version,
        "request_api_version": REQUEST_API_VERSION,
    }
