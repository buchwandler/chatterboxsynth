from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import chatterboxsynth as cs


def test_request_api_contract_and_package_exports() -> None:
    assert cs.REQUEST_API_VERSION == 1
    contract = cs.request_api_contract()
    assert isinstance(contract, cs.RequestApiContract)
    assert contract.version == 1
    assert contract.supports_reference_voice is True
    assert contract.supports_named_voices is False
    assert contract.supports_request_measurement is True
    assert cs.SynthesisRequest is not None
    assert cs.RequestMeasure is not None
    assert cs.DiscoveredBundle is not None
    assert callable(cs.runtime_identity)


def test_synthesis_request_normalizes_english_and_is_immutable() -> None:
    request = cs.SynthesisRequest(id="request-1", text="Hello", language="en-us")
    assert request.language == "en-US"
    with pytest.raises(FrozenInstanceError):
        request.text = "changed"  # type: ignore[misc]

    with pytest.raises(ValueError, match="English"):
        cs.SynthesisRequest(id="request-2", text="Hola", language="es")
    with pytest.raises(ValueError, match="non-empty"):
        cs.SynthesisRequest(id=" ", text="Hello")
    with pytest.raises(TypeError, match="text must be a string"):
        cs.SynthesisRequest(id="request-3", text=1)  # type: ignore[arg-type]


def test_request_measure_copies_details_into_an_immutable_mapping() -> None:
    details = {"model": "nano-v2"}
    measure = cs.RequestMeasure(True, 2, None, details=details)
    details["model"] = "turbo"
    assert measure.details == {"model": "nano-v2"}
    with pytest.raises(TypeError):
        measure.details["model"] = "turbo"  # type: ignore[index]


def test_runtime_identity_has_stable_environment_independent_fields() -> None:
    identity = cs.runtime_identity()
    assert identity["engine"] == "chatterboxsynth"
    assert identity["engine_version"] == cs.__version__
    assert identity["backend"] == "onnxvoice"
    assert identity["onnxvoice_version"]
    assert identity["request_api_version"] == 1
    assert not any("path" in key or "url" in key for key in identity)
