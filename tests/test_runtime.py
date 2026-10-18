from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import chatterboxsynth.runtime as runtime_module
from chatterboxsynth import (
    AudioPrompt,
    ChatterboxRuntime,
    GenerationConfig,
    ModelNotInstalledError,
    SynthesisRequest,
)
from chatterboxsynth.errors import (
    EmptyTextError,
    IncompatibleVoiceError,
    ModelLoadError,
    RuntimeClosedError,
)
from chatterboxsynth.frontend import ChatterboxFrontend

REVISION = "851210533a5bcb4ad78e3ada6d527482dd1a31e7"


class FakeFrontend:
    def __init__(self, *_: Any) -> None:
        pass

    def encode(self, text: str) -> tuple[int, ...]:
        if not text.strip():
            raise EmptyTextError("text must contain non-whitespace characters")
        return (7, 8)


class FakeAdapter:
    def __init__(self) -> None:
        self.prepared: list[np.ndarray] = []
        self.calls: list[dict[str, Any]] = []
        self.close_calls = 0

    def prepare_voice(self, audio: np.ndarray) -> object:
        self.prepared.append(np.array(audio, copy=True))
        return {"prepared": len(self.prepared)}

    def infer(self, token_ids: tuple[int, ...], **kwargs: Any) -> Any:
        self.calls.append({"token_ids": token_ids, **kwargs})
        return SimpleNamespace(
            audio=np.linspace(-0.2, 0.2, 240, dtype=np.float32),
            sample_rate=24_000,
            metadata={"generated_tokens": 3, "stopped_on_eos": True},
        )

    def close(self) -> None:
        self.close_calls += 1


def make_runtime(
    *, model: str = "nano-v2", profile: str = "cpu", owns_adapter: bool = False
) -> tuple[ChatterboxRuntime, FakeAdapter]:
    adapter = FakeAdapter()
    runtime = ChatterboxRuntime(
        adapter,
        FakeFrontend(),  # type: ignore[arg-type]
        model=model,
        profile=profile,
        source_revision=REVISION,
        runtime_metadata={"default_max_new_tokens": 11, "default_repetition_penalty": 1.15},
        owns_adapter=owns_adapter,
    )
    return runtime, adapter


class FakeNotInstalledError(Exception):
    pass


def make_managed_installation(tmp_path: Path, *, profile: str = "cpu") -> Any:
    metadata = {
        "selected_quality": profile,
        "source_revision": REVISION,
        "runtime": {
            "profile": profile,
            "default_max_new_tokens": 12,
            "default_repetition_penalty": 1.1,
        },
        "frontend": {"tokenizer_contract": "nano-v2"},
    }
    return SimpleNamespace(
        id="nano-v2",
        system="chatterbox",
        sample_rate=24_000,
        metadata=metadata,
        artifact=lambda role: SimpleNamespace(path=tmp_path / f"{role}.json"),
    )


def patch_managed_manager(
    monkeypatch: Any,
    installation: Any,
    adapter: FakeAdapter,
    calls: dict[str, Any],
    *,
    missing: bool = False,
) -> None:
    class Manager:
        def __init__(self, **kwargs: Any) -> None:
            calls["manager"] = kwargs

        def resolve(self, ref: str, **kwargs: Any) -> Any:
            calls["resolve"] = (ref, kwargs)
            if missing:
                raise FakeNotInstalledError("missing")
            return installation

        def open(self, selected: Any, **kwargs: Any) -> FakeAdapter:
            calls["open"] = (selected, kwargs)
            return adapter

        def install(self, *_: Any, **__: Any) -> None:
            raise AssertionError("from_installed() must not install")

        def list(self, *_: Any, **__: Any) -> None:
            raise AssertionError("from_installed() must not query the catalog")

    monkeypatch.setattr(
        runtime_module,
        "onnxvoice_module",
        lambda: SimpleNamespace(OnnxVoice=Manager, NotInstalledError=FakeNotInstalledError),
    )
    monkeypatch.setattr(runtime_module, "ChatterboxFrontend", FakeFrontend)


def test_from_installed_resolves_and_opens_local_model_without_acquisition(
    monkeypatch: Any, tmp_path: Path
) -> None:
    installation = make_managed_installation(tmp_path)
    adapter = FakeAdapter()
    calls: dict[str, Any] = {}
    patch_managed_manager(monkeypatch, installation, adapter, calls)
    provider_options = {"CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}}
    session_options = object()

    runtime = ChatterboxRuntime.from_installed(
        "nano",
        quality="cpu",
        provider="cpu",
        provider_options=provider_options,
        session_options=session_options,
        cache_dir=tmp_path / "cache",
    )

    assert calls["manager"] == {"cache_dir": tmp_path / "cache"}
    assert calls["resolve"] == ("chatterbox:nano-v2", {"quality": "cpu"})
    selected, open_options = calls["open"]
    assert selected is installation
    assert open_options["providers"] == "CPUExecutionProvider"
    assert open_options["provider_options"] is provider_options
    assert open_options["session_options"] is session_options
    assert "install" not in calls and "list" not in calls
    assert runtime.model == "nano-v2"
    assert runtime.profile == "cpu"
    assert runtime.source_revision == REVISION
    runtime.close()
    assert adapter.close_calls == 1


def test_from_installed_closes_adapter_when_runtime_construction_fails(
    monkeypatch: Any, tmp_path: Path
) -> None:
    installation = make_managed_installation(tmp_path)
    adapter = FakeAdapter()
    calls: dict[str, Any] = {}
    patch_managed_manager(monkeypatch, installation, adapter, calls)

    def fail_frontend(*_: Any, **__: Any) -> Any:
        raise RuntimeError("frontend construction failed")

    monkeypatch.setattr(runtime_module, "ChatterboxFrontend", fail_frontend)
    with pytest.raises(RuntimeError, match="frontend construction failed"):
        ChatterboxRuntime.from_installed("nano", quality="cpu")
    assert adapter.close_calls == 1


def test_from_installed_raises_stable_error_when_profile_is_absent(
    monkeypatch: Any, tmp_path: Path
) -> None:
    installation = make_managed_installation(tmp_path)
    adapter = FakeAdapter()
    calls: dict[str, Any] = {}
    patch_managed_manager(monkeypatch, installation, adapter, calls, missing=True)

    with pytest.raises(ModelNotInstalledError, match="not installed"):
        ChatterboxRuntime.from_installed("nano", quality="cpu")
    assert "open" not in calls


def test_measure_request_is_exact_and_does_not_require_or_prepare_a_voice() -> None:
    runtime, adapter = make_runtime()

    measure = runtime.measure_request(SynthesisRequest(id="measure-1", text="Hello"))

    assert measure.amount == 2
    assert measure.maximum is None
    assert measure.fits is True
    assert measure.unit == "model_tokens"
    assert measure.details == {"model": "nano-v2", "profile": "cpu"}
    assert adapter.prepared == []
    assert adapter.calls == []


@pytest.mark.parametrize(
    ("model", "profile", "contract", "special_count"),
    [("nano-v2", "cpu", "nano-v2", 1), ("turbo", "fp32", "turbo", 2)],
)
def test_measure_request_includes_export_specific_eot_tokens(
    model: str, profile: str, contract: str, special_count: int
) -> None:
    class Tokenizer:
        def token_to_id(self, token: str) -> int:
            assert token == "<|endoftext|>"
            return 99

        def encode(self, text: str, *, add_special_tokens: bool = True) -> Any:
            ids = [7, 8]
            if add_special_tokens:
                ids.extend([99] * special_count)
            return SimpleNamespace(ids=ids)

    adapter = FakeAdapter()
    frontend = ChatterboxFrontend("unused", contract, tokenizer=Tokenizer())
    runtime = ChatterboxRuntime(
        adapter,
        frontend,
        model=model,
        profile=profile,
        source_revision=REVISION,
        runtime_metadata={"default_max_new_tokens": 11, "default_repetition_penalty": 1.15},
    )

    measure = runtime.measure_request(SynthesisRequest(id="measure-eot", text="Hello"))

    assert measure.amount == 2 + special_count
    assert measure.maximum is None
    assert measure.fits is True
    assert adapter.prepared == []
    assert adapter.calls == []


def test_empty_text_fails_identically_in_measurement_and_synthesis() -> None:
    runtime, adapter = make_runtime()
    voice = runtime.prepare_voice(AudioPrompt(np.zeros(100, dtype=np.float32), 24_000))
    request = SynthesisRequest(id="empty", text=" \n")

    with pytest.raises(EmptyTextError):
        runtime.measure_request(request)
    with pytest.raises(EmptyTextError):
        runtime.synthesize(request.text, voice=voice)
    assert adapter.calls == []


def test_prepared_voice_reuse_strict_generation_and_close_ownership() -> None:
    runtime, adapter = make_runtime(owns_adapter=True)
    prompt = AudioPrompt(np.full((240, 2), 0.1, dtype=np.float32), 24_000)
    voice = runtime.prepare_voice(prompt)
    assert adapter.prepared[0].shape == (240,)

    result = runtime.synthesize("Hello!", voice=voice)
    assert result.sample_rate == 24_000
    assert result.model == "nano-v2"
    assert result.profile == "cpu"
    assert result.generated_tokens == 3
    assert result.stopped_on_eos
    assert adapter.calls[0]["token_ids"] == (7, 8)
    assert adapter.calls[0]["voice_state"] is voice.state
    assert adapter.calls[0]["max_new_tokens"] == 11
    assert adapter.calls[0]["repetition_penalty"] == pytest.approx(1.15)

    runtime.synthesize(
        "Another request.", voice=runtime.prepare_voice(voice), config=GenerationConfig(5, 1.3)
    )
    assert len(adapter.prepared) == 1
    assert adapter.calls[1]["max_new_tokens"] == 5
    assert adapter.calls[1]["repetition_penalty"] == pytest.approx(1.3)

    runtime.close()
    runtime.close()
    assert adapter.close_calls == 1
    assert runtime.closed
    with pytest.raises(RuntimeClosedError):
        runtime.prepare_voice(prompt)


def test_incompatible_prepared_voice_is_rejected_and_external_adapter_is_not_closed() -> None:
    nano, adapter = make_runtime()
    voice = nano.prepare_voice(AudioPrompt(np.zeros(100, dtype=np.float32), 24_000))
    turbo, other_adapter = make_runtime(model="turbo", profile="fp32")
    with pytest.raises(IncompatibleVoiceError):
        turbo.synthesize("Hello", voice=voice)
    turbo.close()
    assert other_adapter.close_calls == 0
    nano.close()
    assert adapter.close_calls == 0


def test_from_pretrained_normalizes_alias_and_forwards_profile_provider(
    monkeypatch, tmp_path
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    metadata = {
        "selected_quality": "cpu",
        "source_revision": REVISION,
        "runtime": {
            "profile": "cpu",
            "default_max_new_tokens": 10,
            "default_repetition_penalty": 1.2,
        },
        "frontend": {"tokenizer_contract": "nano-v2"},
    }
    installation = SimpleNamespace(
        id="nano-v2",
        system="chatterbox",
        sample_rate=24_000,
        metadata=metadata,
        artifact=lambda role: SimpleNamespace(path=tokenizer_path),
    )
    adapter = FakeAdapter()
    calls: dict[str, Any] = {}

    class Manager:
        def __init__(self, **kwargs: Any) -> None:
            calls["manager"] = kwargs

        def install(self, ref: str, **kwargs: Any) -> Any:
            calls["install"] = (ref, kwargs)
            return installation

        def open(self, selected: Any, **kwargs: Any) -> FakeAdapter:
            calls["open"] = (selected, kwargs)
            return adapter

    monkeypatch.setattr(
        runtime_module, "onnxvoice_module", lambda: SimpleNamespace(OnnxVoice=Manager)
    )
    monkeypatch.setattr(runtime_module, "ChatterboxFrontend", FakeFrontend)
    runtime = ChatterboxRuntime.from_pretrained("nano", quality="cpu", provider="cpu")
    assert calls["install"][0] == "chatterbox:nano-v2"
    assert calls["install"][1]["quality"] == "cpu"
    assert calls["open"][1]["providers"] == "CPUExecutionProvider"
    runtime.close()
    assert adapter.close_calls == 1


def _write_installation(root: Path) -> None:
    profiles = {
        "cpu": {
            "speech_encoder": "fp32",
            "embed_tokens": "fp32",
            "language_model": "int8_mixed",
            "conditional_decoder": "int8",
        }
    }
    artifacts: list[dict[str, Any]] = []
    paths = {
        "tokenizer": (None, None, "tokenizer.json"),
        "graph:speech_encoder": ("speech_encoder", "fp32", "onnx/speech_encoder.onnx"),
        "graph:embed_tokens": ("embed_tokens", "fp32", "onnx/embed_tokens.onnx"),
        "graph:language_model": (
            "language_model",
            "int8_mixed",
            "onnx/language_model_int8_mixed.onnx",
        ),
        "graph:conditional_decoder": (
            "conditional_decoder",
            "int8",
            "onnx/conditional_decoder_int8.onnx",
        ),
        "external_data:language_model": (
            "language_model",
            "int8_mixed",
            "onnx/language_model_int8_mixed.onnx.data",
        ),
        "external_data:conditional_decoder": (
            "conditional_decoder",
            "int8",
            "onnx/conditional_decoder_int8.onnx.data",
        ),
    }
    for key, (component, quality, filename) in paths.items():
        destination = root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(("artifact:" + key).encode("utf-8"))
        data = destination.read_bytes()
        role = key.split(":", 1)[0]
        artifacts.append(
            {
                "role": role,
                "component": component,
                "quality": quality,
                "filename": filename,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema": 3,
        "system": "chatterbox",
        "id": "nano-v2",
        "sample_rate": 24_000,
        "metadata": {
            "selected_quality": "cpu",
            "source_revision": REVISION,
            "profiles": profiles,
            "frontend": {"tokenizer_contract": "nano-v2"},
            "runtime": {
                "profile": "cpu",
                "default_max_new_tokens": 10,
                "default_repetition_penalty": 1.2,
            },
        },
        "artifacts": artifacts,
    }
    (root / ".onnxvoice-install.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_uses_only_local_manifest_and_open_local(monkeypatch, tmp_path) -> None:
    _write_installation(tmp_path)
    adapter = FakeAdapter()
    calls: dict[str, Any] = {}

    class OfflineOnlyModule:
        def OnnxVoice(self, **_: Any) -> None:
            raise AssertionError("load() must not construct a catalog manager")

        def open_local(self, **kwargs: Any) -> FakeAdapter:
            calls.update(kwargs)
            return adapter

    monkeypatch.setattr(runtime_module, "onnxvoice_module", lambda: OfflineOnlyModule())
    monkeypatch.setattr(runtime_module, "ChatterboxFrontend", FakeFrontend)
    runtime = ChatterboxRuntime.load(tmp_path, provider="cpu")
    assert set(calls["files"]) >= {
        "tokenizer",
        "graph:speech_encoder",
        "external_data:language_model",
    }
    assert calls["system"] == "chatterbox"
    assert calls["providers"] == "CPUExecutionProvider"
    runtime.close()
    assert adapter.close_calls == 1


def test_load_rejects_tampered_artifact_before_opening(monkeypatch, tmp_path) -> None:
    _write_installation(tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(b"tampered")
    opened = False

    class Module:
        def open_local(self, **_: Any) -> FakeAdapter:
            nonlocal opened
            opened = True
            return FakeAdapter()

    monkeypatch.setattr(runtime_module, "onnxvoice_module", lambda: Module())
    with pytest.raises(ModelLoadError, match="SHA-256"):
        ChatterboxRuntime.load(tmp_path)
    assert not opened
