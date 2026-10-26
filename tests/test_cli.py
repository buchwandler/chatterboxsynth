from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import chatterboxsynth.__main__ as cli
from chatterboxsynth.errors import ModelLoadError


def test_models_command_uses_normalized_onnxvoice_items(monkeypatch, capsys) -> None:
    item = SimpleNamespace(
        id="nano-v2",
        ref="chatterbox:nano-v2",
        metadata={"default_profile": "cpu", "profiles": {"cpu": {}, "fp32": {}}},
        aliases=("nano", "nano-110m"),
        sample_rate=24_000,
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        cli.ChatterboxRuntime,
        "models",
        staticmethod(lambda **kwargs: calls.update(kwargs) or (item,)),
    )
    assert cli.main(["models", "--offline"]) == 0
    assert calls["offline"] is True
    assert "chatterbox:nano-v2" in capsys.readouterr().out


def test_download_command_delegates_to_onnxvoice(monkeypatch, capsys, tmp_path) -> None:
    installation = SimpleNamespace(
        ref="chatterbox:turbo",
        metadata={"selected_quality": "q4f16"},
        path=tmp_path,
    )
    calls: dict[str, Any] = {}

    def download(model: str, **kwargs: Any) -> Any:
        calls.update(model=model, **kwargs)
        return installation

    monkeypatch.setattr(cli.ChatterboxRuntime, "download", staticmethod(download))
    assert cli.main(["download", "turbo", "--quality", "q4f16", "--offline"]) == 0
    assert calls["model"] == "turbo"
    assert calls["quality"] == "q4f16"
    assert calls["offline"] is True
    assert "Installed chatterbox:turbo" in capsys.readouterr().out


def test_cli_turbo_q4f16_cpu_reports_compatibility_error(monkeypatch, capsys) -> None:
    calls: dict[str, Any] = {}
    message = (
        "Chatterbox Turbo q4f16 is not compatible with CPUExecutionProvider "
        "for the published ONNX graph (AveragePool opset 19). Use Turbo fp32 "
        "for CPU, or select an execution provider that supports q4f16."
    )

    def reject(model: str, **kwargs: Any) -> None:
        calls.update(model=model, **kwargs)
        raise ModelLoadError(message)

    monkeypatch.setattr(cli.ChatterboxRuntime, "from_pretrained", staticmethod(reject))
    exit_code = cli.main(
        [
            "synth",
            "--model",
            "turbo",
            "--quality",
            "q4f16",
            "--voice",
            "reference.wav",
            "--text",
            "hello",
            "--output",
            "out.wav",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert calls["model"] == "turbo"
    assert calls["quality"] == "q4f16"
    assert calls["provider"] == "cpu"
    assert "q4f16" in captured.err
    assert "CPUExecutionProvider" in captured.err
    assert "AveragePool" in captured.err
