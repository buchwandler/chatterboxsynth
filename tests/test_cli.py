from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import chatterboxsynth.__main__ as cli


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
