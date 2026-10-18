from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chatterboxsynth.discovery as discovery_module
from chatterboxsynth import DiscoveredBundle, discover_bundles

NANO_REVISION = "851210533a5bcb4ad78e3ada6d527482dd1a31e7"
TURBO_REVISION = "d21799bd0354adb85e348b8a0442a8405110a2cf"


def catalog_item(
    bundle_id: str,
    *,
    aliases: tuple[str, ...],
    profiles: tuple[str, ...],
    default_profile: str,
    revision: str,
) -> Any:
    profile_data = {profile: {} for profile in profiles}
    return SimpleNamespace(
        id=bundle_id,
        sample_rate=24_000,
        aliases=aliases,
        metadata={
            "language": "en",
            "profiles": profile_data,
            "default_profile": default_profile,
            "source_revision": revision,
            "source": {"revision": revision, "requested_revision": "main", "repository": bundle_id},
            "runtime": {"default_max_new_tokens": 1024},
        },
    )


def test_discovery_normalizes_catalog_and_installed_profiles_without_runtime_sessions(
    monkeypatch: Any, tmp_path: Path
) -> None:
    nano = catalog_item(
        "nano-v2",
        aliases=("nano-110m", "nano"),
        profiles=("fp32", "cpu"),
        default_profile="cpu",
        revision=NANO_REVISION,
    )
    turbo = catalog_item(
        "turbo",
        aliases=("turbo-350m",),
        profiles=("q4f16", "fp32"),
        default_profile="fp32",
        revision=TURBO_REVISION,
    )
    installations = [
        SimpleNamespace(id="nano-v2", system="chatterbox", metadata={"selected_quality": "cpu"}),
        SimpleNamespace(
            id="turbo", system="chatterbox", metadata={"runtime": {"profile": "q4f16"}}
        ),
        SimpleNamespace(
            id="nano-v2", system="chatterbox", metadata={"selected_quality": "invalid"}
        ),
    ]
    calls: dict[str, Any] = {"list": []}

    class Manager:
        def __init__(self, **kwargs: Any) -> None:
            calls["manager"] = kwargs

        def list(self, system: str, *, installed: bool = False, refresh: bool = False) -> Any:
            calls["list"].append((system, installed, refresh))
            return installations if installed else [turbo, nano]

        def install(self, *_: Any, **__: Any) -> None:
            raise AssertionError("discovery must not install models")

        def open(self, *_: Any, **__: Any) -> None:
            raise AssertionError("discovery must not open runtime sessions")

    monkeypatch.setattr(
        discovery_module, "onnxvoice_module", lambda: SimpleNamespace(OnnxVoice=Manager)
    )
    sources = {"chatterbox": str(tmp_path / "catalog.json")}

    bundles = discover_bundles(
        cache_dir=tmp_path / "cache",
        offline=True,
        refresh=True,
        catalog_sources=sources,
    )

    assert all(isinstance(bundle, DiscoveredBundle) for bundle in bundles)
    assert [bundle.id for bundle in bundles] == ["nano-v2", "turbo"]
    nano_result, turbo_result = bundles
    assert nano_result.ref == "chatterbox:nano-v2"
    assert nano_result.display_name == "Chatterbox Nano v2"
    assert nano_result.aliases == ("nano", "nano-110m")
    assert nano_result.languages == ("en",)
    assert nano_result.profiles == ("cpu", "fp32")
    assert nano_result.default_profile == "cpu"
    assert nano_result.sample_rate == 24_000
    assert nano_result.source_revision == NANO_REVISION
    assert nano_result.runtime_available is True
    assert nano_result.installed_profiles == ("cpu",)
    assert turbo_result.aliases == ("turbo-350m",)
    assert turbo_result.languages == ("en",)
    assert turbo_result.profiles == ("fp32", "q4f16")
    assert turbo_result.default_profile == "fp32"
    assert turbo_result.source_revision == TURBO_REVISION
    assert turbo_result.runtime_available is True
    assert turbo_result.installed_profiles == ("q4f16",)
    assert calls["manager"] == {
        "cache_dir": tmp_path / "cache",
        "offline": True,
        "catalog_sources": sources,
    }
    assert calls["list"] == [("chatterbox", False, True), ("chatterbox", True, False)]


def test_discovery_reports_no_runtime_when_no_profile_is_installed(monkeypatch: Any) -> None:
    item = catalog_item(
        "nano-v2",
        aliases=("nano", "nano-110m"),
        profiles=("cpu", "fp32"),
        default_profile="cpu",
        revision=NANO_REVISION,
    )

    class Manager:
        def __init__(self, **_: Any) -> None:
            pass

        def list(self, _system: str, *, installed: bool = False, refresh: bool = False) -> Any:
            return [] if installed else [item]

    monkeypatch.setattr(
        discovery_module, "onnxvoice_module", lambda: SimpleNamespace(OnnxVoice=Manager)
    )
    result = discover_bundles()
    assert len(result) == 1
    assert result[0].runtime_available is False
    assert result[0].installed_profiles == ()
