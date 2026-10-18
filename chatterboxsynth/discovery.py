from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._onnxvoice import onnxvoice_module
from .errors import ModelLoadError
from .runtime import _source_revision
from .types import DiscoveredBundle

_BUNDLE_PROFILES = {
    "nano-v2": ("cpu", "fp32"),
    "turbo": ("fp32", "q4f16"),
}
_DISPLAY_NAMES = {"nano-v2": "Chatterbox Nano v2", "turbo": "Chatterbox Turbo"}


def _languages(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    value = metadata.get("languages", metadata.get("language", "en"))
    values: tuple[str, ...]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(item for item in value if isinstance(item, str))
    else:
        values = ()
    return tuple(sorted({language.lower() for language in values}))


def _installed_profiles(installations: Sequence[Any], bundle_id: str) -> set[str]:
    found: set[str] = set()
    for installation in installations:
        if getattr(installation, "system", "chatterbox") != "chatterbox":
            continue
        if getattr(installation, "id", None) != bundle_id:
            continue
        metadata = getattr(installation, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        runtime = metadata.get("runtime")
        profile = metadata.get("selected_quality") or metadata.get("quality")
        if not isinstance(profile, str) and isinstance(runtime, Mapping):
            profile = runtime.get("profile")
        if isinstance(profile, str) and profile in _BUNDLE_PROFILES[bundle_id]:
            found.add(profile)
    return found


def discover_bundles(
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    catalog_sources: dict[str, str] | None = None,
) -> tuple[DiscoveredBundle, ...]:
    """Discover normalized Chatterbox catalog bundles and locally installed profiles."""
    module = onnxvoice_module()
    manager = module.OnnxVoice(
        cache_dir=cache_dir,
        offline=offline,
        catalog_sources=catalog_sources,
    )
    catalog_items = manager.list("chatterbox", refresh=refresh)
    installations = manager.list("chatterbox", installed=True)
    by_id: dict[str, Any] = {}
    for item in catalog_items:
        bundle_id = getattr(item, "id", None)
        if bundle_id in _BUNDLE_PROFILES:
            by_id[bundle_id] = item

    bundles: list[DiscoveredBundle] = []
    for bundle_id in sorted(by_id):
        item = by_id[bundle_id]
        metadata = getattr(item, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise ModelLoadError(f"Catalog bundle {bundle_id!r} has no normalized metadata")
        raw_profiles = metadata.get("profiles")
        if not isinstance(raw_profiles, Mapping):
            raise ModelLoadError(f"Catalog bundle {bundle_id!r} has no normalized profiles")
        profiles = tuple(
            profile for profile in _BUNDLE_PROFILES[bundle_id] if profile in raw_profiles
        )
        default_profile = metadata.get("default_profile")
        if not profiles or not isinstance(default_profile, str) or default_profile not in profiles:
            raise ModelLoadError(f"Catalog bundle {bundle_id!r} has invalid profile metadata")
        sample_rate = getattr(item, "sample_rate", None)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ModelLoadError(f"Catalog bundle {bundle_id!r} has an invalid sample rate")
        installed = _installed_profiles(installations, bundle_id)
        source = metadata.get("source")
        repository = source.get("repository") if isinstance(source, Mapping) else None
        runtime = metadata.get("runtime")
        bundles.append(
            DiscoveredBundle(
                id=bundle_id,
                ref=f"chatterbox:{bundle_id}",
                display_name=_DISPLAY_NAMES[bundle_id],
                aliases=tuple(sorted(getattr(item, "aliases", ()), key=str.casefold)),
                languages=_languages(metadata),
                profiles=profiles,
                default_profile=default_profile,
                sample_rate=sample_rate,
                source_revision=_source_revision(metadata),
                runtime_available=bool(installed),
                installed_profiles=tuple(profile for profile in profiles if profile in installed),
                metadata={
                    "source_repository": repository,
                    "runtime": dict(runtime) if isinstance(runtime, Mapping) else {},
                },
            )
        )
    return tuple(bundles)
