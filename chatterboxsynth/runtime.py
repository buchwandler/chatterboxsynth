from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._onnxvoice import normalize_model_ref, normalize_providers, onnxvoice_module
from .audio import load_reference_audio, normalize_reference_audio
from .config import GenerationConfig
from .errors import (
    IncompatibleVoiceError,
    ModelLoadError,
    RuntimeClosedError,
    SynthesisError,
)
from .frontend import ChatterboxFrontend
from .types import AudioPrompt, PreparedVoice, SynthesisResult

_SAMPLE_RATE = 24_000
_INSTALL_MANIFEST = ".onnxvoice-install.json"
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_COMPONENTS = {"speech_encoder", "embed_tokens", "language_model", "conditional_decoder"}
_PROFILES = {"nano-v2": {"cpu", "fp32"}, "turbo": {"fp32", "q4f16"}}
_EXTERNAL_DATA = {
    "nano-v2": {"cpu": {"language_model", "conditional_decoder"}, "fp32": {"conditional_decoder"}},
    "turbo": {"fp32": _COMPONENTS, "q4f16": _COMPONENTS},
}
_LOCAL_STATIC_ROLES = {
    "tokenizer",
    "tokenizer_config",
    "upstream_manifest",
    "model_config",
    "generation_config",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision(metadata: Mapping[str, Any]) -> str:
    revision = metadata.get("source_revision")
    source = metadata.get("source")
    if not isinstance(revision, str) and isinstance(source, Mapping):
        revision = source.get("revision")
    if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
        raise ModelLoadError("Bundle metadata must contain an exact 40-character source revision")
    return revision


def _installation_fields(
    installation: Any,
) -> tuple[str, str, str, int, dict[str, Any], dict[str, Any]]:
    metadata = getattr(installation, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ModelLoadError("OnnxVoice installation has no normalized Chatterbox metadata")
    bundle_id = getattr(installation, "id", None)
    if bundle_id not in _PROFILES or getattr(installation, "system", None) != "chatterbox":
        raise ModelLoadError("Installation is not a supported Chatterbox Nano v2 or Turbo bundle")
    runtime = metadata.get("runtime")
    frontend = metadata.get("frontend")
    if not isinstance(runtime, Mapping) or not isinstance(frontend, Mapping):
        raise ModelLoadError("Bundle is missing normalized runtime or frontend metadata")
    runtime_values = dict(runtime)
    frontend_values = dict(frontend)
    profile = metadata.get("selected_quality") or runtime_values.get("profile")
    if not isinstance(profile, str) or profile not in _PROFILES[bundle_id]:
        raise ModelLoadError(f"Bundle {bundle_id!r} has an unsupported or missing selected profile")
    rate = getattr(installation, "sample_rate", None)
    if isinstance(rate, bool) or not isinstance(rate, int) or rate != _SAMPLE_RATE:
        raise ModelLoadError(f"Chatterbox bundles must declare {_SAMPLE_RATE} Hz audio")
    contract = frontend_values.get("tokenizer_contract")
    expected_contract = "nano-v2" if bundle_id == "nano-v2" else "turbo"
    if contract != expected_contract:
        raise ModelLoadError("Bundle tokenizer contract does not match its canonical model")
    revision = _source_revision(metadata)
    return bundle_id, profile, revision, rate, runtime_values, frontend_values


def _safe_artifact_path(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename or "\\" in filename or "\x00" in filename:
        raise ModelLoadError("Local installation contains an unsafe artifact filename")
    relative = Path(filename)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ModelLoadError(
            f"Local installation contains an unsafe artifact filename: {filename!r}"
        )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ModelLoadError(f"Local artifact escapes its installation directory: {filename!r}")
    if not resolved.is_file():
        raise ModelLoadError(f"Local installation artifact is missing: {filename!r}")
    return resolved


def _local_files(
    root: Path, manifest: Mapping[str, Any], model: str, profile: str
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    metadata = manifest.get("metadata")
    artifacts = manifest.get("artifacts")
    if not isinstance(metadata, Mapping) or not isinstance(artifacts, list):
        raise ModelLoadError("OnnxVoice installation manifest has invalid metadata or artifacts")
    profiles = metadata.get("profiles")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(profile), Mapping):
        raise ModelLoadError("Local installation does not declare the selected Chatterbox profile")
    expected_profile = profiles[profile]
    if set(expected_profile) != _COMPONENTS:
        raise ModelLoadError("Local installation profile must select exactly four graph components")
    files: dict[str, Path] = {}
    artifact_metadata: dict[str, dict[str, Any]] = {}
    graphs: dict[str, tuple[str, str]] = {}
    external_data: list[tuple[str, str, str]] = []
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            raise ModelLoadError("Local installation contains a malformed artifact record")
        role = entry.get("role")
        component = entry.get("component")
        quality = entry.get("quality")
        filename = entry.get("filename")
        if not isinstance(role, str) or not role:
            raise ModelLoadError("Local installation artifact is missing its role")
        if role not in _LOCAL_STATIC_ROLES | {"graph", "external_data"}:
            raise ModelLoadError(f"Local installation contains unsupported artifact role {role!r}")
        if not isinstance(filename, str) or not filename:
            raise ModelLoadError("Local installation artifact is missing its filename")
        path = _safe_artifact_path(root, filename)
        size = entry.get("size")
        checksum = entry.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or path.stat().st_size != size
            or not isinstance(checksum, str)
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            or _file_sha256(path) != checksum
        ):
            raise ModelLoadError(
                f"Local installation artifact failed size/SHA-256 verification: {filename!r}"
            )
        if role in {"graph", "external_data"}:
            if (
                not isinstance(component, str)
                or component not in _COMPONENTS
                or not isinstance(quality, str)
            ):
                raise ModelLoadError(
                    f"Local {role} artifact has invalid component/quality selectors"
                )
            if expected_profile.get(component) != quality:
                raise ModelLoadError(
                    f"Local {role} artifact is not part of selected profile {profile!r}"
                )
            key = f"{role}:{component}"
            artifact_metadata[key] = {"quality": quality}
            if role == "graph":
                if component in graphs:
                    raise ModelLoadError(
                        f"Local installation contains duplicate {component} graphs"
                    )
                graphs[component] = (quality, filename)
            else:
                external_data.append((component, quality, filename))
        else:
            if component is not None or quality is not None:
                raise ModelLoadError(f"Static artifact {filename!r} has graph selectors")
            key = role
        if key in files:
            raise ModelLoadError(f"Local installation contains duplicate artifact role {key!r}")
        files[key] = path
    expected_external = {
        (component, expected_profile[component]) for component in _EXTERNAL_DATA[model][profile]
    }
    actual_external = {(component, quality) for component, quality, _ in external_data}
    if actual_external != expected_external:
        raise ModelLoadError(
            "Local installation external-data inventory differs from its graph profile"
        )
    if set(graphs) != _COMPONENTS or "tokenizer" not in files:
        raise ModelLoadError("Local installation must contain four graphs and one tokenizer")
    for component, quality, filename in external_data:
        graph_quality, graph_filename = graphs[component]
        if quality != graph_quality or filename not in {
            graph_filename + ".data",
            graph_filename + "_data",
        }:
            raise ModelLoadError(
                f"External-data artifact {filename!r} does not match its graph companion"
            )
    return files, artifact_metadata


class ChatterboxRuntime:
    """User-facing lifecycle and synthesis API over OnnxVoice's Chatterbox adapter."""

    def __init__(
        self,
        adapter: Any,
        frontend: ChatterboxFrontend,
        *,
        model: str,
        profile: str,
        source_revision: str,
        runtime_metadata: Mapping[str, Any],
        sample_rate: int = _SAMPLE_RATE,
        owns_adapter: bool = False,
    ) -> None:
        if model not in _PROFILES or profile not in _PROFILES[model]:
            raise ModelLoadError(f"Unsupported Chatterbox model/profile: {model!r}/{profile!r}")
        if sample_rate != _SAMPLE_RATE:
            raise ModelLoadError(f"Chatterbox runtime must produce {_SAMPLE_RATE} Hz audio")
        if not isinstance(source_revision, str) or _REVISION_RE.fullmatch(source_revision) is None:
            raise ModelLoadError("source_revision must be an exact 40-character SHA")
        if not isinstance(runtime_metadata, Mapping):
            raise ModelLoadError("runtime_metadata must be a mapping")
        self._adapter = adapter
        self._frontend = frontend
        self.model = model
        self.profile = profile
        self.sample_rate = sample_rate
        self.source_revision = source_revision
        self._runtime_metadata = dict(runtime_metadata)
        self._owns_adapter = owns_adapter
        self._closed = False
        self._runtime_fingerprint = hashlib.sha256(
            f"chatterbox\0{model}\0{profile}\0{source_revision}\0{sample_rate}".encode()
        ).hexdigest()

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        quality: str | None = None,
        provider: str | Sequence[str] | None = "cpu",
        provider_options: Any | None = None,
        session_options: Any | None = None,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        force_download: bool = False,
        catalog_sources: dict[str, str] | None = None,
        progress: Any | None = None,
    ) -> ChatterboxRuntime:
        """Install through OnnxVoice and open its cleanroom graph adapter."""
        ref, bundle_id = normalize_model_ref(model)
        if quality is not None and quality not in _PROFILES[bundle_id]:
            raise ValueError(f"Unsupported profile {quality!r} for {bundle_id}")
        module = onnxvoice_module()
        manager = module.OnnxVoice(
            cache_dir=cache_dir,
            offline=offline,
            catalog_sources=catalog_sources,
        )
        installation = manager.install(
            ref,
            quality=quality,
            force=force_download,
            progress=progress,
        )
        adapter = manager.open(
            installation,
            providers=normalize_providers(provider),
            provider_options=provider_options,
            session_options=session_options,
        )
        try:
            model_id, profile, revision, rate, runtime, frontend_metadata = _installation_fields(
                installation
            )
            tokenizer_path = installation.artifact("tokenizer").path
            frontend = ChatterboxFrontend(tokenizer_path, frontend_metadata["tokenizer_contract"])
            return cls(
                adapter,
                frontend,
                model=model_id,
                profile=profile,
                source_revision=revision,
                runtime_metadata=runtime,
                sample_rate=rate,
                owns_adapter=True,
            )
        except BaseException:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
            raise

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        quality: str | None = None,
        provider: str | Sequence[str] | None = "cpu",
        provider_options: Any | None = None,
        session_options: Any | None = None,
    ) -> ChatterboxRuntime:
        """Open a local OnnxVoice installation without catalog or network access."""
        root = Path(path).expanduser().resolve()
        manifest_path = root / _INSTALL_MANIFEST
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(
                f"Could not read OnnxVoice installation manifest at {manifest_path}"
            ) from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema") != 3:
            raise ModelLoadError("Unsupported or malformed OnnxVoice installation manifest")
        bundle_id = manifest.get("id")
        if manifest.get("system") != "chatterbox" or bundle_id not in _PROFILES:
            raise ModelLoadError("Local installation is not a supported Chatterbox bundle")
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ModelLoadError("Local installation is missing normalized bundle metadata")
        runtime = metadata.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ModelLoadError("Local installation is missing runtime metadata")
        profile = metadata.get("selected_quality") or runtime.get("profile")
        if not isinstance(profile, str) or profile not in _PROFILES[bundle_id]:
            raise ModelLoadError("Local installation is missing its selected Chatterbox profile")
        if quality is not None and quality != profile:
            raise ModelLoadError(
                f"Local installation contains profile {profile!r}, not the requested {quality!r}"
            )
        installation_view = type(
            "LocalInstallationView",
            (),
            {
                "id": bundle_id,
                "system": "chatterbox",
                "metadata": metadata,
                "sample_rate": manifest.get("sample_rate"),
            },
        )()
        model_id, selected, revision, rate, runtime_values, frontend_metadata = (
            _installation_fields(installation_view)
        )
        files, artifact_metadata = _local_files(root, manifest, model_id, selected)
        module = onnxvoice_module()
        adapter = module.open_local(
            system="chatterbox",
            files=files,
            artifact_metadata=artifact_metadata,
            metadata=dict(metadata),
            runtime=runtime_values,
            sample_rate=rate,
            providers=normalize_providers(provider),
            provider_options=provider_options,
            session_options=session_options,
        )
        try:
            frontend = ChatterboxFrontend(
                files["tokenizer"], frontend_metadata["tokenizer_contract"]
            )
            return cls(
                adapter,
                frontend,
                model=model_id,
                profile=selected,
                source_revision=revision,
                runtime_metadata=runtime_values,
                sample_rate=rate,
                owns_adapter=True,
            )
        except BaseException:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
            raise

    @classmethod
    def models(
        cls,
        *,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        catalog_sources: dict[str, str] | None = None,
        refresh: bool = False,
    ) -> tuple[Any, ...]:
        """List normalized Chatterbox items from OnnxVoice's catalog."""
        module = onnxvoice_module()
        manager = module.OnnxVoice(
            cache_dir=cache_dir,
            offline=offline,
            catalog_sources=catalog_sources,
        )
        return tuple(manager.list("chatterbox", refresh=refresh))

    @classmethod
    def download(
        cls,
        model: str,
        *,
        quality: str | None = None,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        force_download: bool = False,
        catalog_sources: dict[str, str] | None = None,
        progress: Any | None = None,
    ) -> Any:
        """Install a selected profile through OnnxVoice without opening sessions."""
        ref, bundle_id = normalize_model_ref(model)
        if quality is not None and quality not in _PROFILES[bundle_id]:
            raise ValueError(f"Unsupported profile {quality!r} for {bundle_id}")
        module = onnxvoice_module()
        manager = module.OnnxVoice(
            cache_dir=cache_dir,
            offline=offline,
            catalog_sources=catalog_sources,
        )
        return manager.install(
            ref,
            quality=quality,
            force=force_download,
            progress=progress,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeClosedError("ChatterboxRuntime is closed")

    def prepare_voice(self, source: str | Path | AudioPrompt | PreparedVoice) -> PreparedVoice:
        """Prepare reference audio once, returning an immutable reusable voice state."""
        self._ensure_open()
        if isinstance(source, PreparedVoice):
            self._validate_voice(source)
            return source
        if isinstance(source, AudioPrompt):
            audio = normalize_reference_audio(
                source.samples,
                source.sample_rate,
                target_sample_rate=self.sample_rate,
            )
            label = None
        elif isinstance(source, (str, Path)):
            audio = load_reference_audio(source, target_sample_rate=self.sample_rate)
            label = str(Path(source).expanduser())
        else:
            raise TypeError("source must be a path, AudioPrompt, or compatible PreparedVoice")
        state = self._adapter.prepare_voice(audio)
        return PreparedVoice(
            state=state,
            sample_rate=self.sample_rate,
            model=self.model,
            profile=self.profile,
            source_revision=self.source_revision,
            runtime_fingerprint=self._runtime_fingerprint,
            source=label,
        )

    def _validate_voice(self, voice: PreparedVoice) -> None:
        if (
            voice.model != self.model
            or voice.profile != self.profile
            or voice.source_revision != self.source_revision
            or voice.sample_rate != self.sample_rate
            or voice.runtime_fingerprint != self._runtime_fingerprint
        ):
            raise IncompatibleVoiceError(
                "PreparedVoice belongs to an incompatible Chatterbox model, profile, or source revision"
            )

    def synthesize(
        self,
        text: str,
        *,
        voice: PreparedVoice,
        config: GenerationConfig | None = None,
    ) -> SynthesisResult:
        """Generate one deterministic greedy synthesis result without text chunking."""
        self._ensure_open()
        if not isinstance(voice, PreparedVoice):
            raise TypeError("voice must be a PreparedVoice returned by prepare_voice()")
        self._validate_voice(voice)
        if config is not None and not isinstance(config, GenerationConfig):
            raise TypeError("config must be GenerationConfig or None")
        token_ids = self._frontend.encode(text)
        max_new_tokens = (
            config.max_new_tokens
            if config and config.max_new_tokens is not None
            else self._runtime_metadata.get("default_max_new_tokens")
        )
        repetition_penalty = (
            config.repetition_penalty
            if config and config.repetition_penalty is not None
            else self._runtime_metadata.get("default_repetition_penalty")
        )
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
            or isinstance(repetition_penalty, bool)
            or not isinstance(repetition_penalty, (int, float))
            or not math.isfinite(float(repetition_penalty))
            or repetition_penalty <= 0
        ):
            raise ModelLoadError("Bundle has invalid default Chatterbox generation settings")
        inference = self._adapter.infer(
            token_ids,
            voice_state=voice.state,
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(repetition_penalty),
        )
        result_metadata = getattr(inference, "metadata", None)
        if not isinstance(result_metadata, Mapping):
            raise SynthesisError("OnnxVoice adapter returned no generation metadata")
        output_rate = getattr(inference, "sample_rate", None)
        if output_rate != self.sample_rate:
            raise SynthesisError(
                f"OnnxVoice adapter returned {output_rate!r} Hz audio; expected {self.sample_rate} Hz"
            )
        generated_tokens = result_metadata.get("generated_tokens")
        stopped_on_eos = result_metadata.get("stopped_on_eos")
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens < 0
        ):
            raise SynthesisError("OnnxVoice adapter returned invalid generated-token metadata")
        if not isinstance(stopped_on_eos, bool):
            raise SynthesisError("OnnxVoice adapter returned invalid end-of-sequence metadata")
        return SynthesisResult(
            audio=getattr(inference, "audio", None),
            sample_rate=self.sample_rate,
            model=self.model,
            profile=self.profile,
            generated_tokens=generated_tokens,
            stopped_on_eos=stopped_on_eos,
            text=text,
        )

    def close(self) -> None:
        """Close only the adapter/session resources this runtime created."""
        if self._closed:
            return
        self._closed = True
        if self._owns_adapter:
            close = getattr(self._adapter, "close", None)
            if callable(close):
                close()

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> ChatterboxRuntime:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
