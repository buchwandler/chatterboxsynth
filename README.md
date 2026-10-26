# ChatterboxSynth

Cleanroom Python synthesis frontend for the revision-pinned Chatterbox Nano v2 and Turbo ONNX bundles. OnnxVoice owns catalog access, installation, ONNX sessions, and the four-graph runtime; ChatterboxSynth owns tokenizer policy, reference audio, prepared voices, synthesis results, WAV output, and CLI.

## Install

```bash
pip install "chatterboxsynth[cpu]"
```

ChatterboxSynth 0.1.x requires a Chatterbox-capable OnnxVoice 0.2.x release.
Install `chatterboxsynth[cpu]` for the default ONNX Runtime CPU provider or
`chatterboxsynth[gpu]` for CUDA.

ChatterboxSynth currently pins Python-SoXR 1.1.0 for reference-audio resampling.

## Quick start

```python
from chatterboxsynth import ChatterboxRuntime, GenerationConfig

with ChatterboxRuntime.from_pretrained("nano", quality="cpu", provider="cpu") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    result = runtime.synthesize(
        "Hello from Chatterbox Nano.",
        voice=voice,
        config=GenerationConfig(max_new_tokens=1024, repetition_penalty=1.2),
    )
    result.save_wav("hello.wav")
```

Turbo supports `[chuckle]`, `[laugh]`, and other model-native tags:

```python
with ChatterboxRuntime.from_pretrained("turbo", quality="fp32") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    runtime.synthesize("Hello [chuckle] from Turbo.", voice=voice).save_wav("turbo.wav")
```

## Turbo quality profiles

- **Turbo fp32:** CPU-capable baseline, subject to available device memory.
- **Turbo q4f16:** Provider-dependent, not a CPUExecutionProvider profile. The currently pinned published speech encoder requires `AveragePool` opset 19, which CPUExecutionProvider does not support. An explicit `quality="q4f16"` request with CPU is rejected before download; use Turbo fp32 for CPU or select an execution provider that supports the graph. No particular GPU provider is promised without end-to-end validation.

Use `ChatterboxRuntime.load("/path/to/onnxvoice/install")` for an already-installed OnnxVoice bundle. This local path reads the install manifest and verifies artifact sizes and SHA-256 values; it does not query a catalog or access the network.

For render-time opening of a managed model, use `ChatterboxRuntime.from_installed("nano", quality="cpu")`. It resolves and opens only a locally installed profile and never installs a model or refreshes a catalog. By contrast, `from_pretrained()` is the convenience path that may acquire and install a model.

## CLI

```bash
chatterboxsynth models
chatterboxsynth download nano --quality cpu
chatterboxsynth synth --model nano --quality cpu --voice reference.wav --text "Hello" --output hello.wav
chatterboxsynth synth --model turbo --quality fp32 --voice reference.wav --text "Hello [chuckle]" --output turbo.wav
```

Common options include `--provider cpu|cuda|auto`, `--offline`, `--cache-dir`, `--max-new-tokens`, and `--repetition-penalty`. All catalog, cache, and model-install behavior is delegated to OnnxVoice.

## Audio and request policy

`prepare_voice()` accepts an audio filename, `AudioPrompt(samples, sample_rate)`, or a compatible `PreparedVoice`. Stereo audio is deterministically averaged to mono, validated as finite and within `[-1, 1]`, and resampled to 24 kHz with `soxr` HQ. Prepared voice states cannot be reused across incompatible model/profile/source revisions.

Synthesis is deterministic greedy decoding. Empty text is rejected. The MVP does not split long text, stitch chunks, use random sampling, apply PerTh, or persist voice states. `SynthesisResult` validates non-empty finite mono float32 output at the bundle's 24 kHz rate; PCM16 WAV conversion clips only at the save boundary.

## Request measurement and discovery

The public request contract is available from the package root: `REQUEST_API_VERSION`, `request_api_contract()`, `SynthesisRequest`, `RequestMeasure`, `discover_bundles()`, and `runtime_identity()`. `runtime.measure_request(SynthesisRequest(id="request-1", text="Hello"))` runs the same frontend encoding used by synthesis and reports the exact model-token count, including the export-specific EOT tokens. Measurement is tokenizer-only: it does not require a prepared voice, run ONNX inference, or acquire a model. Current bundles declare no finite text-input maximum, so accepted requests report `maximum=None` and `fits=True`. `max_new_tokens` and the bundle's `default_max_new_tokens` cap generated speech tokens; they are not text-input limits.

`discover_bundles()` normalizes the Chatterbox catalog and annotates profiles already present in the local OnnxVoice inventory. Discovery may refresh catalog metadata when requested, but it does not install models or open ONNX sessions.

## Cleanroom and development

See [docs/cleanroom.md](docs/cleanroom.md) and [docs/architecture.md](docs/architecture.md). Runtime dependencies exclude Chatterbox Python packages, Torch, Torchaudio, and Transformers. Tokenizers are loaded from the installed bundle, never package-global data.

```bash
pip install -e ".[cpu,dev]"
pytest
ruff check chatterboxsynth tests examples
mypy
python -m build
```

`examples/first_wav.py --voice reference.wav` uses a user-provided short WAV. No voice recording or model weights are included. `setuptools_scm` resolves package versions from Git metadata; a raw source ZIP without `.git` may report the documented fallback `0.0.0`. Build publishable wheels and sdists from tagged Git checkouts.
