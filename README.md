# ChatterboxSynth

Cleanroom Python synthesis frontend for the revision-pinned Chatterbox Nano v2 and Turbo ONNX bundles. OnnxVoice owns catalog access, installation, ONNX sessions, and the four-graph runtime; ChatterboxSynth owns tokenizer policy, reference audio, prepared voices, synthesis results, WAV output, and CLI.

## Install

```bash
pip install "chatterboxsynth[cpu]"
```

ChatterboxSynth 0.1.x requires a Chatterbox-capable OnnxVoice 0.2.x release.
Install `chatterboxsynth[cpu]` for the default ONNX Runtime CPU provider or
`chatterboxsynth[gpu]` for CUDA.

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

Use `ChatterboxRuntime.load("/path/to/onnxvoice/install")` for an already-installed OnnxVoice bundle. This local path reads the install manifest and verifies artifact sizes and SHA-256 values; it does not query a catalog or access the network.

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
