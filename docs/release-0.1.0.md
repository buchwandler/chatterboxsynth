# ChatterboxSynth 0.1.0 Release Checklist

This checklist prepares and verifies the first public release. It does not publish a GitHub Release, create/push a release tag, configure PyPI, or upload a package. Complete the external checks below before publishing. ChatterboxSynth 0.1.x requires the Chatterbox-capable OnnxVoice release `>=0.2.11,<0.3`.

## Release candidate: real-model gates

Run these checks on a supported Linux CPU environment using the published release candidate and a user-provided, licensed `reference.wav`. Do not commit the recording or model artifacts. The first test covers Nano CPU; the second covers Turbo FP32. The optional Turbo q4f16 profile is not a universal CPU guarantee.

Create an isolated environment and install the public release candidate:

```bash
python -m venv .venv-rc
. .venv-rc/bin/activate
python -m pip install --upgrade pip
python -m pip install "chatterboxsynth[cpu]==0.1.0rc1"
```

For a locally built wheel instead of an RC upload, install it together with the published OnnxVoice CPU extra:

```bash
python -m pip install "onnxvoice[cpu]>=0.2.11,<0.3" dist/chatterboxsynth-0.1.0-py3-none-any.whl
```

### Nano CPU

Run the following from a directory containing `reference.wav`:

```python
from pathlib import Path

import numpy as np
import soundfile as sf
from chatterboxsynth import ChatterboxRuntime

with ChatterboxRuntime.from_pretrained("nano", quality="cpu") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    first = runtime.synthesize(
        "This is the first release smoke test.", voice=voice
    )
    second = runtime.synthesize(
        "The prepared voice is reused for a second sentence.", voice=voice
    )

    for result in (first, second):
        assert result.sample_rate == 24_000
        assert result.audio.ndim == 1 and result.audio.size > 0
        assert np.isfinite(result.audio).all()
        assert result.model == "nano"
        assert result.profile == "cpu"
        assert isinstance(result.generated_tokens, int) and result.generated_tokens >= 0
        assert isinstance(result.stopped_on_eos, bool)

    first.save_wav("nano-smoke.wav")
    second.save_wav("nano-reuse-smoke.wav")

for output in ("nano-smoke.wav", "nano-reuse-smoke.wav"):
    samples, sample_rate = sf.read(Path(output), dtype="float32")
    assert sample_rate == 24_000
    assert samples.ndim == 1 and samples.size > 0
    assert np.isfinite(samples).all()
```

Pass only if installation, reference preparation, both synthesis calls, reuse of the same prepared voice, finite audio and metadata checks, PCM16 WAV reopening at 24 kHz, and clean context-manager shutdown all succeed without a runtime/provider error.

### Turbo FP32

Repeat in the same isolated environment with Turbo and a tag-bearing sentence:

```python
import numpy as np
import soundfile as sf
from chatterboxsynth import ChatterboxRuntime

with ChatterboxRuntime.from_pretrained("turbo", quality="fp32") as runtime:
    voice = runtime.prepare_voice("reference.wav")
    result = runtime.synthesize(
        "Hello [chuckle] from the ChatterboxSynth release candidate.",
        voice=voice,
    )
    assert result.sample_rate == 24_000
    assert result.audio.ndim == 1 and result.audio.size > 0
    assert np.isfinite(result.audio).all()
    assert result.model == "turbo"
    assert result.profile == "fp32"
    assert isinstance(result.generated_tokens, int) and result.generated_tokens >= 0
    assert isinstance(result.stopped_on_eos, bool)
    result.save_wav("turbo-smoke.wav")

samples, sample_rate = sf.read("turbo-smoke.wav", dtype="float32")
assert sample_rate == 24_000
assert samples.ndim == 1 and samples.size > 0
assert np.isfinite(samples).all()
```

Pass only if the real published Turbo export accepts the tag-containing text, synthesis completes, WAV output reopens correctly, and the runtime closes cleanly. Test `turbo/q4f16` only on a provider known to support its operators; do not treat it as a CPU release guarantee.

## Metadata consistency checklist

Confirm the release notes, README, package metadata, documentation, and tagged distribution agree on these public contracts:

- Package name: `chatterboxsynth`; public release tag: `v0.1.0`; tagged package version: `0.1.0`.
- Supported runtime models: Chatterbox Nano v2 and Turbo only.
- Supported Python versions: 3.10 through 3.13.
- Native output sample rate: 24 kHz.
- Nano profiles: `cpu` and `fp32`; Turbo profiles: `fp32` and `q4f16` (subject to provider support).
- Minimum Chatterbox-capable OnnxVoice: `0.2.11`; runtime remains cleanroom and does not depend on Chatterbox Python packages, Torch, Torchaudio, or Transformers.

## Tagged package and dependency checks

From a clean checkout of the release tag, verify the checkout and version:

```bash
git status --short
git describe --tags --exact-match
```

Expected tag: `v0.1.0`; the working tree should be clean. Then run:

```bash
python -m pip install -e ".[cpu,dev]"
python -m pytest -q
python -m ruff check chatterboxsynth tests examples
python -m mypy
python -m build
python -m twine check dist/*
```

Inspect both archives:

```bash
python -m zipfile -l dist/chatterboxsynth-0.1.0-py3-none-any.whl
tar -tzf dist/chatterboxsynth-0.1.0.tar.gz
```

The wheel should contain the package modules (`__init__.py`, `__main__.py`, `_onnxvoice.py`, `_version.py`, `audio.py`, `config.py`, `errors.py`, `frontend.py`, `runtime.py`, `types.py`) and `py.typed`. Neither artifact may contain ONNX model files, reference recordings, Chatterbox Python source, Torch artifacts, local cache data, or test-download artifacts.

In a fresh environment, verify resolution against the public dependency:

```bash
python -m pip install "chatterboxsynth[cpu]==0.1.0"
python - <<'PY'
from packaging.version import Version
import chatterboxsynth
import onnxvoice

assert chatterboxsynth.__version__ == "0.1.0"
assert Version(onnxvoice.__version__) >= Version("0.2.11")
assert Version(onnxvoice.__version__) < Version("0.3")
print("chatterboxsynth", chatterboxsynth.__version__)
print("onnxvoice", onnxvoice.__version__)
PY
chatterboxsynth --help
chatterboxsynth models
```

The `models` command may contact the catalog; for an online release smoke, confirm that the public Chatterbox catalog resolves successfully. The publish workflow independently checks that the release tag (after removing an optional leading `v`) equals `chatterboxsynth.__version__`, runs tests/Ruff/mypy/build/Twine checks, and installs the built wheel in a clean virtual environment for import and CLI-help smoke tests before its final Trusted Publishing action.

## GitHub and PyPI gates

Before publishing the GitHub Release, an authorized project operator must confirm that the PyPI Trusted Publisher is configured for all of the following:

- GitHub owner and repository: `buchwandler/chatterboxsynth`.
- Workflow filename: `.github/workflows/python-publish.yml`.
- PyPI environment name: `pypi`.

Also verify the GitHub release tag is exactly `v0.1.0` and all CI and real-model gates above have passed. The workflow listens only for a published GitHub Release and is designed to stop before upload if any preceding dependency, version, quality, build, metadata, or clean-wheel check fails. Do not publish the GitHub Release until the Trusted Publisher configuration has been confirmed; do not treat documentation or local smoke tests as evidence that the external publisher is configured.
