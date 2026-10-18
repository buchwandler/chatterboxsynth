# Cleanroom boundary

ChatterboxSynth is an independent ONNX synthesis frontend. It does not import, vendor, or call Resemble AI's Chatterbox Python implementation.

## Forbidden runtime imports and dependencies

The runtime must not import or depend on `chatterbox`, `chatterbox-tts`, `chatterbox_tts`, `torch`, `torchaudio`, or `transformers`. The standalone `tokenizers` package reads the installed bundle's pinned `tokenizer.json`; `numpy`, `soundfile`, and `soxr` cover numeric audio, WAV I/O, and resampling. ONNX model acquisition and graph execution remain in OnnxVoice.

A CI AST guard scans every package module plus the OnnxVoice Chatterbox catalog/runtime modules when that sibling checkout is available. Avoid dynamic imports of forbidden packages as well as normal import statements.

## Contract evidence

- Nano v2 export: `KitsuMate/chatterbox-nano-v2-onnx`, commit `851210533a5bcb4ad78e3ada6d527482dd1a31e7`.
- Turbo export: `ResembleAI/chatterbox-turbo-ONNX`, commit `d21799bd0354adb85e348b8a0442a8405110a2cf`.
- The bundle catalog records the tokenizer URLs, byte sizes, SHA-256 values, graph contracts, runtime defaults, and source revisions. The Nano v2 model card documents one trailing end-of-text token; the pinned Turbo tokenizer JSON declares two trailing end-of-text tokens in its template post-processor.
- Token-ID parity snapshots are derived independently from the exact public `tokenizer.json` files above. CI downloads only these tokenizer artifacts, verifies their catalog-pinned SHA-256 values, and checks the snapshots with the Rust `tokenizers` library; no model weights or reference voices are fetched for tokenizer tests.
- Reference-audio processing uses public file formats and `soundfile`/`soxr`; graph feeds and outputs are owned and validated by OnnxVoice.

## Contributor checklist

1. Read only public model metadata, ONNX graph signatures, the pinned tokenizer JSON, and independently generated fixtures when establishing behavior.
2. Keep generation, KV-cache handling, session ownership, catalog installation, and ONNX decoding in OnnxVoice.
3. Keep tokenizer behavior export-specific; never add global special-token policy without matching pinned-revision evidence.
4. Run `pytest`, Ruff, mypy, and the cleanroom AST guard before changing runtime behavior.
5. Update the recorded source revision and parity fixtures together when a model export changes; never substitute a moving `main` URL for a pinned artifact.
