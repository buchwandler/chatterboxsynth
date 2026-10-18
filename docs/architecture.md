# Architecture

ChatterboxSynth is a thin user-facing synthesis layer over OnnxVoice.

- **OnnxVoice** resolves and installs catalog bundles, verifies artifacts, creates ONNX sessions, and owns the four-graph Chatterbox adapter.
- **ChatterboxSynth** selects the export-specific tokenizer policy, loads/downmixes/resamples reference audio, manages immutable prepared voices, exposes deterministic generation settings, validates results, and writes WAV files.
- The data-only `chatterbox-onnx-bundles` repository owns source-pinned bundle metadata and catalog refreshes.

`ChatterboxRuntime.from_pretrained()` delegates model lookup and installation to OnnxVoice. `ChatterboxRuntime.load()` reads an OnnxVoice installation manifest and opens its files with `OnnxVoice.open_local`; it does not query a catalog or access the network. Neither path duplicates a downloader, asset cache, session factory, graph loop, or decoder.

The initial API supports English Nano v2 (`cpu`, `fp32`) and Turbo (`fp32`, `q4f16`) only. Synthesis is greedy and single-request: no random sampling, sentence splitting, chunk stitching, watermark, or persistent voice cache is added.
