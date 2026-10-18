from __future__ import annotations

import argparse
from pathlib import Path

from chatterboxsynth import ChatterboxRuntime, GenerationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize a first Chatterbox ONNX WAV")
    parser.add_argument("--model", choices=("nano", "turbo"), default="nano")
    parser.add_argument("--quality")
    parser.add_argument("--voice", type=Path, required=True, help="path to a short reference WAV")
    parser.add_argument("--output", type=Path, default=Path("first_wav.wav"))
    parser.add_argument("--text", default="Hello from the Chatterbox ONNX cleanroom MVP.")
    args = parser.parse_args()

    with ChatterboxRuntime.from_pretrained(
        args.model, quality=args.quality, provider="cpu"
    ) as runtime:
        voice = runtime.prepare_voice(args.voice)
        result = runtime.synthesize(args.text, voice=voice, config=GenerationConfig())
        result.save_wav(args.output)
    print(f"Wrote {args.output} at {result.sample_rate} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
