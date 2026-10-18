from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import GenerationConfig
from .runtime import ChatterboxRuntime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chatterboxsynth")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="list normalized Chatterbox catalog entries")
    models.add_argument("--cache-dir", type=Path)
    models.add_argument("--offline", action="store_true")
    models.add_argument("--refresh", action="store_true")

    download = subparsers.add_parser("download", help="install a model profile through OnnxVoice")
    download.add_argument("model")
    download.add_argument("--quality")
    download.add_argument("--cache-dir", type=Path)
    download.add_argument("--offline", action="store_true")
    download.add_argument("--force", action="store_true")

    synth = subparsers.add_parser("synth", help="synthesize one WAV without automatic chunking")
    synth.add_argument("--model", required=True)
    synth.add_argument("--quality")
    synth.add_argument("--provider", choices=("cpu", "cuda", "auto"), default="cpu")
    synth.add_argument("--cache-dir", type=Path)
    synth.add_argument("--offline", action="store_true")
    synth.add_argument("--voice", required=True, type=Path)
    synth.add_argument("--text", required=True)
    synth.add_argument("--output", required=True, type=Path)
    synth.add_argument("--max-new-tokens", type=int)
    synth.add_argument("--repetition-penalty", type=float)
    return parser


def _show_models(args: argparse.Namespace) -> int:
    items = ChatterboxRuntime.models(
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh=args.refresh,
    )
    for item in items:
        profiles = item.metadata.get("profiles", {})
        default_profile = item.metadata.get("default_profile", "unknown")
        profile_names = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else "unknown"
        aliases = ", ".join(item.aliases) if item.aliases else "none"
        print(
            f"{item.ref}\tdefault={default_profile}\tprofiles={profile_names}"
            f"\taliases={aliases}\tsample_rate={item.sample_rate}"
        )
    return 0


def _download(args: argparse.Namespace) -> int:
    installation = ChatterboxRuntime.download(
        args.model,
        quality=args.quality,
        cache_dir=args.cache_dir,
        offline=args.offline,
        force_download=args.force,
    )
    print(
        f"Installed {installation.ref} profile={installation.metadata.get('selected_quality')} "
        f"at {installation.path}"
    )
    return 0


def _synthesize(args: argparse.Namespace) -> int:
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    with ChatterboxRuntime.from_pretrained(
        args.model,
        quality=args.quality,
        provider=args.provider,
        cache_dir=args.cache_dir,
        offline=args.offline,
    ) as runtime:
        voice = runtime.prepare_voice(args.voice)
        result = runtime.synthesize(args.text, voice=voice, config=config)
        result.save_wav(args.output)
    print(
        f"Wrote {args.output} ({result.sample_rate} Hz, {result.duration_seconds:.2f} s, "
        f"{result.generated_tokens} speech tokens)"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    actions: dict[str, Any] = {"models": _show_models, "download": _download, "synth": _synthesize}
    try:
        return actions[args.command](args)
    except Exception as exc:
        print(f"chatterboxsynth: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
