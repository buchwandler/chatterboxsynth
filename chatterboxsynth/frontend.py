from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import EmptyTextError, OptionalDependencyError, SynthesisError

_NANO_CONTRACT = "nano-v2"
_TURBO_CONTRACT = "turbo"
_END_OF_TEXT = "<|endoftext|>"


class ChatterboxFrontend:
    """Standalone tokenizer frontend with export-specific EOT handling."""

    def __init__(
        self,
        tokenizer_path: str | Path,
        tokenizer_contract: str,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        if tokenizer_contract not in {_NANO_CONTRACT, _TURBO_CONTRACT}:
            raise ValueError(f"Unsupported Chatterbox tokenizer contract: {tokenizer_contract!r}")
        if tokenizer is None:
            try:
                from tokenizers import Tokenizer
            except ModuleNotFoundError as exc:
                raise OptionalDependencyError(
                    "Chatterbox tokenization requires tokenizers; install chatterboxsynth."
                ) from exc
            try:
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
            except (OSError, ValueError) as exc:
                raise SynthesisError(f"Could not load tokenizer {tokenizer_path}: {exc}") from exc
        self.tokenizer_path = Path(tokenizer_path)
        self.tokenizer_contract = tokenizer_contract
        self._tokenizer = tokenizer
        try:
            self._eot_id = tokenizer.token_to_id(_END_OF_TEXT)
        except AttributeError as exc:
            raise SynthesisError("Tokenizer does not expose token_to_id()") from exc
        if not isinstance(self._eot_id, int) or isinstance(self._eot_id, bool):
            raise SynthesisError("Pinned tokenizer does not contain the expected end-of-text token")

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode exact user text without generic special-token insertion.

        The pinned Nano v2 tokenizer post-processor adds one trailing EOT token;
        the pinned Turbo tokenizer post-processor adds two. Both policies are
        verified against the export-specific contract. User-supplied EOT tokens
        are rejected so text cannot inject a generation stop.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise EmptyTextError("text must contain non-whitespace characters")
        try:
            raw_ids = tuple(
                int(value) for value in self._tokenizer.encode(text, add_special_tokens=False).ids
            )
            token_ids = tuple(
                int(value) for value in self._tokenizer.encode(text, add_special_tokens=True).ids
            )
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise SynthesisError(f"Tokenizer could not encode text: {exc}") from exc
        if not raw_ids:
            raise EmptyTextError("text produced no tokenizer IDs")
        if self._eot_id in raw_ids:
            raise SynthesisError("text contains the reserved end-of-text token")
        expected_eot_count = 1 if self.tokenizer_contract == _NANO_CONTRACT else 2
        expected_suffix = (self._eot_id,) * expected_eot_count
        if token_ids[: len(raw_ids)] != raw_ids or token_ids[len(raw_ids) :] != expected_suffix:
            raise SynthesisError("Pinned tokenizer special-token post-processor changed")
        return token_ids
