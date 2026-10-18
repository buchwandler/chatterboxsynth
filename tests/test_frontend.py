from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chatterboxsynth.errors import EmptyTextError, SynthesisError
from chatterboxsynth.frontend import ChatterboxFrontend


class FakeTokenizer:
    def __init__(self, *, special_count: int = 1) -> None:
        self.special_count = special_count

    def token_to_id(self, token: str) -> int | None:
        return 99 if token == "<|endoftext|>" else None

    def encode(self, text: str, *, add_special_tokens: bool = True) -> Any:
        if text == "<|endoftext|>":
            return SimpleNamespace(ids=[99])
        token_ids = [1, 2]
        if add_special_tokens:
            token_ids.extend([99] * self.special_count)
        return SimpleNamespace(ids=token_ids)


def test_export_specific_eot_post_processors_are_preserved() -> None:
    nano = ChatterboxFrontend("unused", "nano-v2", tokenizer=FakeTokenizer(special_count=1))
    turbo = ChatterboxFrontend("unused", "turbo", tokenizer=FakeTokenizer(special_count=2))
    assert nano.encode("hello") == (1, 2, 99)
    assert turbo.encode("hello") == (1, 2, 99, 99)


def test_whitespace_and_user_injected_eot_are_rejected() -> None:
    frontend = ChatterboxFrontend("unused", "nano-v2", tokenizer=FakeTokenizer())
    with pytest.raises(EmptyTextError):
        frontend.encode(" \n\t")
    with pytest.raises(SynthesisError, match="reserved"):
        frontend.encode("<|endoftext|>")


def test_pinned_tokenizer_id_fixtures() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "tokenizer-parity.json"
    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    tokenizers = {
        "nano-v2": os.environ.get("CHATTERBOX_NANO_TOKENIZER"),
        "turbo": os.environ.get("CHATTERBOX_TURBO_TOKENIZER"),
    }
    missing = [model for model, path in tokenizers.items() if not path]
    if missing:
        pytest.skip(f"pinned tokenizer artifacts not supplied for: {', '.join(missing)}")
    for model, raw_path in tokenizers.items():
        path = Path(raw_path or "")
        expected = fixtures[model]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
        contract = "nano-v2" if model == "nano-v2" else "turbo"
        frontend = ChatterboxFrontend(path, contract)
        for text, token_ids in expected["texts"].items():
            assert list(frontend.encode(text)) == token_ids, (model, text)
