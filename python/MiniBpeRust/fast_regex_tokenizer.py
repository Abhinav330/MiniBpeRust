"""
FastRegexTokenizer: minbpe's RegexTokenizer with training and encoding done in
Rust.

Only train(), train_files(), encode_ordinary() and encode_batch() are new.
save(), load(), decode() and the special-token handling are inherited
unchanged, so `.model` and `.vocab` files stay fully compatible with minbpe.
"""

from __future__ import annotations

import regex as re

from . import rustminibpe
from .regex import RegexTokenizer


class FastRegexTokenizer(RegexTokenizer):
    """RegexTokenizer whose training and encoding run in Rust."""

    def __init__(self, pattern: str | None = None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        """
        super().__init__(pattern=pattern)
        self._encoder: rustminibpe.RustEncoder | None = None

    def _set_merges(self, merge_list) -> None:
        """Rebuild merges/vocab from Rust's [((id1, id2), count), ...] output."""
        merges = {}
        for i, (pair, _count) in enumerate(merge_list):
            merges[tuple(pair)] = 256 + i
        self.merges = merges
        self.vocab = self._build_vocab()
        self._encoder = None

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train on an in-memory string. Raises ValueError if vocab_size < 256."""
        self._set_merges(
            rustminibpe.train_on_text(text, self.pattern, vocab_size, verbose)
        )

    def train_files(self, paths, vocab_size: int, verbose: bool = False) -> None:
        """
        Train on text files that Rust reads itself, so a huge corpus never has
        to become a Python string. Files must be UTF-8.
        """
        self._set_merges(
            rustminibpe.train_on_files(
                [str(path) for path in paths], self.pattern, vocab_size, verbose
            )
        )

    def load(self, model_file: str) -> None:
        """Load a `minbpe v1` .model file written by save()."""
        super().load(model_file)
        self.compiled_pattern = re.compile(self.pattern)
        self._encoder = None

    @property
    def encoder(self) -> rustminibpe.RustEncoder:
        """The compiled Rust encoder, built on first use and reused afterwards."""
        return self._get_encoder()

    def _get_encoder(self) -> rustminibpe.RustEncoder:
        if self._encoder is None:
            ordered = sorted(self.merges, key=self.merges.get)
            self._encoder = rustminibpe.RustEncoder(self.pattern, ordered)
        return self._encoder

    def encode_ordinary(self, text: str) -> list[int]:
        """Encoding that ignores any special tokens."""
        return self._get_encoder().encode_ordinary(text)

    def encode_batch(self, texts) -> list[list[int]]:
        """Encode many documents in parallel on all cores. Ignores special tokens."""
        return self._get_encoder().encode_batch(list(texts))
