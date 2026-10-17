"""
MiniBpeRust - minbpe's byte-level BPE tokenizers, with training and encoding in Rust.

Public API
----------
Tokenizer           base class; implements save()/load() for `minbpe v1` files
BasicTokenizer      byte-level BPE without regex splitting
RegexTokenizer      GPT-2/GPT-4 style regex-split BPE, pure Python
FastRegexTokenizer  RegexTokenizer trained and encoded by Rust
GPT4Tokenizer       pretrained cl100k_base tokenizer (needs `MiniBpeRust[gpt4]`)

The native extension is available as `MiniBpeRust.rustminibpe`.
"""

from .base import Tokenizer
from .basic import BasicTokenizer
from .fast_regex_tokenizer import FastRegexTokenizer
from .regex import GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN, RegexTokenizer
from .rustminibpe import __version__

__all__ = [
    "Tokenizer",
    "BasicTokenizer",
    "RegexTokenizer",
    "FastRegexTokenizer",
    "GPT4Tokenizer",
    "GPT2_SPLIT_PATTERN",
    "GPT4_SPLIT_PATTERN",
    "__version__",
]


def __getattr__(name):
    # GPT4Tokenizer pulls in tiktoken, which is an optional dependency, so it
    # is imported on first use instead of at package import time.
    if name == "GPT4Tokenizer":
        from .gpt4 import GPT4Tokenizer

        return GPT4Tokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
