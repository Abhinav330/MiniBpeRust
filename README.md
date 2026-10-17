# MiniBpeRust

[![CI](https://github.com/Abhinav330/MiniBpeRust/actions/workflows/CI.yml/badge.svg)](https://github.com/Abhinav330/MiniBpeRust/actions/workflows/CI.yml)
[![PyPI](https://img.shields.io/pypi/v/MiniBpeRust.svg)](https://pypi.org/project/MiniBpeRust/)
[![Python versions](https://img.shields.io/pypi/pyversions/MiniBpeRust.svg)](https://pypi.org/project/MiniBpeRust/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fast byte-level BPE tokenization: the [minbpe](https://github.com/karpathy/minbpe)
`RegexTokenizer` API, with **training and encoding in Rust**.

```python
from MiniBpeRust import FastRegexTokenizer

tokenizer = FastRegexTokenizer()          # GPT-4 split pattern by default
tokenizer.train_files(["corpus.txt"], vocab_size=50_000, verbose=True)
tokenizer.register_special_tokens({"<|endoftext|>": 50_000})
tokenizer.save("tokenizer")

ids = tokenizer.encode("Hello, world!", allowed_special="all")
tokenizer.decode(ids) == "Hello, world!"
```

## What you get

* **Drop-in tokenizer** – `FastRegexTokenizer` subclasses minbpe's
  `RegexTokenizer`, so `encode`, `decode`, `save`, `load` and
  `register_special_tokens` behave exactly like before.
* **Rust training** – `train(text)` for in-memory text and `train_files(paths)`
  for corpora that Rust reads itself, so a 450 MB file never becomes a Python
  string.
* **Rust encoding** – compiled once from the merge list and reused, with a cache
  for repeated chunks and `encode_batch()` running on all cores via rayon.
* **Byte-identical models** – merges (including minbpe's tie-breaking on equal
  counts) and `.model` files are identical to the pure-Python implementation.
  Existing minbpe models load as-is.
* **GIL released** – training and encoding do not block Python threads.
* **Batteries included** – `Tokenizer`, `BasicTokenizer`, `RegexTokenizer`,
  `FastRegexTokenizer`, optional `GPT4Tokenizer`, and the native functions in
  `MiniBpeRust.rustminibpe`.
* **Stable ABI wheels** – built as `cp39-abi3`, so one wheel per platform works
  on every CPython 3.9+ (Linux, macOS, Windows).

## Installation

```bash
pip install MiniBpeRust
```

Optional extras:

```bash
pip install "MiniBpeRust[gpt4]"   # GPT4Tokenizer (needs tiktoken)
pip install "MiniBpeRust[dev]"    # pytest
```

Install the latest commit straight from GitHub:

```bash
pip install "git+https://github.com/Abhinav330/MiniBpeRust.git"
```

If no wheel exists for your platform, pip builds from source. That needs
**Rust 1.85+** and the usual maturin prerequisites.

## Quick start

### Train from memory

```python
from MiniBpeRust import FastRegexTokenizer

text = open("input.txt", encoding="utf-8").read()

tokenizer = FastRegexTokenizer()
tokenizer.train(text, vocab_size=1024, verbose=True)

print(tokenizer.encode("hello world"))
print(tokenizer.decode(tokenizer.encode("hello world")))
```

### Train from files (recommended for big corpora)

```python
from MiniBpeRust import FastRegexTokenizer

tokenizer = FastRegexTokenizer()
tokenizer.train_files(
    ["corpus_v1.txt", "corpus_v2.txt"],
    vocab_size=200_256,
    verbose=True,
)
tokenizer.register_special_tokens({"<|endoftext|>": 200_256})
tokenizer.save("tok200k")
```

`save()` writes `tok200k.model` (load it later) and `tok200k.vocab` (for humans),
exactly like minbpe.

### Load and encode

```python
from MiniBpeRust import FastRegexTokenizer

tokenizer = FastRegexTokenizer()
tokenizer.load("tok200k.model")

ids = tokenizer.encode("नमस्ते भारत", allowed_special="all")
tokenizer.decode(ids)

# Parallel encoding: one list of ids per document, ignores special tokens
batch_ids = tokenizer.encode_batch(["first document", "second document"])
```

### Custom split pattern

```python
from MiniBpeRust import FastRegexTokenizer

INDIC_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{M}\p{N}]?+[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

tokenizer = FastRegexTokenizer(pattern=INDIC_PATTERN)
tokenizer.train_files(["hindi_corpus.txt"], vocab_size=50_000)
```

The pattern is stored in the model file, so `load()` restores it automatically.

### Rust functions directly

```python
from MiniBpeRust import rustminibpe

merges = rustminibpe.train_on_text("hello hello world", r"\s+|.", 260)
# [((104, 101), 4), ((108, 108), 3), ...]  ->  ((id1, id2), count) in merge order

ordered = [pair for pair, _count in merges]
encoder = rustminibpe.RustEncoder(r"\s+|.", ordered)
encoder.encode_ordinary("hello world")
```

## API

| Object | Purpose |
| --- | --- |
| `Tokenizer` | Base class; `save()` / `load()` for `minbpe v1` `.model` files |
| `BasicTokenizer` | Byte-level BPE without regex splitting (pure Python) |
| `RegexTokenizer` | GPT-2/GPT-4 style regex-split BPE (pure Python) |
| `FastRegexTokenizer` | `RegexTokenizer` with Rust training and encoding |
| `GPT4Tokenizer` | Pretrained `cl100k_base` tokenizer (optional, needs `tiktoken`) |
| `rustminibpe` | Native module: `train_on_text`, `train_on_files`, `RustEncoder` |

`FastRegexTokenizer` methods:

| Method | Description |
| --- | --- |
| `train(text, vocab_size, verbose=False)` | Train on a Python string |
| `train_files(paths, vocab_size, verbose=False)` | Train on UTF-8 files read by Rust |
| `encode(text, allowed_special="none_raise")` | Encode with minbpe's special-token rules |
| `encode_ordinary(text)` | Encode, ignoring special tokens |
| `encode_batch(texts)` | Encode many documents in parallel (special tokens ignored) |
| `decode(ids)` | Decode token ids back to text |
| `save(prefix)` / `load(model_file)` | Write/read minbpe `v1` model files |
| `register_special_tokens(dict)` | Map special-token strings to ids |
| `encoder` | The cached `rustminibpe.RustEncoder` |

Both `train` and `encode_ordinary` process the same regex chunks as minbpe, so
token ids are interchangeable between the two implementations.

## Compatibility with minbpe

* The model file format is unchanged (`minbpe v1`), so models trained with
  minbpe load in `FastRegexTokenizer` and the other way round.
* `tests/test_minibperust.py` asserts that merges, vocabularies and even the
  written `.model` bytes are identical to the pure-Python tokenizer, for both
  the GPT-4 pattern and a Devanagari-aware pattern.
* `save()` and `load()` live in `Tokenizer` and are used unmodified.

## Development

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install maturin pytest
maturin develop --release
pytest
```

Layout:

```
MiniBpeRust/
├── Cargo.toml                     # Rust extension crate
├── pyproject.toml                 # maturin build config + PyPI metadata
├── src/
│   ├── lib.rs                     # PyO3 bindings
│   ├── train.rs                   # BPE training (minbpe-identical tie-breaking)
│   └── encode.rs                  # encoder + rayon batch encoding
├── python/MiniBpeRust/
│   ├── __init__.py                # public API
│   ├── base.py                    # Tokenizer + save/load
│   ├── basic.py                   # BasicTokenizer
│   ├── regex.py                   # RegexTokenizer
│   ├── gpt4.py                    # GPT4Tokenizer (optional)
│   ├── fast_regex_tokenizer.py    # FastRegexTokenizer
│   └── rustminibpe.pyi            # type stubs for the native module
├── tests/
└── examples/
```

### Build and publish

```bash
maturin build --release --out dist     # abi3 wheels
maturin sdist --out dist               # source distribution
uv publish dist/*                      # or: twine upload dist/*
```

Tagging the repo (`git tag v0.1.0 && git push --tags`) runs
`.github/workflows/CI.yml`, which builds every wheel and publishes to PyPI using
the `PYPI_API_TOKEN` repository secret.

## Notes and limitations

* `train_files()` requires UTF-8 input. Invalid UTF-8 raises `ValueError`
  instead of silently replacing bytes.
* Training parity with minbpe assumes the same corpus, the same order of files
  and the same regex pattern.
* Patterns use [fancy-regex](https://docs.rs/fancy-regex), which supports the
  GPT-2/GPT-4 and Devanagari patterns used here, but is not a byte-for-byte
  reimplementation of Python's `regex` module for every exotic construct.
* `GPT4Tokenizer` is pretrained and cannot be trained, saved or loaded — same
  as in minbpe.

## License

MIT. Portions are derived from
[minbpe](https://github.com/karpathy/minbpe) (MIT, Andrej Karpathy) and from
[tiktoken](https://github.com/openai/tiktoken)'s GPT-2/GPT-4 split patterns.
See [LICENSE](LICENSE).

