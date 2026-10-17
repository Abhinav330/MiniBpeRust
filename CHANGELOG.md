# Changelog

All notable changes to MiniBpeRust are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-17

### Added

- `FastRegexTokenizer`: minbpe's `RegexTokenizer` with training and encoding in
  Rust, fully compatible with `minbpe v1` model files.
- `train(text, vocab_size, verbose=False)` using the Rust trainer.
- `train_files(paths, vocab_size, verbose=False)`: Rust reads UTF-8 corpus files
  directly, so large corpora never become Python strings.
- `encode_batch(texts)`: rayon-parallel encoding on all cores.
- `MiniBpeRust.rustminibpe` native module with `train_on_text`,
  `train_on_files` and `RustEncoder`.
- Bundled pure-Python `Tokenizer`, `BasicTokenizer`, `RegexTokenizer` and
  optional `GPT4Tokenizer` (`MiniBpeRust[gpt4]`).
- Type stubs for the native module.
- CI: abi3 wheels for Linux, musllinux, macOS and Windows, plus PyPI publishing
  on tags.
