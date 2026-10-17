"""Tests for MiniBpeRust: Rust/Python parity, save/load compatibility, API."""

import pytest

import MiniBpeRust
from MiniBpeRust import (
    BasicTokenizer,
    FastRegexTokenizer,
    GPT4_SPLIT_PATTERN,
    RegexTokenizer,
    Tokenizer,
)
from MiniBpeRust import rustminibpe
from MiniBpeRust.fast_regex_tokenizer import FastRegexTokenizer as FastFromModule

INDIC_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{M}\p{N}]?+[\p{L}\p{M}]+|\p{N}{1,3}| ?[^\s\p{L}\p{M}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 20
    + "Antidisestablishmentarianism is a long word.\n"
    + "नमस्ते भारत, यह हिन्दी पाठ है।\n"
    + "1234567890 42 3.14159\n"
    + "def add(a, b):\n    return a + b\n"
)


def test_public_api():
    assert FastFromModule is FastRegexTokenizer
    assert MiniBpeRust.__version__ == rustminibpe.__version__
    assert FastRegexTokenizer().pattern == GPT4_SPLIT_PATTERN
    assert isinstance(FastRegexTokenizer(), RegexTokenizer)
    assert isinstance(RegexTokenizer(), Tokenizer)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="has no attribute 'does_not_exist'"):
        MiniBpeRust.does_not_exist


@pytest.mark.parametrize("pattern", [GPT4_SPLIT_PATTERN, INDIC_PATTERN])
def test_train_matches_pure_python(pattern):
    slow = RegexTokenizer(pattern=pattern)
    fast = FastRegexTokenizer(pattern=pattern)
    slow.train(CORPUS, 320)
    fast.train(CORPUS, 320)

    assert fast.merges == slow.merges
    assert fast.vocab == slow.vocab

    for text in ["hello world", "नमस्ते भारत", "1234567890", "def add(a, b):"]:
        assert fast.encode(text) == slow.encode(text)
        assert fast.encode_ordinary(text) == slow.encode_ordinary(text)


def test_encode_decode_roundtrip():
    tokenizer = FastRegexTokenizer(pattern=INDIC_PATTERN)
    tokenizer.train(CORPUS, 320)
    for text in ["the quick brown fox", "नमस्ते भारत", "42 3.14159"]:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text


def test_encode_batch_matches_encode_ordinary():
    tokenizer = FastRegexTokenizer()
    tokenizer.train(CORPUS, 300)
    documents = ["hello world", "", "नमस्ते", "abc 123", "hello world"]
    assert tokenizer.encode_batch(documents) == [
        tokenizer.encode_ordinary(document) for document in documents
    ]


def test_train_files_matches_in_memory_training(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    # write bytes so Windows' newline translation cannot change the corpus
    corpus_file.write_bytes(CORPUS.encode("utf-8"))

    from_text = FastRegexTokenizer(pattern=INDIC_PATTERN)
    from_text.train(CORPUS, 350)

    from_files = FastRegexTokenizer(pattern=INDIC_PATTERN)
    from_files.train_files([corpus_file], 350)

    assert from_files.merges == from_text.merges



def test_train_files_rejects_invalid_utf8(tmp_path):
    corpus_file = tmp_path / "invalid.txt"
    corpus_file.write_bytes(b"hello \xff\xfe world")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        FastRegexTokenizer().train_files([corpus_file], 300)


def test_train_rejects_small_vocab_size():
    with pytest.raises(ValueError, match="vocab_size must be >= 256"):
        FastRegexTokenizer().train("hello world", 128)


def test_save_load_roundtrip(tmp_path):
    tokenizer = FastRegexTokenizer(pattern=INDIC_PATTERN)
    tokenizer.train(CORPUS, 400)
    tokenizer.register_special_tokens({"<|endoftext|>": 400})
    prefix = str(tmp_path / "tokenizer")
    tokenizer.save(prefix)

    loaded = FastRegexTokenizer()
    loaded.load(prefix + ".model")

    assert loaded.pattern == INDIC_PATTERN
    assert loaded.merges == tokenizer.merges
    # register_special_tokens() does not rebuild vocab (minbpe behaviour), but
    # load() does, so compare against the rebuilt vocab
    assert loaded.vocab == tokenizer._build_vocab()
    assert loaded.special_tokens == {"<|endoftext|>": 400}

    text = "hello <|endoftext|> world"
    assert loaded.encode(text, allowed_special="all") == tokenizer.encode(
        text, allowed_special="all"
    )


def test_model_file_is_identical_to_pure_python(tmp_path):
    fast = FastRegexTokenizer(pattern=INDIC_PATTERN)
    slow = RegexTokenizer(pattern=INDIC_PATTERN)
    fast.train(CORPUS, 320)
    slow.train(CORPUS, 320)
    fast.register_special_tokens({"<|endoftext|>": 320})
    slow.register_special_tokens({"<|endoftext|>": 320})
    fast.save(str(tmp_path / "fast"))
    slow.save(str(tmp_path / "slow"))

    assert (tmp_path / "fast.model").read_bytes() == (tmp_path / "slow.model").read_bytes()


def test_special_tokens():
    tokenizer = FastRegexTokenizer()
    tokenizer.train(CORPUS, 300)
    tokenizer.register_special_tokens({"<|endoftext|>": 300})

    ids = tokenizer.encode("hello <|endoftext|> world", allowed_special="all")
    assert 300 in ids
    assert tokenizer.decode(ids) == "hello <|endoftext|> world"

    with pytest.raises(AssertionError):
        tokenizer.encode("hello <|endoftext|> world")


def test_rust_encoder_directly():
    tokenizer = FastRegexTokenizer()
    tokenizer.train(CORPUS, 300)
    ordered = sorted(tokenizer.merges, key=tokenizer.merges.get)

    encoder = rustminibpe.RustEncoder(tokenizer.pattern, ordered)
    assert encoder.num_merges == len(ordered)
    assert len(encoder) == len(ordered)
    assert repr(encoder) == f"RustEncoder(num_merges={len(ordered)})"

    slow = RegexTokenizer(pattern=tokenizer.pattern)
    slow.merges = dict(tokenizer.merges)
    slow.vocab = slow._build_vocab()

    for text in ["hello world", "नमस्ते भारत", "abc 123"]:
        assert encoder.encode_ordinary(text) == slow.encode_ordinary(text)
    assert encoder.encode_batch(["hello world", "abc 123"]) == [
        slow.encode_ordinary("hello world"),
        slow.encode_ordinary("abc 123"),
    ]


def test_encoder_property_is_cached():
    tokenizer = FastRegexTokenizer()
    tokenizer.train(CORPUS, 280)
    assert tokenizer.encoder is tokenizer.encoder
    assert tokenizer.encoder.num_merges == 280 - 256


def test_basic_tokenizer_roundtrip():
    tokenizer = BasicTokenizer()
    tokenizer.train(CORPUS, 300)
    text = "hello hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text
