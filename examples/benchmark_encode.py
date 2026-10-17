"""Compare pure-Python RegexTokenizer encoding with MiniBpeRust's Rust encoder.

Example:
    python examples/benchmark_encode.py --vocab-size 1024 --repeats 3
"""

import argparse
import time

from MiniBpeRust import FastRegexTokenizer, GPT4_SPLIT_PATTERN, RegexTokenizer

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Tokenizers turn text into integers, and integers back into text. "
    "नमस्ते भारत, यह हिन्दी पाठ है। "
    "def add(a, b):\n    return a + b\n"
) * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per tokenizer")
    return parser.parse_args()


def best_time(tokenizer, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        ids = tokenizer.encode_ordinary(TEXT)
        timings.append(time.perf_counter() - start)
    if not ids:
        raise RuntimeError("encoder produced no tokens")
    return min(timings)


def main() -> None:
    args = parse_args()

    slow = RegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
    fast = FastRegexTokenizer(pattern=GPT4_SPLIT_PATTERN)
    slow.train(TEXT, args.vocab_size)
    fast.train(TEXT, args.vocab_size)

    slow_ids = slow.encode_ordinary(TEXT)
    fast_ids = fast.encode_ordinary(TEXT)
    if slow_ids != fast_ids:
        raise RuntimeError("token ids differ between the Python and Rust encoders")

    slow_seconds = best_time(slow, args.repeats)
    fast_seconds = best_time(fast, args.repeats)
    size_mb = len(TEXT.encode("utf-8")) / 1_000_000

    print(f"tokens: {len(slow_ids)}")
    print(f"python: {slow_seconds:.3f}s ({size_mb / slow_seconds:.2f} MB/s)")
    print(f"rust:   {fast_seconds:.3f}s ({size_mb / fast_seconds:.2f} MB/s)")
    print(f"speedup: {slow_seconds / fast_seconds:.1f}x")


if __name__ == "__main__":
    main()
