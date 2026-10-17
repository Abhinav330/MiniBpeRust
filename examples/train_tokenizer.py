"""Train a MiniBpeRust tokenizer from UTF-8 text files.

Example:
    python examples/train_tokenizer.py corpus.txt --vocab-size 8192 --out tokenizer
"""

import argparse

from MiniBpeRust import FastRegexTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MiniBpeRust tokenizer and save it as minbpe v1 files."
    )
    parser.add_argument("corpus", nargs="+", help="UTF-8 text files to train on")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument(
        "--out",
        default="tokenizer",
        help="output prefix; writes <out>.model and <out>.vocab",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="optional regex split pattern (defaults to the GPT-4 pattern)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=[],
        metavar="TOKEN",
        help="register a special token; ids continue after the vocabulary",
    )
    parser.add_argument("--verbose", action="store_true", help="print every merge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = FastRegexTokenizer(pattern=args.pattern)
    print(f"training on {len(args.corpus)} file(s) with vocab_size={args.vocab_size}")
    tokenizer.train_files(args.corpus, vocab_size=args.vocab_size, verbose=args.verbose)

    if args.special_token:
        specials = {
            token: args.vocab_size + index
            for index, token in enumerate(args.special_token)
        }
        tokenizer.register_special_tokens(specials)
        print(f"registered special tokens: {specials}")

    tokenizer.save(args.out)
    print(f"saved {args.out}.model and {args.out}.vocab ({len(tokenizer.merges)} merges)")


if __name__ == "__main__":
    main()
