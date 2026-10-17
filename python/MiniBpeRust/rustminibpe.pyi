"""Type stubs for the compiled `MiniBpeRust.rustminibpe` extension."""

from collections.abc import Sequence

__version__: str

def train_on_text(
    text: str,
    pattern: str,
    vocab_size: int,
    verbose: bool = False,
) -> list[tuple[tuple[int, int], int]]: ...
def train_on_files(
    paths: Sequence[str],
    pattern: str,
    vocab_size: int,
    verbose: bool = False,
) -> list[tuple[tuple[int, int], int]]: ...

class RustEncoder:
    def __init__(self, pattern: str, merges: Sequence[tuple[int, int]]) -> None: ...
    @property
    def num_merges(self) -> int: ...
    def encode_ordinary(self, text: str) -> list[int]: ...
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
