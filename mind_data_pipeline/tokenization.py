"""
Tokenization helpers for chunking raw text into 500-token windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import tiktoken


def build_tokenizer(model_name: str | None = None, encoding_name: str = "cl100k_base"):
    """
    Build a tiktoken encoder. Falls back to cl100k_base if the model is unknown.
    """

    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            pass
    return tiktoken.get_encoding(encoding_name)


@dataclass
class ChunkerConfig:
    chunk_size: int = 500
    overlap: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")


class TextChunker:
    """Split documents into overlapping token windows."""

    def __init__(self, encoding, config: ChunkerConfig | None = None) -> None:
        self.encoding = encoding
        self.config = config or ChunkerConfig()
        self._stride = self.config.chunk_size - self.config.overlap

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def chunk_text(self, text: str) -> Iterator[str]:
        tokens = self.encoding.encode(text, disallowed_special=())
        if not tokens:
            return

        start = 0
        total = len(tokens)
        while start < total:
            end = min(start + self.config.chunk_size, total)
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            yield self.encoding.decode(chunk_tokens)
            start += self._stride


def token_count(encoding, text: str) -> int:
    """Convenience helper."""

    return len(encoding.encode(text, disallowed_special=()))

