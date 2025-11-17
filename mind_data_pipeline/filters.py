"""
Lightweight heuristics used to accept or reject generations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .tokenization import TextChunker


class ConversationFilter(Protocol):
    def accepts(self, text: str) -> bool:
        ...


@dataclass
class MinTokenFilter:
    chunker: TextChunker
    min_tokens: int = 50

    def accepts(self, text: str) -> bool:
        return self.chunker.count_tokens(text) >= self.min_tokens

