"""
Hugging Face dataset helpers for the Open Web Math subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

from datasets import IterableDatasetDict, load_dataset


@dataclass
class DatasetConfig:
    name: str = "brando/small-open-web-math-dataset"
    split: str = "train"
    streaming: bool = False
    text_field: str = "text"
    max_samples: Optional[int] = None
    revision: Optional[str] = None
    token: Optional[str] = None


class DatasetStream:
    """Iterate over raw text documents."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._dataset = None

    def _load(self):
        if self._dataset is not None:
            return self._dataset
        self._dataset = load_dataset(
            self.config.name,
            split=self.config.split,
            streaming=self.config.streaming,
            revision=self.config.revision,
            token=self.config.token,
        )
        return self._dataset

    def iter_texts(self) -> Iterator[Tuple[int, str]]:
        dataset = self._load()
        iterator = dataset

        if isinstance(dataset, IterableDatasetDict):
            iterator = dataset[self.config.split]

        for idx, row in enumerate(iterator):
            if self.config.max_samples is not None and idx >= self.config.max_samples:
                break
            text = row.get(self.config.text_field)
            if not text:
                continue
            yield idx, text

