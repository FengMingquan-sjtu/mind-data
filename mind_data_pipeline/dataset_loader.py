"""
Local parquet dataset helpers for the math_en_0527 corpus.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


def _default_token_columns() -> Tuple[str, ...]:
    return ("tokens", "input_ids")


@dataclass
class DatasetConfig:
    dataset_dir: str = "/prodcpfs/user/fengmingquan/dataset/raw/books_math_en_0527"
    tokenizer_path: str = "/prodcpfs/user/fengmingquan/model/Qwen2-0.5B"
    token_columns: Sequence[str] = field(default_factory=_default_token_columns)
    text_column: Optional[str] = "text"
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.dataset_dir:
            raise ValueError("dataset_dir must be provided.")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path must be provided.")
        if not self.token_columns:
            raise ValueError("At least one token column name must be provided.")
        # Ensure we store an immutable tuple for downstream use.
        self.token_columns = tuple(self.token_columns)


class DatasetStream:
    """Iterate over raw text decoded from local parquet files."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._parquet_files: Optional[List[str]] = None

    def _load_tokenizer(self):
        if self._tokenizer is None:
            LOGGER.info("Loading tokenizer from %s", self.config.tokenizer_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        return self._tokenizer

    def _resolve_parquet_files(self) -> List[str]:
        if self._parquet_files is None:
            pattern = os.path.join(self.config.dataset_dir, "*.parquet")
            self._parquet_files = sorted(glob.glob(pattern))
            if not self._parquet_files:
                raise FileNotFoundError(
                    f"No parquet files found under {self.config.dataset_dir}"
                )
            LOGGER.info(
                "Discovered %d parquet files under %s",
                len(self._parquet_files),
                self.config.dataset_dir,
            )
        return self._parquet_files

    @staticmethod
    def _normalize_tokens(value) -> Optional[List[int]]:
        if not isinstance(value, (list, tuple, np.ndarray)):
            return None
        if not value:
            return None
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError):
            return None

    def _row_to_text(self, row: dict, tokenizer) -> Optional[str]:
        for candidate in self.config.token_columns:
            tokens = row.get(candidate)
            normalized = self._normalize_tokens(tokens)
            if normalized:
                return tokenizer.decode(normalized, skip_special_tokens=False)

        for name, value in row.items():
            lowered = name.lower()
            if ("token" in lowered or "id" in lowered):
                normalized = self._normalize_tokens(value)
                if normalized:
                    return tokenizer.decode(normalized, skip_special_tokens=False)

        if self.config.text_column:
            text_value = row.get(self.config.text_column)
            if isinstance(text_value, str) and text_value.strip():
                return text_value

        return None

    @staticmethod
    def _batch_to_rows(batch) -> Iterator[dict]:
        columns = batch.schema.names
        if not columns:
            return
        column_data = [batch.column(name).to_pylist() for name in columns]
        row_count = len(column_data[0]) if column_data else 0
        for row_idx in range(row_count):
            yield {
                name: column_data[col_idx][row_idx]
                for col_idx, name in enumerate(columns)
            }

    def iter_texts(self) -> Iterator[Tuple[int, str]]:
        files = self._resolve_parquet_files()
        #tokenizer = self._load_tokenizer()

        emitted = 0

        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches():
                for row in self._batch_to_rows(batch):
                    if self.config.max_samples is not None and emitted >= self.config.max_samples:
                        return
                    text = row.get("text") # read raw text, no tokenization
                    #text = self._row_to_text(row, tokenizer)
                    if not text:
                        print(f"No text found in row, skipping")
                        continue
                    yield emitted, text
                    emitted += 1

