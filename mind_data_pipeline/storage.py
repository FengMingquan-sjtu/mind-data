"""
Helpers for writing JSONL outputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class JsonlWriter:
    path: Path
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(
                f"{self.path} already exists. Pass overwrite=True to replace it."
            )
        self._fp = self.path.open("w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        json.dump(record, self._fp, ensure_ascii=False)
        self._fp.write("\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

