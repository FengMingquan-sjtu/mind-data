"""
High-level orchestration for math-informed synthetic data generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional

from .dataset_loader import DatasetConfig, DatasetStream
from .filters import ConversationFilter, MinTokenFilter
from .llm_api import ConversationLLM, LLMConfig
from .prompts import PromptTemplate
from .storage import JsonlWriter
from .tokenization import ChunkerConfig, TextChunker, build_tokenizer

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    min_conversation_tokens: int = 50
    tokenizer_model: Optional[str] = None
    dry_run: bool = False


class MindDataPipeline:
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        llm_config: Optional[LLMConfig],
        prompts: List[PromptTemplate],
        output_path,
        overwrite: bool = False,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.prompts = prompts
        encoding = build_tokenizer(model_name=pipeline_config.tokenizer_model)
        self.chunker = TextChunker(encoding, pipeline_config.chunker)
        self.dataset_stream = DatasetStream(pipeline_config.dataset)
        self.writer = JsonlWriter(output_path, overwrite=overwrite)
        self.filter: ConversationFilter = MinTokenFilter(
            chunker=self.chunker, min_tokens=pipeline_config.min_conversation_tokens
        )
        self.dry_run = pipeline_config.dry_run
        if self.dry_run:
            self.llm = None
        else:
            if llm_config is None:
                raise ValueError("llm_config must be provided when dry_run is False.")
            self.llm = ConversationLLM(llm_config)

    def _dry_run_generation(self, instruction: str, context: str) -> str:
        snippet = context.strip().split("\n", maxsplit=1)[0][:200]
        return f"[DRY RUN] {instruction[:60]}... | snippet: {snippet}"

    def _generate(self, prompt: PromptTemplate, context: str) -> Optional[str]:
        if self.dry_run:
            return self._dry_run_generation(prompt.instruction, context)
        if not self.llm:
            raise RuntimeError("LLM client is not initialized.")
        return self.llm.generate(prompt.instruction, context)

    def run(self) -> dict:
        stats = {
            "samples_seen": 0,
            "chunks_generated": 0,
            "conversations_kept": 0,
            "conversations_filtered": 0,
        }
        try:
            for sample_idx, raw_text in self.dataset_stream.iter_texts():
                stats["samples_seen"] += 1
                for chunk_idx, chunk in enumerate(self.chunker.chunk_text(raw_text)):
                    stats["chunks_generated"] += 1
                    for prompt in self.prompts:
                        conversation = self._generate(prompt, chunk)
                        if not conversation:
                            continue
                        if not self.filter.accepts(conversation):
                            stats["conversations_filtered"] += 1
                            continue
                        stats["conversations_kept"] += 1
                        record = {
                            "sample_index": sample_idx,
                            "chunk_index": chunk_idx,
                            "prompt_style": prompt.name,
                            "context": chunk,
                            "conversation": conversation,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        self.writer.write(record)
                if stats["samples_seen"] % 10 == 0:
                    LOGGER.info(
                        "Processed %d samples | kept %d conversations",
                        stats["samples_seen"],
                        stats["conversations_kept"],
                    )
        finally:
            self.writer.close()
        LOGGER.info("Generation complete: %s", stats)
        return stats

