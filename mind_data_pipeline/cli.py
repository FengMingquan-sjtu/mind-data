"""
Command-line interface for generating Math Informed syNthetic Dialogue data.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv

from .dataset_loader import DatasetConfig
from .llm_api import LLMConfig, resolve_env_api_key, resolve_env_base_url
from .pipeline import MindDataPipeline, PipelineConfig
from .prompts import DEFAULT_PROMPT_ORDER, PromptTemplate, validate_prompt_names
from .tokenization import ChunkerConfig


def _build_prompts(selected: Optional[List[str]]) -> List[PromptTemplate]:
    names = selected or DEFAULT_PROMPT_ORDER
    return validate_prompt_names(names)


def _resolve_api_key(cli_value: Optional[str]) -> Optional[str]:
    return cli_value or resolve_env_api_key()


def _resolve_base_url(cli_value: Optional[str]) -> Optional[str]:
    return cli_value or resolve_env_base_url()


def generate_conversations(
    output_path: Path = typer.Option(
        Path("data/mind_conversations.jsonl"),
        help="Path to the JSONL file where synthetic conversations will be stored.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite the output file if it already exists."
    ),
    prompts: Optional[List[str]] = typer.Option(
        None,
        "--prompt-style",
        help="Subset of prompt styles to use (repeat flag for multiple). "
        f"Choices: {', '.join(DEFAULT_PROMPT_ORDER)}",
    ),
    max_samples: Optional[int] = typer.Option(
        None, help="Stop after consuming this many raw samples from the dataset."
    ),
    chunk_size: int = typer.Option(500, help="Token window size for raw contexts."),
    chunk_overlap: int = typer.Option(
        50, help="Token overlap between adjacent raw chunks to preserve continuity."
    ),
    min_conversation_tokens: int = typer.Option(
        50, help="Discard generations whose token length is below this threshold."
    ),
    dataset_dir: Path = typer.Option(
        Path("/prodcpfs/user/fengmingquan/dataset/raw/books_math_en_0527"),
        help="Directory containing math_en_0527 parquet shards.",
    ),
    tokenizer_path: Path = typer.Option(
        Path("/prodcpfs/user/fengmingquan/model/Qwen2-0.5B"),
        help="Tokenizer checkpoint used to decode stored token IDs.",
    ),
    token_columns: Optional[List[str]] = typer.Option(
        None,
        "--token-column",
        help="Column(s) that hold token IDs. Provide multiple times for fallbacks.",
    ),
    text_column: Optional[str] = typer.Option(
        None,
        help="Optional column that already contains raw text (used if tokens missing).",
    ),
    tokenizer_model: Optional[str] = typer.Option(
        None, help="Model name passed to tiktoken when building the tokenizer."
    ),
    model: str = typer.Option(
        os.environ.get("MIND_LLM_MODEL", "deepseek-r1"),
        help="Model name passed to the tmp.py-compatible API.",
    ),
    temperature: float = typer.Option(1.0, help="LLM sampling temperature."),
    top_p: float = typer.Option(0.9, help="LLM nucleus sampling parameter."),
    max_output_tokens: int = typer.Option(2048, help="Max tokens to generate per call."),
    request_timeout: float = typer.Option(120.0, help="Request timeout in seconds."),
    api_key: Optional[str] = typer.Option(
        None,
        help="API key for the tmp.py OpenAI-compatible endpoint. "
        "Falls back to MIND_API_KEY/OPENAI_API_KEY.",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        help="Base URL for the tmp.py endpoint. Falls back to MIND_BASE_URL/OPENAI_BASE_URL.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Skip LLM calls and emit placeholder conversations for testing.",
    ),
    log_level: str = typer.Option("INFO", help="Python logging level (e.g., INFO, DEBUG)."),
):
    """
    Generate synthetic math-focused conversations from the local math_en_0527 corpus.
    """

    load_dotenv()
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    prompts_to_use = _build_prompts(prompts)

    dataset_kwargs = {
        "dataset_dir": str(dataset_dir),
        "tokenizer_path": str(tokenizer_path),
        "max_samples": max_samples,
        "text_column": text_column,
    }
    if token_columns:
        dataset_kwargs["token_columns"] = token_columns

    pipeline_config = PipelineConfig(
        dataset=DatasetConfig(**dataset_kwargs),
        chunker=ChunkerConfig(chunk_size=chunk_size, overlap=chunk_overlap),
        min_conversation_tokens=min_conversation_tokens,
        tokenizer_model=tokenizer_model,
        dry_run=dry_run,
    )

    llm_config = None
    if not dry_run:
        llm_config = LLMConfig(
            api_key=_resolve_api_key(api_key),
            base_url=_resolve_base_url(base_url),
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            request_timeout=request_timeout,
        )

    pipeline = MindDataPipeline(
        pipeline_config=pipeline_config,
        llm_config=llm_config,
        prompts=prompts_to_use,
        output_path=output_path,
        overwrite=overwrite,
    )

    stats = pipeline.run()
    typer.echo(f"Generation finished. Stats: {stats}")


def run():
    typer.run(generate_conversations)


if __name__ == "__main__":
    run()

