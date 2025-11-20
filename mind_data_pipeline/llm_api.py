"""
Wrapper around the OpenAI-compatible API described in tmp.py.
"""

from __future__ import annotations

import os
import random
import time
import asyncio
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, AsyncOpenAI

DEFAULT_SYSTEM_PROMPT = (
    "You are a mathematics-focused assistant that converts raw documents into synthetic "
    "multi-turn conversations. Preserve every detail from the original context while "
    "adding structured reasoning. Never introduce information that is not grounded in the "
    "provided context."
)

ENV_API_KEY = "MIND_API_KEY"
ENV_BASE_URL = "MIND_BASE_URL"
ENV_MODEL = "MIND_LLM_MODEL"


@dataclass
class LLMConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "deepseek-r1"
    temperature: float = 1.0
    top_p: float = 0.9
    max_output_tokens: int = 2048
    request_timeout: float = 120.0
    max_attempts: int = 3
    backoff_seconds: float = 2.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


def resolve_env_api_key() -> Optional[str]:
    return (
        os.environ.get(ENV_API_KEY)
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DIRECTLLM_API_KEY")
    )


def resolve_env_base_url() -> Optional[str]:
    return os.environ.get(ENV_BASE_URL) or os.environ.get("OPENAI_BASE_URL")


def create_openai_client(api_key: Optional[str], base_url: Optional[str], timeout: float) -> OpenAI:
    if not api_key:
        raise ValueError(
            "API key is missing. Please set MIND_API_KEY or OPENAI_API_KEY (or pass via --api-key)."
        )
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def create_async_openai_client(api_key: Optional[str], base_url: Optional[str], timeout: float) -> AsyncOpenAI:
    if not api_key:
        raise ValueError(
            "API key is missing. Please set MIND_API_KEY or OPENAI_API_KEY (or pass via --api-key)."
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


class ConversationLLM:
    """LLM wrapper that adheres to the tmp.py API usage."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        api_key = config.api_key or resolve_env_api_key()
        base_url = config.base_url or resolve_env_base_url()
        if not api_key:
            raise ValueError(
                "API key not provided. Set MIND_API_KEY/OPENAI_API_KEY or supply --api-key."
            )
        self.client = create_openai_client(api_key, base_url, timeout=config.request_timeout)
        self.async_client = create_async_openai_client(api_key, base_url, timeout=config.request_timeout)

    def _build_messages(self, instruction: str, context: str):
        user_message = (
            f"{instruction}\n\n"
            "Context:\n"
            "<context>\n"
            f"{context.strip()}\n"
            "</context>\n\n"
            "Return only the conversation with speaker tags. Use the same language as the context."
        )
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def generate(self, instruction: str, context: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_output_tokens,
                    messages=self._build_messages(instruction, context),
                )
                choice = response.choices[0]
                message = choice.message.content if choice and choice.message else None
                if not message:
                    raise RuntimeError("Empty response from LLM.")
                return message.strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.config.max_attempts:
                    break
                sleep_for = self.config.backoff_seconds * attempt + random.random()
                time.sleep(sleep_for)
        raise RuntimeError("LLM generation failed.") from last_error

    async def generate_async(self, instruction: str, context: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=self.config.max_output_tokens,
                    messages=self._build_messages(instruction, context),
                )
                choice = response.choices[0]
                message = choice.message.content if choice and choice.message else None
                if not message:
                    raise RuntimeError("Empty response from LLM.")
                return message.strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == self.config.max_attempts:
                    break
                sleep_for = self.config.backoff_seconds * attempt + random.random()
                await asyncio.sleep(sleep_for)
        raise RuntimeError("LLM generation failed.") from last_error

