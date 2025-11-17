"""
Example client usage for the DirectLLM-compatible API.
"""

from __future__ import annotations

from dotenv import load_dotenv

from mind_data_pipeline.llm_api import (
    create_openai_client,
    resolve_env_api_key,
    resolve_env_base_url,
)

DEFAULT_BASE_URL = "https://maas.devops.xiaohongshu.com/v1"


def build_client():
    load_dotenv()
    api_key = resolve_env_api_key()
    base_url = resolve_env_base_url() or DEFAULT_BASE_URL
    return create_openai_client(api_key, base_url, timeout=120)


def streaming_demo(client):
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": "帮我制定一份日本的五天四夜的旅游攻略，小红书风格"},
        ],
        stream=True,
        max_tokens=4096,
        temperature=0.9,
    )
    for chunk in completion:
        print(chunk.model_dump_json())


def non_streaming_demo(client):
    completion = client.chat.completions.create(
        model="deepseek-r1",
        messages=[
            {"role": "system", "content": "你是一个AI助手"},
            {"role": "user", "content": "帮我制定一份日本的五天四夜的旅游攻略，小红书风格"},
        ],
        stream=False,
        max_tokens=4096,
        temperature=0.9,
    )
    print(completion.model_dump_json())


if __name__ == "__main__":
    client = build_client()
    non_streaming_demo(client)
