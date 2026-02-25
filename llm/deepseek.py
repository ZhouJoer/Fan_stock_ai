from __future__ import annotations

"""DeepSeek 后端：从 env 读取 DEEPSEEK_*，提供 get_llm。"""

import os

from langchain_openai import ChatOpenAI


def get_llm(*, temperature: float = 0.3, model: str | None = None) -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
    default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Set env var DEEPSEEK_API_KEY to use DeepSeek Chat."
        )
    return ChatOpenAI(
        model=model or default_model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


__all__ = ["get_llm"]
