from __future__ import annotations

"""OpenAI 后端：从 env 读取 OPENAI_*，提供 get_llm。"""

import os

from langchain_openai import ChatOpenAI


def get_llm(*, temperature: float = 0.3, model: str | None = None) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set env var OPENAI_API_KEY to use OpenAI Chat."
        )
    kwargs: dict = {
        "model": model or default_model,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


__all__ = ["get_llm"]
