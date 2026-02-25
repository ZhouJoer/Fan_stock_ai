from __future__ import annotations

"""LLM 统一入口：根据 LLM_PROVIDER 选择后端，导出 llm、tool_llm、get_llm。"""

import os

# 导入 llm 包时自动加载项目根目录的 .env，确保 DEEPSEEK_API_KEY 等生效
try:
    from dotenv import load_dotenv
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env")
except ImportError:
    pass

_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()

if _PROVIDER == "deepseek":
    from llm.deepseek import get_llm
elif _PROVIDER == "openai":
    from llm.openai_ import get_llm
else:
    raise RuntimeError(
        f"Unsupported LLM_PROVIDER={_PROVIDER!r}. Use 'deepseek' or 'openai'."
    )

llm = get_llm(temperature=0.3)
tool_llm = get_llm(temperature=0.1)

__all__ = ["get_llm", "llm", "tool_llm"]
