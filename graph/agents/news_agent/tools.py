"""新闻 Agent 工具（仅支持中国 A 股与港股）。

实现原则：
- 只依赖本项目现有依赖（requirements.txt），不引入新的第三方库。
- AkShare 的具体接口在不同版本可能有差异，因此这里采用‘多接口尝试 + 失败降级’的方式。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Optional

from langchain_core.tools import tool


def _pick_first_existing(obj: Any, names: Iterable[str]) -> Optional[Callable[..., Any]]:
    for name in names:
        func = getattr(obj, name, None)
        if callable(func):
            return func
    return None


def _df_to_text(df, limit: int = 20) -> str:
    """把 DataFrame 转成适合给 LLM 的紧凑文本。"""
    try:
        return df.head(limit).to_csv(index=False)
    except Exception:
        return str(df)


@tool
def get_ashare_market_news(limit: int = 20) -> str:
    """获取 A 股市场相关新闻（尽力而为）。"""
    try:
        import akshare as ak

        candidates = [
            # 不同 AkShare 版本可能存在不同接口名
            "stock_news_em",
            "stock_zh_a_news_em",
            "stock_market_news_em",
        ]
        func = _pick_first_existing(ak, candidates)
        if not func:
            return "未找到可用的 A 股新闻接口（AkShare 版本可能不包含相关方法）。"

        # 部分接口不需要参数
        df = func()
        return _df_to_text(df, limit=limit)

    except Exception as e:
        return f"获取 A 股市场新闻失败: {e}"


@tool
def get_hk_market_news(limit: int = 20) -> str:
    """获取港股市场相关新闻（尽力而为）。"""
    try:
        import akshare as ak

        candidates = [
            "stock_hk_news_em",
            "stock_hk_news",
        ]
        func = _pick_first_existing(ak, candidates)
        if not func:
            return "未找到可用的港股新闻接口（AkShare 版本可能不包含相关方法）。"

        df = func()
        return _df_to_text(df, limit=limit)

    except Exception as e:
        return f"获取港股市场新闻失败: {e}"


@tool
def get_ashare_stock_news(symbol: str, limit: int = 20) -> str:
    """获取指定 A 股股票相关新闻（尽力而为）。

    Args:
        symbol: 6 位股票代码，例如 600519。
    """
    try:
        import akshare as ak

        # 一些接口可能叫 stock_news_em(symbol=...), 或者 stock_zh_a_news_em(symbol=...)
        candidates = [
            "stock_news_em",
            "stock_zh_a_news_em",
            "stock_news_stock_jqka",
        ]
        func = _pick_first_existing(ak, candidates)
        if not func:
            return "未找到可用的 A 股个股新闻接口（AkShare 版本可能不包含相关方法）。"

        try:
            df = func(symbol=symbol)
        except TypeError:
            # 兼容参数名差异
            df = func(stock=symbol)

        return _df_to_text(df, limit=limit)

    except Exception as e:
        return f"获取 A 股个股新闻失败: {e}"


@tool
def get_hk_stock_news(symbol: str, limit: int = 20) -> str:
    """获取指定港股股票相关新闻（尽力而为）。

    Args:
        symbol: 港股代码，通常为 5 位数字字符串，例如 00700。
    """
    try:
        import akshare as ak

        candidates = [
            "stock_hk_news_em",
            "stock_hk_stock_news_em",
        ]
        func = _pick_first_existing(ak, candidates)
        if not func:
            return "未找到可用的港股个股新闻接口（AkShare 版本可能不包含相关方法）。"

        try:
            df = func(symbol=symbol)
        except TypeError:
            df = func(stock=symbol)

        return _df_to_text(df, limit=limit)

    except Exception as e:
        return f"获取港股个股新闻失败: {e}"
