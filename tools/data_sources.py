"""
行情数据源模块：统一封装 akshare / baostock 拉取逻辑。
"""

from __future__ import annotations

import io
import time
import contextlib
import pandas as pd

SUPPORTED_STOCK_SOURCES = ("akshare", "baostock")
_stock_source_priority = ["akshare", "baostock"]
_source_unavailable_until: dict[str, float] = {k: 0.0 for k in SUPPORTED_STOCK_SOURCES}


def get_stock_source_priority() -> list[str]:
    """获取当前股票日线数据源优先级（按顺序尝试）。"""
    return list(_stock_source_priority)


def set_stock_source_priority(order: list[str] | tuple[str, ...]) -> list[str]:
    """设置股票日线数据源优先级，自动清洗无效项并补齐未提供源。"""
    global _stock_source_priority
    cleaned: list[str] = []
    for s in (order or []):
        key = str(s).strip().lower()
        if key in SUPPORTED_STOCK_SOURCES and key not in cleaned:
            cleaned.append(key)
    for s in SUPPORTED_STOCK_SOURCES:
        if s not in cleaned:
            cleaned.append(s)
    _stock_source_priority = cleaned
    return list(_stock_source_priority)


def mark_source_unavailable(source: str, cooldown_seconds: int = 300) -> None:
    key = str(source).strip().lower()
    if key in _source_unavailable_until:
        _source_unavailable_until[key] = time.time() + max(1, int(cooldown_seconds))


def clear_source_unavailable(source: str) -> None:
    key = str(source).strip().lower()
    if key in _source_unavailable_until:
        _source_unavailable_until[key] = 0.0


def is_source_available(source: str) -> bool:
    key = str(source).strip().lower()
    if key not in _source_unavailable_until:
        return False
    return time.time() >= float(_source_unavailable_until.get(key, 0.0) or 0.0)


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名到: 日期, 开盘, 最高, 最低, 收盘, 成交量, 涨跌幅"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename_map = {}
    for src, dst in [
        ("date", "日期"),
        ("open", "开盘"),
        ("high", "最高"),
        ("low", "最低"),
        ("close", "收盘"),
        ("volume", "成交量"),
        ("pct_chg", "涨跌幅"),
        ("pctChg", "涨跌幅"),
        ("changepercent", "涨跌幅"),
    ]:
        if src in out.columns and dst not in out.columns:
            rename_map[src] = dst
    if rename_map:
        out = out.rename(columns=rename_map)
    required = ["日期", "开盘", "最高", "最低", "收盘", "成交量", "涨跌幅"]
    for c in required:
        if c not in out.columns:
            out[c] = 0.0
    # 缺少涨跌幅时按收盘价计算（百分比）
    if "涨跌幅" in out.columns:
        out["涨跌幅"] = pd.to_numeric(out["涨跌幅"], errors="coerce")
        need_fill = out["涨跌幅"].isna().all()
    else:
        need_fill = True
    if need_fill:
        close = pd.to_numeric(out["收盘"], errors="coerce")
        out["涨跌幅"] = close.pct_change().fillna(0.0) * 100.0
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    out = out.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    return out[required]


def fetch_stock_hist_akshare(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """通过 akshare 拉取 A 股日线。"""
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=str(symbol).strip(),
        period="daily",
        start_date=str(start_date),
        end_date=str(end_date),
        adjust=adjust,
    )
    return _normalize_ohlcv_df(df)


def fetch_stock_hist_baostock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """通过 baostock 拉取 A 股日线（复权：前复权）。"""
    try:
        import baostock as bs
    except Exception as e:
        raise RuntimeError(f"baostock 不可用: {e}")

    code = str(symbol).strip()
    if code.startswith("6"):
        bs_code = f"sh.{code}"
    else:
        bs_code = f"sz.{code}"

    # baostock SDK 会在 login/logout 时直接 print，重定向 stdout 以免污染后端日志
    with contextlib.redirect_stdout(io.StringIO()):
        lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock 登录失败: {lg.error_code} {lg.error_msg}")
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
            end_date=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}",
            frequency="d",
            adjustflag="2",  # 2: 前复权
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock 查询失败: {rs.error_code} {rs.error_msg}")
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=rs.fields)
        # baostock 返回字符串，统一转数值
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return _normalize_ohlcv_df(df)
    finally:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bs.logout()
        except Exception:
            pass


def fetch_stock_hist_by_source(
    source: str,
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """按 source 路由拉取股票日线。"""
    key = str(source).strip().lower()
    if key == "akshare":
        return fetch_stock_hist_akshare(symbol, start_date, end_date, adjust=adjust)
    if key == "baostock":
        return fetch_stock_hist_baostock(symbol, start_date, end_date)
    raise ValueError(f"不支持的数据源: {source}")

