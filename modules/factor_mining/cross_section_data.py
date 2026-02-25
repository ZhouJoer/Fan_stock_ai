"""
截面数据：PE、换手率等按 (股票代码, 日期) 获取，供因子样本构建时 join。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def fetch_pe_turnover_for_codes(
    codes: List[str],
    start_date: str,
    end_date: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    批量获取指定股票在日期区间内的市盈率、换手率截面数据。
    start_date / end_date 格式 YYYY-MM-DD 或 YYYYMMDD。
    返回: {(code, date_str): {"pe_ratio": float, "turnover_ratio": float}, ...}
    换手率：akshare 返回的为百分比，此处转为小数（如 1.5 -> 0.015）。
    """
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    if not codes:
        return out

    def _norm_d(s: str) -> str:
        s = str(s).strip()[:10]
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    start_norm = _norm_d(start_date)
    end_norm = _norm_d(end_date)
    start_ymd = start_norm.replace("-", "")
    end_ymd = end_norm.replace("-", "")

    try:
        import akshare as ak
    except ImportError:
        return out

    for code in codes:
        code = str(code).strip()
        if not code or len(code) < 6:
            continue
        # 换手率：日线接口带 换手率 列
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_ymd,
                end_date=end_ymd,
                adjust="",
            )
            if df is not None and not df.empty and "日期" in df.columns:
                date_col = "日期"
                for _, row in df.iterrows():
                    d = row[date_col]
                    d_str = _norm_d(str(pd.Timestamp(d)) if hasattr(d, "strftime") else str(d))
                    turn = row.get("换手率")
                    if turn is not None and pd.notna(turn):
                        try:
                            turn_f = float(turn)
                            if turn_f < 0 or turn_f > 100:
                                turn_f = 0.0
                            else:
                                turn_f = turn_f / 100.0
                        except (TypeError, ValueError):
                            turn_f = 0.0
                    else:
                        turn_f = 0.0
                    key = (code, d_str)
                    if key not in out:
                        out[key] = {"pe_ratio": 0.0, "turnover_ratio": 0.0}
                    out[key]["turnover_ratio"] = turn_f
        except Exception:
            pass

        # 市盈率：乐咕乐股接口（按股票取日度）
        try:
            if hasattr(ak, "stock_a_lg_indicator"):
                ind = ak.stock_a_lg_indicator(stock=code)
            elif hasattr(ak, "stock_a_indicator_lg"):
                ind = ak.stock_a_indicator_lg(stock=code)
            else:
                ind = None
            if ind is not None and not ind.empty:
                date_col = next((c for c in ind.columns if "date" in c.lower() or "日期" in c or "trade" in c), ind.columns[0])
                pe_col = next((c for c in ind.columns if "pe" in c.lower() or "市盈" in str(c)), None)
                if pe_col:
                    for _, row in ind.iterrows():
                        d = row[date_col]
                        d_str = _norm_d(str(pd.Timestamp(d)) if hasattr(d, "strftime") else str(d))
                        if d_str < start_norm or d_str > end_norm:
                            continue
                        try:
                            pe_val = float(row[pe_col])
                            if pe_val <= 0 or pd.isna(pe_val):
                                pe_val = 0.0
                        except (TypeError, ValueError):
                            pe_val = 0.0
                        key = (code, d_str)
                        if key not in out:
                            out[key] = {"pe_ratio": 0.0, "turnover_ratio": 0.0}
                        out[key]["pe_ratio"] = pe_val
        except Exception:
            pass

    return out
