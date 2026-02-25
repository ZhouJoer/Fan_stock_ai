"""
本地股票列表存储：保存股票代码、名称及属性（市值、是否 ST、市场等）。
文件：data/stock_list.json，与 stock_name_code_map 互补（映射表侧重名称查代码，本模块侧重属性与列表）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_STOCK_LIST_FILE = _DATA_DIR / "stock_list.json"

# 内存缓存：code -> 属性 dict
_store: Dict[str, Dict[str, Any]] = {}
_loaded = False


def _normalize_code(code: str) -> str:
    s = (code or "").strip()
    if "." in s:
        s = s.split(".")[0]
    s = re.sub(r"[^\d]", "", s)
    return s.zfill(6)[:6] if len(s) >= 6 else s.zfill(5)[:5]


def _load() -> None:
    global _store, _loaded
    if _loaded:
        return
    _loaded = True
    _store = {}
    if not _STOCK_LIST_FILE.exists():
        return
    try:
        with open(_STOCK_LIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else (data.get("stocks") or data.get("list") or [])
        for item in items:
            if isinstance(item, dict) and item.get("code"):
                code = _normalize_code(str(item["code"]))
                _store[code] = dict(item)
    except Exception:
        pass


def _save() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    items = list(_store.values())
    with open(_STOCK_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def get_stock_list() -> List[Dict[str, Any]]:
    """返回全部本地股票列表，每项含 code、name 及已有属性。"""
    _load()
    return [dict(v) for v in _store.values()]


def get_stock_by_code(code: str) -> Optional[Dict[str, Any]]:
    """按代码查询一条，未找到返回 None。"""
    if not code:
        return None
    _load()
    c = _normalize_code(str(code))
    if c in _store:
        return dict(_store[c])
    return _store.get(str(code).strip())


def upsert_stock(
    code: str,
    name: str = "",
    *,
    market: str = "",
    market_cap: Optional[float] = None,
    is_st: Optional[bool] = None,
    updated_at: str = "",
    **extra: Any,
) -> None:
    """按代码插入或更新一条；name/code 必填，其余可选。"""
    if not code:
        return
    _load()
    from datetime import datetime
    c = _normalize_code(str(code))
    now = updated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if c in _store:
        rec = _store[c]
        if name:
            rec["name"] = str(name).strip()
        if market:
            rec["market"] = market
        if market_cap is not None:
            rec["market_cap"] = market_cap
        if is_st is not None:
            rec["is_st"] = is_st
        rec["updated_at"] = now
        for k, v in extra.items():
            rec[k] = v
    else:
        rec = {
            "code": c,
            "name": str(name).strip() if name else "",
            "market": market or "",
            "market_cap": market_cap,
            "is_st": is_st if is_st is not None else ("ST" in (name or "").upper() or "*ST" in (name or "").upper()),
            "updated_at": now,
        }
        rec.update(extra)
        _store[c] = rec
    _save()


def save_stocks(stocks: List[Dict[str, Any]]) -> None:
    """用给定列表覆盖本地存储（每项需含 code，建议含 name）。"""
    global _store, _loaded
    _store = {}
    for item in stocks:
        if isinstance(item, dict) and item.get("code"):
            c = _normalize_code(str(item["code"]))
            _store[c] = dict(item)
    _loaded = True
    _save()


def update_from_spot() -> int:
    """
    从 akshare 全市场行情刷新本地列表（代码、名称、总市值等），返回更新条数。
    可选依赖 akshare；若无则返回 0。
    """
    try:
        import akshare as ak
        from datetime import datetime
    except ImportError:
        return 0
    try:
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty or "代码" not in df.columns or "名称" not in df.columns:
            return 0
        df = df.copy()
        df["_code"] = df["代码"].astype(str).str.strip().str.zfill(6)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated = 0
        for _, row in df.iterrows():
            code = row["_code"]
            name = str(row.get("名称", "")).strip()
            mv = row.get("总市值")
            if mv is not None:
                try:
                    mv = float(mv)
                except (TypeError, ValueError):
                    mv = None
            is_st = "ST" in name.upper() or "*ST" in name.upper()
            upsert_stock(code, name, market_cap=mv, is_st=is_st, updated_at=now)
            updated += 1
        return updated
    except Exception:
        return 0
