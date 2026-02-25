"""
本地股票名称与代码映射表。
在所有解析股票名称/代码的实现中优先查此表，查不到再走 akshare 等接口。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# 映射表文件：项目根目录下 data/stock_name_code_map.json
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_MAP_FILE = _DATA_DIR / "stock_name_code_map.json"

# 内存缓存：按市场区分，A 股 6 位、港股 5 位
# _by_name["a_share"] / _by_name["hk"]: name -> code
# _by_code: code -> name（统一，便于按代码反查）
_by_name: dict[str, dict[str, str]] = {"a_share": {}, "hk": {}}
_by_code: dict[str, str] = {}
_loaded = False


def _normalize_code(code: str, market: str = "a_share") -> str:
    """规范化代码：去掉后缀，A 股 6 位，港股 5 位。"""
    s = (code or "").strip()
    if "." in s:
        s = s.split(".")[0]
    s = re.sub(r"[^\d]", "", s)
    if not s:
        return (code or "").strip()
    if market == "hk" or len(s) <= 5:
        return s.zfill(5)[:5]
    return s.zfill(6)[:6]


def _load() -> None:
    global _by_name, _by_code, _loaded
    if _loaded:
        return
    _loaded = True
    _by_name = {"a_share": {}, "hk": {}}
    _by_code = {}
    if not _MAP_FILE.exists():
        return
    try:
        with open(_MAP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 新格式：a_share / hk
        for m in ("a_share", "hk"):
            for k, v in (data.get(m) or {}).items():
                if k and v:
                    _by_name[m][str(k).strip()] = _normalize_code(str(v), m)
        # 兼容旧版 by_name / by_code
        if "by_name" in data and isinstance(data["by_name"], dict):
            for k, v in data["by_name"].items():
                if k and v:
                    c = _normalize_code(str(v), "a_share")
                    _by_name["a_share"][str(k).strip()] = c
        for name, code in _by_name["a_share"].items():
            _by_code[code] = name
        for name, code in _by_name["hk"].items():
            _by_code[code] = name
    except Exception:
        pass


def _save() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump({"a_share": _by_name["a_share"], "hk": _by_name["hk"]}, f, ensure_ascii=False, indent=2)


def get_code_by_name(name: str, market: str = "a_share") -> str | None:
    """根据股票名称查代码，优先本地表。market: 'a_share' | 'hk'。未找到返回 None。"""
    if not name or not str(name).strip():
        return None
    _load()
    key = str(name).strip()
    m = "a_share" if market == "a_share" else "hk"
    if key in _by_name[m]:
        return _by_name[m][key]
    for n, c in _by_name[m].items():
        if key in n or n in key:
            return c
    return None


def get_name_by_code(code: str) -> str | None:
    """根据股票代码查名称，优先本地映射表，其次本地股票列表。未找到返回 None。"""
    if not code:
        return None
    _load()
    c = _normalize_code(str(code), "a_share")
    if c in _by_code:
        return _by_code[c]
    c5 = _normalize_code(str(code), "hk")
    if c5 in _by_code:
        return _by_code[c5]
    try:
        from tools.stock_local_store import get_stock_by_code
        rec = get_stock_by_code(code)
        if rec and rec.get("name"):
            return rec["name"]
    except Exception:
        pass
    return None


def set_mapping(name: str, code: str, market: str = "a_share") -> None:
    """添加或更新一条名称-代码映射并持久化；同时写入本地股票列表（代码、名称及属性）。market: 'a_share' | 'hk'。"""
    if not name or not code:
        return
    _load()
    m = "a_share" if market == "a_share" else "hk"
    key_name = str(name).strip()
    key_code = _normalize_code(str(code), m)
    _by_name[m][key_name] = key_code
    _by_code[key_code] = key_name
    _save()
    try:
        from tools.stock_local_store import upsert_stock
        if m == "a_share":
            market = "sh" if key_code.startswith(("6", "5")) else "sz" if key_code.startswith(("0", "3")) else ""
        else:
            market = "hk"
        upsert_stock(key_code, key_name, market=market)
    except Exception:
        pass


def add_mappings(name_code_pairs: list[tuple[str, str]], market: str = "a_share") -> None:
    """批量添加映射并持久化。"""
    if not name_code_pairs:
        return
    _load()
    m = "a_share" if market == "a_share" else "hk"
    for name, code in name_code_pairs:
        if name and code:
            key_name = str(name).strip()
            key_code = _normalize_code(str(code), m)
            _by_name[m][key_name] = key_code
            _by_code[key_code] = key_name
    _save()


def get_all_mappings() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """返回 (by_market name->code, code->name) 的副本。"""
    _load()
    return {k: dict(v) for k, v in _by_name.items()}, dict(_by_code)
