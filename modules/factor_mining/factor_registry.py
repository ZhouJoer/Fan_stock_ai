"""
因子注册表：可选因子的唯一配置入口，便于前端展示与后端增减因子。

因子大类（内部 key → 前端展示名）：
- style → 风格与估值：动量、波动、估值（PE/换手）等
- trading → 情绪与交易：成交量、反转、情绪指标（RSI/量价背离等）等

子类（sub_category）对应中文：momentum→动量, volatility→波动, valuation→估值,
volume→成交量, reversal→反转, sentiment_misc→情绪指标。

- 新增因子：在 AVAILABLE_FACTORS 中增加一项，并在 factor_definitions.build_factor_values 中实现计算（若为量价因子）。
- 删除因子：从 AVAILABLE_FACTORS 移除，并从 factor_definitions 中移除对应计算即可。
"""

from __future__ import annotations

from typing import Dict, List, Any

# 大类展示名（内部 key：style=风格与估值，trading=情绪与交易）
CATEGORY_LABEL_ZH: Dict[str, str] = {
    "style": "风格与估值",
    "trading": "情绪与交易",
}
# 子类展示名
SUB_CATEGORY_LABEL_ZH: Dict[str, str] = {
    "momentum": "动量",
    "volatility": "波动",
    "valuation": "估值",
    "volume": "成交量",
    "reversal": "反转",
    "sentiment_misc": "情绪指标",
}

# 可选因子列表：id 与 factor_definitions / factor_trainer 中使用的列名一致
# category 为内部 key（style=风格与估值，trading=情绪与交易），sub_category 为子类 key
AVAILABLE_FACTORS: List[Dict[str, Any]] = [
    # 风格与估值
    {"id": "momentum_20", "name_zh": "20日动量", "category": "style", "sub_category": "momentum", "description": "过去20日收益率"},
    {"id": "momentum_60", "name_zh": "60日动量", "category": "style", "sub_category": "momentum", "description": "过去60日收益率"},
    {"id": "trend_strength", "name_zh": "趋势强度", "category": "style", "sub_category": "volatility", "description": "价格线性趋势斜率相对均值"},
    {"id": "low_volatility", "name_zh": "低波动", "category": "style", "sub_category": "volatility", "description": "负的日收益波动率"},
    {"id": "pe_ratio", "name_zh": "市盈率", "category": "style", "sub_category": "valuation", "description": "PE（需截面数据）"},
    {"id": "turnover_ratio", "name_zh": "换手率", "category": "style", "sub_category": "valuation", "description": "真实换手（需截面数据）"},
    # 情绪与交易
    {"id": "volume_surge", "name_zh": "成交量突增", "category": "trading", "sub_category": "volume", "description": "近期成交量相对均线比例"},
    {"id": "turnover_proxy", "name_zh": "换手代理", "category": "trading", "sub_category": "volume", "description": "短长期成交量比"},
    {"id": "short_reversal", "name_zh": "短期反转", "category": "trading", "sub_category": "reversal", "description": "短期涨幅取负"},
    {"id": "intraday_amplitude", "name_zh": "日内振幅", "category": "trading", "sub_category": "reversal", "description": "(高-低)/收 均值"},
    {"id": "rsi", "name_zh": "RSI", "category": "trading", "sub_category": "sentiment_misc", "description": "RSI(14)"},
    {"id": "price_volume_divergence", "name_zh": "量价背离", "category": "trading", "sub_category": "sentiment_misc", "description": "价与量变化相关系数取负"},
    {"id": "high_position_volume", "name_zh": "高位成交", "category": "trading", "sub_category": "sentiment_misc", "description": "成交在高位的加权分位"},
    {"id": "volume_oscillation", "name_zh": "成交量震荡", "category": "trading", "sub_category": "volume", "description": "成交量变异系数"},
]

# factor_set -> 因子 id 列表（style=风格与估值，trading=情绪与交易），与 factor_trainer 一致
_FACTOR_SET_IDS: Dict[str, List[str]] = {
    "style": [f["id"] for f in AVAILABLE_FACTORS if f["category"] == "style"],
    "trading": [f["id"] for f in AVAILABLE_FACTORS if f["category"] == "trading"],
    "momentum": ["momentum_20", "momentum_60"],
    "volatility": ["low_volatility", "trend_strength"],
    "volume": ["volume_surge", "turnover_proxy"],
    "reversal": ["short_reversal", "intraday_amplitude"],
}
_all_ids = [f["id"] for f in AVAILABLE_FACTORS]
_FACTOR_SET_IDS["hybrid"] = _all_ids


def get_available_factors() -> List[Dict[str, Any]]:
    """返回可选因子列表，含 id、name_zh、category、sub_category、description，以及前端展示用 category_label_zh、sub_category_label_zh。"""
    out = []
    for f in AVAILABLE_FACTORS:
        row = dict(f)
        row["category_label_zh"] = CATEGORY_LABEL_ZH.get(f["category"], f["category"])
        row["sub_category_label_zh"] = SUB_CATEGORY_LABEL_ZH.get(f["sub_category"], f["sub_category"])
        out.append(row)
    return out


def get_factor_ids_by_set(factor_set: str) -> List[str]:
    """按 factor_set 返回因子 id 列表，供 factor_trainer 等使用。支持 style | trading | momentum | volatility | volume | reversal | hybrid；兼容旧名 risk→style, sentiment→trading。"""
    key = str(factor_set).strip().lower()
    if key == "risk":
        key = "style"
    elif key == "sentiment":
        key = "trading"
    return list(_FACTOR_SET_IDS.get(key, _all_ids))


def get_all_factor_ids() -> List[str]:
    """返回全部因子 id。"""
    return list(_all_ids)
