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
    {"id": "ema5", "name_zh": "EMA5偏离", "category": "style", "sub_category": "momentum", "description": "收盘价相对EMA5偏离"},
    {"id": "ema10", "name_zh": "EMA10偏离", "category": "style", "sub_category": "momentum", "description": "收盘价相对EMA10偏离"},
    {"id": "ema12", "name_zh": "EMA12偏离", "category": "style", "sub_category": "momentum", "description": "收盘价相对EMA12偏离"},
    {"id": "ema20", "name_zh": "EMA20偏离", "category": "style", "sub_category": "momentum", "description": "收盘价相对EMA20偏离"},
    {"id": "ema26", "name_zh": "EMA26偏离", "category": "style", "sub_category": "momentum", "description": "收盘价相对EMA26偏离"},
    {"id": "ema120", "name_zh": "EMA120偏离", "category": "style", "sub_category": "momentum", "description": "收盘价相对EMA120偏离"},
    {"id": "bias5", "name_zh": "5日乖离率", "category": "style", "sub_category": "momentum", "description": "收盘价相对5日均线偏离"},
    {"id": "bias10", "name_zh": "10日乖离率", "category": "style", "sub_category": "momentum", "description": "收盘价相对10日均线偏离"},
    {"id": "bias20", "name_zh": "20日乖离率", "category": "style", "sub_category": "momentum", "description": "收盘价相对20日均线偏离"},
    {"id": "bias60", "name_zh": "60日乖离率", "category": "style", "sub_category": "momentum", "description": "收盘价相对60日均线偏离"},
    {"id": "roc6", "name_zh": "6日变动速率", "category": "style", "sub_category": "momentum", "description": "6日价格变动速率"},
    {"id": "roc12", "name_zh": "12日变动速率", "category": "style", "sub_category": "momentum", "description": "12日价格变动速率"},
    {"id": "roc60", "name_zh": "60日变动速率", "category": "style", "sub_category": "momentum", "description": "60日价格变动速率"},
    {"id": "roc120", "name_zh": "120日变动速率", "category": "style", "sub_category": "momentum", "description": "120日价格变动速率"},
    {"id": "price1m", "name_zh": "1月价格动量", "category": "style", "sub_category": "momentum", "description": "当前价相对近1月均价"},
    {"id": "price3m", "name_zh": "3月价格动量", "category": "style", "sub_category": "momentum", "description": "当前价相对近3月均价"},
    {"id": "price1y", "name_zh": "1年价格动量", "category": "style", "sub_category": "momentum", "description": "当前价相对近1年均价"},
    {"id": "trix5", "name_zh": "TRIX5", "category": "style", "sub_category": "momentum", "description": "5日TRIX终极指标"},
    {"id": "trix10", "name_zh": "TRIX10", "category": "style", "sub_category": "momentum", "description": "10日TRIX终极指标"},
    {"id": "cci20", "name_zh": "CCI20", "category": "style", "sub_category": "momentum", "description": "20日顺势指标"},
    {"id": "trend_strength", "name_zh": "趋势强度", "category": "style", "sub_category": "volatility", "description": "价格线性趋势斜率相对均值"},
    {"id": "low_volatility", "name_zh": "低波动", "category": "style", "sub_category": "volatility", "description": "负的日收益波动率"},
    {"id": "variance20", "name_zh": "20日年化方差", "category": "style", "sub_category": "volatility", "description": "20日收益年化方差"},
    {"id": "variance60", "name_zh": "60日年化方差", "category": "style", "sub_category": "volatility", "description": "60日收益年化方差"},
    {"id": "variance120", "name_zh": "120日年化方差", "category": "style", "sub_category": "volatility", "description": "120日收益年化方差"},
    {"id": "skewness_20", "name_zh": "20日偏度", "category": "style", "sub_category": "volatility", "description": "20日收益分布偏度"},
    {"id": "skewness_60", "name_zh": "60日偏度", "category": "style", "sub_category": "volatility", "description": "60日收益分布偏度"},
    {"id": "skewness_120", "name_zh": "120日偏度", "category": "style", "sub_category": "volatility", "description": "120日收益分布偏度"},
    {"id": "kurtosis_20", "name_zh": "20日峰度", "category": "style", "sub_category": "volatility", "description": "20日收益分布峰度"},
    {"id": "kurtosis_60", "name_zh": "60日峰度", "category": "style", "sub_category": "volatility", "description": "60日收益分布峰度"},
    {"id": "kurtosis_120", "name_zh": "120日峰度", "category": "style", "sub_category": "volatility", "description": "120日收益分布峰度"},
    {"id": "sharpe_ratio_20", "name_zh": "20日夏普", "category": "style", "sub_category": "volatility", "description": "20日收益夏普比率"},
    {"id": "sharpe_ratio_60", "name_zh": "60日夏普", "category": "style", "sub_category": "volatility", "description": "60日收益夏普比率"},
    {"id": "sharpe_ratio_120", "name_zh": "120日夏普", "category": "style", "sub_category": "volatility", "description": "120日收益夏普比率"},
    {"id": "boll_up", "name_zh": "布林上轨偏离", "category": "style", "sub_category": "volatility", "description": "布林上轨相对当前价偏离"},
    {"id": "boll_down", "name_zh": "布林下轨偏离", "category": "style", "sub_category": "volatility", "description": "当前价相对布林下轨偏离"},
    {"id": "atr6", "name_zh": "ATR6", "category": "style", "sub_category": "volatility", "description": "6日平均真实波幅(归一化)"},
    {"id": "atr14", "name_zh": "ATR14", "category": "style", "sub_category": "volatility", "description": "14日平均真实波幅(归一化)"},
    {"id": "pe_ratio", "name_zh": "市盈率", "category": "style", "sub_category": "valuation", "description": "PE（需截面数据）"},
    {"id": "turnover_ratio", "name_zh": "换手率", "category": "style", "sub_category": "valuation", "description": "真实换手（需截面数据）"},
    # 情绪与交易
    {"id": "volume_surge", "name_zh": "成交量突增", "category": "trading", "sub_category": "volume", "description": "近期成交量相对均线比例"},
    {"id": "turnover_proxy", "name_zh": "换手代理", "category": "trading", "sub_category": "volume", "description": "短长期成交量比"},
    {"id": "vol5", "name_zh": "5日成交量均值", "category": "trading", "sub_category": "volume", "description": "5日平均成交量"},
    {"id": "vol10", "name_zh": "10日成交量均值", "category": "trading", "sub_category": "volume", "description": "10日平均成交量"},
    {"id": "vol20", "name_zh": "20日成交量均值", "category": "trading", "sub_category": "volume", "description": "20日平均成交量"},
    {"id": "vol60", "name_zh": "60日成交量均值", "category": "trading", "sub_category": "volume", "description": "60日平均成交量"},
    {"id": "vol120", "name_zh": "120日成交量均值", "category": "trading", "sub_category": "volume", "description": "120日平均成交量"},
    {"id": "vol240", "name_zh": "240日成交量均值", "category": "trading", "sub_category": "volume", "description": "240日平均成交量"},
    {"id": "vroc6", "name_zh": "6日成交量变动速率", "category": "trading", "sub_category": "volume", "description": "6日成交量变动速率"},
    {"id": "vroc12", "name_zh": "12日成交量变动速率", "category": "trading", "sub_category": "volume", "description": "12日成交量变动速率"},
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
    "momentum": [
        "momentum_20", "momentum_60",
        "ema5", "ema10", "ema12", "ema20", "ema26", "ema120",
        "bias5", "bias10", "bias20", "bias60",
        "roc6", "roc12", "roc60", "roc120",
        "price1m", "price3m", "price1y",
        "trix5", "trix10", "cci20",
    ],
    "volatility": [
        "low_volatility", "trend_strength",
        "variance20", "variance60", "variance120",
        "skewness_20", "skewness_60", "skewness_120",
        "kurtosis_20", "kurtosis_60", "kurtosis_120",
        "sharpe_ratio_20", "sharpe_ratio_60", "sharpe_ratio_120",
        "boll_up", "boll_down", "atr6", "atr14",
    ],
    "volume": [
        "volume_surge", "turnover_proxy",
        "vol5", "vol10", "vol20", "vol60", "vol120", "vol240",
        "vroc6", "vroc12",
    ],
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
