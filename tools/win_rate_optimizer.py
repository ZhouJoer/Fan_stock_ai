"""
胜率优化：re-export 自 modules.strategy_config，保持向后兼容。
业务实现已迁至 modules/strategy_config/win_rate_optimizer.py。
"""
from modules.strategy_config.win_rate_optimizer import (
    high_win_rate_decision,
    get_optimized_config,
    HIGH_WIN_RATE_CONFIG,
    calculate_trend_filter,
    calculate_volume_confirmation,
    count_bullish_indicators,
    count_bearish_indicators,
    EquityCurveFilter,
    confirm_signal_with_delay,
    WIN_RATE_IMPROVEMENT_TIPS,
)

__all__ = [
    "high_win_rate_decision",
    "get_optimized_config",
    "HIGH_WIN_RATE_CONFIG",
    "calculate_trend_filter",
    "calculate_volume_confirmation",
    "count_bullish_indicators",
    "count_bearish_indicators",
    "EquityCurveFilter",
    "confirm_signal_with_delay",
    "WIN_RATE_IMPROVEMENT_TIPS",
]
