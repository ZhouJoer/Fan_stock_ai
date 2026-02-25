"""
一次性回填脚本：基于 stock_daily 生成 stock_weekly / stock_monthly。

用法：
python -m tools.backfill_weekly_monthly
python -m tools.backfill_weekly_monthly 600519
"""

from __future__ import annotations

import sys

from db.manager import backfill_weekly_monthly


def main():
    symbol = sys.argv[1].strip() if len(sys.argv) > 1 else None
    result = backfill_weekly_monthly(symbol=symbol)
    if "error" in result:
        print(f"[backfill] failed: {result['error']}")
        raise SystemExit(1)
    print(f"[backfill] done, updated_symbols={result.get('updated_symbols', 0)}")


if __name__ == "__main__":
    main()
