import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "stock_data.db")

_LEVEL_TABLE_MAP = {
    "daily": "stock_daily",
    "weekly": "stock_weekly",
    "monthly": "stock_monthly",
}


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_daily (
            symbol TEXT,
            trade_date TEXT,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            pct_chg REAL,
            PRIMARY KEY (symbol, trade_date)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_weekly (
            symbol TEXT,
            trade_date TEXT,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            pct_chg REAL,
            PRIMARY KEY (symbol, trade_date)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_monthly (
            symbol TEXT,
            trade_date TEXT,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            pct_chg REAL,
            PRIMARY KEY (symbol, trade_date)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS kline_sync_state (
            symbol TEXT PRIMARY KEY,
            daily_latest TEXT,
            weekly_latest TEXT,
            monthly_latest TEXT,
            last_quality_status TEXT,
            updated_at TEXT
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_daily_symbol_date ON stock_daily(symbol, trade_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_weekly_symbol_date ON stock_weekly(symbol, trade_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock_monthly_symbol_date ON stock_monthly(symbol, trade_date)")
    conn.commit()
    conn.close()
    # 因子表（关系型存储）
    try:
        from db.factor_storage import init_factor_tables
        init_factor_tables()
    except Exception as e:
        print(f"Warning: init_factor_tables: {e}")


def _normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"missing column: {col}")
    out = df[required_cols].copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["日期", "开盘", "收盘", "最高", "最低"])
    out = out.drop_duplicates(subset=["日期"], keep="last").sort_values("日期")
    return out.reset_index(drop=True)


def _latest_date_in_table(conn: sqlite3.Connection, table: str, symbol: str) -> Optional[str]:
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(trade_date) FROM {table} WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    return result[0] if result and result[0] else None


def _aggregate_from_daily(df_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    # freq: "W-FRI" or "M"
    if df_daily.empty:
        return pd.DataFrame()
    d = df_daily.copy()
    d["dt"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d = d.dropna(subset=["dt"]).sort_values("dt")
    if d.empty:
        return pd.DataFrame()
    d["period"] = d["dt"].dt.to_period(freq)
    agg = (
        d.groupby("period")
        .agg(
            trade_date=("trade_date", "max"),
            open=("open", "first"),
            close=("close", "last"),
            high=("high", "max"),
            low=("low", "min"),
            volume=("volume", "sum"),
        )
        .reset_index(drop=True)
    )
    base_close = d.groupby("period")["close"].first().reset_index(drop=True)
    agg["pct_chg"] = (agg["close"] / base_close - 1.0) * 100.0
    # 质量校验：不合法K线剔除
    agg = agg[(agg["high"] >= agg[["open", "close"]].max(axis=1)) & (agg["low"] <= agg[["open", "close"]].min(axis=1))]
    return agg.reset_index(drop=True)


def _upsert_kline_rows(conn: sqlite3.Connection, table: str, symbol: str, df_rows: pd.DataFrame):
    if df_rows.empty:
        return
    values = [
        (
            symbol,
            str(r["trade_date"]),
            float(r["open"]),
            float(r["close"]),
            float(r["high"]),
            float(r["low"]),
            float(r["volume"]),
            float(r["pct_chg"]) if pd.notna(r["pct_chg"]) else 0.0,
        )
        for _, r in df_rows.iterrows()
    ]
    cur = conn.cursor()
    cur.executemany(
        f"""
        INSERT OR REPLACE INTO {table} (symbol, trade_date, open, close, high, low, volume, pct_chg)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )


def _rebuild_weekly_monthly_for_symbol(conn: sqlite3.Connection, symbol: str):
    daily_latest = _latest_date_in_table(conn, "stock_daily", symbol)
    if not daily_latest:
        return
    weekly_latest = _latest_date_in_table(conn, "stock_weekly", symbol)
    monthly_latest = _latest_date_in_table(conn, "stock_monthly", symbol)

    weekly_start = None
    monthly_start = None
    if weekly_latest:
        weekly_start = (pd.to_datetime(weekly_latest) - timedelta(days=10)).strftime("%Y-%m-%d")
    if monthly_latest:
        monthly_start = (pd.to_datetime(monthly_latest) - timedelta(days=40)).strftime("%Y-%m-%d")

    cur = conn.cursor()
    if weekly_start or monthly_start:
        start_dt = min([d for d in [weekly_start, monthly_start] if d is not None])
        df_daily = pd.read_sql_query(
            """
            SELECT trade_date, open, close, high, low, volume, pct_chg
            FROM stock_daily
            WHERE symbol = ? AND trade_date >= ?
            ORDER BY trade_date ASC
            """,
            conn,
            params=(symbol, start_dt),
        )
    else:
        df_daily = pd.read_sql_query(
            """
            SELECT trade_date, open, close, high, low, volume, pct_chg
            FROM stock_daily
            WHERE symbol = ?
            ORDER BY trade_date ASC
            """,
            conn,
            params=(symbol,),
        )
    if df_daily.empty:
        return

    weekly = _aggregate_from_daily(df_daily, "W-FRI")
    monthly = _aggregate_from_daily(df_daily, "M")

    if not weekly.empty:
        min_w = str(weekly["trade_date"].min())
        cur.execute("DELETE FROM stock_weekly WHERE symbol = ? AND trade_date >= ?", (symbol, min_w))
        _upsert_kline_rows(conn, "stock_weekly", symbol, weekly)
    if not monthly.empty:
        min_m = str(monthly["trade_date"].min())
        cur.execute("DELETE FROM stock_monthly WHERE symbol = ? AND trade_date >= ?", (symbol, min_m))
        _upsert_kline_rows(conn, "stock_monthly", symbol, monthly)

    quality_status = "ok"
    if weekly.empty or monthly.empty:
        quality_status = "partial"
    cur.execute(
        """
        INSERT OR REPLACE INTO kline_sync_state (
            symbol, daily_latest, weekly_latest, monthly_latest, last_quality_status, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            daily_latest,
            _latest_date_in_table(conn, "stock_weekly", symbol),
            _latest_date_in_table(conn, "stock_monthly", symbol),
            quality_status,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )


def save_daily_data(symbol: str, df: pd.DataFrame):
    """保存日线并增量重建周/月线。"""
    if df.empty:
        return
    try:
        normalized = _normalize_daily_frame(df)
    except Exception as e:
        print(f"Error normalize daily data: {e}")
        return
    if normalized.empty:
        return

    conn = get_connection()
    try:
        cur = conn.cursor()
        values = [
            (
                symbol,
                row["日期"],
                float(row["开盘"]),
                float(row["收盘"]),
                float(row["最高"]),
                float(row["最低"]),
                float(row["成交量"]) if pd.notna(row["成交量"]) else 0.0,
                float(row["涨跌幅"]) if pd.notna(row["涨跌幅"]) else 0.0,
            )
            for _, row in normalized.iterrows()
        ]
        cur.executemany(
            """
            INSERT OR REPLACE INTO stock_daily (symbol, trade_date, open, close, high, low, volume, pct_chg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        _rebuild_weekly_monthly_for_symbol(conn, symbol)
        conn.commit()
    except Exception as e:
        print(f"Error saving data: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_latest_date(symbol: str, level: str = "daily") -> Optional[str]:
    table = _LEVEL_TABLE_MAP.get(level, "stock_daily")
    conn = get_connection()
    try:
        return _latest_date_in_table(conn, table, symbol)
    finally:
        conn.close()


def count_cached_data(symbol: str, level: str = "daily") -> int:
    table = _LEVEL_TABLE_MAP.get(level, "stock_daily")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", (symbol,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0


def get_cached_history(symbol: str, limit: int = 30, level: str = "daily") -> pd.DataFrame:
    """从数据库获取最近行情，支持 daily/weekly/monthly。"""
    table = _LEVEL_TABLE_MAP.get(level, "stock_daily")
    conn = get_connection()
    query = f"""
    SELECT trade_date as 日期, open as 开盘, close as 收盘,
           high as 最高, low as 最低, volume as 成交量, pct_chg as 涨跌幅
    FROM {table}
    WHERE symbol = ?
    ORDER BY trade_date DESC
    LIMIT ?
    """
    try:
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        return df.sort_values(by="日期", ascending=True)
    except Exception as e:
        print(f"Error reading cache: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def list_cached_symbols() -> list[str]:
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol ASC", conn)
        return df["symbol"].tolist() if "symbol" in df.columns else []
    except Exception:
        return []
    finally:
        conn.close()


def backfill_weekly_monthly(symbol: Optional[str] = None) -> dict:
    conn = get_connection()
    try:
        symbols = [symbol] if symbol else list_cached_symbols()
        updated = 0
        for s in symbols:
            if not s:
                continue
            _rebuild_weekly_monthly_for_symbol(conn, s)
            updated += 1
        conn.commit()
        return {"updated_symbols": updated}
    except Exception as e:
        conn.rollback()
        return {"updated_symbols": 0, "error": str(e)}
    finally:
        conn.close()


# Initialize DB on module load
init_db()
