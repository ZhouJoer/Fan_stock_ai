"""
因子数据的关系型存储：保存/加载截面因子与远期收益，基于 SQLite。
表：factor_values(symbol, trade_date, factor_name, value)、forward_returns(symbol, trade_date, label_horizon, y)。
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from db.manager import get_connection


def init_factor_tables(conn=None):
    """创建因子相关表（若不存在）。"""
    own = conn is None
    if own:
        conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_values (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                factor_name TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (symbol, trade_date, factor_name)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_returns (
                symbol TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                label_horizon INTEGER NOT NULL,
                y REAL NOT NULL,
                PRIMARY KEY (symbol, trade_date, label_horizon)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_factor_values_symbol_date ON factor_values(symbol, trade_date)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_factor_values_name_date ON factor_values(factor_name, trade_date)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_forward_returns_symbol_date ON forward_returns(symbol, trade_date)"
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def save_factor_df(
    factor_df: pd.DataFrame,
    factor_columns: Optional[List[str]] = None,
    label_horizon: int = 1,
) -> int:
    """
    将因子 DataFrame 持久化到数据库。
    factor_df 需含列 date, stock_code, y 以及若干因子列。
    factor_columns: 要保存的因子名列表，默认取除 date/stock_code/y 外的所有列。
    返回写入行数（因子值行数 + 远期收益行数）。
    """
    if factor_df is None or factor_df.empty:
        return 0
    if "date" not in factor_df.columns or "stock_code" not in factor_df.columns or "y" not in factor_df.columns:
        return 0
    if factor_columns is None:
        factor_columns = [
            c for c in factor_df.columns
            if c not in ("date", "stock_code", "y")
        ]
    if not factor_columns:
        return 0

    conn = get_connection()
    written = 0
    try:
        cur = conn.cursor()
        horizon = max(1, int(label_horizon))
        # 远期收益
        for _, row in factor_df.iterrows():
            try:
                d = str(row["date"]).strip()[:10]
                code = str(row["stock_code"]).strip()
                y_val = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue
            cur.execute(
                """
                INSERT OR REPLACE INTO forward_returns (symbol, trade_date, label_horizon, y)
                VALUES (?, ?, ?, ?)
                """,
                (code, d, horizon, y_val),
            )
            written += 1
        # 因子值（长表）
        for _, row in factor_df.iterrows():
            try:
                d = str(row["date"]).strip()[:10]
                code = str(row["stock_code"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            for f in factor_columns:
                if f not in factor_df.columns:
                    continue
                try:
                    v = float(row[f])
                except (TypeError, ValueError):
                    continue
                cur.execute(
                    """
                    INSERT OR REPLACE INTO factor_values (symbol, trade_date, factor_name, value)
                    VALUES (?, ?, ?, ?)
                    """,
                    (code, d, f, v),
                )
                written += 1
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
    return written


def load_factor_df(
    symbols: List[str],
    start_date: str,
    end_date: str,
    factor_names: List[str],
    label_horizon: int = 1,
) -> Optional[pd.DataFrame]:
    """
    从数据库加载因子与远期收益，拼成与 build_training_samples 输出同构的 DataFrame。
    列：date, stock_code, y, factor1, factor2, ...
    若缺数据则返回 None；若部分有则只返回能拼齐 (date, stock_code) 且含 y 与全部 factor_names 的行。
    """
    if not symbols or not factor_names:
        return None
    horizon = max(1, int(label_horizon))
    conn = get_connection()
    try:
        place_s = ",".join("?" * len(symbols))
        # 远期收益
        qy = f"""
        SELECT symbol, trade_date, y
        FROM forward_returns
        WHERE symbol IN ({place_s}) AND label_horizon = ?
          AND trade_date >= ? AND trade_date <= ?
        """
        df_y = pd.read_sql_query(
            qy,
            conn,
            params=(*symbols, horizon, start_date, end_date),
        )
        if df_y.empty:
            return None
        df_y = df_y.rename(columns={"symbol": "stock_code", "trade_date": "date"})
        # 因子值：按因子名分别查再 pivot，或一次查所有 factor_name IN (...)
        place_f = ",".join("?" * len(factor_names))
        qf = f"""
        SELECT symbol, trade_date, factor_name, value
        FROM factor_values
        WHERE symbol IN ({place_s}) AND factor_name IN ({place_f})
          AND trade_date >= ? AND trade_date <= ?
        """
        df_f = pd.read_sql_query(
            qf,
            conn,
            params=(*symbols, *factor_names, start_date, end_date),
        )
        if df_f.empty:
            return None
        # pivot: (symbol, trade_date) -> 列 factor_name
        df_pivot = df_f.pivot_table(
            index=["symbol", "trade_date"],
            columns="factor_name",
            values="value",
            aggfunc="first",
        ).reset_index()
        df_pivot = df_pivot.rename(columns={"symbol": "stock_code", "trade_date": "date"})
        # 必须有全部因子列
        missing = [f for f in factor_names if f not in df_pivot.columns]
        if missing:
            return None
        merged = df_y.merge(
            df_pivot,
            on=["stock_code", "date"],
            how="inner",
        )
        if merged.empty:
            return None
        merged = merged[["date", "stock_code", "y"] + factor_names]
        return merged.dropna().sort_values(["date", "stock_code"]).reset_index(drop=True)
    except Exception:
        return None
    finally:
        conn.close()


def get_stored_factor_names() -> List[str]:
    """返回数据库中已出现过的因子名列表（去重）。"""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT factor_name FROM factor_values ORDER BY factor_name",
            conn,
        )
        return df["factor_name"].tolist() if "factor_name" in df.columns else []
    except Exception:
        return []
    finally:
        conn.close()


def get_factor_date_range(symbol: Optional[str] = None, factor_name: Optional[str] = None) -> Optional[tuple]:
    """
    返回 (min_date, max_date) 或 None。
    若 symbol 给定则按该股票；若 factor_name 给定则按该因子；都给定则交集。
    """
    conn = get_connection()
    try:
        q = "SELECT MIN(trade_date) AS min_d, MAX(trade_date) AS max_d FROM factor_values WHERE 1=1"
        params = []
        if symbol:
            q += " AND symbol = ?"
            params.append(symbol)
        if factor_name:
            q += " AND factor_name = ?"
            params.append(factor_name)
        df = pd.read_sql_query(q, conn, params=params if params else None)
        if df.empty or pd.isna(df.iloc[0]["min_d"]) or pd.isna(df.iloc[0]["max_d"]):
            return None
        return (str(df.iloc[0]["min_d"]), str(df.iloc[0]["max_d"]))
    except Exception:
        return None
    finally:
        conn.close()


def delete_factor_data(
    symbol: Optional[str] = None,
    trade_date_from: Optional[str] = None,
    trade_date_to: Optional[str] = None,
    factor_name: Optional[str] = None,
) -> int:
    """
    按条件删除因子/远期收益数据，返回删除行数。
    不传则不删（至少传一个条件更安全，这里允许全删用于测试）。
    """
    conn = get_connection()
    deleted = 0
    try:
        cur = conn.cursor()
        for table in ("factor_values", "forward_returns"):
            q = f"DELETE FROM {table} WHERE 1=1"
            params = []
            if symbol:
                col = "symbol"
                q += f" AND {col} = ?"
                params.append(symbol)
            if trade_date_from:
                col = "trade_date"
                q += f" AND {col} >= ?"
                params.append(trade_date_from)
            if trade_date_to:
                col = "trade_date"
                q += f" AND {col} <= ?"
                params.append(trade_date_to)
            if factor_name and table == "factor_values":
                q += " AND factor_name = ?"
                params.append(factor_name)
            cur.execute(q, params)
            deleted += cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return deleted
