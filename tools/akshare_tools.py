from langchain_core.tools import tool
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import re
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from tools.stock_name_code_map import get_code_by_name, set_mapping


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """从 DataFrame 中按候选列名顺序取第一个存在的列。"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_hk_symbol(symbol: str) -> str:
    """规范化港股代码为 5 位数字字符串（例如 700 -> 00700）。"""
    digits = re.sub(r"[^\d]", "", symbol or "")
    if not digits:
        return symbol
    return digits.zfill(5)[:5]

@tool
def get_stock_code(symbol_name: str) -> str:
    """
    根据股票名称查询股票代码。
    例如输入 "贵州茅台"，返回 "600519"。
    """
    try:
        # 优先查本地名称-代码映射表
        code = get_code_by_name(symbol_name, "a_share")
        if code:
            name = symbol_name
            return f"{name} 的代码是 {code}"
        # 获取所有A股实时行情数据作为基础列表
        df = ak.stock_zh_a_spot_em()
        # 模糊匹配
        result = df[df['名称'].str.contains(symbol_name)]
        if result.empty:
            return f"未找到名称包含 {symbol_name} 的股票。"
        # 返回第一条匹配结果并写入本地表
        code = result.iloc[0]['代码']
        name = result.iloc[0]['名称']
        set_mapping(name, code, "a_share")
        return f"{name} 的代码是 {code}"
    except Exception as e:
        return f"查询股票代码失败: {e}"


@tool
def get_hk_stock_code(symbol_name: str) -> str:
    """根据港股名称查询港股代码。

    说明：
    - AkShare 的港股实时行情接口在不同版本可能存在差异，这里会尝试多个候选接口。
    - 结果尽量返回 5 位数字代码（例如：腾讯控股 -> 00700）。
    """
    try:
        # 优先查本地名称-代码映射表
        code = get_code_by_name(symbol_name, "hk")
        if code:
            return f"{symbol_name} 的港股代码是 {code}"
        # 候选：不同 AkShare 版本可能存在不同接口
        spot_func = None
        for name in ("stock_hk_spot_em", "stock_hk_spot"):
            spot_func = getattr(ak, name, None)
            if callable(spot_func):
                break

        if not callable(spot_func):
            return "当前 AkShare 版本未提供港股实时行情列表接口。"

        df = spot_func()
        if df is None or getattr(df, "empty", True):
            return "港股行情列表为空，无法查询代码。"

        name_col = _first_existing_column(df, ["名称", "name", "股票名称", "公司名称"])
        code_col = _first_existing_column(df, ["代码", "symbol", "股票代码", "代码(01)"])
        if not name_col or not code_col:
            return f"港股行情数据列不匹配，无法解析名称/代码字段。可用列: {list(df.columns)}"

        result = df[df[name_col].astype(str).str.contains(symbol_name, na=False)]
        if result.empty:
            return f"未找到名称包含 {symbol_name} 的港股。"

        code_raw = str(result.iloc[0][code_col])
        name = str(result.iloc[0][name_col])
        code = _normalize_hk_symbol(code_raw)
        set_mapping(name, code, "hk")
        return f"{name} 的港股代码是 {code}"

    except Exception as e:
        return f"查询港股代码失败: {e}"

from db.manager import save_daily_data, get_cached_history, get_latest_date

@tool
def get_stock_history(symbol: str, period: str = "daily") -> str:
    """
    获取股票的历史行情数据。
    Args:
        symbol: 股票代码，例如 "600519"。
        period: 周期，默认为 "daily"。
    Returns:
        包含最近30天开盘、收盘、最高、最低价的CSV字符串。
    """
    try:
        if period != "daily":
             # 简单起见，目前只缓存日线
             start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
             end_date = datetime.now().strftime("%Y%m%d")
             df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
             if df.empty: return f"未找到股票 {symbol} 的历史数据。"
             columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '涨跌幅']
             return df[columns].tail(30).to_csv(index=False)

        # 检查缓存逻辑
        today_str = datetime.now().strftime("%Y-%m-%d")
        latest_db_date = get_latest_date(symbol)
        
        # 如果缓存数据较新（例如是今天的），直接从数据库读
        # 注意：这里简化逻辑，如果数据库最后日期 >= 今天（或昨天），就认为够用了
        # 实际可能需要更复杂的交易日历判断
        need_fetch = True
        if latest_db_date:
            # 简单的日期比较字符串
             if latest_db_date >= (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"):
                 need_fetch = False
        
        if need_fetch:
            print(f"[Tools] Fetching data from Akshare for {symbol}...")
            # akshare 的 stock_zh_a_hist 接口
            # start_date 设为最近 60 天，保证覆盖所需的30天
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")
            end_date = datetime.now().strftime("%Y%m%d")
            
            df = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
            
            if not df.empty:
                # 只有成功获取才保存
                save_daily_data(symbol, df)
        else:
            print(f"[Tools] Using cached data for {symbol}.")
            
        # 从数据库读取（无论是刚存的还是本来就有的）
        df_subset = get_cached_history(symbol, limit=30)
        
        if df_subset.empty:
            return f"未找到股票 {symbol} 的历史数据（即使尝试更新后）。"
        
        # --- 计算技术指标 ---
        try:
             # 为了计算指标需要更多数据，这里我们假设数据库返回的数据量能够满足基本计算要求
             # 如果数据库只保留了limit条，计算长周期EMA可能不准确。
             # 理想情况是: 读更多数据 -> 计算 -> 切片返回
             
             # 重新读取更多历史数据用于指标计算 (比如 100 天)
             df_calc = get_cached_history(symbol, limit=100) 
             if df_calc.empty: df_calc = df_subset
             
             # 收盘价
             close = df_calc['收盘']
             
             # MACD
             macd = MACD(close=close)
             df_calc['MACD'] = macd.macd()
             df_calc['MACD_SIGNAL'] = macd.macd_signal()
             df_calc['MACD_HIST'] = macd.macd_diff()
             
             # Bollinger Bands
             bollinger = BollingerBands(close=close, window=20, window_dev=2)
             df_calc['BOLL_HIGH'] = bollinger.bollinger_hband()
             df_calc['BOLL_LOW'] = bollinger.bollinger_lband()
             
             # EMA
             df_calc['EMA12'] = EMAIndicator(close=close, window=12).ema_indicator()
             df_calc['EMA26'] = EMAIndicator(close=close, window=26).ema_indicator()
             
             # 只保留要求的最后30天
             df_subset = df_calc.tail(30)
             
        except Exception as tech_e:
             print(f"[Tools] Indicator calculation failed: {tech_e}")
        
        return df_subset.to_csv(index=False)
    except Exception as e:
        return f"获取历史数据失败: {e}"


@tool
def get_hk_stock_history(symbol: str, period: str = "daily") -> str:
    """获取港股历史行情数据（仅做最基础的日线抓取与返回）。

    Args:
        symbol: 港股代码（建议 5 位数字字符串，例如 "00700"）。
        period: 目前仅保证支持 "daily"。

    Returns:
        最近 30 条记录的 CSV 字符串。
    """
    try:
        symbol = _normalize_hk_symbol(symbol)
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        # 候选接口（按常见命名尝试）
        hist_func = None
        for name in (
            "stock_hk_hist",
            "stock_hk_hist_em",
            "stock_hk_daily",
        ):
            hist_func = getattr(ak, name, None)
            if callable(hist_func):
                break

        if not callable(hist_func):
            return "当前 AkShare 版本未提供港股历史行情接口。"

        # 尽量兼容参数差异
        try:
            df = hist_func(symbol=symbol, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
        except TypeError:
            try:
                df = hist_func(symbol=symbol, start_date=start_date, end_date=end_date)
            except TypeError:
                df = hist_func(symbol)

        if df is None or getattr(df, "empty", True):
            return f"未找到港股 {symbol} 的历史数据。"

        # 常见列名兼容：日期/开盘/收盘/最高/最低/成交量/涨跌幅
        date_col = _first_existing_column(df, ["日期", "date", "交易日期"])
        open_col = _first_existing_column(df, ["开盘", "open", "开盘价"])
        close_col = _first_existing_column(df, ["收盘", "close", "收盘价"])
        high_col = _first_existing_column(df, ["最高", "high", "最高价"])
        low_col = _first_existing_column(df, ["最低", "low", "最低价"])
        vol_col = _first_existing_column(df, ["成交量", "volume", "成交股数", "成交量(股)"])
        pct_col = _first_existing_column(df, ["涨跌幅", "pct_change", "涨跌幅(%)", "涨跌幅%"])

        cols = [c for c in [date_col, open_col, close_col, high_col, low_col, vol_col, pct_col] if c]
        if cols:
            return df[cols].tail(30).to_csv(index=False)
        return df.tail(30).to_csv(index=False)

    except Exception as e:
        return f"获取港股历史数据失败: {e}"

@tool
def get_stock_info(symbol: str) -> str:
    """
    获取股票的基本财务指标和公司信息（如市盈率、总市值等）。
    Args:
        symbol: 股票代码，例如 "600519"。
    """
    try:
        # 使用实时行情接口获取当前指标
        df = ak.stock_zh_a_spot_em()
        result = df[df['代码'] == symbol]
        
        if result.empty:
            return f"未找到股票 {symbol} 的实时信息。"
            
        data = result.iloc[0]
        info = {
            "代码": data['代码'],
            "名称": data['名称'],
            "最新价": data['最新价'],
            "涨跌幅": data['涨跌幅'],
            "成交量": data['成交量'],
            "成交额": data['成交额'],
            "市盈率(动态)": data.get('市盈率-动态', 'N/A'),
            "市净率": data.get('市净率', 'N/A'),
            "总市值": data.get('总市值', 'N/A')
        }
        return str(info)
    except Exception as e:
        return f"获取股票基本信息失败: {e}"


@tool
def get_hk_stock_info(symbol: str) -> str:
    """获取港股的基本信息（基于实时行情列表字段，尽力而为）。"""
    try:
        symbol = _normalize_hk_symbol(symbol)

        spot_func = None
        for name in ("stock_hk_spot_em", "stock_hk_spot"):
            spot_func = getattr(ak, name, None)
            if callable(spot_func):
                break

        if not callable(spot_func):
            return "当前 AkShare 版本未提供港股实时行情接口。"

        df = spot_func()
        if df is None or getattr(df, "empty", True):
            return "港股实时信息为空。"

        code_col = _first_existing_column(df, ["代码", "symbol", "股票代码"])
        name_col = _first_existing_column(df, ["名称", "name", "股票名称"])
        if not code_col:
            return f"港股行情数据列不匹配，无法解析代码字段。可用列: {list(df.columns)}"

        # 统一代码格式后再匹配
        df_code_norm = df[code_col].astype(str).map(_normalize_hk_symbol)
        result = df[df_code_norm == symbol]
        if result.empty:
            return f"未找到港股 {symbol} 的实时信息。"

        row = result.iloc[0]

        # 不同版本字段不一致：这里尽可能挑常见字段
        price_col = _first_existing_column(df, ["最新价", "现价", "price", "最新"])
        pct_col = _first_existing_column(df, ["涨跌幅", "涨跌幅(%)", "pct_change", "涨跌%"])
        amount_col = _first_existing_column(df, ["成交额", "amount", "成交金额"])
        vol_col = _first_existing_column(df, ["成交量", "volume", "成交股数"])

        info = {
            "代码": symbol,
            "名称": str(row[name_col]) if name_col else "N/A",
            "最新价": row[price_col] if price_col else "N/A",
            "涨跌幅": row[pct_col] if pct_col else "N/A",
            "成交量": row[vol_col] if vol_col else "N/A",
            "成交额": row[amount_col] if amount_col else "N/A",
        }

        return str(info)

    except Exception as e:
        return f"获取港股基本信息失败: {e}"

if __name__ == "__main__":
    # Test
    print(get_stock_code.invoke("茅台"))
