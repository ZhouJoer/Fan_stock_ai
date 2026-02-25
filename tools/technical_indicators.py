"""
技术指标计算模块
提供各种技术分析指标的计算功能
"""
import pandas as pd
import numpy as np


def calculate_technical_indicators(stock_data):
    """
    计算技术指标
    
    参数：
        stock_data: 股票历史数据（列表或DataFrame）
    
    返回：
        包含各种技术指标的字典
    """
    df = pd.DataFrame(stock_data) if not isinstance(stock_data, pd.DataFrame) else stock_data
    indicators = {}
    
    # 移动平均线
    indicators['MA5'] = df['close'].rolling(window=5).mean().iloc[-1]
    indicators['MA10'] = df['close'].rolling(window=10).mean().iloc[-1]
    indicators['MA20'] = df['close'].rolling(window=20).mean().iloc[-1]
    indicators['MA60'] = df['close'].rolling(window=60).mean().iloc[-1] if len(df) >= 60 else df['close'].mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['RSI'] = (100 - (100 / (1 + rs))).iloc[-1]
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    indicators['MACD'] = macd.iloc[-1]
    indicators['MACD_Signal'] = signal.iloc[-1]
    indicators['MACD_Hist'] = (macd - signal).iloc[-1]
    
    # 布林带
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    indicators['BB_Upper'] = (bb_middle + 2 * bb_std).iloc[-1]
    indicators['BB_Middle'] = bb_middle.iloc[-1]
    indicators['BB_Lower'] = (bb_middle - 2 * bb_std).iloc[-1]
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    indicators['ATR'] = tr.rolling(window=14).mean().iloc[-1]
    
    # 当前价格
    indicators['Current_Price'] = df['close'].iloc[-1]
    
    return indicators


def add_technical_indicators_to_df(df):
    """
    为 DataFrame 添加所有技术指标列
    
    参数：
        df: 包含 OHLCV 数据的 DataFrame
    
    返回：
        添加了技术指标列的 DataFrame
    """
    # 移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 布林带
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    return df
