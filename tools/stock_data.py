"""
股票数据获取模块
提供股票数据的获取、缓存和转换功能
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from db.manager import save_daily_data, get_cached_history, get_latest_date, count_cached_data
from tools.stock_name_code_map import get_code_by_name, set_mapping


def get_stock_code_from_name(stock_input):
    """
    将股票名称转换为股票代码
    
    支持：
    - 纯数字代码：600875 -> 600875
    - 带后缀代码：600875.SH -> 600875
    - 股票名称：东方电气 -> 600875
    
    参数：
        stock_input: 股票代码或名称
    
    返回：
        股票代码（纯数字）
    """
    # 如果是纯数字或带后缀的代码，直接提取
    stock_input = str(stock_input).strip()
    
    # 提取纯数字代码
    if '.' in stock_input:
        code = stock_input.split('.')[0]
        if code.isdigit():
            return code
    
    if stock_input.isdigit():
        return stock_input
    
    # 优先查本地名称-代码映射表
    code = get_code_by_name(stock_input, "a_share")
    if code:
        print(f"[Data] 本地表命中: {stock_input} -> {code}")
        return code

    # 否则尝试通过股票名称查找代码
    try:
        print(f"[Data] 正在查找股票名称 '{stock_input}' 对应的代码...")
        # 获取A股股票列表
        stock_list = ak.stock_zh_a_spot_em()
        
        # 精确匹配股票名称
        match = stock_list[stock_list['名称'] == stock_input]
        if not match.empty:
            code = match.iloc[0]['代码']
            print(f"[Data] 找到股票: {stock_input} -> {code}")
            set_mapping(stock_input, code, "a_share")
            return code
        
        # 模糊匹配（包含）
        match = stock_list[stock_list['名称'].str.contains(stock_input, na=False)]
        if not match.empty:
            code = match.iloc[0]['代码']
            name = match.iloc[0]['名称']
            print(f"[Data] 模糊匹配到股票: {stock_input} -> {name}({code})")
            set_mapping(name, code, "a_share")
            return code
        
        print(f"[Data] 未找到股票 '{stock_input}'，将尝试直接使用")
        return stock_input
        
    except Exception as e:
        print(f"[Data] 查找股票代码失败: {e}")
        return stock_input


def _convert_cache_to_list(df_cache):
    """将缓存的 DataFrame 转换为标准列表格式"""
    stock_data = []
    for _, row in df_cache.iterrows():
        stock_data.append({
            'date': pd.to_datetime(row['日期']),
            'open': float(row['开盘']),
            'high': float(row['最高']),
            'low': float(row['最低']),
            'close': float(row['收盘']),
            'volume': int(row['成交量']) if pd.notna(row['成交量']) else 0
        })
    
    if stock_data:
        print(f"[Data] 成功获取 {len(stock_data)} 条数据，最新收盘价：¥{stock_data[-1]['close']:.2f}")
    return stock_data


def _is_etf_code(code: str) -> bool:
    """
    判断是否为ETF代码
    
    ETF代码特征：
    - 上交所ETF：51开头（如510300、510500）
    - 深交所ETF：159开头（如159915、159919）
    """
    code_str = str(code).strip()
    if code_str.startswith('51') and len(code_str) == 6:
        return True
    if code_str.startswith('159') and len(code_str) == 6:
        return True
    return False


def _get_etf_market_suffix(code: str) -> str:
    """
    获取ETF的市场后缀
    
    参数：
        code: ETF代码
    
    返回：
        ".SH" 或 ".SZ"
    """
    code_str = str(code).strip()
    if code_str.startswith('51'):
        return '.SH'
    elif code_str.startswith('159'):
        return '.SZ'
    else:
        # 默认尝试上交所
        return '.SH'


def fetch_stock_data(stock_code, days=120, use_cache=True):
    """
    从 akshare 获取真实股票数据，支持数据库缓存
    
    参数：
        stock_code: 股票代码或名称，如 "600000.SH"、"600000" 或 "东方电气"
        days: 需要获取的「交易日」数量（K 线根数），非日历天
        use_cache: 是否使用缓存
    
    返回：
        包含 OHLCV 数据的列表，失败时抛出异常
    """
    # 将股票名称转换为代码
    symbol = get_stock_code_from_name(stock_code)
    
    # 判断是否为ETF，ETF需要特殊处理
    is_etf = _is_etf_code(symbol)
    
    try:
        need_fetch = True
        
        if use_cache:
            # 先检查缓存：条数足够则优先使用缓存，不再请求网络（与因子挖掘等模块一致，减少网络失败）
            latest_db_date = get_latest_date(symbol)
            cached_count = count_cached_data(symbol)
            if latest_db_date and cached_count >= days:
                try:
                    latest_dt = datetime.strptime(str(latest_db_date)[:10], "%Y-%m-%d").date()
                    today = datetime.now().date()
                    days_ago = (today - latest_dt).days
                    # 最新一条在 15 个日历日内视为可用缓存（与因子挖掘一致：周末/节假日/限流时少打网络）
                    if days_ago <= 15:
                        need_fetch = False
                        print(f"[Data] ✓ 使用缓存 {symbol}（最新：{latest_db_date}，共{cached_count}条）")
                except Exception:
                    pass
            if need_fetch and latest_db_date and cached_count < days:
                print(f"[Data] 缓存数据不足 {symbol}，需要{days}条，缓存{cached_count}条，将请求网络")
        
        if need_fetch:
            # 日期范围：入参 days 为「交易日」数，换算为日历天（252 交易日/年 ≈ 365 天）
            # 使用 1.5 倍 + 60 天缓冲，避免数据源单次条数限制或区间边界导致不足
            calendar_days = int(days * 1.5) + 60
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=calendar_days)).strftime("%Y%m%d")
            
            print(f"[Data] 从 akshare 获取 {symbol}（{days}交易日≈{calendar_days}日历天，{'ETF' if is_etf else '股票'}）...")
            
            df = None
            
            # ETF需要使用专用接口（网络不稳定时重试）
            if is_etf:
                print(f"[Data] 检测到ETF代码，使用ETF专用接口...")
                
                df = None
                last_error = None
                _max_etf_retries = 2  # 连接断开时重试次数
                
                for _etf_retry in range(_max_etf_retries):
                    if df is not None and not df.empty:
                        break
                    if _etf_retry > 0:
                        import time
                        time.sleep(2)  # 重试前等待 2 秒
                        print(f"[Data] ETF 数据获取重试 {_etf_retry}/{_max_etf_retries - 1}...")
                
                    # 方法1: 尝试 fund_etf_hist_em（东方财富源，推荐）
                    try:
                        if hasattr(ak, 'fund_etf_hist_em'):
                            df = ak.fund_etf_hist_em(
                                symbol=symbol,
                                period="daily",
                                start_date=start_date,
                                end_date=end_date,
                                adjust="qfq"
                            )
                            if df is not None and not df.empty:
                                print(f"[Data] 使用fund_etf_hist_em接口获取ETF数据成功，{len(df)}条")
                                break
                    except Exception as e1:
                        last_error = e1
                        print(f"[Data] fund_etf_hist_em接口失败: {e1}")
                
                # 方法2: 如果方法1失败，尝试新浪源（需要市场前缀）
                if df is None or df.empty:
                    try:
                        market_suffix = _get_etf_market_suffix(symbol)
                        if market_suffix == '.SH':
                            symbol_with_prefix = f"sh{symbol}"
                        else:
                            symbol_with_prefix = f"sz{symbol}"
                        
                        if hasattr(ak, 'fund_etf_hist_sina'):
                            # akshare 部分版本 fund_etf_hist_sina 无 period 参数，先试无 period
                            try:
                                df_sina = ak.fund_etf_hist_sina(symbol=symbol_with_prefix, period="daily")
                            except TypeError:
                                df_sina = ak.fund_etf_hist_sina(symbol=symbol_with_prefix)
                            if df_sina is not None and not df_sina.empty:
                                # 转换列名以匹配标准格式
                                column_mapping = {
                                    'date': '日期',
                                    'close': '收盘',
                                    'open': '开盘',
                                    'high': '最高',
                                    'low': '最低',
                                    'volume': '成交量'
                                }
                                for old_col, new_col in column_mapping.items():
                                    if old_col in df_sina.columns:
                                        df_sina = df_sina.rename(columns={old_col: new_col})
                                # 过滤日期范围
                                if '日期' in df_sina.columns:
                                    df_sina['日期'] = pd.to_datetime(df_sina['日期'])
                                    start_dt = pd.to_datetime(start_date)
                                    end_dt = pd.to_datetime(end_date)
                                    df_sina = df_sina[(df_sina['日期'] >= start_dt) & 
                                                      (df_sina['日期'] <= end_dt)]
                                df = df_sina
                                print(f"[Data] 使用fund_etf_hist_sina接口获取ETF数据成功，{len(df)}条")
                    except Exception as e2:
                        last_error = e2
                        print(f"[Data] fund_etf_hist_sina接口失败: {e2}")
                
                # 方法3: 如果前两个都失败，尝试普通股票接口（某些ETF可能也能用）
                if df is None or df.empty:
                    try:
                        df = ak.stock_zh_a_hist(
                            symbol=symbol, 
                            period="daily", 
                            start_date=start_date, 
                            end_date=end_date, 
                            adjust="qfq"
                        )
                        if df is not None and not df.empty:
                            print(f"[Data] 使用stock_zh_a_hist接口获取ETF数据成功，{len(df)}条")
                    except Exception as e3:
                        last_error = e3
                        print(f"[Data] stock_zh_a_hist接口也失败: {e3}")
                
                # 如果所有方法都失败
                if df is None or df.empty:
                    raise ValueError(f"所有ETF接口都失败，最后错误: {last_error}")
            else:
                # 普通股票 - 添加重试机制（指数退避）
                max_retries = 3
                base_delay = 3  # 基础延迟3秒
                df = None
                last_error = None
                
                for retry in range(max_retries):
                    try:
                        # 在重试前添加延迟（指数退避：3秒、6秒、12秒）
                        if retry > 0:
                            delay = base_delay * (2 ** (retry - 1))
                            print(f"[Data] 等待 {delay} 秒后重试...")
                            import time
                            time.sleep(delay)
                        
                        df = ak.stock_zh_a_hist(
                            symbol=symbol, 
                            period="daily", 
                            start_date=start_date, 
                            end_date=end_date, 
                            adjust="qfq"  # 前复权
                        )
                        if df is not None and not df.empty:
                            print(f"[Data] ✓ {symbol} 获取成功（第{retry+1}次尝试）")
                            # 便于排查不足：打印本次返回条数与日期区间
                            _date_col = "日期" if "日期" in df.columns else (df.columns[0] if len(df.columns) else None)
                            if _date_col is not None:
                                _min_d = df[_date_col].min()
                                _max_d = df[_date_col].max()
                                print(f"[Data] 本次返回 {len(df)} 条，区间 {_min_d} 至 {_max_d}")
                            # 若数据源单次返回条数不足（常见为接口约 300 条上限），用更早区间补拉一次并合并
                            if len(df) < days and _date_col is not None:
                                try:
                                    _min_dt = pd.to_datetime(df[_date_col].min())
                                    _start_earlier = (_min_dt - timedelta(days=min(400, (days - len(df)) * 2))).strftime("%Y%m%d")
                                    _end_earlier = (_min_dt - timedelta(days=1)).strftime("%Y%m%d")
                                    df2 = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=_start_earlier, end_date=_end_earlier, adjust="qfq")
                                    if df2 is not None and not df2.empty:
                                        df = pd.concat([df2, df], ignore_index=True)
                                        df = df.drop_duplicates(subset=[_date_col], keep="last").sort_values(_date_col).reset_index(drop=True)
                                        print(f"[Data] 补充拉取更早区间后共 {len(df)} 条")
                                except Exception as _e:
                                    print(f"[Data] 补充拉取更早区间失败（可忽略）: {_e}")
                            break  # 成功获取数据，退出重试循环
                        else:
                            last_error = ValueError("返回数据为空")
                    except Exception as e:
                        last_error = e
                        error_msg = str(e)
                        # 判断错误类型
                        if 'Connection' in error_msg or 'Remote' in error_msg:
                            error_type = "网络连接错误"
                        elif 'timeout' in error_msg.lower():
                            error_type = "请求超时"
                        elif '限流' in error_msg or 'rate limit' in error_msg.lower():
                            error_type = "请求限流"
                        else:
                            error_type = "未知错误"
                        
                        if retry < max_retries - 1:
                            print(f"[Data] ⚠ 第{retry+1}次获取 {symbol} 失败（{error_type}），将重试...")
                        else:
                            print(f"[Data] ✗ {symbol} 重试{max_retries}次后仍失败（{error_type}）")
                
                # 如果所有重试都失败，尝试使用缓存
                if df is None or df.empty:
                    print(f"[Data] 尝试使用缓存数据作为降级方案...")
                    df_cache = get_cached_history(symbol, limit=days)
                    if not df_cache.empty:
                        cache_count = len(df_cache)
                        print(f"[Data] ✓ 使用缓存数据 {symbol}（{cache_count}条，可能需要{days}条）")
                        if cache_count >= 60:  # 至少60条数据才能使用
                            return _convert_cache_to_list(df_cache)
                        else:
                            print(f"[Data] ⚠ 缓存数据不足（{cache_count}条 < 60条），无法使用")
                    
                    # 如果缓存也不可用，抛出错误
                    error_detail = f"获取股票 {symbol} 数据失败（重试{max_retries}次）"
                    if last_error:
                        error_detail += f": {str(last_error)[:100]}"
                    error_detail += "\n可能原因：1) 网络连接不稳定 2) 数据源限流 3) 数据源服务器暂时不可用"
                    error_detail += "\n建议：1) 检查网络连接 2) 稍后重试 3) 使用缓存数据（如果可用）"
                    raise ValueError(error_detail)
            
            if df is None or df.empty:
                print(f"[Data] 警告：未获取到 {symbol} 的数据")
                # 尝试从缓存读取
                df_cache = get_cached_history(symbol, limit=days)
                if not df_cache.empty:
                    print(f"[Data] 使用缓存数据")
                    return _convert_cache_to_list(df_cache)
                raise ValueError(f"无法获取{'ETF' if is_etf else '股票'} {symbol} 的数据")
            
            # 保存到数据库
            save_daily_data(symbol, df)
            print(f"[Data] 数据已缓存到数据库，共 {len(df)} 条")
        
        # 从缓存读取数据
        df_cache = get_cached_history(symbol, limit=days)
        if df_cache.empty:
            raise ValueError(f"缓存中无{'ETF' if is_etf else '股票'} {symbol} 的数据")
        
        # 检查获取到的数据量
        actual_count = len(df_cache)
        if actual_count < days:
            print(f"[Data] 警告：{symbol} 只获取到 {actual_count} 条数据，少于请求的 {days} 条")
            print(f"[Data] 可能原因：1) 该股票在请求区间内上市较晚、可交易天数不足；2) 数据源单次返回条数限制（如约 300 条）；3) 部分日期因缺失/异常被过滤。已全部拉取并缓存。")
            
        return _convert_cache_to_list(df_cache)
        
    except Exception as e:
        error_msg = str(e)
        print(f"[Data] 获取真实数据失败: {e}")
        
        # 尝试从缓存读取（即使数据不足也使用）
        try:
            df_cache = get_cached_history(symbol, limit=days)
            if not df_cache.empty:
                cache_count = len(df_cache)
                print(f"[Data] 网络失败，使用缓存数据（{cache_count}条，可能需要{days}条）")
                if cache_count >= 60:  # 至少60条数据才能使用
                    return _convert_cache_to_list(df_cache)
                else:
                    print(f"[Data] 警告：缓存数据不足（{cache_count}条 < 60条），无法使用")
        except Exception as cache_error:
            print(f"[Data] 读取缓存也失败: {cache_error}")
        
        # 如果是网络连接错误，提供更友好的提示
        if 'Connection' in error_msg or 'Remote' in error_msg or 'timeout' in error_msg.lower():
            raise ValueError(f"获取{'ETF' if is_etf else '股票'} {stock_code} 数据失败: 网络连接错误（可能是网络不稳定或数据源限流）。请稍后重试，或检查网络连接。")
        else:
            raise ValueError(f"获取{'ETF' if is_etf else '股票'} {stock_code} 数据失败: {e}")


def get_latest_close(stock_code, use_cache=True):
    """
    获取股票最新收盘价（与因子挖掘等模块一致：优先缓存，再拉取，失败时用旧缓存兜底）。
    用于模拟仓、账户市值等仅需最新价的场景，不要求最少 60 条。

    参数：
        stock_code: 股票代码或名称
        use_cache: 是否使用缓存

    返回：
        float: 最新收盘价

    异常：
        ValueError: 无法获取任何数据时
    """
    symbol = get_stock_code_from_name(stock_code)
    # 1) 优先用缓存（任意条数均可）
    if use_cache:
        cached = get_cached_history(symbol, limit=1)
        if not cached.empty:
            return float(cached.iloc[-1]["收盘"])
    # 2) 拉取数据（与因子挖掘相同：fetch_stock_data 含重试与缓存写入）
    try:
        data = fetch_stock_data(stock_code, days=60, use_cache=use_cache)
        if data and len(data) > 0:
            return float(data[-1]["close"])
    except Exception:
        pass
    # 3) 拉取失败时用旧缓存兜底
    if use_cache:
        cached = get_cached_history(symbol, limit=500)
        if not cached.empty:
            return float(cached.iloc[-1]["收盘"])
    raise ValueError(f"无法获取 {stock_code} 的最新价格（网络失败且无缓存）")


def get_stock_data(stock_code, days=120, use_cache=True):
    """
    获取股票数据（兼容旧接口）
    
    参数：
        stock_code: 股票代码或名称
        days: 需要获取的「交易日」数量（K 线根数），内部会换算为日历天请求
        use_cache: 是否使用缓存
    
    返回：
        包含 OHLCV 数据的列表
    
    异常：
        ValueError: 当数据获取失败时
    """
    data = fetch_stock_data(stock_code, days, use_cache)
    
    if data is None or len(data) < 60:
        raise ValueError(f"获取 {stock_code} 数据不足，需要至少60条，获取到 {len(data) if data else 0} 条")
    
    return data


def get_stock_data_by_level(stock_code, days=120, level="daily", use_cache=True):
    """
    获取不同周期数据（daily/weekly/monthly）。

    说明：
    - daily 直接复用 get_stock_data 逻辑。
    - weekly/monthly 优先读本地聚合缓存；
      若缓存不足，则先补拉日线并触发聚合后再读取。
    """
    lv = str(level or "daily").strip().lower()
    if lv == "daily":
        return get_stock_data(stock_code, days=days, use_cache=use_cache)

    if lv not in ("weekly", "monthly"):
        raise ValueError(f"不支持的 level: {level}")

    symbol = get_stock_code_from_name(stock_code)
    if use_cache:
        cached = get_cached_history(symbol, limit=days, level=lv)
        if not cached.empty and len(cached) >= max(10, int(days * 0.6)):
            return _convert_cache_to_list(cached)

    # 周/月数据由日线聚合得到，先补齐日线缓存
    daily_days = max(240, int(days) * 25)
    fetch_stock_data(symbol, days=daily_days, use_cache=use_cache)
    cached = get_cached_history(symbol, limit=days, level=lv)
    if cached.empty:
        raise ValueError(f"获取 {stock_code} 的 {lv} 数据失败")
    return _convert_cache_to_list(cached)


def get_market_context_for_llm(days: int = 5, end_date: str = None) -> dict:
    """
    获取A股市场环境摘要，供LLM做趋势判断使用。
    包含：北向资金近期流向、大盘/成交量概况。
    
    参数:
        days: 取最近几个交易日
        end_date: 截止日期 YYYY-MM-DD，None 表示今天（用于回测时传入当日）
    
    返回:
        {"northbound_text": "北向资金描述", "volume_text": "成交量描述", "summary": "整体描述"}
        获取失败时返回空 dict，调用方可不传 market_context。
    """
    out = {}
    try:
        # 北向资金：仅连接成功且拿到数据时才写入，失败则不注入市场和资金数据
        if hasattr(ak, 'stock_hsgt_hist_em'):
            df_nb = ak.stock_hsgt_hist_em(symbol="北向资金")
            if df_nb is not None and not df_nb.empty and '日期' in df_nb.columns:
                df_nb['日期'] = pd.to_datetime(df_nb['日期']).dt.strftime('%Y-%m-%d')
                if end_date:
                    df_nb = df_nb[df_nb['日期'] <= end_date]
                df_nb = df_nb.tail(days)
                if not df_nb.empty:
                    flow_col = '当日成交净买额' if '当日成交净买额' in df_nb.columns else ('当日资金流入' if '当日资金流入' in df_nb.columns else None)
                    if flow_col:
                        total = df_nb[flow_col].sum()
                        out['northbound_text'] = f"近{days}日北向资金净流入合计: {total:.1f}亿元"
                    else:
                        out['northbound_text'] = f"近{days}日北向资金数据已获取，请结合日期与净流入列判断流向"
    except Exception as e:
        print(f"[Data] 北向资金获取失败: {e}")
    try:
        # 上证指数：优先易连的数据源（新浪/腾讯），东财易被限流放最后；每源只试一次，成功即停
        df_idx = None
        end_d = datetime.now() if not end_date else datetime.strptime(end_date[:10], "%Y-%m-%d")
        start_d = (end_d - timedelta(days=days + 30)).strftime("%Y%m%d")
        end_d_str = end_d.strftime("%Y%m%d")
        sources = [
            ("stock_zh_index_daily", lambda: ak.stock_zh_index_daily(symbol="sh000001") if hasattr(ak, 'stock_zh_index_daily') else None),
            ("stock_zh_index_daily_tx", lambda: ak.stock_zh_index_daily_tx(symbol="sh000001") if hasattr(ak, 'stock_zh_index_daily_tx') else None),
            ("index_zh_a_hist", lambda: ak.index_zh_a_hist(symbol="000001", period="daily", start_date=start_d, end_date=end_d_str) if hasattr(ak, 'index_zh_a_hist') else None),
            ("stock_zh_index_daily_em", lambda: ak.stock_zh_index_daily_em(symbol="sh000001") if hasattr(ak, 'stock_zh_index_daily_em') else None),
        ]
        last_err = None
        for name, fetch in sources:
            if df_idx is not None and not df_idx.empty:
                break
            try:
                data = fetch()
                if data is not None and not data.empty:
                    df_idx = data
                    break
            except Exception as e:
                last_err = e
        if (df_idx is None or df_idx.empty) and last_err is not None:
            print(f"[Data] 大盘多源均失败: {last_err}")
        # 仅连接成功且拿到数据时才写入大盘描述，失败则不注入市场和资金数据
        if df_idx is not None and not df_idx.empty:
            date_col = next((c for c in df_idx.columns if '日期' in str(c) or c == 'date'), df_idx.columns[0])
            if end_date:
                df_idx['_dt'] = pd.to_datetime(df_idx[date_col]).dt.strftime('%Y-%m-%d')
                df_idx = df_idx[df_idx['_dt'] <= end_date].drop(columns=['_dt'], errors='ignore')
            df_idx = df_idx.tail(days)
            if not df_idx.empty:
                close_col = next((c for c in df_idx.columns if '收' in str(c) or c == 'close'), None)
                vol_col = next((c for c in df_idx.columns if '量' in str(c) or c == 'volume'), None)
                if close_col and vol_col:
                    last_close = float(df_idx[close_col].iloc[-1])
                    first_close = float(df_idx[close_col].iloc[0])
                    pct = (last_close / first_close - 1) * 100 if first_close else 0
                    vol_sum = float(df_idx[vol_col].sum())
                    out['volume_text'] = f"上证指数近{days}日涨跌: {pct:+.2f}%，区间成交总量: {vol_sum/1e8:.1f}亿"
                else:
                    out['volume_text'] = f"上证指数近{days}日数据已获取"
    except Exception as e:
        print(f"[Data] 大盘成交量获取失败: {e}")
    # 仅用「有效」描述生成 summary，不含「异常/暂不可用/缺失」等，避免 LLM 复述
    valid_parts = [
        v for k, v in out.items()
        if k != 'summary' and isinstance(v, str)
        and '异常' not in v and '暂不可用' not in v and '数据缺失' not in v and '暂无' not in v
    ]
    if valid_parts:
        out['summary'] = "；".join(valid_parts)
    else:
        out.pop('summary', None)
    return out
