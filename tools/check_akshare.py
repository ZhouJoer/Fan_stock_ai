#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 akshare 与数据源（东方财富等）的连接是否有效。
用法：在项目根目录执行  python -m tools.check_akshare（推荐 Python 3.10）
"""
import sys
import os
from datetime import datetime, timedelta

def _env(name):
    v = os.environ.get(name) or os.environ.get(name.upper())
    return v or "(未设置)"

def main():
    print("=" * 60)
    print("akshare 连接检查")
    print("=" * 60)

    # 1. 环境
    print("\n[1] 环境")
    print("  Python:", sys.version.split()[0])
    print("  HTTP_PROXY:", _env("HTTP_PROXY"))
    print("  HTTPS_PROXY:", _env("HTTPS_PROXY"))
    print("  NO_PROXY:", _env("NO_PROXY"))
    if _env("HTTP_PROXY") != "(未设置)" or _env("HTTPS_PROXY") != "(未设置)":
        print("  提示: 若代理不可达或代理阻止东方财富，可临时取消代理再试：")
        print("    set HTTP_PROXY=  && set HTTPS_PROXY=  (Windows)")
        print("    unset HTTP_PROXY HTTPS_PROXY  (Linux/Mac)")

    # 2. 基础网络（可选）
    print("\n[2] 基础网络（测试能否访问公网）")
    try:
        import urllib.request
        req = urllib.request.Request("https://www.baidu.com", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            print("  访问 https://www.baidu.com -> 状态", r.status)
    except Exception as e:
        print("  访问公网失败:", type(e).__name__, str(e)[:80])
        print("  可能原因: 本机网络不通、代理错误、防火墙拦截。请先解决基础网络再测 akshare。")
    else:
        print("  基础网络正常。")

    # 3. akshare 版本与导入
    print("\n[3] akshare")
    try:
        import akshare as ak
        try:
            ver = ak.__version__
        except AttributeError:
            ver = "未知"
        print("  版本:", ver)
    except ImportError as e:
        print("  导入失败:", e)
        print("  请执行: pip install akshare")
        return 1

    # 4. 实际请求（与回测相同接口：stock_zh_a_hist）
    print("\n[4] 数据源请求（东方财富 A 股日线，与回测一致）")
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
    symbol = "000001"  # 平安银行，常用测试代码
    print("  请求: stock_zh_a_hist(symbol=%s, start_date=%s, end_date=%s)" % (symbol, start_date, end_date))
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            print("  结果: 成功，获取 %d 条数据" % len(df))
            print("  示例日期:", df["日期"].iloc[-1] if "日期" in df.columns else "N/A")
        else:
            print("  结果: 返回为空（数据源可能无数据或限流）")
    except Exception as e:
        print("  结果: 失败")
        print("  异常类型:", type(e).__name__)
        print("  异常信息:", str(e)[:200])
        if "Connection" in str(e) or "Remote" in str(e) or "timeout" in str(e).lower():
            print("\n  可能原因与建议:")
            print("  1. 东方财富限流/封禁: 稍后重试；或浏览器打开 https://www.eastmoney.com/ 登录后再运行本脚本。")
            print("  2. 代理/防火墙: 若使用了 HTTP_PROXY/HTTPS_PROXY，尝试取消后再试。")
            print("  3. 升级 akshare: pip install akshare --upgrade")
            print("  4. 换网络: 换手机热点或更换网络环境。")
        return 1

    print("\n" + "=" * 60)
    print("akshare 连接检查通过，可以正常拉取数据。")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
