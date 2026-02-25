"""CNINFO 数据源工具（A 股定期报告/公告）。

目标：参考附件 sources.py 的 CNINFO 调用方式，但在本项目中保持依赖最小化：
- 仅使用 Python 标准库进行 HTTP 请求（urllib）
- 返回适合 LLM 消费的结构化文本/JSON 字符串
- 自动将获取的公告存入知识库供语义检索

注意：
- CNINFO 接口为公开接口，偶尔会有频控/字段变更；本工具尽量做容错与降级。
- 本工具只覆盖 A 股（深市/沪市），不覆盖港股。
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

from langchain_core.tools import tool


_CNINFO_SEARCH_URL = "http://www.cninfo.com.cn/new/information/topSearch/query"
_CNINFO_HIS_ANNOUNCEMENT_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
_STATIC_PREFIX = "http://static.cninfo.com.cn/"


def _normalize_stock_code(stock_code: str) -> str:
    code = re.sub(r"[^\d]", "", stock_code or "")
    if len(code) < 6:
        return code.zfill(6)
    return code[:6]


def _extract_quarter_from_title(title: str) -> Optional[int]:
    if not title:
        return None

    patterns = [
        (r"第一季|一季|1季|Q1", 1),
        (r"第二季|二季|2季|Q2|半年度|中期", 2),
        (r"第三季|三季|3季|Q3", 3),
        (r"第四季|四季|4季|Q4|年度报告|年报", 4),
    ]

    for pattern, q in patterns:
        if re.search(pattern, title, re.IGNORECASE):
            return q
    return None


def _post_form(url: str, data: dict[str, Any], headers: dict[str, str]) -> Any:
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def _default_headers() -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "http://www.cninfo.com.cn",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search&lastPage=index",
    }


def _get_org_id(stock_code: str) -> Optional[str]:
    headers = _default_headers()
    data = {"keyWord": stock_code}
    try:
        result = _post_form(_CNINFO_SEARCH_URL, data=data, headers=headers)
        if not result:
            return None
        for item in result:
            if str(item.get("code")) == stock_code:
                return item.get("orgId")
        # 没找到精确匹配则用第一条
        return result[0].get("orgId")
    except Exception:
        return None


def _get_market_params(stock_code: str) -> tuple[str, str, str]:
    """返回 (column, plate, market_name)"""
    if stock_code.startswith(("000", "002", "300")):
        return "szse", "sz", "SZSE"
    return "sse", "sh", "SSE"


def _category_for_report_type(report_type: str) -> str:
    mapping = {
        "annual": "category_ndbg_szsh",
        "semi-annual": "category_bndbg_szsh",
        "quarterly": "category_sjdbg_szsh",
    }
    return mapping.get(report_type, "category_ndbg_szsh")


@tool
def fetch_cninfo_filings(
    stock_code: str,
    report_type: str = "annual",
    year: Optional[int] = None,
    quarter: Optional[int] = None,
    limit: int = 10,
) -> str:
    """从 CNINFO 获取 A 股定期报告/公告（返回 PDF 链接与关键信息）。

    Args:
        stock_code: 6 位股票代码，例如 600519。
        report_type: 仅支持英文枚举："annual" | "semi-annual" | "quarterly"。
        year: 年份过滤（按公告发布时间窗口粗略过滤）。不填则默认近 3 年。
        quarter: 季度过滤（1-4），仅当 report_type="quarterly" 时生效。
        limit: 最多返回条数。

    Returns:
        JSON 字符串：[{"title":..., "date":..., "pdf_url":..., "company":..., "code":...}, ...]

    说明：
        CNINFO 返回的字段可能随时间变化，本工具会尽量兼容；若失败会返回可读的错误文本。
    """

    try:
        stock_code = _normalize_stock_code(stock_code)
        if report_type not in ("annual", "semi-annual", "quarterly"):
            return "report_type 仅支持：annual / semi-annual / quarterly（必须英文）。"
        if quarter is not None and report_type != "quarterly":
            return "quarter 参数仅在 report_type=quarterly 时可用。"
        if quarter is not None and (quarter < 1 or quarter > 4):
            return "quarter 必须在 1-4 之间。"

        org_id = _get_org_id(stock_code)
        if not org_id:
            return f"未能从 CNINFO 查询到 orgId（stock_code={stock_code}）。"

        column, plate, market_name = _get_market_params(stock_code)
        category = _category_for_report_type(report_type)

        headers = _default_headers()

        current_year = datetime.now().year
        years = [year] if year else [current_year, current_year - 1, current_year - 2]

        results: list[dict[str, Any]] = []

        for y in years:
            if len(results) >= limit:
                break

            # CNINFO 用 seDate 控制查询窗口
            start_date = f"{y}-01-01"
            end_date = f"{y + 1}-01-01"
            se_date = f"{start_date}~{end_date}"

            form = {
                "pageNum": "1",
                "pageSize": "30",
                "column": column,
                "tabName": "fulltext",
                "plate": plate,
                "stock": f"{stock_code},{org_id}",
                "searchkey": "",
                "secid": "",
                "category": f"{category};",
                "trade": "",
                "seDate": se_date,
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            }

            payload = _post_form(_CNINFO_HIS_ANNOUNCEMENT_URL, data=form, headers=headers)
            announcements = (payload or {}).get("announcements") or []

            for a in announcements:
                if len(results) >= limit:
                    break

                title = str(a.get("announcementTitle", ""))
                if report_type == "quarterly" and quarter is not None:
                    q = _extract_quarter_from_title(title)
                    if q is not None and q != quarter:
                        continue

                adjunct_url = a.get("adjunctUrl") or ""
                pdf_url = f"{_STATIC_PREFIX}{adjunct_url}" if adjunct_url else ""
                
                # 处理日期：将时间戳转为可读日期格式
                date_raw = a.get("announcementTime", "")
                date_str = ""
                try:
                    if date_raw:
                        # CNINFO返回的是毫秒时间戳
                        timestamp = int(date_raw) / 1000 if len(str(date_raw)) > 10 else int(date_raw)
                        from datetime import datetime as dt
                        date_str = dt.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                except:
                    date_str = str(date_raw)[:10]

                results.append(
                    {
                        "code": str(a.get("secCode", stock_code)),
                        "company": str(a.get("secName", "")),
                        "market": market_name,
                        "title": title,
                        "date": date_str,
                        "pdf_url": pdf_url,
                    }
                )

        # 自动将前几条结果存入知识库（后台线程）
        if results:
            try:
                from graph.agents.research_agent.knowledge import add_cninfo_filing_to_knowledge
                
                logger.info(f"Adding up to 3 CNINFO filings to knowledge base...")
                # 在后台线程处理，不阻塞工具返回
                import threading
                
                def _background_kb_save():
                    for filing in results[:3]:
                        try:
                            add_cninfo_filing_to_knowledge(filing)
                        except Exception as e:
                            logger.warning(f"Failed to save filing to KB: {e}")
                
                threading.Thread(target=_background_kb_save, daemon=True).start()
                    
            except Exception as kb_err:
                logger.warning(f"Failed to add CNINFO results to knowledge base: {kb_err}")

        return json.dumps(results, ensure_ascii=False)

    except Exception as e:
        return f"CNINFO 获取失败: {e}"
